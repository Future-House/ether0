import re
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from functools import wraps
from itertools import starmap
from typing import Any, ParamSpec, TypeVar, cast

from ether0.model_prompts import (
    ANSWER_END,
    ANSWER_START,
    THINK_END,
    THINK_START,
    ProblemPrompt,
    SysPrompt,
    extract_answer_loose,
)
from ether0.rewards import accuracy_reward, format_reward

P = ParamSpec("P")
R = TypeVar("R")


def wrap_reward_func(func: Callable[P, R], **wrap_kwargs: Any) -> Callable[P, R]:
    @wraps(func)  # needed by GRPOTrainer for logging
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(*args, **wrap_kwargs, **kwargs)

    return wrapped


@dataclass
class ChatArguments:
    """Arguments for making a chat conversation for SFT or RL training."""

    sys_prompt: SysPrompt | None = field(
        default=None,
        metadata={
            "help": (
                "If provided, use this system prompt. If not provided, the chat"
                " template may inject one."
            )
        },
    )

    problem_prompt: ProblemPrompt = field(
        default=ProblemPrompt.NONE,
        metadata={
            "help": (
                "Prompt to put before the problem in the first user message, relevant"
                " for both RL or SFT. Make sure this matches between SFT and RL, so if"
                " the SFT'd model wasn't passed this during SFT, don't pass this to RL."
            )
        },
    )

    reasoning: bool = field(
        default=True,
        metadata={
            "help": (
                "If True (default), it is assumed that the model's response contains"
                f" reasoning enclosed in `{THINK_START}` and `{THINK_END}`."
            )
        },
    )

    def make_rl_conversation(
        self, row: MutableMapping[str, str | list[str]]
    ) -> dict[str, list[dict] | list[list[dict]]]:
        """Format a dataset row into a chat-like conversation structure.

        This will add a `messages` key to the dataset. Unlike make_sft_convo,
        the answer will not be included.
        """
        if not self.sys_prompt:
            msgs: list[dict] = []
        else:
            msgs = [{
                "role": "system",
                "content": SysPrompt(self.sys_prompt).get_sys_prompt(),
            }]
        problem_prompt = ProblemPrompt(self.problem_prompt).get_prompt()
        if problem_prompt:
            problem_prompt += "\n\n"

        def add_user(problem: str) -> list[dict]:
            return [*msgs, {"role": "user", "content": problem_prompt + problem}]

        if isinstance(row["problem"], str):  # Single
            all_msgs: list[dict] | list[list[dict]] = add_user(row["problem"])
        else:  # Batched
            all_msgs = [add_user(p) for p in row["problem"]]
        return {"prompt": all_msgs}

    def make_sft_conversation(
        self, row: MutableMapping[str, str | list[str]]
    ) -> dict[str, list[dict] | list[list[dict]]]:
        """Format a dataset row into a chat-like conversation structure.

        This will add a `messages` key to the dataset.
        """
        if (
            self.reasoning
            and ProblemPrompt(self.problem_prompt) == ProblemPrompt.ANSWER
        ):
            raise ValueError(
                "It does not make sense to include reasoning in the SFT traces,"
                " but then only prompt about answer XML (without thoughts)."
            )

        def add_assistant(
            raw_answer: str, thought: str, prior_msgs: list[dict]
        ) -> list[dict]:
            if re.search(r"<\/answer>", raw_answer):
                # Remove prelude and postlude plus XML tags,
                # because an OpenRouter-hosted DeepSeek R1 can give answer
                # with a prelude and XML tags, but our training expects just an answer
                # > The reaction involves sodium borohydride ([BH4-].[Na+]), <redacted>.
                # > Under these conditions, <redacted>.
                # > <answer>N1(CCOCC1)C1=CC=C(C(O))C=C1</answer>
                answer = extract_answer_loose(raw_answer)
                if not answer:
                    raise ValueError(
                        "Failed to extract just the answer from the answer"
                        f" {raw_answer!r}."
                    )
            else:
                answer = raw_answer

            return [
                *prior_msgs,
                {
                    "role": "assistant",
                    "content": (
                        (f"{THINK_START}{thought}{THINK_END}" if self.reasoning else "")
                        + f"{ANSWER_START}{answer}{ANSWER_END}"
                    ),
                },
            ]

        # The first part will be the same as the RL conversation
        msgs = self.make_rl_conversation(row)["prompt"]
        # Now add the answer, with optional thinking
        if isinstance(row["problem"], str):  # Single
            all_msgs: list[dict] | list[list[dict]] = add_assistant(
                cast(str, row["answer"]),
                cast(str, row["thought"]),
                cast(list[dict], msgs),
            )
        else:  # Batched
            all_msgs = list(
                starmap(
                    add_assistant, zip(row["answer"], row["thought"], msgs, strict=True)
                )
            )
        return {"messages": all_msgs}

    def get_reward_funcs(
        self,
        format_reward_value: float = 1.0,
        soft: bool = False,
        test: bool = False,
        good_molecule_bonus: float = 0.0,
    ) -> list[Callable]:
        return [
            wrap_reward_func(
                format_reward,
                reasoning=self.reasoning,
                reward=format_reward_value,
            ),
            wrap_reward_func(
                accuracy_reward,
                reasoning=self.reasoning,
                soft=soft,
                test=test,
                good_molecule_bonus=good_molecule_bonus,
            ),
        ]
