import pytest

from ether0.chat import ChatArguments
from ether0.model_prompts import ProblemPrompt, SysPrompt


class TestChatArguments:
    @pytest.mark.parametrize(
        ("args", "row", "expected"),
        [
            (
                ChatArguments(problem_prompt=ProblemPrompt.NONE),
                {"problem": "stub problem"},
                {"prompt": [{"content": "stub problem", "role": "user"}]},
            ),
            (
                ChatArguments(problem_prompt=ProblemPrompt.NONE),
                {"problem": ["stub problem", "stub problem 2"]},
                {
                    "prompt": [
                        [{"content": "stub problem", "role": "user"}],
                        [{"content": "stub problem 2", "role": "user"}],
                    ]
                },
            ),
            (
                ChatArguments(
                    sys_prompt=SysPrompt.SCIENTIFIC_AI,
                    problem_prompt=ProblemPrompt.THINK_ANSWER,
                ),
                {"problem": "stub problem"},
                {
                    "prompt": [
                        {
                            "role": "system",
                            "content": "You are a scientific reasoning AI assistant.",
                        },
                        {
                            "role": "user",
                            "content": (
                                "A conversation between User and Assistant. The user"
                                " asks a question, and the Assistant solves it. The"
                                " assistant first thinks about the reasoning process in"
                                " the mind and then provides the user with the answer."
                                " The reasoning process and answer are enclosed within"
                                " <|think_start|> <|think_end|>"
                                " and <|answer_start|> <|answer_end|> tags,"
                                " respectively, i.e., <|think_start|> reasoning process here"
                                " <|think_end|><|answer_start|> answer here <|answer_end|>"
                                "\n\nstub problem"
                            ),
                        },
                    ]
                },
            ),
        ],
    )
    def test_rl_conversation(
        self, args: ChatArguments, row: dict, expected: dict
    ) -> None:
        assert args.make_rl_conversation(row) == expected
