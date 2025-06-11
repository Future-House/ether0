import asyncio
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Self
from uuid import UUID

from aviary.core import Message
from aviary.utils import (
    EvalAnswerMode,
    MultipleChoiceEvaluation,
    MultipleChoiceQuestion,
    eval_answer,
)
from datasets import Dataset
from lmi import LiteLLMModel
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

ETHER0_QA_PACKAGE_DIR = pathlib.Path(__file__).parent
REPO_QA_EVAL_JSONL_PATH = ETHER0_QA_PACKAGE_DIR / "repo_qa_eval.jsonl"
REPO_QA_STEP0_JSONL_PATH = ETHER0_QA_PACKAGE_DIR / "repo_qa_s0.jsonl"
REPO_ROOT_DIR = ETHER0_QA_PACKAGE_DIR.parent.parent.parent.parent
REPO_ROOT_FILES = [str(f) for f in REPO_ROOT_DIR.iterdir() if not f.is_dir()]
README_PATH = REPO_ROOT_DIR / "README.md"
ETHER0_PACKAGE_DIR = REPO_ROOT_DIR / "src" / "ether0"
REWARDS_PATH = ETHER0_PACKAGE_DIR / "rewards.py"
CHAT_PATH = ETHER0_PACKAGE_DIR / "chat.py"


LLM_SCORE_EVAL_CONFIG: dict[str, Any] = {
    "prompt": (
        "Here is a question, the correct answer to the question,"
        " and a proposed answer to the question."
        " We are going to grade the proposed answer as"
        " correct (1.0), incorrect (0.0), or unsure (-1.0)."
        " If the proposed answer matches the correct answer,"
        " respond with just 1.0."
        " If the proposed answer does not match the correct answer,"
        " respond with just 0.0."
        " Otherwise if unsure about the match,"
        " respond with -1.0."
        "\n\nQuestion: {question}"
        "\n\nCorrect answer: {correct_answer}"
        "\n\nProposed answer: {proposed_answer}"
    ),
    "max_score": 1.0,
    "model": "gpt-4o",
    "temperature": 0.0,
}


class Ether0RepoOpenAnswer(MultipleChoiceQuestion):
    prompt_without_id: bool = True
    prompt_without_options: bool = True
    options: Sequence[str] = Field(default_factory=list)
    unsure_answer: str | None = None

    @classmethod
    def from_ds(cls, row: Mapping[str, str]) -> Self:
        return cls(
            question_id=UUID(row["id"]),
            question=row["question"],
            ideal_answer=row["answer"],
        )

    async def grade(
        self, proposed_answer: str
    ) -> tuple[MultipleChoiceEvaluation, str | None]:
        evaluation = await eval_answer(
            proposed=proposed_answer,
            correct=self.ideal_answer,
            question=self.question_prompt,
            eval_mode=EvalAnswerMode.LLM_SCORE,
            llm_eval_config=LLM_SCORE_EVAL_CONFIG,
        )
        if evaluation == 1:
            return MultipleChoiceEvaluation.CORRECT, str(evaluation)
        if evaluation == 0:
            return MultipleChoiceEvaluation.INCORRECT, str(evaluation)
        if evaluation == -1:
            return MultipleChoiceEvaluation.UNSURE, str(evaluation)
        raise NotImplementedError(f"Didn't handle {evaluation=}.")


OPEN_ANSWER_PREFIX = (
    "We are working with a code repository."
    " Please answer the below short answer question."
)


def format_zero_shot_prompt(question: str, *paths: pathlib.Path) -> str:
    """Format a message with optional context files for zero shotting LLMs."""
    return "\n\n".join(
        [OPEN_ANSWER_PREFIX]
        + [
            f"For context, here is the {path.relative_to(REPO_ROOT_DIR)!s}:"
            f"\n\n{path.read_text(encoding='utf-8').strip()}"
            for path in paths
        ]
        + [question]
    )


async def zero_shot_then_grade(
    model: LiteLLMModel, qa: Ether0RepoOpenAnswer, *paths: pathlib.Path
) -> MultipleChoiceEvaluation:
    # NOTE: this code has many extra locals to enable convenient debugging
    question, correct_answer = qa.question_prompt, qa.ideal_answer  # noqa: F841
    prompt = format_zero_shot_prompt(question, *paths)
    llm_result = await model.acompletion([Message(content=prompt)])
    if len(llm_result) != 1 or not llm_result[0].text:
        raise NotImplementedError(f"Unexpected shape of LLM result {llm_result}.")
    proposed_answer = llm_result[0].text
    evaluation = (await qa.grade(proposed_answer=proposed_answer))[0]
    _ = 0  # Debug here
    return evaluation


async def bulk_zero_shot_then_grade(
    model: LiteLLMModel,
    open_answers: Iterable[Ether0RepoOpenAnswer],
    *paths: pathlib.Path,
) -> tuple[float, float]:
    """Evaluate on the eval dataset and return a two-tuple of accuracy and precision."""
    evaluations = await tqdm_asyncio.gather(
        *(zero_shot_then_grade(model, oa, *paths) for oa in open_answers),
        desc="Running evaluation",
        ncols=0,
    )
    return MultipleChoiceEvaluation.calculate_accuracy_precision(evaluations)


async def main() -> None:
    model = LiteLLMModel(name="gpt-4o")
    open_answers = [
        Ether0RepoOpenAnswer.from_ds(row)
        for row in Dataset.from_json(str(REPO_QA_EVAL_JSONL_PATH))
    ]
    for world in (
        (),
        (README_PATH,),
        (REWARDS_PATH,),
        (README_PATH, REWARDS_PATH),
        (README_PATH, REWARDS_PATH, CHAT_PATH),
    ):
        accuracy, precision = await bulk_zero_shot_then_grade(
            model, open_answers, *world
        )
        world_name = (
            " + ".join([str(w.relative_to(REPO_ROOT_DIR)) for w in world])
            if world
            else "Null"
        )
        print(
            f"{world_name!r} world..."
            f" accuracy: {accuracy:.2%}, precision: {precision:.2%}."
        )


if __name__ == "__main__":
    asyncio.run(main())
