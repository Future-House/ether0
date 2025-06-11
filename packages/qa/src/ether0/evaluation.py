import pathlib
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Self, assert_never
from uuid import UUID

from aviary.utils import (
    EvalAnswerMode,
    MultipleChoiceEvaluation,
    MultipleChoiceQuestion,
    eval_answer,
)
from datasets import Dataset
from pydantic import Field

ETHER0_QA_PACKAGE_DIR = pathlib.Path(__file__).parent
REPO_QA_EVAL_JSONL_PATH = ETHER0_QA_PACKAGE_DIR / "repo_qa_eval.jsonl"


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


class Ether0RepoTaskSplit(StrEnum):
    EVAL = "repo_qa_eval"

    def get_task(self) -> Dataset:
        if self == Ether0RepoTaskSplit.EVAL:
            return Dataset.from_json(str(REPO_QA_EVAL_JSONL_PATH))
        assert_never(self)
