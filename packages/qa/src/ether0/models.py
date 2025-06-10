import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
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

REPO_QA_JSONL_PATH = Path(__file__).parent / "repo_qa.jsonl"


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


class Ether0OpenAnswer(MultipleChoiceQuestion):
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


async def run_grade_eval(
    model: LiteLLMModel, qa: Ether0OpenAnswer
) -> MultipleChoiceEvaluation:
    # NOTE: this code has many extra locals to enable convenient debugging
    question, correct_answer = qa.question_prompt, qa.ideal_answer  # noqa: F841
    llm_result = await model.acompletion(
        [Message(content=f"{OPEN_ANSWER_PREFIX}\n\n{question}")]
    )
    if len(llm_result) != 1 or not llm_result[0].text:
        raise NotImplementedError(f"Unexpected shape of LLM result {llm_result}.")
    proposed_answer = llm_result[0].text
    evaluation = (await qa.grade(proposed_answer=proposed_answer))[0]
    _ = 0  # Debug here
    return evaluation


async def run_eval() -> tuple[float, float]:
    """Evaluate on the eval dataset and return a two-tuple of accuracy and precision."""
    open_answers = [
        Ether0OpenAnswer.from_ds(row)
        for row in Dataset.from_json(str(REPO_QA_JSONL_PATH))
    ]

    model = LiteLLMModel(name="gpt-4o")
    evaluations = await tqdm_asyncio.gather(
        *(run_grade_eval(model, oa) for oa in open_answers),
        desc="Running evaluation",
    )
    return MultipleChoiceEvaluation.calculate_accuracy_precision(evaluations)


async def main() -> None:
    accuracy, precision = await run_eval()
    print(f"Accuracy: {accuracy:.2%}, Precision: {precision:.2%}.")


if __name__ == "__main__":
    asyncio.run(main())
