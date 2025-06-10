import logging
import statistics
from collections.abc import Collection, Iterable
from typing import Any

from datasets import load_dataset

try:
    from aenum import extend_enum
    from lighteval.metrics.metrics import (
        MetricCategory,
        Metrics,
        MetricUseCase,
        SampleLevelMetric,
    )
    from lighteval.tasks.lighteval_task import LightevalTaskConfig
    from lighteval.tasks.requests import Doc
except ImportError as exc:
    raise ImportError(
        "To use ether0's LightEval tasks, please install the 'lighteval' extra via:"
        " `pip install ether0[lighteval]`."
    ) from exc

from ether0.data import get_problem_category
from ether0.model_prompts import LOOSE_XML_ANSWER_USER_PROMPT, ProblemPrompt
from ether0.models import make_problem_types_filter
from ether0.rewards import accuracy_reward, format_reward

logger = logging.getLogger(__name__)

ETHER0_ACCURACY_METRIC_NAME = "ether0_accuracy"
ETHER0_FORMAT_METRIC_NAME = "ether0_format"


def evaluate_ether0_accuracy(
    predictions: list[str],
    formatted_doc: Doc,
    golds: list[str] | None = None,  # noqa: ARG001
) -> float:
    if len(predictions) != 1:
        raise NotImplementedError(
            "Didn't handle anything besides one prediction"
            f" for doc {formatted_doc}, got {predictions}."
        )
    return accuracy_reward(
        completions=predictions,
        solution=[formatted_doc.specific["solution"]],
        reasoning=formatted_doc.specific["reasoning"],
        soft=formatted_doc.specific["soft"],
        test=formatted_doc.specific["test"],
    )[0]


def evaluate_ether0_format(
    predictions: list[str],
    formatted_doc: Doc,
    golds: list[str] | None = None,  # noqa: ARG001
) -> float:
    if len(predictions) != 1:
        raise NotImplementedError(
            "Didn't handle anything besides one prediction"
            f" for doc {formatted_doc}, got {predictions}."
        )
    if formatted_doc.specific["test"]:
        logger.warning("ether0's format reward is only applicable at training time.")
    return format_reward(
        completions=predictions,
        reasoning=formatted_doc.specific["reasoning"],
    )[0]


for metric_name, metric_eval_fn in (
    (ETHER0_ACCURACY_METRIC_NAME, evaluate_ether0_accuracy),
    (ETHER0_FORMAT_METRIC_NAME, evaluate_ether0_format),
):
    if (  # Work around https://github.com/huggingface/lighteval/issues/805
        metric_name not in Metrics.__members__
    ):
        extend_enum(
            Metrics,
            metric_name,
            SampleLevelMetric(
                metric_name=metric_name,
                higher_is_better=True,
                category=MetricCategory.GENERATIVE,
                use_case=MetricUseCase.ACCURACY,
                sample_level_fn=metric_eval_fn,
                corpus_level_fn=statistics.mean,
            ),
        )


KEYS_TO_STORE_IN_DOC = {"id", "solution"}


def make_ether0_task(
    name: str,
    soft: bool,
    test: bool,
    reasoning: bool,
    problem_types: str | Collection[str] | None = None,
    metric_names: Iterable[str] | None = None,
    **kwargs,
) -> LightevalTaskConfig:
    """Create LightEval task for the ether0-benchmark dataset."""
    reward_fn_kwargs = {"soft": soft, "test": test, "reasoning": reasoning}
    if not test:
        prob_prompt = ProblemPrompt.THINK_ANSWER if reasoning else ProblemPrompt.ANSWER
        prompt_prefix: str = prob_prompt.get_prompt()
    else:
        prompt_prefix = LOOSE_XML_ANSWER_USER_PROMPT

    def row_to_doc(row: dict[str, Any], task_name: str) -> Doc:
        """Convert an ether0-benchmark dataset row to a LightEval Doc."""
        return Doc(
            query="\n\n".join((prompt_prefix, row["problem"])),
            task_name=task_name,
            choices=[""],  # Placeholder for non-QA tasks
            gold_index=0,  # Points to above placeholder
            specific={k: row[k] for k in KEYS_TO_STORE_IN_DOC} | reward_fn_kwargs,
        )

    if metric_names is None:
        metric_names = (
            (ETHER0_ACCURACY_METRIC_NAME, ETHER0_FORMAT_METRIC_NAME)
            if not test
            else (ETHER0_ACCURACY_METRIC_NAME,)
        )
    return LightevalTaskConfig(
        name=name,
        prompt_function=row_to_doc,
        suite=["community"],
        hf_repo="futurehouse/ether0-benchmark",
        hf_subset="default",
        hf_filter=(
            make_problem_types_filter(problem_types, type_col="problem_type")
            if problem_types is not None
            else None
        ),
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        metric=[getattr(Metrics, metric_name) for metric_name in metric_names],
        **kwargs,
    )


# TASKS_TABLE is required by LightEval for --custom-tasks CLI arg
TASKS_TABLE = [  # Add general tasks
    make_ether0_task(
        f"ether0:{nickname}{':soft' if is_soft else ''}",
        soft=is_soft,
        test=kwargs["test"],
        reasoning=kwargs["reasoning"],
    )
    for is_soft in (False, True)
    for nickname, kwargs in (
        ("loose", {"test": True, "reasoning": False}),
        ("strict:no_reasoning", {"test": False, "reasoning": False}),
        ("strict", {"test": False, "reasoning": True}),
    )
]
TASKS_TABLE.extend([  # Add problem type-specific tasks
    make_ether0_task(
        f"ether0:{nickname}{':soft' if is_soft else ''}:{prob_cat}",
        soft=is_soft,
        test=kwargs["test"],
        reasoning=kwargs["reasoning"],
        problem_types=f"re:^{prob_cat}.*$",
    )
    for is_soft in (False, True)
    for nickname, kwargs in (
        ("loose", {"test": True, "reasoning": False}),
        ("strict:no_reasoning", {"test": False, "reasoning": False}),
        ("strict", {"test": False, "reasoning": True}),
    )
    for prob_cat in {
        get_problem_category(pt)
        for pt in load_dataset("futurehouse/ether0-benchmark", split="test")[
            "problem_type"
        ]
    }
])
