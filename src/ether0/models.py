import re
from collections.abc import Callable, Collection, Mapping
from enum import StrEnum, auto
from typing import Any

from datasets import DatasetDict
from pydantic import BaseModel, Field, model_validator

from ether0.utils import TDataset

REWARD_REASON_KEY = "reward_reason"  # Sentinel key


class RewardReason(StrEnum):
    FORMAT_FAILED = auto()
    INVALID_MOL = auto()
    # Catch-all for invalid values that aren't a molecule or a reaction
    INVALID_VALUE = auto()

    # Oracle regression values
    WRONG_NUMERICAL_ANSWER = auto()

    # Reaction/retro-synthesis failures
    INVALID_RXN = auto()
    WRONG_PRODUCT = auto()
    PRODUCT_IS_REACTANT = auto()
    NOT_PURCHASABLE = auto()

    # Molecule formula/functional group failures
    WRONG_FORMULA = auto()
    FAILED_CONSTRAINT = auto()

    # Unreasonable molecules
    FAILED_REOS_CHECK = auto()
    FAILED_RING_CHECK = auto()
    FAILED_COUNTERION_CHECK = auto()

    # Really this is a bug, but we don't want to blow up training if a
    # few bad examples slip through.
    INVALID_GROUND_TRUTH = auto()

    # Failover reason if we have an exception during a reward function.
    # NOTE: not using "failed" or "error" since an unhandled exception
    # may be something else
    REWARD_FUNCTION_EXCEPTION = auto()

    # These are automatically added if no other reason is given
    WRONG_ANSWER = auto()
    RIGHT_ANSWER = auto()

    def set_reason(self, metadata: dict | None) -> None:
        if metadata is not None:
            metadata[REWARD_REASON_KEY] = self.value

    @classmethod
    def set_default_reason(cls, reward: float, metadata: dict | None) -> None:
        if metadata is not None and REWARD_REASON_KEY not in metadata:
            (cls.RIGHT_ANSWER if reward >= 1.0 else cls.WRONG_ANSWER).set_reason(
                metadata
            )


SOLUTION_DELIMITER = "!:!"


class RewardFunctionInfo(BaseModel):
    """Metadata used by a reward function to evaluate a solution."""

    fxn_name: str = Field(description="Name of the reward function to use.")
    answer_info: str = Field(
        description="Serialized metadata used by the reward function."
    )
    problem_type: str = Field(description="Problem type, for reference.")

    @model_validator(mode="before")
    @classmethod
    def check_card_number_not_present(cls, data: Any) -> Any:
        if isinstance(data, str):
            # Deserialize from a string 3-tuple
            fn, ainfo, pt = data.split(SOLUTION_DELIMITER, maxsplit=2)
            return {"fxn_name": fn, "answer_info": ainfo, "problem_type": pt}
        return data


class QAExample(BaseModel):
    """Question-answer example with reward function info."""

    id: str = Field(description="Unique identifier for this example.")
    problem: str = Field(description="Problem to solve.")
    problem_type: str = Field(description="Problem type, for reference or filtering.")
    solution: RewardFunctionInfo = Field(
        description="Metadata for the reward function."
    )
    ideal: str | None = Field(
        description=(
            "An optional ideal answer. This could be a candidate SMILES, a log10 of"
            " water solubility, or None if having an ideal does not make sense."
        )
    )
    unformatted: str | None = Field(
        description=(
            "Optional raw data used to generate the problem, used for traceability."
        )
    )


def make_problem_types_filter(
    problem_types: str | Collection[str], type_col: str
) -> Callable[[Mapping[str, Any]], bool]:
    """Make a filtration function to filter a dataset by problem types.

    Args:
        problem_types: A string or collection of strings specifying the problem
            types to filter by.
            - If a string or a collection of strings:
                - Strings starting with "re:" are treated as regex patterns.
                  If a regex filter is provided, then it must be the only filter.
                - Strings starting with "!" are treated as problem types to exclude.
                - Other strings are treated as exact problem types to include.
                - Mixing inclusion and exclusion rules (e.g. ["type_a", "!type_b"])
                  is not allowed.
        type_col: The column name in the dataset that contains the problem type.

    Returns:
        Callable that returns True to keep a row, otherwise False to filter it out.
    """
    if isinstance(problem_types, str):  # Assume single problem type as a string
        problem_types = [problem_types]
    problem_types = {pt.strip() for pt in problem_types}

    if any(pt.startswith("re:") for pt in problem_types):
        # A regex was passed in
        if len(problem_types) != 1:
            raise ValueError(
                "If filtering by regex, only one filter is supported,"
                f" passed {problem_types}."
            )
        regex = re.compile(next(iter(problem_types)).removeprefix("re:"))

        def filter_func(x):
            return regex.match(x[type_col]) is not None

    else:
        # Treat as exact string match
        valid_problem_types = {pt for pt in problem_types if not pt.startswith("!")}
        invalid_problem_types = {
            pt.removeprefix("!") for pt in problem_types if pt.startswith("!")
        }
        if valid_problem_types:
            if invalid_problem_types:
                raise ValueError(
                    "Cannot specify both problem types to keep and to exclude,"
                    f" passed {problem_types}."
                )

            def filter_func(x):
                return x[type_col] in valid_problem_types

        else:

            def filter_func(x):
                return x[type_col] not in invalid_problem_types

    return filter_func


def filter_problem_types(
    dataset: TDataset, problem_types: str | Collection[str] | None
) -> TDataset:
    """Filter a dataset by problem types.

    Args:
        dataset: The dataset to filter. Can be a single Dataset or a DatasetDict.
        problem_types: See make_problem_types_filter.__doc__.

    Returns:
        The filtered dataset.
    """
    if problem_types is None:
        return dataset

    columns = (
        next(iter(dataset.values())) if isinstance(dataset, DatasetDict) else dataset
    ).column_names
    # ether0-benchmark uses 'problem_type'; some variants may use 'type'
    type_col = "problem_type" if "problem_type" in columns else "type"

    return dataset.filter(
        make_problem_types_filter(problem_types, type_col),
        desc="Filtering problem types",
    )
