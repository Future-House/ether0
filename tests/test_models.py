import pytest
from datasets import Dataset

from ether0.models import QAExample, RewardFunctionInfo, filter_problem_types


class TestModels:
    def test_load(self, ether0_benchmark_test: Dataset) -> None:
        ether0_parsed = [QAExample(**r) for r in ether0_benchmark_test]

        ex_0 = ether0_parsed[0]
        assert isinstance(ex_0, QAExample)
        assert ex_0.id == "00c8bc2d-0bb3-53c2-8bdf-cd19616d4536"
        assert (
            ex_0.problem
            == "Generate a SMILES representation for a molecule containing groups:"
            " charged and nitro. It should also have formula C13H12N6O5."
        )
        assert ex_0.problem_type == "functional-group"
        assert ex_0.ideal == "Cc1ncc([N+](=O)[O-])n1CC(=O)N/N=C/c1ccc([N+](=O)[O-])cc1"
        assert ex_0.unformatted == "C13H12N6O5,['charged', 'nitro']"
        assert isinstance(ex_0.solution, RewardFunctionInfo)
        ex0_sol = ex_0.solution
        assert (
            (ex0_sol.fxn_name, ex0_sol.answer_info, ex0_sol.problem_type)
            == tuple(ex0_sol.model_dump().values())
            == (
                "functional_group_eval",
                "('C13H12N6O5', ['charged', 'nitro'])",
                "functional-group",
            )
        )


# NOTE: the num_expected_types numbers may have to be adjusted if we add
# more problem types to the dataset.
@pytest.mark.parametrize(
    ("filters", "should_remove_rows", "num_expected_types", "should_raise"),
    [
        pytest.param([], False, 70, False, id="no-filter-1"),
        pytest.param(None, False, 70, False, id="no-filter-2"),
        pytest.param(["reaction-prediction"], True, 1, False, id="include-1"),
        pytest.param(
            ["reaction-prediction", "retro-synthesis"],
            True,
            2,
            False,
            id="include-2",
        ),
        pytest.param(["!reaction-prediction"], True, 69, False, id="exclude-1"),
        pytest.param(
            ["!reaction-prediction", "molecule-name"],
            # Note that in this case, should_remove_rows and num_expected are just
            # dummy values. Filtering should fail before we get there.
            True,
            999,
            True,
            id="exclude-include",
        ),
    ],
)
def test_filter_problem_types(
    ether0_benchmark_test: Dataset,
    filters: list[str] | None,
    should_remove_rows: bool,
    num_expected_types: int,
    should_raise: bool,
) -> None:
    if should_raise:
        with pytest.raises(
            ValueError,
            match="Cannot specify both problem types to keep and to exclude",
        ):
            filter_problem_types(ether0_benchmark_test, filters)
        return

    filtered = filter_problem_types(ether0_benchmark_test, filters)
    problem_types = set(filtered["problem_type"])

    assert len(problem_types) == num_expected_types
    assert (len(filtered) < len(ether0_benchmark_test)) == should_remove_rows
