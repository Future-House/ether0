from typing import TYPE_CHECKING
from unittest.mock import patch

import httpx
import pytest
from ether0.rewards import oracle_solubility_eval
from pydantic import JsonValue

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    ("yhat", "y", "expected"),
    [
        pytest.param(
            "c1c(O)nc2ccc(CN)cc2c1OC1CCCC1",
            '("scaffold", "c1ccc2c(OC3CCCC3)ccnc2c1", -3.844724178314209, "increase")',
            1.0,
            id="match-scaffold",
        ),
        pytest.param(
            "Oc1c(O)nc2ccc(C[NH3])cc2c1OC1CCCC1O",
            '("scaffold", "c1ccc2c(OC3CCCC3)ccnc2c1", -3.844724178314209, "decrease")',
            0.0,
            id="match-scaffold-bad-solubility",
        ),
        pytest.param(
            "CCCCCC=CCCCN(C)CCC",
            '("groups", ["cis double bond", "hetero N basic H"],  -4.693881511688232, "decrease")',  # noqa: E501
            1.0,
            id="match-groups",
        ),
        pytest.param(
            "CCCCCCCCCCN(C)N[NH]CNCC",
            '("groups", ["cis double bond", "hetero N basic H"],  -1.9085578918457031, "decrease")',  # noqa: E501
            0.0,
            id="match-groups-bad-groups",
        ),
        pytest.param(
            "CCCCN(CCCC)C(=O)C1c2ccccc2Oc2ccccc21",
            '("tanimoto", "CCCN(CCC)C(=O)C1c2ccccc2Oc2ccccc21", -5.273194313049316, "decrease")',
            1.0,
            id="match-tanimoto",
        ),
        pytest.param(
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCN(CCCC)C(=O)C1c2ccccc2Oc2ccccc21",
            '("tanimoto", "CCCN(CCC)C(=O)C1c2ccccc2Oc2ccccc21", -5.273194313049316, "decrease")',
            0.0,
            id="match-tanimoto-too-far",
        ),
        pytest.param(
            "CCCCCCCCCCCCCCCCCCCCCCN(CCC)C(=O)C1c2ccccc2Oc2ccccc21",
            '("tanimoto", "CCCN(CCC)C(=O)C1c2ccccc2Oc2ccccc21", -7.45, "decrease")',
            0.0,
            id="match-tanimoto-hacked-dist",
        ),
        pytest.param(
            "CN(C)C(=O)C1c2ccccc2Oc2ccccc21",
            '("tanimoto", "CCCN(CCC)C(=O)C1c2ccccc2Oc2ccccc21", -4.273194313049316, "decrease")',
            0.0,
            id="match-tanimoto-bad-solubility",
        ),
        pytest.param(
            "CN1CCN(CCCCNc2ncc3cc(-c4c(Cl)cccc4Cl)c(=O)n(C)c3n2)CC1.CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            '("tanimoto", "CN1CCN(CCCCNc2ncc3cc(-c4c(Cl)cccc4Cl)c(=O)n(C)c3n2)CC1", -4.273194313049316, "decrease")',  # noqa: E501
            0.0,
            id="match-tanimoto-bad-structure",
        ),
        pytest.param(
            "C[C@@H]1CC[C@@]2(CC[C@@]3(C(=CC[C@H]4[C@]3(CC[C@@H]5[C@@]4(C[C@H]([C@@H]([C@@]5(C)CO)O)O)C)C)[C@@H]2[C@H]1C)C)C(=O)O[C@H]6[C@@H]([C@H]([C@@H]([C@H](O6)CO[C@H]7[C@@H]([C@H]([C@@H]([C@H](O7)CO)O[C@H]8[C@@H]([C@@H]([C@H]([C@@H](O8)C)O)O)O)O)O)O)O)O",
            '("groups", ["secondary alcohol", "primary alcohol", "hydroxylated heteroatom substituted glycosidic ring"],  -5.921097755432129, "increase")',  # noqa: E501
            1.0,
            id="problematic-groups",
        ),
        pytest.param(
            "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1-c1ccc(C#CCCCC(=O)NO)o1",
            '("tanimoto", "CCCC", -6.25, "increase")',
            0.0,
            id="identical-increase",
        ),
        pytest.param(
            "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1-c1ccc(C#CCCCC(=O)NO)o1",
            '("tanimoto", "CCCC", -7.25, "decrease")',
            0.0,
            id="identical-decrease",
        ),
        pytest.param(
            "OOCCCN(CCC)C(=O)C1c2ccccc2Oc2ccccc21",
            '("tanimoto", "OCCCN(CCC)C(=O)C1c2ccccc2Oc2ccccc21", -5.273194313049316, "decrease")',  # noqa: E501
            0.0,
            id="unreasonable-molecule-failure",
        ),
    ],
)
def test_oracle_solubility_eval(
    test_client: "TestClient", yhat: str, y: str, expected: float
) -> None:
    expl: dict[str, JsonValue] = {}
    with patch.object(httpx, "post", test_client.post):
        result = oracle_solubility_eval(yhat, y, metadata=expl)
    assert result == expected, f"Expected {expected}, got {result}. Explanation: {expl}"
