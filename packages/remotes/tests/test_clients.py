import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import httpx
import pytest
from ether0.clients import fetch_forward_rxn, fetch_purchasable, fetch_solubility

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.parametrize(
    ("smiles", "purchasable"),
    [
        ("CC(=O)OC1=CC=CC=C1C(=O)O", True),
        ("CCO", True),
        ("C1=CC=C(C=C1)C(=O)O", True),
        ("OCN1C=CC=C1C(=O)O", False),
    ],
)
def test_fetch_purchasable(
    test_client: "TestClient", smiles: str, purchasable: bool
) -> None:
    with patch.object(httpx, "post", test_client.post):
        assert fetch_purchasable(smiles)[smiles] == purchasable


@pytest.mark.parametrize(
    ("smiles", "solubility"),
    [
        ("CC(=O)OC1=CC=CC=C1C(=O)O", -2.5),
        ("O=C(NC1CCCC1)C(C1CC1)S1C(=N)C(C2=CC=NC3=CC=CC=C23)N=C1", -5.9),
    ],
)
def test_fetch_solubility(
    test_client: "TestClient", smiles: str, solubility: float
) -> None:
    with patch.object(httpx, "post", test_client.post):
        result = fetch_solubility(smiles)
    assert "solubility" in result
    assert pytest.approx(result["solubility"], abs=0.1) == solubility


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Requires loading MolTrans model, too large for CI"
)
@pytest.mark.parametrize(
    ("precursor", "product", "correct"),
    [
        pytest.param("CC=O.O=C1CCC1Cl>[Mg].CCOCC>", "CC(O)C1(O)CCC1Cl", True),
        pytest.param(
            "CCC=O.CC1(C)CC(N)C(=O)N1>[B-](OC(=O)C)(OC(=O)C)OC(=O)C.[Na+].C=O>",
            "CCCN(C)C1CC(C)(C)NC1=O",
            True,
        ),
        pytest.param("CCCC=O.O=C1CC=C(Br)S1>[Mg].CCOCC>", "CCCC(O)C1=CCC(=O)S1", True),
        pytest.param("CCCC=O.COC(=O)C1CC1Br>[Mg].CCOCC>", "CCCC(O)C1CC1C(=O)OC", True),
        pytest.param(
            "CCCC=O.NC1CCCNC1=O>[B-](OC(=O)C)(OC(=O)C)OC(=O)C.[Na+].C=O>",
            "O=C1NCCCC1N1CNCCCC1=O",
            True,
        ),
        pytest.param("CC=O.O=C1CCC1Cl.[Mg].CCOCC", None, False, id="missing_arrow"),
        pytest.param(
            "CC=O.O=C1CCC1Cl > [Mg].CCOCC", None, False, id="space_in_reaction"
        ),
        pytest.param("not a > reaction", None, False, id="invalid_reaction"),
        pytest.param(
            "CCCC=O.COC(=O)C1CC1Br>[Mg].CCOCC", None, False, id="trailing_arrow"
        ),
    ],
)
def test_fetch_forward_rxn(
    test_client: "TestClient", precursor: str, product: str | None, correct: bool
) -> None:
    with patch.object(httpx, "post", test_client.post):
        result = fetch_forward_rxn(precursor)
    if correct:
        assert (
            result.get("product") == product
        ), f"Failed to get expected {product=} in {result=}."
        assert "error" not in result
    else:
        assert result.get(
            "error"
        ), f"Expected an error given {precursor=} and {correct=}"
        assert "syntax error" in result["error"].lower()
        assert "product" not in result
