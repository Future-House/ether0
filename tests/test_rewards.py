from unittest.mock import patch

import pytest
from pydantic import JsonValue

from ether0.data import is_reasonable_fp, is_reasonable_ring_system, mol_from_smiles
from ether0.models import RewardReason
from ether0.rewards import (
    caption_eval,
    formula_diff,
    formula_eval,
    functional_group_eval,
    oracle_rxn_eval,
    product_eval,
    rxn_eval,
    str_eval,
    valid_mol_eval,
    valid_molecule_eval,
)


@pytest.mark.parametrize(
    ("yhat", "y", "expected"),
    [
        pytest.param(
            "methyl 2-(ethylcarbamoyl)-1,3-dioxo-2,3-dihydro-1H-pyrrolo[3,4-c]pyridine-5-carboxylate",  # noqa: E501
            "methyl 2-(ethylcarbamoyl)-1,3-dioxo-2,3-dihydro-1(H)-pyrrolo[3,4-c]pyridine-5-carboxylate",  # noqa: E501
            1.0,
            id="parentheses",
        ),
        pytest.param(
            "methyl 2-(ethylcarbamoyl)-1,3-dioxo-2,3-dihydro-1H-pyrrolo[3,4-c]pyridine-5-carboxylate",  # noqa: E501
            "methyl 2-(ethylcarbamoyl)-1,3-dioxo-2,3-dihydro-1{H}-pyrrolo[3,4-c]pyridine-5-carboxylate",  # noqa: E501
            1.0,
            id="culies parentheses",
        ),
        pytest.param(
            "methyl 2-(ethylcarbamoyl)-1,3-dioxo-2,3-dihydro-1H-pyrrolo[3,4-c]pyridine-5-carboxylate",  # noqa: E501
            "methyl 2-(ethylcarbamoyl)-1,3-dioxo-2,3-dihydro-1H-pyrrolo[3,4-c]pyridine-5-carboxylate",  # noqa: E501
            1.0,
            id="same",
        ),
        pytest.param(
            " methyl 2-(ethylcarbamoyl)-1,3-dioxo-2,3-dihydro-1H-pyrrolo[3,4-c]pyridine-5-carboxylate",  # noqa: E501
            "methyl 2-(ethylcarbamoyl)-1,3-dioxo-2,3-dihydro-1H-pyrrolo[3,4-c]pyridine-5-carboxylate  ",  # noqa: E501
            1.0,
            id="spacing",
        ),
        pytest.param(
            "methyl 3-(ethylcarbamoyl)-1,3-dioxo-2,3-dihydro-1H-pyrrolo[3,4-c]pyridine-5-carboxylate",  # noqa: E501
            "methyl 2-(ethylcarbamoyl)-1,3-dioxo-2,3-dihydro-1H-pyrrolo[3,4-c]pyridine-5-carboxylate",  # noqa: E501
            0.0,
            id="different",
        ),
        pytest.param(
            "(5S,8R,9S,10R,13S,14R,17S)-17-[(1R,2S,3R,4S,7R,9S,10S,12R,15S)-3-(benzoylamino)-2-hydroxy-3-phenylpropanoyl]oxy-5,9-dihydroxy-4,10,13-trimethyl-11-oxo-6-oxatetracyclo[11.3.1.0^{3,10}.0^{4,7}]heptadec-14-en-8-yl (2R,3S)-3-benzamido-2-hydroxy-3-phenylpropanoate",  # noqa: E501
            " (5~S~,8~R~,9~S~,10R,13S,14R,17S)-17-[(1R,2S,3R,4S,7R,9S,10S,12R,15S)-3-(benzoylamino)-2-hydroxy-3-phenylpropanoyl]oxy-5,9-dihydroxy-4,10,13-trimethyl-11-oxo-6-oxatetracyclo[11.3.1.0^{3,10}.0^{4,7}]heptadec-14-en-8-yl (2R,3S)-3-benzamido-2-hydroxy-3-phenylpropanoate",  # noqa: E501
            1.0,
            id="italics",
        ),
        pytest.param(
            "(5S,8R,9S,10R,13S,14R,17S)-17-[(1R,2S,3R,4S,7R,9S,10S,12R,15S)-3-(benzoylamino)-2-hydroxy-3-phenylpropanoyl]oxy-5,9-dihydroxy-4,10,13-trimethyl-11-oxo-6-oxatetracyclo[11.3.1.0^{3,10}.0(4,7)]heptadec-14-en-8-yl (2R,3S)-3-benzamido-2-hydroxy-3-phenylpropanoate",  # noqa: E501
            " (5~S~,8~R~,9~S~,10R,13S,14R,17S)-17-[(1R,2S,3R,4S,7R,9S,10S,12R,15S)-3-(benzoylamino)-2-hydroxy-3-phenylpropanoyl]oxy-5,9-dihydroxy-4,10,13-trimethyl-11-oxo-6-oxatetracyclo[11.3.1.0^(3,10).0^{4,7}]heptadec-14-en-8-yl (2R,3S)-3-benzamido-2-hydroxy-3-phenylpropanoate",  # noqa: E501
            1.0,
            id="curlies and carrots",
        ),
        pytest.param(
            "(5S,8R,9S,10R,13S,14R,17S)-17-[(1R,2S,3R,4S,7R,9S,10S,12R,15S)-3-benzoylamino-2-hydroxy-3-phenylpropanoyl]oxy-5,9-dihydroxy-4,10,13-trimethyl-11-oxo-6-oxatetracyclo[11.3.1.0^{3,10}.0(4,7)]heptadec-14-en-8-yl (2R,3S)-3-benzamido-2-hydroxy-3-phenylpropanoate",  # noqa: E501
            " (5~S~,8~R~,9~S~,10R,13S,14R,17S)-17-[(1R,2S,3R,4S,7R,9S,10S,12R,15S)-3-(benzoylamino)-2-hydroxy-3-phenylpropanoyl]oxy-5,9-dihydroxy-4,10,13-trimethyl-11-oxo-6-oxatetracyclo[11.3.1.0^(3,10).0^{4,7}]heptadec-14-en-8-yl (2R,3S)-3-benzamido-2-hydroxy-3-phenylpropanoate",  # noqa: E501
            1.0,
            id="more parentheses",
        ),
        pytest.param(
            "(5S,8R,9S,10R,13S,14R,17S)-17-[(1R,2S,3R,4S,7R,9S,10S,12R,15S)-3-(benzoylamino)-2-hydroxy-3-phenylpropanoyl]oxy-5,9-dihydroxy-4,10,13-trimethyl-11-oxo-6-oxatetracyclo[11.3.1.0^{3,10}.0(4,7)]heptadec-14-en-8-yl (2R,3S)-3-benzamido-2-hydroxy-3-phenylpropanoate",  # noqa: E501
            " (5~S~,8~R~,9~S~,10R,13S,14R,17S)-17-(1R,2S,3R,4S,7R,9S,10S,12R,15S)-3-(benzoylamino)-2-hydroxy-3-phenylpropanoyloxy-5,9-dihydroxy-4,10,13-trimethyl-11-oxo-6-oxatetracyclo[11.3.1.0^(3,10).0^{4,7}]heptadec-14-en-8-yl (2R,3S)-3-benzamido-2-hydroxy-3-phenylpropanoate",  # noqa: E501
            0.0,
            id="bad-parentheses",
        ),
    ],
)
def test_str_eval(yhat: str, y: str, expected: float) -> None:
    assert str_eval(yhat, y) == expected


@pytest.mark.parametrize(
    ("yhat", "y", "expected"),
    [
        pytest.param(
            "Buchwald-Hartwig amination",
            "Buchwald-Hartwig amination",
            1.0,
            id="same rxn",
        ),
        pytest.param(
            "buchwald hartwig amination",
            "Buchwald-Hartwig amination",
            1.0,
            id="caps/hyphens",
        ),
        pytest.param(
            "BuchwaldHartwigAmination",
            "Buchwald-Hartwig amination",
            1.0,
            id="no spaces",
        ),
        pytest.param(
            "Buchwald\u2013Hartwig amination",
            "Buchwald-Hartwig amination",
            1.0,
            id="en dash",
        ),
        pytest.param(
            "Buchwald\u2013Hartwig animation",
            "Buchwald-Hartwig amination",
            0.0,
            id="false positive",
        ),
    ],
)
def test_rxn_eval(yhat: str, y: str, expected: float) -> None:
    assert rxn_eval(yhat, y) == expected


@pytest.mark.parametrize(
    ("yhat", "y", "expected"),
    [
        pytest.param(
            "O=C(OC1C(OC(=O)C=2C=CC=CC2)C3(O)C(C)(C)CCCC3(C)C4CC=5OC=CC5C(C)C14)C=6C=CC=CC6",
            "O=C(OC1C(OC(=O)C=2C=CC=CC2)C3(O)C(C)(C)CCCC3(C)C4CC=5OC=CC5C(C)C14",
            1.0,
            id="full-answer",
        ),
        pytest.param(
            ")C=6C=CC=CC6",
            "O=C(OC1C(OC(=O)C=2C=CC=CC2)C3(O)C(C)(C)CCCC3(C)C4CC=5OC=CC5C(C)C14",
            1.0,
            id="partial-answer",
        ),
        pytest.param(
            "",
            "O=C(OC1C(OC(=O)C=2C=CC=CC2)C3(O)C(C)(C)CCCC3(C)C4CC=5OC=CC5C(C)C14",
            0.0,
            id="empty-generation",
        ),
        pytest.param(
            "CCC",
            "O=C(OC1C(OC(=O)C=2C=CC=CC2)C3(O)C(C)(C)CCCC3(C)C4CC=5OC=CC5C(C)C14",
            0.0,
            id="wrong-valid-SMILES",
        ),
        pytest.param(
            "applesauce",
            "O=C(OC1C(OC(=O)C=2C=CC=CC2)C3(O)C(C)(C)CCCC3(C)C4CC=5OC=CC5C(C)C14",
            0.0,
            id="non-SMILES-yhat",
        ),
    ],
)
def test_valid_mol_eval(yhat: str, y: str, expected: float) -> None:
    metadata: dict[str, JsonValue] = {}
    assert (
        valid_mol_eval(yhat, y, metadata=metadata) == expected
    ), f"Reason for failure: {metadata}"


@pytest.mark.parametrize(
    ("yhat", "y", "expected_reward", "expected_reason"),
    [
        pytest.param(
            "CCCO",
            "CCCO",
            1.0,
            None,
            id="exact-match",
        ),
        pytest.param(
            "CCCO",
            "C#N",
            0.0,
            RewardReason.INVALID_GROUND_TRUTH,
            id="chembench-8ee3546d-a3b8-4c7b-90ef-ead9ff11a50d-removed",
        ),
    ],
)
def test_product_eval(
    yhat: str,
    y: str,
    expected_reward: float,
    expected_reason: RewardReason | None,
) -> None:
    metadata: dict[str, JsonValue] = {}
    assert product_eval(yhat, y, metadata=metadata) == expected_reward
    assert metadata.get("reward_reason") == expected_reason
    # Also testing caption_eval here since it's the same
    assert caption_eval(yhat, y, metadata=metadata) == expected_reward


@pytest.mark.parametrize(
    ("yhat", "y", "expected"),
    [
        pytest.param(
            r"C/C=C(/C)\C(=O)O[C@@H]1C[C@@]2(C(=O)C=C(O2)/C(=C\[C@@H]3[C@@H]1C(=C)C(=O)O3)/CO)C",
            "C=C1C(=O)O[C@@H]2/C=C(/CO)C3=CC(=O)[C@@](C)(C[C@@H](OC(=O)C(C)=CC)[C@@H]12)O3",
            1.0,
            id="match",
        ),
        pytest.param(
            "CC1=CC(=C(C(=C1C(=O)O)O)C)OC(=O)C2=C(C(=C(C=C2C)OC)C)OC",
            "C=C1C(=O)O[C@@H]2/C=C(/CO)C3=CC(=O)[C@@](C)(C[C@@H](OC(=O)C(C)=CC)[C@@H]12)O3",
            0.05,
            id="formula-match",
        ),
        pytest.param(
            "CC1=CC(=C(C(=C1C(=O)O)O)C)OC(=O",
            "C=C1C(=O)O[C@@H]2/C=C(/CO)C3=CC(=O)[C@@](C)(C[C@@H](OC(=O)C(C)=CC)[C@@H]12)O3",
            0.0,
            id="bad-mol",
        ),
        pytest.param(
            "CC1=C[C@@H]2O[C@H]3C[C@H]4OC(=O)C=CC=CC(=O)OCC[C@@]5(C)O[C@@H]5C(=O)OC[C@]2(CC1)[C@@]4(C)[C@]31CO1",
            "CC1=C[C@@H]2O[C@H]3C[C@H]4OC(=O)C=CC=CC(=O)OCC[C@@]5(C)O[C@@H]5C(=O)OC[C@]2(CC1)[C@@]4(C)[C@]31CO1",
            1.0,
            id="wild-molecule",
        ),
    ],
)
def test_formula_eval(yhat: str, y: str, expected: float) -> None:
    metadata: dict[str, JsonValue] = {}
    assert (
        formula_eval(yhat, y, soft=True, metadata=metadata) >= expected
    ), f"Reason for failure: {metadata}"


@pytest.mark.parametrize(
    ("yhat", "y", "expected"),
    [
        pytest.param(
            r"Cc1nc(NC(=O)[C@@H](N)CO)sc1-c1cnc(Cl)c(NS(=O)(=O)c2ccccc2)c1",
            "('C18H18ClN5O4S2', ['imidoylhalide cyclic'])",
            1.0,
            id="match",
        ),
        pytest.param(
            r"Cc1nc(NC(=O)[C@@H](N)CO)sc1-c1cnc(Cl)c(NS(=O)(=O)c2ccccc2)c1",
            "('C18H18ClN5O4S2', ['imidoylhalide cyclic', 'non-existing'])",
            0.0,
            id="bad groups",
        ),
        pytest.param(
            r"Cc1nc(NC(=O)[C@@H](N)CO)sc1-c1cnc(Cl)c(NS(=O)(=O)c2ccccc2)c1",
            "('C18H18ClN5O4S3', ['imidoylhalide cyclic'])",
            0.0,
            id="bad formula",
        ),
        pytest.param(
            r"CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@](O)(c3ccccc3)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(O)C[C@@H](C)CN[C@H](C)[C@@H](O)[C@]1(C)O",
            "('C43H74N2O12', ['1,2-Aminoalcohol', 'hydroxylated heteroatom substituted glycosidic ring', 'tertiary alcohol'])",  # noqa: E501
            1.0,
            id="renamed-groups",
        ),
        pytest.param(r"CCC", "('C3H8', [])", 1.0, id="no-groups"),
        pytest.param(r"CCCNNNNN", "('C3H13N5', [])", 0.0, id="unreasonable-molecule"),
        pytest.param(r"C1CCCCC2C1CCCCCCCCC2", "('C16H30', [])", 0.0, id="bad-ring"),
        pytest.param(
            "CCCCCBr", "('C5H11Br',['alkylbromide'])", 1.0, id="observed-problem"
        ),
    ],
)
def test_functional_group_eval(yhat: str, y: str, expected: float) -> None:
    metadata: dict[str, JsonValue] = {}
    assert (
        functional_group_eval(yhat, y, metadata=metadata) == expected
    ), f"Reason for failure: {metadata}"


@pytest.mark.parametrize(
    ("yhat", "y", "expected"),
    [
        pytest.param(
            "CCC=O.CC1(C)CC(N)C(=O)N1>[B-](OC(=O)C)(OC(=O)C)OC(=O)C.[Na+].C=O>",
            "CCCN(C)C1CC(C)(C)NC1=O",
            1.0,
            id="match",
        ),
        pytest.param(
            "CCC=O.CC1(C)CC(N)C(=O)N1>[B-](OC(=O)C)(OC(=O)C)OC(=O)C.[Na+].C=O>CCCN(C)C1CC(C)(C)NC1=O",
            "CCCN(C)C1CC(C)(C)NC1=O",
            1.0,
            id="match-w-product",
        ),
        pytest.param(
            "CCC=O.CC1(C)CC(N)C(=O)N1>[B-](OC(=O)C)(OC(=O)C)OC(=O)C.[Na+].C=O>CCCCN(C=O)C1CC(C)(C)N(C(=O)C)O1",
            "CCCN(C)C1CC(C)(C)NC1=O",
            0.0,
            id="match-w-non-matching-product",
        ),
        pytest.param(
            "CCC=O.CC1(C)CC(N)C(=O)N1>[B-](OC(=O)C)(OC(=O)C)OC(=O)C.[Na+].C=O>CCCXeN(C=O)C1CC(C)(C)N(C(=O)C)O1",
            "CCCN(C)C1CC(C)(C)NC1=O",
            0.0,
            id="match-w-invalid-product",
        ),
        pytest.param(
            "CCC=O.CC1(C)CC(N)C(=O)N1>[B-](OC(=O)C)(OC(=O)C)OC(=O)C.[Na+].C=O",
            "CCCN(C)C1CC(C)(C)NC1=O",
            0.0,
            id="match-wo-trailing",
        ),
        pytest.param(
            "CCC=O.CC1(C)CC(N)C(=O)N1>[B-](OC(=O)C)(OC(=O)C)OC(=O)C.[Na+].C=O>>>>",
            "CCCN(C)C1CC(C)(C)NC1=O",
            0.0,
            id="no-match-w-many-trailing",
        ),
        pytest.param(
            "CCC=O.CC1(C)CC(N)C(=O)N1",
            "CCCN(C=O)C1CC(C)(C)N(C(=O)C)O1",
            0.0,
            id="invalid",
        ),
        pytest.param(
            "C(P)(P)(P)CC=O.CC1(C)(C)CC(N)C(=O)N1>[B-](OC(=O)C)(OC(=O)C)OC(=O)C.[Na+].C=O>",
            "CCCN(C=O)C1CC(C)(C)N(C(=O)C)O1",
            0.0,
            id="no-purchase",
        ),
        pytest.param(
            "OB(O)c1cc(C2CC2)cnc1Cl.Cl -> OB(O)c1cc(C2CC2)cnc1Cl + HBr + HIO2 + HIO3S + CH3COOH || 3s | 3*375I | 9*63BrI | 3*55Br | 3*657s*3*6I | 3*3*7Br*I*P | 3s*369I | 3*7*6s",  # noqa: E501
            "OB(O)c1cc(C2CC2)cnc1Cl",
            0.0,
            id="insane-reward-hacking",
        ),
        pytest.param(
            "CNCCC1CC1(F)F>CC#CC>",
            "CNCCC1CC1(F)F",
            0.0,
            id="trivial-reactants",
        ),
        pytest.param(
            "CC(C)CN1CC(O)C1.CC(C)CN1CC(O)CBr.CCO>CC#CC>",
            "CC(C)CN1CC(O)C1",
            0.0,
            id="disallow-product-in-reactants",
        ),
        pytest.param(
            "N#N.CCO>CC#CC.CC(C)CN1CC(O)C1>",
            "CC(C)CN1CC(O)C1",
            0.0,
            id="disallow-product-in-reagents",
        ),
        pytest.param(
            "C1(CN(C1)CC(C)C)O.CC(C)CN1CC(O)CBr.CCO>CC#CC>",
            "CC(C)CN1CC(O)C1",
            0.0,
            id="disallow-product-in-reactants-with-different-smiles",
        ),
        pytest.param(
            "C=CCNC(=O)Br.BrC#Cc1ccccc1.CCO>[Mg].c1ccccc1>",
            "C=CCNC(=O)C#Cc1ccccc1",
            0.0,
            id="hacked-purchasability",
        ),
        pytest.param(
            "CCC=O.CC1(C)CC(N)C(=O)N1>[B-](OC(=O)C)(OC(=O)C)OC(=O)C.[Na+].C=O.[THF]>CCCN(C=O)C1CC(C)(C)N(C(=O)C)O1",
            "CCCN(C=O)C1CC(C)(C)N(C(=O)C)O1",
            0.0,
            id="invalid-reagent",
        ),
    ],
)
def test_oracle_rxn_eval(yhat: str, y: str, expected: float) -> None:
    # Create a mock dictionary for purchasable molecules
    # Some of these are actually purchasable (or not purchasable),
    # but it's easier to just make it all explicit here.
    # Especially if we change our definition of purchasable in the future.
    mock_purchasable = {
        "CC1(C)CC(N)C(=O)N1": True,
        "XeCC1(C)CC(N)C(=O)N1": False,
        "C=CCNC(=O)Br": False,
        "CC(C)CN1CC(O)C1": True,
        "CC1(C)(C)CC(N)C(=O)N1": False,
        "C(P)(P)(P)CC=O": False,
    }
    with (
        patch("ether0.rewards.fetch_purchasable", return_value=mock_purchasable),
        patch("ether0.rewards.fetch_forward_rxn", return_value={"product": y}),
    ):
        metadata: dict[str, JsonValue] = {}
        result = oracle_rxn_eval(yhat, y, metadata=metadata)
        assert result == expected, (
            f"Given {yhat=} and {y=}, expected {expected} but got {result} with"
            f" {metadata=}."
        )


@pytest.mark.parametrize(
    ("f1", "f2", "expected"),
    [
        pytest.param("C1", "C2", 1.0, id="simple-1"),
        pytest.param("C1", "C1H1", 1.0, id="simple-2"),
        pytest.param("C1H2", "C1H2", 0.0, id="simple-3"),
        pytest.param("N2", "O2", 8**0.5, id="simple-4"),
        pytest.param("X100C1", "X100C2", 1.0, id="bad-element-5"),
        pytest.param("C100", "C100H100", 100, id="big-digits"),
        pytest.param("CH2", "H2", 1, id="implicit"),
    ],
)
def test_formula_diff(f1: str, f2: str, expected: float) -> None:
    assert formula_diff(f1, f2) == expected


@pytest.mark.parametrize(
    ("mol", "ref_mol", "expected"),
    [
        pytest.param(
            "O=C(/C=C/C1=CC=CC=C1)OC[C@H]1O[C@@H](O[C@@H]2O[C@@H]3C[C@H]4[C@H](O)[C@@H](O)[C@@](O)(CO3)[C@@H]24)[C@H](O)[C@@H](O)[C@@H]1O",
            None,
            1,
            id="passing-1",
        ),
        pytest.param(
            "CC(C)C[C@H](NC(=O)[C@H](Cc1c[nH]cn1)NC(=O)[C@H](Cc1ccccc1)NC(=O)OC(C)(C)C)[C@@H](O)[C@@H](O)CC(C)C",
            None,
            1,
            id="passing-2",
        ),
        pytest.param("CCCCCBr", "CCCCCBr", 1, id="passing-3"),
    ],
)
def test_is_reasonable_ring_system(
    mol: str, ref_mol: str | None, expected: float
) -> None:
    mol_ = mol_from_smiles(mol)
    assert mol_ is not None
    assert (
        is_reasonable_ring_system(mol_, mol_from_smiles(ref_mol) if ref_mol else None)
        == expected
    )


@pytest.mark.parametrize(
    ("mol", "ref_mol", "expected"),
    [
        pytest.param(
            "O=C1OC2=CC=CC=C2C=C1c3ccc(O)c(O)c3c4ccc(O)cc4OCC=CCCCCCCC(N)(N)NS",
            None,
            False,
            id="weird-nitrogen-group",
        ),
        pytest.param(
            "O=S(=O)(N)c1c(Cl)cc2c(c1)S(=O)(=O)NCN2",
            None,
            True,
            id="sulfonamide",
        ),
        pytest.param(
            "C1=NC=NC=C1OCC=CCCC(N)S",
            None,
            False,
            id="weird-S-C-N-group",
        ),
        pytest.param(
            "CCC",
            None,
            True,
            id="simple-alkane",
        ),
    ],
)
def test_is_reasonable_fp(mol: str, ref_mol: str | None, expected: bool) -> None:
    mol_ = mol_from_smiles(mol)
    assert mol_ is not None
    assert (
        is_reasonable_fp(mol_, ref_mol=mol_from_smiles(ref_mol) if ref_mol else None)
        == expected
    )


@pytest.mark.parametrize(
    ("yhat", "expected"),
    [
        ("CC(C)CCC", 1.0),
        ("CC(C)(C)(C)C", 0.0),
        ("", 0.0),
        ("INVALID", 0.0),
    ],
)
def test_valid_molecule_eval(yhat, expected):
    assert valid_molecule_eval(yhat, y="") == expected
