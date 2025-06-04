from collections.abc import Collection

import pytest
from datasets import Dataset
from pydantic import JsonValue

from ether0.data import (
    SMILES_PATTERN,
    get_problem_categories_from_datasets,
    get_problem_category,
)
from ether0.models import RewardFunctionInfo
from ether0.rewards import EVAL_FUNCTIONS


def test_get_problem_categories_from_datasets(ether0_benchmark_test: Dataset) -> None:
    assert get_problem_categories_from_datasets(ether0_benchmark_test) == {
        "functional-group",
        "molecule-completion",
        "molecule-formula",
        "molecule-name",
        "oracle-solubility",
        "property-cat-eve",
        "property-cat-safety",
        "property-cat-smell",
        "property-regression-adme",
        "property-regression-ld50",
        "property-regression-pka",
        "reaction-prediction",
        "retro-synthesis",
        "simple-formula",
    }


UNVERIFIABLE_PROBLEM_CATEGORY_PREFIXES_TO_EXCLUDE: Collection[str] = {
    "oracle-solubility",  # 'ideal' is not actually an answer
    "retro-synthesis",  # 'ideal' is not actually an answer
}


def test_evals(ether0_benchmark_test: Dataset) -> None:
    failures = []
    for row in ether0_benchmark_test:
        reward_info = RewardFunctionInfo.model_validate(row["solution"])
        fxn_name, answer_info, problem_type = tuple(reward_info.model_dump().values())
        problem_category = get_problem_category(problem_type)
        if (
            problem_category in UNVERIFIABLE_PROBLEM_CATEGORY_PREFIXES_TO_EXCLUDE
            or problem_category
            == "molecule-completion"  # Molc had no 'ideal's when this was made
        ):
            continue
        metadata: dict[str, JsonValue] = {}
        try:
            if problem_category.startswith("property"):
                yhat = answer_info
            else:
                assert row["ideal"]
                yhat = row["ideal"]
            assert (
                EVAL_FUNCTIONS[fxn_name](yhat=yhat, y=answer_info, metadata=metadata)
                == 1.0
            )
        except AssertionError:
            failures.append((problem_category, row["id"], metadata))
    assert not failures


TEST_REASONING_TEXT = (
    "Let's analyze the given molecules and try to predict their LD50 values. LD50"
    " refers to the lethal dose at which 50% of the test organisms die. A lower LD50"
    " means higher toxicity, and a higher LD50 indicates lower toxicity. We need to"
    " identify structural features that relate to toxicity.\n\nThe question leaves open"
    " the possibility that none of the compounds have an LD50 of 320 mg/kg. Let's"
    " consider each molecule individually:\n\n1."
    " ClC1=C(C=CC(=C1)Cl)C1(OCC(O1)COC1=CC=C(C=C1)N1CCN(CC1)C(C)=O)CN1C=NC=C1: This"
    " molecule appears to be quite complex. It has a dichloro-substituted aromatic"
    " ring, an ether linkage, a morpholine ring, a piperazine ring, and an imidazole"
    " ring. The presence of two chlorine atoms on the phenyl ring could suggest some"
    " interaction with biological targets. The molecule also has a morpholine and"
    " piperazine moiety which could contribute to binding with receptors or enzymes."
    " The presence of an amide group might indicate some polarity, but the overall"
    " structure looks relatively lipophilic (nonpolar) given the aromatic rings and"
    " alkyl chains.\n\n2."
    " ClC1=C(C=CC(=C1)Cl)[C@]1(OC[C@@H](O1)COC1=CC=C(C=C1)N1CCN(CC1)C1=CC=C(C=C1)N1C(N(N=C1)[C@H](CC)C)=O)CN1N=CN=C1:"  # noqa: E501
    " This is a very complex molecule, with multiple rings, stereocenters, and"
    " heteroatoms. It's a distinct structure and appears to be larger than the first"
    " molecule. We can see a furan ring, a pyrazole ring, an amide group, and other"
    " major differences. This change in the rings and other functional groups is likely"
    " to significantly change the molecular properties compared to the first"
    " molecule.\n\n3."
    " [2H]C(C(=O)N1CCN(CC1)C1=CC=C(C=C1)OCC1O[C@@](OC1)(CN1C=NC=C1)C1=C(C=C(C=C1)Cl)Cl)([2H])[2H]:"  # noqa: E501
    " This molecule, labeled with deuterium, has multiple rings including a piperazine,"
    " furan, a substituted imidazole, and a dichlorinated phenyl ring. It also includes"
    " an ester group which is sometimes associated with higher toxicity compared to"
    " simple ethers.\n\nThinking about general principles of toxicity, lipophilicity"
    " (fat solubility) is often related to higher toxicity. A molecule with a marked"
    " lipophilic character can often accumulate in fatty tissues and interact with the"
    " cell membrane, affect cellular transport or receptor activity. This could lead to"
    " higher toxicity by interfering with normal cellular function. Similarly, the"
    " presence of chlorine atoms can sometimes contribute to toxicity due to possible"
    " metabolic activation to reactive intermediates. However, the position and nature"
    " of other substituents and functional groups can influence how chlorine"
    " substitutions modulate toxicity. For example, some chlorinated compounds are"
    " relatively non-toxic.\n\nConsidering the size and complexity of the molecules, we"
    " should think about their potential metabolic pathways. Large molecules can be"
    " metabolized through various pathways, potentially leading to reactive"
    " intermediates that interact with biological molecules. Metabolites of these"
    " compounds might be more or less toxic than the initial molecules, and the"
    " metabolic pathways themselves might be quite different. Perhaps one of the"
    " metabolites could be the reason for an LD50 of 320 mg/kg. Alternatively, a"
    " compound might be relatively non-toxic in itself, but its presence can alter"
    " enzyme activity or other metabolic processes and indirectly lead to cell"
    " damage.\n\nComparing the three molecules. Molecules 1 and 2 share some structural"
    " features like the dichloro-substituted aromatic ring and the presence of a"
    " morpholine ring system. However, they also have distinct differences in the"
    " connectivity and presence of additional rings, including likely some more polar"
    " and/or sterically bulky substituents. Molecule 3 has different ring systems and"
    " the addition of both a deuterated methyl group and an ester group which adds"
    " polar character and can often activate adjacent portions of the molecule by"
    " metabolic oxygenation.\n\nLet's think about bioreactivity beyond simple chemical"
    " interactions. Structures can influence how a molecule interacts with biological"
    " receptors or enzymes. The size and shape of these molecules and the nature of the"
    " functional groups can determine the extent of the molecule's binding interactions"
    " with biomolecules. Some conformationally adaptable structures might bind strongly"
    " to targets and interfere with crucial pathways, which can lead to toxicity."
    " Therefore, weaknesses in essential molecular machinery could have similar"
    " negative effects if bound by those biomolecules.\n\nIf one of these molecules has"
    " an LD50 of 320 mg/kg, it suggests moderate toxicity. It could be that one of the"
    " molecules doesn't have the necessary structural features to interact strongly"
    " with critical biological targets for high toxicity, and/or it might be"
    " metabolized to relatively non-toxic products, such as carbon dioxide and water."
    " Thus, while the molecules share some features with other potentially bioactive"
    " molecules, it could be that they themselves are not exceptionally potent."
)

NO_SMILES_TEXT = "This text does not contain any SMILES"


@pytest.mark.parametrize(
    ("text", "expected_answer"),
    [
        (
            TEST_REASONING_TEXT,
            [
                "ClC1=C(C=CC(=C1)Cl)C1(OCC(O1)COC1=CC=C(C=C1)N1CCN(CC1)C(C)=O)CN1C=NC=C1",
                "ClC1=C(C=CC(=C1)Cl)[C@]1(OC[C@@H](O1)COC1=CC=C(C=C1)N1CCN(CC1)C1=CC=C(C=C1)N1C(N(N=C1)[C@H](CC)C)=O)CN1N=CN=C1",
                "[2H]C(C(=O)N1CCN(CC1)C1=CC=C(C=C1)OCC1O[C@@](OC1)(CN1C=NC=C1)C1=C(C=C(C=C1)Cl)Cl)([2H])[2H]",
            ],
        ),
        (
            NO_SMILES_TEXT,
            [],
        ),
    ],
)
def test_extract_smiles_from_text(text: str, expected_answer: list[str]) -> None:
    assert sorted(SMILES_PATTERN.findall(text)) == sorted(expected_answer)
