import ast
import logging
import operator
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, cast

import exmol
from pydantic import JsonValue
from rdkit import Chem, DataStructs
from rdkit.Chem import GetMolFrags, SanitizeMol  # pylint: disable=no-name-in-module
from rdkit.Chem.rdMolDescriptors import (  # pylint: disable=no-name-in-module
    CalcMolFormula,
    GetMorganFingerprintAsBitVect,
)
from rdkit.Chem.rdmolfiles import MolToSmiles  # pylint: disable=no-name-in-module
from rdkit.rdBase import BlockLogs

from ether0.clients import fetch_forward_rxn, fetch_purchasable, fetch_solubility
from ether0.data import is_reasonable_fp, is_reasonable_ring_system, mol_from_smiles
from ether0.model_prompts import extract_thought_answer_strict
from ether0.models import RewardFunctionInfo, RewardReason

block = BlockLogs()

logger = logging.getLogger(__name__)


class RewardEvalFn(Protocol):
    def __call__(
        self,
        yhat: str,
        y: str,
        soft: bool = False,
        test: bool = False,
        metadata: dict[str, JsonValue] | None = None,
    ) -> float: ...


def formula_diff(formula1: str, formula2: str) -> float:
    """Calculate l2 norm between two molecular formulas."""
    # important = elements we care about in organic chem
    important_elements = {"C", "H", "O", "N", "F", "Cl", "Br", "P", "S"}
    pattern = re.compile(r"([A-Z][a-z]?)(\d*)")
    counts1 = dict.fromkeys(important_elements, 0)
    counts2 = dict.fromkeys(important_elements, 0)
    for m in pattern.finditer(formula1):
        element = m.group(1)
        count = int(m.group(2)) if m.group(2) else 1
        if element in important_elements:
            counts1[element] += count
    for m in pattern.finditer(formula2):
        element = m.group(1)
        count = int(m.group(2)) if m.group(2) else 1
        if element in important_elements:
            counts2[element] += count
    d2 = sum((counts1[k] - counts2[k]) ** 2 for k in important_elements)
    return d2**0.5


def format_reward(
    completions,
    reasoning: bool,
    reward: float = 1.0,
    **kwargs,  # noqa: ARG001
) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    if isinstance(completions[0], list):
        completion_contents = [completion[0]["content"] for completion in completions]
    else:
        completion_contents = completions
    # Note we check `answer is not None` since empty answer still counts as valid
    # formatting.
    return [
        reward if answer is not None else 0.0
        for answer in (
            extract_thought_answer_strict(c, reasoning)[1] for c in completion_contents
        )
    ]


SUPERSCRIPT_PATTERN = re.compile(r"\^{([\d,]+)}")
ITALICS_PATTERN = re.compile(r"{([a-zA-Z])}")
# parentheses that aren't nested or contain hyphens
# https://regex101.com/r/6c8smX/1
USELESS_PARENTHESES = re.compile(r"([-\d])[\(\[{]([A-Za-z0-9]+)[\]\)}]-")


def normalize_iupac(s: str) -> str:
    """Normalize an IUPAC name by removing special formatting and characters.

    Args:
        s: Original IUPAC name.

    Returns:
        A normalized IUPAC name without special characters.
    """
    s = s.strip().casefold()
    # replace ^{n} with ^(n)
    s = SUPERSCRIPT_PATTERN.sub(r"^(\1)", s)
    # remove italicized pattern - but don't match ^{1,5} (by avoiding matching commas)
    s = ITALICS_PATTERN.sub(r"\1", s)
    # remove garbage
    s = s.replace("$", "").replace("~", "")  # noqa: FURB184
    # remove parentheses that aren't nested or contain hyphens
    s = USELESS_PARENTHESES.sub(r"\1\2-", s)
    # ok to ignore carrots and hpyhens for comparison
    return s.replace("^", "").replace(" ", "-")  # noqa: FURB184


def normalize_unicodes(s: str) -> str:
    """Normalize all Unicode dashes/hyphens to regular hyphen.

    Args:
        s: Input string with potential Unicode characters.

    Returns:
        Unicode-normalized string.
    """
    s = unicodedata.normalize("NFKC", s)
    s = "".join("-" if unicodedata.category(c) in {"Pd", "Po"} else c for c in s)
    return s.replace("-", "")  # minus sign  # noqa: FURB184


def is_reasonable_molecule(
    mol: Chem.Mol,
    metadata: dict[str, JsonValue] | None,
    test: bool,  # noqa: ARG001
    ref_mol: Chem.Mol | None = None,
) -> bool:
    """Returns True if the molecule passes heuristics for being a reasonable molecule."""
    # always check valence
    try:
        SanitizeMol(mol)
    except Exception:
        RewardReason.INVALID_MOL.set_reason(metadata)
        return False

    # We have decided that the convention will be to check the
    # same at test time and train time.

    # determine if we have counter-ions (which is fine), but we want to
    # evaluate the largest molecule only. We only consider single molecules
    # or single molecules + a counterion as valid responses
    sorted_frags = sorted(  # sort by size
        GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumAtoms(), reverse=True
    )
    if len(sorted_frags) > 2:  # noqa: PLR2004
        # not a counter-ion
        RewardReason.FAILED_COUNTERION_CHECK.set_reason(metadata)
        return False
    if len(sorted_frags) == 2:  # noqa: PLR2004
        # If 2, assume first is counter-ion, and double check it's small
        cmol = sorted_frags[1]
        if cmol.GetNumHeavyAtoms() > 5:  # noqa: PLR2004
            RewardReason.FAILED_COUNTERION_CHECK.set_reason(metadata)
            return False

    mol = sorted_frags[0]

    ring_status = is_reasonable_ring_system(mol, ref_mol)

    if not ring_status:
        RewardReason.FAILED_RING_CHECK.set_reason(metadata)
        return False

    failure = is_reasonable_fp(mol, ref_mol)
    if not failure:
        RewardReason.FAILED_REOS_CHECK.set_reason(metadata)
        return False
    return True


FULL_SMILES_KEY = "full_smiles"


def set_full_smiles(smiles: str, metadata: dict[str, JsonValue] | None) -> None:
    if metadata is not None:
        metadata[FULL_SMILES_KEY] = smiles


BAD_SMARTS_PATTERNS = [
    "[#16]-[#16]-[#16]",  # More than a thiol bond
    "[#8]~[#8]",  # Peroxides
    "[#7]-[NH2]",  # Hydrazines
    "[#7]-[NH3]",  # weird charged amine
    "[#7]~[#7]~[#7]",  # 3 or more amines
    "[NX2](=[OX1])[O;$([X2]),$([X1-])]",  # Nitrite
    "[SX2][NX2]=[OX1]",  # Thionitrite
    "[$([NX3](=[OX1])(=[OX1])[O;$([X2]),$([X1-])]),$([NX3+]([OX1-])(=[OX1])[O;$([X2]),$([X1-])])]",  # Nitrate  # noqa: E501
    "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",  # Nitro
    "[NX2](=[OX1])[!#7;!#8]",  # Nitroso
    "[CX4]" + ("-[CX4]" * 6),  # Long chain of carbons (7 or more)
]


def contains_bad_substruct(mol: Chem.Mol) -> bool:
    return any(
        mol.HasSubstructMatch(Chem.MolFromSmarts(pat)) for pat in BAD_SMARTS_PATTERNS
    )


def rxn_eval(
    yhat: str,
    y: str,
    soft: bool = False,  # noqa: ARG001
    test: bool = False,  # noqa: ARG001
    metadata: dict[str, JsonValue] | None = None,  # noqa: ARG001
) -> float:
    """Returns 1.0 if strings match (case-insensitive), otherwise 0.0."""
    # some normalization for IUPAC names - shouldn't affect others

    if normalize_iupac(yhat) == normalize_iupac(y):
        return 1.0

    # If that fails (would return 0), try normalizing further
    return (
        1.0
        if normalize_unicodes(normalize_iupac(yhat))
        == normalize_unicodes(normalize_iupac(y))
        else 0.0
    )


def str_eval(
    yhat: str,
    y: str,
    soft: bool = False,  # noqa: ARG001
    test: bool = False,  # noqa: ARG001
    metadata: dict[str, JsonValue] | None = None,
) -> float:
    """Returns 1.0 if strings match (case-insensitive), otherwise 0.0."""
    set_full_smiles(yhat, metadata)
    return 1.0 if normalize_iupac(yhat) == normalize_iupac(y) else 0.0


def valid_mol_eval(
    yhat: str,
    y: str,
    soft: bool = False,  # noqa: ARG001
    test: bool = False,
    metadata: dict[str, JsonValue] | None = None,
) -> float:
    """Validate if yhat is a valid SMILES string, when appended to y.

    Args:
        yhat: Model-predicted SMILES string or partial completion.
        y: Base SMILES string (e.g. "O=C1CCC2=CC=C(O)C(OC)=C2C#CCC2=CC3=C4") to append
            yhat and check validity.
        test: unused
        soft: unused
        metadata: optional metadata dictionary

    Returns:
        1.0 if `y + yhat` is a valid SMILES string, 0.0 otherwise.
    """
    if not yhat:
        RewardReason.INVALID_MOL.set_reason(metadata)
        return 0.0

    # First attempt yhat alone (assuming full SMILES), then try y+yhat (assuming
    # partial) if that fails
    for smiles in (yhat, y + yhat):
        if not smiles.startswith(y):
            # only accept a solution containing the answer
            continue
        try:
            mol = mol_from_smiles(smiles)
        except Exception:
            logger.exception(
                f"Failed to construct molecule from SMILES string {yhat!r}."
            )
            continue
        if mol is not None:
            set_full_smiles(smiles, metadata)
            if not is_reasonable_molecule(mol, metadata, test):
                return 0.0
            return 1.0

    # Nothing worked - mark as invalid
    RewardReason.INVALID_MOL.set_reason(metadata)
    return 0.0


SMOOTH_THRESHOLD_TANIMOTO_SIMILARITY = 0.7  # close enough


def tanimoto_similarity(
    m1: Chem.Mol | None, m2: Chem.Mol | None, atom_threshold: float = 10.0
) -> float:
    """Calculate Tanimoto similarity between two molecules.

    The `atom_threshold` parameter is a relative fraction (e.g., `0.2` for 20%)
    that sets a threshold for degenerate cases when the fingerprints are similar,
    but there are many more atoms in one molecule.

    Default is 10.0, which corresponds to a 1000% difference and has no practical effect.
    """
    if m1 is None or m2 is None:
        return 0.0
    fp1 = GetMorganFingerprintAsBitVect(m1, 2)
    fp2 = GetMorganFingerprintAsBitVect(m2, 2)

    # heavy atom threshold
    atoms1 = m1.GetNumHeavyAtoms()
    atoms2 = m2.GetNumHeavyAtoms()
    if (denom := max(atoms1, atoms2)) > 0:
        # Do not apply the atom diff check if there are no heavy atoms.
        # This is always safe, since the only way to avoid
        # this block is if m1=m2=H2, which would pass anyway.
        atom_diff = abs(atoms1 - atoms2) / denom
        if atom_diff > atom_threshold:
            return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def exact_mol_match(m1: Chem.Mol, m2: Chem.Mol) -> float:
    s1 = MolToSmiles(m1, canonical=True, isomericSmiles=True)  # noqa: FURB120
    s2 = MolToSmiles(m2, canonical=True, isomericSmiles=True)  # noqa: FURB120
    return 1.0 if s1 == s2 else 0.0


def get_largest_mol(smiles: str) -> Chem.Mol | None:
    parts = smiles.split(".")
    # Filter out small fragments (removes counter-ions) and invalid SMILES
    mols = [
        mol_from_smiles(p)
        for p in parts
        if (len(p) > 3 and mol_from_smiles(p) is not None)  # noqa: PLR2004
    ]
    if not mols:
        return None
    mols_atoms = []
    for mol in mols:
        n_atoms = None if mol is None else mol.GetNumAtoms()
        if n_atoms is None:
            raise NotImplementedError(f"Didn't handle {mol=} having None atoms.")
        mols_atoms.append((mol, n_atoms))
    return max(mols_atoms, key=operator.itemgetter(1))[0]


def product_eval(
    yhat: str,
    y: str,
    soft: bool = False,
    test: bool = False,  # noqa: ARG001
    metadata: dict[str, JsonValue] | None = None,
) -> float:
    """Computes the Tanimoto similarity of the largest fragments from two SMILES (if soft) or exact match (if not soft).

    Returns:
        Reward in [0, 1].
    """  # noqa: E501,W505
    m1 = get_largest_mol(yhat)
    m2 = get_largest_mol(y)

    if m1 is None:
        RewardReason.INVALID_MOL.set_reason(metadata)
        return 0.0
    if m2 is None:
        RewardReason.INVALID_GROUND_TRUTH.set_reason(metadata)
        logger.warning(f"Invalid ground truth molecule {y!r}.")
        return 0.0

    # don't use yhat directly since it may have multiple molecules
    set_full_smiles(MolToSmiles(m1), metadata)

    if soft:
        return tanimoto_similarity(m1, m2)

    return exact_mol_match(m1, m2)  # exact match for non-soft mode


def caption_eval(
    yhat: str,
    y: str,
    soft: bool = False,
    test: bool = False,
    metadata: dict[str, JsonValue] | None = None,
) -> float:
    """Currently forwards to product_eval, but also stores Tanimoto in metadata."""
    if metadata is not None:
        m1 = get_largest_mol(yhat)
        m2 = get_largest_mol(y)
        metadata["tanimoto"] = (
            tanimoto_similarity(m1, m2) if (m1 is not None and m2 is not None) else 0.0
        )
    return product_eval(yhat, y, soft, test, metadata)


def formula_eval(
    yhat: str,
    y: str,
    soft: bool = False,
    test: bool = False,
    metadata: dict[str, JsonValue] | None = None,
) -> float:
    """Check correct formula and Tanimoto similarity, giving a reward in [0, 1]."""
    set_full_smiles(yhat, metadata)
    mhat = mol_from_smiles(yhat)
    m = mol_from_smiles(y)
    if mhat is None:
        RewardReason.INVALID_MOL.set_reason(metadata)
        return 0.0
    if m is None:
        RewardReason.INVALID_GROUND_TRUTH.set_reason(metadata)
        logger.warning(f"Invalid ground truth molecule {y!r}.")
        return 0.0

    fhat = CalcMolFormula(mhat)
    f = CalcMolFormula(m)
    if fhat != f:
        RewardReason.WRONG_FORMULA.set_reason(metadata)
        return 0.0

    if not is_reasonable_molecule(mhat, metadata, test, ref_mol=m):
        return 0.0

    return (
        1.0
        if tanimoto_similarity(mhat, m) >= SMOOTH_THRESHOLD_TANIMOTO_SIMILARITY
        # Give partial credit if soft=True and we got the right formula
        else (0.5 if soft else 0.0)
    )


def functional_group_eval(
    yhat: str,
    y: str,
    soft: bool = False,
    test: bool = False,
    metadata: dict[str, JsonValue] | None = None,
) -> float:
    """Match functional group and formula, giving a reward in [0, 1]."""
    set_full_smiles(yhat, metadata)
    mhat = mol_from_smiles(yhat)
    if mhat is None:
        RewardReason.INVALID_MOL.set_reason(metadata)
        return 0.0

    y_args: tuple[str, list[str]] = ast.literal_eval(y)

    formula = y_args[0]
    groups = {g.lower() for g in y_args[1]}

    fhat = CalcMolFormula(mhat)
    if fhat != formula:
        RewardReason.WRONG_FORMULA.set_reason(metadata)
        return 0.0

    groupshat: set[str] = {
        f.lower() for f in exmol.get_functional_groups(mhat, return_all=True)
    }

    if not is_reasonable_molecule(mhat, metadata, test):
        return 0.0

    return (
        1.0
        if groups <= groupshat
        # Give partial credit if soft=True and we got the right formula
        else (0.5 if soft else 0.0)
    )


def oracle_solubility_eval(
    yhat: str,
    y: str,
    soft: bool = False,  # noqa: ARG001
    test: bool = False,
    metadata: dict[str, JsonValue] | None = None,
) -> float:
    """Evaluate solubility prediction using remote, giving a reward in [0, 1]."""
    set_full_smiles(yhat, metadata)
    # we only want single molecules
    if "." in yhat:
        return 0.0
    mhat = mol_from_smiles(yhat)
    if mhat is None:
        RewardReason.INVALID_MOL.set_reason(metadata)
        return 0.0

    y_args: tuple[str, str | list[str], float, str] = ast.literal_eval(y)
    constraint_type, constraint_data, target = y_args[:3]
    # Unused: direction = y_args[3]  # noqa: ERA001

    ref_mol: Chem.Mol | None = None

    # first check constraint
    if constraint_type == "scaffold":
        ref_mol = mol_from_smiles(cast(str, constraint_data))
        if ref_mol is None:
            raise NotImplementedError(
                f"Didn't handle when {constraint_data=} is invalid."
            )
        if not mhat.HasSubstructMatch(ref_mol):
            RewardReason.FAILED_CONSTRAINT.set_reason(metadata)
            return 0.0
    elif constraint_type == "groups":
        groups = [g.lower() for g in exmol.get_functional_groups(mhat, return_all=True)]
        if not any(group.lower() in groups for group in constraint_data):
            RewardReason.FAILED_CONSTRAINT.set_reason(metadata)
            return 0.0
    elif constraint_type == "tanimoto":
        ref_mol = mol_from_smiles(cast(str, constraint_data))
        if (
            tanimoto_similarity(mhat, ref_mol, atom_threshold=0.2)
            < SMOOTH_THRESHOLD_TANIMOTO_SIMILARITY
        ):
            RewardReason.FAILED_CONSTRAINT.set_reason(metadata)
            return 0.0
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")

    if not is_reasonable_molecule(mhat, metadata, test, ref_mol=ref_mol):
        return 0.0

    # make sure we hit the target
    result = fetch_solubility(yhat)
    if "solubility" in result:
        solubility = result["solubility"]
        delta = solubility - target
        # hard coded to typical solubility accuracies
        # we subtract 0.01 because some questions ask for
        # 0.5 change and we don't want restatements to
        # be matches
        if abs(delta) > (0.5 - 0.01):
            RewardReason.WRONG_NUMERICAL_ANSWER.set_reason(metadata)
            return 0.0
        return 1.0
    RewardReason.INVALID_MOL.set_reason(metadata)
    return 0.0


def oracle_rxn_eval(
    yhat: str,
    y: str,
    soft: bool = False,
    test: bool = False,  # noqa: ARG001
    metadata: dict[str, JsonValue] | None = None,
) -> float:
    """Evaluate forward reaction prediction using remote, giving a reward in [0, 1]."""
    if ">" not in yhat or "." not in yhat:
        RewardReason.INVALID_RXN.set_reason(metadata)
        return 0.0

    # make sure there are not more than two angle brackets
    if yhat.count(">") > 2:  # noqa: PLR2004
        RewardReason.INVALID_RXN.set_reason(metadata)
        return 0.0

    # ok now do real check on regex after heuristic checks
    # adapted partly from https://gist.github.com/lsauer/1312860/264ae813c2bd2c27a769d261c8c6b38da34e22fb
    # https://regex101.com/r/9bdE6H/1
    # basically SMILES_THINGS>SMILES_THINGS | empty>
    if not re.match(
        r"^[^J][a-z0-9@+\-\[\]\(\)\\\/%=#$\.]{6,}>[a-z0-9@+\-\[\]\(\)\\\/%=#$\.]{0,}>",
        yhat,
        re.IGNORECASE,  # lower = aromatic, which we're fine matching
    ):
        RewardReason.INVALID_RXN.set_reason(metadata)
        return 0.0

    ymol = mol_from_smiles(y)
    if ymol is None:
        RewardReason.INVALID_GROUND_TRUTH.set_reason(metadata)
        logger.warning(f"Invalid ground truth molecule {y!r}.")
        return 0.0

    reactant_smi = yhat.split(">")[0].split(".")
    reactants = [mol_from_smiles(r) for r in reactant_smi]
    if not all(x is not None for x in reactants):
        RewardReason.INVALID_MOL.set_reason(metadata)
        return 0.0

    reagents = [mol_from_smiles(r) for r in yhat.split(">")[1].split(".") if r.strip()]
    if not all(x is not None for x in reagents):
        RewardReason.INVALID_MOL.set_reason(metadata)
        return 0.0

    # check products, if present, contain the desired product
    products = [mol_from_smiles(r) for r in yhat.split(">")[2].split(".") if r.strip()]
    # notice we pass if there are no products
    if products:
        if not all(x is not None for x in products):
            RewardReason.INVALID_MOL.set_reason(metadata)
            return 0.0
        if not any(exact_mol_match(m, ymol) == 1.0 for m in products):  # type: ignore[arg-type]
            RewardReason.INVALID_RXN.set_reason(metadata)
            return 0.0

    # Disallow products in the reactants or reagents
    if any(exact_mol_match(m, ymol) == 1.0 for m in (reactants + reagents)):  # type: ignore[arg-type]
        RewardReason.PRODUCT_IS_REACTANT.set_reason(metadata)
        return 0.0

    # check that the reactants are purchasable

    def is_small_so_probably_purchasable(smi: str) -> bool:
        mol = mol_from_smiles(smi)
        # Molecules with <= 4 heavy atoms are likely purchasable,
        # since they include solvents and counterions
        return mol is not None and mol.GetNumHeavyAtoms() <= 4  # noqa: PLR2004

    purchasable_results = fetch_purchasable(reactant_smi)
    if not all(
        purchasable_results.get(r, False) or is_small_so_probably_purchasable(r)
        for r in reactant_smi
    ):
        RewardReason.NOT_PURCHASABLE.set_reason(metadata)
        return 0.0

    result = fetch_forward_rxn(yhat)
    if "product" in result:
        product = result["product"]
        pmol = mol_from_smiles(product)
        if pmol is None:
            RewardReason.INVALID_MOL.set_reason(metadata)
            return 0.0
        if soft:
            return tanimoto_similarity(pmol, ymol)
        if exact_mol_match(pmol, ymol) == 1.0:
            return 1.0
        RewardReason.WRONG_PRODUCT.set_reason(metadata)
        return 0.0
    RewardReason.INVALID_RXN.set_reason(metadata)
    return 0.0


def valid_molecule_eval(
    yhat: str,
    y: str,  # noqa: ARG001
    soft: bool = False,  # noqa: ARG001
    test: bool = False,  # noqa: ARG001
    metadata: dict[str, JsonValue] | None = None,  # noqa: ARG001
) -> float:
    """Evaluate if yhat is valid molecule."""
    if not yhat:
        return 0.0
    mol = mol_from_smiles(yhat, sanitize=True)
    return float(mol is not None)


EVAL_FUNCTIONS: Mapping[str, RewardEvalFn] = {
    "str_eval": str_eval,
    "valid_mol_eval": valid_mol_eval,
    "caption_eval": caption_eval,
    "product_eval": product_eval,
    "rxn_eval": rxn_eval,
    "formula_eval": formula_eval,
    "functional_group_eval": functional_group_eval,
    "sol_eval": oracle_solubility_eval,
    "rxn_forward": oracle_rxn_eval,
    "should_not_answer_eval": str_eval,
    "should_answer_eval": valid_molecule_eval,
}


# These correspond to open-ended problems that do not have a
# unique molecule as answer.
APPLY_GOOD_MOLECULE_CHECK: set[str] = {
    "valid_mol_eval",
    "formula_eval",
    "functional_group_eval",
    "sol_eval",
}


def accuracy_reward(
    completions: Sequence[list[Mapping[str, str]]] | Sequence[str],
    solution: Iterable[str],
    reasoning: bool,
    metadata: list[dict[str, JsonValue]] | None = None,
    soft: bool = False,
    test: bool = False,
    good_molecule_bonus: float = 0.0,
    **kwargs,  # noqa: ARG001
) -> list[float]:
    """Reward function that checks if the completion is the same as the ground truth."""
    if isinstance(completions[0], list):
        messages = cast(Sequence[list[Mapping[str, str]]], completions)
        contents: Sequence[str] = [m[0]["content"] for m in messages]
    else:
        contents = cast(Sequence[str], completions)
    if soft and test:
        raise ValueError("Soft mode is not supported for test time accuracy reward.")
    rewards = []
    problem_types: list[str | None] = []

    if metadata is None:
        # Create empty metadata that we can use internal to this function
        metadata = [{} for _ in contents]
    else:
        if metadata:
            raise NotImplementedError(f"Received non-empty metadata {metadata}.")
        metadata.extend([{} for _ in contents])

    for content, info, meta in zip(contents, solution, metadata, strict=True):
        reward = 0.0
        reward_info = RewardFunctionInfo.model_validate(info)
        fxn_name, answer_info, problem_type = tuple(reward_info.model_dump().values())
        try:
            if test:
                answer: str | None = (
                    content.split("<answer>")[1].split("</answer>")[0]
                    if "<answer>" in content
                    else content
                )
            else:
                answer = extract_thought_answer_strict(content, reasoning=reasoning)[1]
            if answer is not None:
                # During test time, see if full SMILES string was given as input
                if problem_type == "valid_mol_eval" and test:
                    # If we're testing, we only allow full SMILES strings
                    reward = EVAL_FUNCTIONS[fxn_name](
                        answer, answer_info, test=test, metadata=meta
                    )
                else:
                    reward = EVAL_FUNCTIONS[fxn_name](
                        answer, answer_info, soft=soft, metadata=meta
                    )
                RewardReason.set_default_reason(reward, meta)

                if reward == 1.0 and fxn_name in APPLY_GOOD_MOLECULE_CHECK:
                    if FULL_SMILES_KEY not in meta:
                        raise ValueError(  # noqa: TRY301
                            f"Missing full SMILES key in metadata {meta}"
                            f" with reward function {fxn_name}."
                        )
                    full_smiles = cast(str, meta[FULL_SMILES_KEY])
                    mol = mol_from_smiles(full_smiles)
                    if mol is None:
                        raise ValueError(  # noqa: TRY301
                            f"Invalid full SMILES {full_smiles}"
                            f" with reward function {fxn_name}."
                        )
                    meta["is_good_molecule"] = not contains_bad_substruct(mol)
                    if meta["is_good_molecule"]:
                        reward += good_molecule_bonus

            else:
                RewardReason.FORMAT_FAILED.set_reason(meta)
            rewards.append(reward)
            problem_types.append(problem_type)
        except Exception:
            logger.exception(
                f"Unhandled exception in {fxn_name=} for {problem_type=}"
                f" with inputs {content=}, {answer_info=} {soft=}, and {test=}."
            )
            RewardReason.REWARD_FUNCTION_EXCEPTION.set_reason(meta)
            rewards.append(reward)
            problem_types.append(None)

    return rewards
