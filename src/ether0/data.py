import logging
import re
from collections.abc import Collection, Mapping
from pathlib import Path

from datasets import Dataset
from molbloom import BloomFilter, canon
from rdkit import Chem
from rdkit.Chem.Draw import MolDraw2D, MolDraw2DSVG  # pylint: disable=no-name-in-module
from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DCairo
from rdkit.Chem.rdChemReactions import (  # pylint: disable=no-name-in-module
    ReactionFromSmarts,
)
from rdkit.Chem.rdDepictor import (  # pylint: disable=no-name-in-module
    Compute2DCoords,
    StraightenDepiction,
)
from rdkit.Chem.rdMolDescriptors import (  # pylint: disable=no-name-in-module
    GetMorganFingerprint,
)
from rdkit.Chem.rdmolfiles import MolFromSmiles  # pylint: disable=no-name-in-module

logger = logging.getLogger(__name__)


PROBLEM_CATEGORY_TO_NICKNAME: Mapping[str, str] = {
    "functional-group": "functional group",
    "molecule-caption": "molecule caption",
    "molecule-completion": "SMILES completion",
    "molecule-formula": "elucidation",
    "molecule-name": "IUPAC name",
    "oracle-solubility": "solubility edit",
    "property": "multiple choice",
    "property-cat-brain": "BBB permeability",
    "property-cat-eve": "Human receptor binding",
    "property-cat-safety": "safety",
    "property-cat-smell": "scent",
    "property-regression-pka": "pKa",
    "property-regression-ld50": "LD50",
    "property-regression-adme": "ADME",
    "reaction-prediction": "reaction prediction",
    "retro-synthesis": "retrosynthesis",
    "simple-formula": "molecular formula",
    "property-regression-adme/log_hlm_clint": "log of HLM CL$_{\\text{int}}$",
    "property-regression-adme/log_mdr1-mdck_er": "log of MDR1-MDCK ER",
    "property-regression-adme/log_rlm_clint": "log of RLM CL$_{\\text{int}}$",
    "property-regression-adme/log_solubility": "log of aqueous solubility",
}


def get_problem_type(row: Mapping[str, str]) -> str:
    return row.get("problem_type") or row["type"]


def get_problem_category(problem_type: str | None) -> str:
    return (problem_type or "").split("/", maxsplit=1)[0]


def get_problem_categories_from_datasets(*datasets: Dataset) -> Collection[str]:
    return {
        get_problem_category(pt)
        for dataset in datasets
        for pt in (dataset.hf_dataset if hasattr(dataset, "hf_dataset") else dataset)[
            "problem_type"
        ]
    }


# Use this regex with findall to extract SMILES strings from text.
# Note this function currently fails on counterions e.g.
# Cc1ccc(-c2ccc3c(c2)c2ccccc2c[n+]3C)cc1.[Cl-]
SMILES_PATTERN = re.compile(
    r"(?<!\w)(?:(?:Cl|Br|[BCNOPSFIC]|[cnops]|\[[^\]]+?\]|[0-9@+\-=#\\/()%])){4,}(?!\w)"
)


def make_sized_d2d(w: int = 400, h: int = 300) -> MolDraw2DCairo:
    return MolDraw2DCairo(w, h)


def draw_molecule(
    smiles: str, bg_opacity: float = 1.0, d2d: MolDraw2D | None = None
) -> str:
    """Draw a SMILES molecule and return the drawing string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to convert {smiles=} to a molecule.")
    Compute2DCoords(mol)
    StraightenDepiction(mol)
    if d2d is None:
        d2d = MolDraw2DSVG(-1, -1)
    dopts = d2d.drawOptions()
    dopts.useBWAtomPalette()
    dopts.setBackgroundColour((*dopts.getBackgroundColour(), bg_opacity))
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()


def draw_reaction(
    rxn_smiles: str, bg_opacity: float = 1.0, d2d: MolDraw2D | None = None
) -> str:
    rxn = ReactionFromSmarts(rxn_smiles, useSmiles=True)
    if d2d is None:
        d2d = MolDraw2DSVG(-1, -1)
    dopts = d2d.drawOptions()
    dopts.useBWAtomPalette()
    dopts.setBackgroundColour((*dopts.getBackgroundColour(), bg_opacity))
    d2d.DrawReaction(rxn)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()


# Precompiled SMARTS patterns for protected bonds and ring atoms
_ring_db_pat = Chem.MolFromSmarts("[#6R,#16R]=[OR0,SR0,CR0,NR0]")
_ring_atom_pat = Chem.MolFromSmarts("[R]")


bloom_filters: dict[str, BloomFilter] = {}


def _get_bits(mol: Chem.Mol) -> set[str]:
    """Get the fingerprint bits from a molecule."""
    # the keys are the actual bits
    bi: dict[int, tuple[tuple[int, int], ...]] = {}
    GetMorganFingerprint(mol, 2, bitInfo=bi)  # type: ignore[arg-type]
    return {str(k) for k in bi}


ETHER0_DIR = Path(__file__).parent


def _get_bloom_filter(name: str) -> BloomFilter:
    if name in bloom_filters:
        return bloom_filters[name]
    bloom_filters[name] = BloomFilter(str(ETHER0_DIR / f"{name}.bloom"))
    return bloom_filters[name]


def get_ring_system(mol: Chem.Mol) -> list[str]:
    """
    Extracts ring systems from an RDKit molecule and returns a list of SMILES.
    Bonds not in rings and not protected (e.g., ring carbonyls) are cleaved.

    Source: https://github.com/PatWalters/useful_rdkit_utils/blob/edb126e3fd71870ae2d1c9440b904106e3ef97a2/useful_rdkit_utils/ring_systems.py#L13
    Which has a MIT license, copyright 2021-2025 PatWalters.
    """  # noqa: D205
    # Copy to avoid mutating original
    mol = Chem.Mol(mol)

    # Tag protected bonds
    for bond in mol.GetBonds():
        bond.SetBoolProp("protected", False)  # noqa: FBT003
    for a1, a2 in mol.GetSubstructMatches(_ring_db_pat):
        b = mol.GetBondBetweenAtoms(a1, a2)
        b.SetBoolProp("protected", True)  # noqa: FBT003

    # Cleave linker bonds
    cleave_idxs = [
        b.GetIdx()
        for b in mol.GetBonds()
        if not b.IsInRing()
        and not b.GetBoolProp("protected")
        and b.GetBondType() == Chem.BondType.SINGLE
    ]
    if cleave_idxs:
        frag_mol = Chem.FragmentOnBonds(mol, cleave_idxs)
        Chem.SanitizeMol(frag_mol)
    else:
        frag_mol = mol

    # Split into fragments and clean up
    ring_smiles: list[str] = []
    for frag in Chem.GetMolFrags(frag_mol, asMols=True):
        if frag.HasSubstructMatch(_ring_atom_pat):
            for atom in frag.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomicNum(1)
                    atom.SetIsotope(0)
            frag = Chem.RemoveAllHs(frag)  # noqa: PLW2901
            # Fix stereo on terminal double bonds
            for bd in frag.GetBonds():
                if bd.GetBondType() == Chem.BondType.DOUBLE and (
                    1 in {bd.GetBeginAtom().GetDegree(), bd.GetEndAtom().GetDegree()}
                ):
                    bd.SetStereo(Chem.BondStereo.STEREONONE)
            ring_smiles.append(Chem.MolToSmiles(frag))

    return ring_smiles


def is_reasonable_ring_system(mol: Chem.Mol, ref_mol: Chem.Mol | None = None) -> bool:
    """
    Check if a molecule has a reasonable ring system.

    Either no rings or the ring system is found in known rings.
    If reference is provided, thsos are assumed valid.
    """
    bloom_filter = _get_bloom_filter("rings")
    ring_systems = [canon(r) for r in get_ring_system(mol)]
    # remove from consideration all rings in ref_mol, since we'll always assume they're correct
    if ref_mol:
        ref_ring_systems = [canon(r) for r in get_ring_system(ref_mol)]
        ring_systems = [ring for ring in ring_systems if ring not in ref_ring_systems]
    return all((r in bloom_filter) for r in ring_systems)


def is_reasonable_fp(mol: Chem.Mol, ref_mol: Chem.Mol | None = None) -> bool:
    """
    Check if a molecule has a reasonable fingerprint.

    If reference is provided, those fingerprints are assumed valid.
    """
    bloom_filter = _get_bloom_filter("fingerprints")
    bits: Collection[str] = _get_bits(mol)
    # remove from consideration all rings in ref_mol, since we'll always assume they're correct
    if ref_mol:
        ref_bits = _get_bits(ref_mol)
        bits = [bit for bit in bits if bit not in ref_bits]
    return all((b in bloom_filter) for b in bits)


def mol_from_smiles(smiles: str, *args, **kwargs) -> Chem.Mol | None:
    """MolFromSmiles is type-hinted to always return Mol, but can return None."""
    return MolFromSmiles(smiles, *args, **kwargs)
