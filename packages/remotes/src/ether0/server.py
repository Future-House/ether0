import argparse
import os
import re
import secrets
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import numpy.typing as npt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from molbloom import buy
from molsol import KDESol
from onmt import opts
from onmt.translate.translator import build_translator
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.utils.parse import ArgumentParser
from pydantic import BaseModel
from rdkit import Chem

ETHER0_DIR = Path(__file__).parent

auth_scheme = HTTPBearer()


def validate_token(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),  # noqa: B008
) -> str:
    # NOTE: don't use os.environ.get() to avoid possible empty string matches, and
    # to have clearer server failures if the AUTH_TOKEN env var isn't present
    if not secrets.compare_digest(
        credentials.credentials, os.environ["ETHER0_REMOTES_API_TOKEN"]
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


app = FastAPI(title="ether0 remotes server", dependencies=[Depends(validate_token)])


class MolecularTransformer:
    """Uses code from https://doi.org/10.1021/acscentsci.9b00576."""

    DEFAULT_MOLTRANS_MODEL_PATH: ClassVar[Path] = (
        ETHER0_DIR / "USPTO480k_model_step_400000.pt"
    )

    def __init__(self):
        # Use `or None` to deny setting empty string to the environment variable
        os_environ_model_path = (
            os.environ.get("ETHER0_REMOTES_MOLTRANS_MODEL_PATH") or None
        )
        self.model_path = os_environ_model_path or str(self.DEFAULT_MOLTRANS_MODEL_PATH)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"MolTrans model not found"
                f"{f', did you misconfigure the path {os_environ_model_path}?' if os_environ_model_path else '.'}"  # noqa: E501
                " Please properly configure the environment variable"
                " 'ETHER0_REMOTES_MOLTRANS_MODEL_PATH',"
                f" or the default path checked is {self.DEFAULT_MOLTRANS_MODEL_PATH}."
            )

    @staticmethod
    def translate(opt: argparse.Namespace) -> None:
        ArgumentParser.validate_translate_opts(opt)
        logger = init_logger(opt.log_file)

        translator = build_translator(opt, logger=logger, report_score=True)
        src_shards = split_corpus(opt.src, opt.shard_size)
        tgt_shards = split_corpus(opt.tgt, opt.shard_size)
        features_shards = []
        features_names = []
        for feat_name, feat_path in opt.src_feats.items():
            features_shards.append(split_corpus(feat_path, opt.shard_size))
            features_names.append(feat_name)
        shard_pairs = zip(src_shards, tgt_shards, *features_shards)  # noqa: B905

        for (src_shard, tgt_shard, *features_shard) in shard_pairs:
            features_shard_ = defaultdict(list)
            for j, x in enumerate(features_shard):
                features_shard_[features_names[j]] = x
            translator.translate(
                src=src_shard,
                src_feats=features_shard_,
                tgt=tgt_shard,
                batch_size=opt.batch_size,
                batch_type=opt.batch_type,
                attn_debug=opt.attn_debug,
                align_debug=opt.align_debug,
            )

    @staticmethod
    def smiles_tokenizer(smiles: str) -> str:
        smiles_regex = re.compile(
            r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
        )
        tokens = list(smiles_regex.findall(smiles))
        return " ".join(tokens)

    @staticmethod
    def canonicalize_smiles(smiles: str) -> str:
        # Try to use canonical smiles because original uspto is distributed in canonical form.
        # If fails, we trust the augmentation and use the original smiles.
        try:
            return Chem.MolToSmiles(
                Chem.MolFromSmiles(smiles), isomericSmiles=True, canonical=True
            )
        except Exception as err:
            # If rdkit failed, it means some molecule is invalid.
            # Here we catch which ones are invalid so we inform what's wrong
            # on the error message.
            invalid_smiles = []
            for mol in smiles.split("."):
                try:
                    Chem.MolToSmiles(
                        Chem.MolFromSmiles(mol), isomericSmiles=True, canonical=True
                    )
                except:  # noqa: E722
                    invalid_smiles.append(mol)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "The reaction could not be parsed by RDKit. The following"
                    f" SMILES were invalid: {', '.join(invalid_smiles)}"
                ),
            ) from err

    def run(self, reaction: str) -> tuple[str, uuid.UUID]:
        """Translates SMILES reaction strings using MolTrans model.

        Args:
            reaction: SMILES representation of a chemical reaction

        Returns:
            SMILES representation of the predicted product and a job ID
        """
        # Create a unique ID for the request
        job_id = uuid.uuid4()

        # Create temporary files for use in mol moltransformer
        with (
            tempfile.NamedTemporaryFile(
                mode="w+", delete=False, encoding="utf-8"
            ) as precursor_file,
            tempfile.NamedTemporaryFile(
                mode="w+", delete=False, encoding="utf-8"
            ) as output_file,
        ):

            # Write tokenized reaction to the precursor file
            precursor_file.write(MolecularTransformer.smiles_tokenizer(reaction))
            precursor_file.flush()

            # OpenNMT expects to receive a list of arguments to translate
            parser = ArgumentParser()
            opts.config_opts(parser)
            opts.translate_opts(parser)

            args_dict = {
                "model": self.model_path,
                "src": precursor_file.name,
                "output": output_file.name,
                "batch_size": "64",
                "beam_size": "50",
                "max_length": "300",
            }
            args_list = [f"--{k}={v}" for k, v in args_dict.items()]
            opt = parser.parse_args(args_list)

            MolecularTransformer.translate(opt)

            output_file.close()
            prediction = Path(output_file.name).read_text(encoding="utf-8")

            # Clean up temporary files
            # we don't care if a failure leaves them dangling,
            # since they are in a temp dir
            os.unlink(precursor_file.name)
            os.unlink(output_file.name)

        return prediction.replace(" ", "").strip(), job_id


class MolBloom:
    """Uses code from https://doi.org/10.1186/s13321-023-00765-1."""

    def __init__(self) -> None:
        # trigger eager loading of the bloom filter
        buy("C1=CC=CC=C1", catalog="zinc20")
        self.bloom = buy

    def run(self, smiles: str) -> bool:
        """Checks if a molecule is purchasable using MolBloom.

        Args:
            smiles: SMILES representation of a molecule

        Returns:
            True if the molecule is purchasable, False otherwise
        """
        return self.bloom(smiles, canonicalize=True, catalog="zinc20")


class Solubility:
    """Uses code from https://doi.org/10.1039/D3DD00217A."""

    def __init__(self) -> None:
        self.sol = KDESol()

    def run(self, smiles: str) -> npt.NDArray[np.float32] | Literal[False]:
        """Computes solubility prediction for a molecule using KDESol.

        Args:
            smiles: SMILES representation of a molecule.

        Returns:
            Numpy array containing the mean predicted solubility,
                aleatoric uncertainty (au), and epistemic uncertainty (eu).
        """
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return False  # type: ignore[unreachable]
        prediction = self.sol(Chem.MolToSmiles(m, canonical=True, isomericSmiles=False))
        if prediction is None:
            # Try without canonicalization.
            # The model is an LSTM that uses tokens generated from SELFIES tokens.
            # Depending on the SMILES notation, the model might not have the necessary tokens
            # in its vocabulary to describe the molecule.
            prediction = self.sol(smiles)
        return prediction if prediction is not None else False


class MolTransRequest(BaseModel):
    reaction: str


@app.post("/translate")
def translate_endpoint(request: MolTransRequest) -> dict[str, str | uuid.UUID]:
    reaction = request.reaction.replace(" ", "")
    if not reaction.count(">") == 2:  # noqa: PLR2004
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Syntax error in the reaction SMILES: {reaction}\n"
                "The reaction should have two '>' characters, and no spaces."
            ),
        )
    rxn = reaction.split(">")[:-1]
    query_reaction = MolecularTransformer.canonicalize_smiles(
        ".".join([r for r in rxn if r])
    )

    product, job_id = MolecularTransformer().run(query_reaction)
    return {
        "product": product,
        "id": job_id,
        "reaction": query_reaction + ">>" + product,
    }


class MolBloomRequest(BaseModel):
    smiles: list[str] | str


@app.post("/is_purchasable")
def is_purchasable_endpoint(request: MolBloomRequest) -> dict[str, bool]:
    is_purchasable = MolBloom().run
    smiles = request.smiles
    if isinstance(smiles, str):
        smiles = [smiles]
    return {s: is_purchasable(s) for s in smiles}


class SmilesRequest(BaseModel):
    smiles: str


@app.post("/compute_solubility")
def compute_solubility_endpoint(
    request: SmilesRequest,
) -> dict[str, float] | dict[str, str]:
    if "." in request.smiles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only single molecules are supported",
        )
    prediction = Solubility().run(smiles=request.smiles)
    if prediction is False:
        return {"error": "Solubility prediction failed."}
    mean, au, eu = prediction.tolist()
    return {"mean": mean, "au": au, "eu": eu}


def main() -> None:
    """Run uvicorn to serve the FastAPI app."""
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Serving requires the 'serve' extra for the `uvicorn` package. Please:"
            " `pip install ether0.remotes[serve]`."
        ) from exc

    uvicorn.run("ether0.server:app")


if __name__ == "__main__":
    main()
