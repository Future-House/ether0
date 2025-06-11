import asyncio
import pathlib
from typing import TYPE_CHECKING
from uuid import UUID

import litellm
from aviary.core import TASK_DATASET_REGISTRY
from aviary.envs.litqa import GradablePaperQAEnvironment, LitQATaskDataset
from ldp.agent import SimpleAgent
from ldp.alg import (
    Evaluator,
    EvaluatorConfig,
    MeanMetricsCallback,
    StoreTrajectoriesCallback,
)
from paperqa import Docs, Settings
from paperqa.agents.search import FAILED_DOCUMENT_ADD_ID, get_directory_index
from paperqa.settings import (
    AgentSettings,
    IndexSettings,
    MaybeSettings,
    ParsingSettings,
)
from tqdm.asyncio import tqdm_asyncio

from ether0.evaluation import Ether0RepoOpenAnswer, Ether0RepoTaskSplit
from ether0.zero_shot import ETHER0_PACKAGE_DIR, REPO_ROOT_DIR, REPO_ROOT_FILES

if TYPE_CHECKING:
    import anyio


def update_litellm_max_callbacks(value: int = 1000) -> None:
    """Update litellm's MAX_CALLBACKS limit, can call with default to defeat this limit.

    SEE: https://github.com/BerriAI/litellm/issues/9792
    """
    litellm.litellm_core_utils.logging_callback_manager.LoggingCallbackManager.MAX_CALLBACKS = (
        value
    )


class Ether0RepoBuilderDocs(Docs):
    """Docs specific to the ether0 repo/package."""

    async def bulk_aadd(
        self, *files: pathlib.Path, settings: MaybeSettings | None = None
    ) -> None:
        names = [str(f.relative_to(REPO_ROOT_DIR)) for f in files]
        await tqdm_asyncio.gather(
            *(
                self.aadd(path=f, docname=name, citation=name, settings=settings)
                for f, name in zip(files, names, strict=True)
            ),
            desc="Adding files",
            ncols=0,
        )


class Ether0RepoTaskDataset(LitQATaskDataset):
    """LitQA task variant for ether0 repo QA tasks."""

    def __init__(
        self,
        *args,
        split: str | Ether0RepoTaskSplit = Ether0RepoTaskSplit.EVAL,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data = Ether0RepoTaskSplit(split).get_task()

    def _make_gradable_environment(
        self,
        ideal_answer: str,
        distractors: str | list[str],
        question_id: str | UUID,
        question: str,
        sources: str | list[str] | None = None,
    ) -> GradablePaperQAEnvironment:
        if distractors:
            raise ValueError(f"{type(self).__name__} does not support distractors.")
        return GradablePaperQAEnvironment(
            query=Ether0RepoOpenAnswer(
                question_id=(
                    UUID(question_id) if isinstance(question_id, str) else question_id
                ),
                question=question,
                ideal_answer=ideal_answer,
                **(self._question_kwargs or {}),
            ),
            settings=self._settings,
            docs=self._base_docs.model_copy(),
            sources=sources,
            rewards=self._rewards,
            **self._env_kwargs,
        )

    def get_new_env_by_idx(self, idx: int) -> GradablePaperQAEnvironment:
        sources: list[str] = []  # TODO: add sources to measure recall
        return self._make_gradable_environment(
            ideal_answer=self.data[idx]["answer"],
            distractors=[],  # Placeholder
            question_id=self.data[idx]["id"],
            question=self.data[idx]["question"],
            sources=sources,
        )

    def __len__(self) -> int:
        return len(self.data)


TASK_DATASET_NAME = "ether0-repo-qa"
TASK_DATASET_REGISTRY[TASK_DATASET_NAME] = (
    Ether0RepoTaskDataset.__module__,
    Ether0RepoTaskDataset.__name__,
)


SUFFIX_TO_DISALLOW = {
    "uv.lock",  # Changes too often
    ".pyc",  # Not human-readable
    ".bloom",  # Not human-readable
}


def filter_valid_files(file: "anyio.Path | pathlib.Path") -> bool:
    """Filter out files that should not be indexed."""
    file_ = pathlib.Path(str(file))  # Use sync to avoid nest-asyncio with is_dir()
    try:
        return (
            not file_.is_dir()
            and file_.suffix not in SUFFIX_TO_DISALLOW
            and (
                str(file_) in REPO_ROOT_FILES
                or bool(file_.relative_to(ETHER0_PACKAGE_DIR))
            )
        )
    except ValueError:  # Not a relative path of ETHER0_PACKAGE_DIR
        return False


def make_parsing_and_index_settings() -> tuple[ParsingSettings, IndexSettings]:
    return ParsingSettings(
        # Disable doc details upgrade to avoid failed citation acquisition
        # on files such as pyproject.toml or non-descript Python files
        use_doc_details=False
    ), IndexSettings(
        name="ether0-all",
        paper_directory=REPO_ROOT_DIR,
        files_filter=filter_valid_files,
    )


async def main() -> None:
    update_litellm_max_callbacks()

    parsing_settings, index_settings = make_parsing_and_index_settings()
    settings = Settings(
        parsing=parsing_settings,
        agent=AgentSettings(index=index_settings),
    )

    # Force index build up front
    search_index = await get_directory_index(settings=settings)

    # Show how to audit a Docs object in the built index
    index_files = await search_index.index_files
    docs: Docs = await search_index.get_saved_object(  # type: ignore[assignment]
        next(
            iter({k: v for k, v in index_files.items() if v != FAILED_DOCUMENT_ADD_ID})
        )
    )
    print(f"Sample Docs object has docnames {docs.docnames}.")

    dataset = Ether0RepoTaskDataset(settings=settings)
    mcb = MeanMetricsCallback(eval_dataset=dataset, track_tool_usage=True)
    tcb = StoreTrajectoriesCallback()
    evaluator = Evaluator(
        config=EvaluatorConfig(batch_size=128),
        agent=SimpleAgent(),
        dataset=dataset,
        callbacks=[mcb, tcb],
    )
    await evaluator.evaluate()
    print(f"Eval-set performance: {mcb.eval_means}.")


if __name__ == "__main__":
    asyncio.run(main())
