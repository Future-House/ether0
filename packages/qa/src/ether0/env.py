from typing import Self, cast
from uuid import UUID, uuid4

from aviary.core import Tool
from futurehouse_client.models.rest import WorldModel
from paperqa import Docs
from paperqa.agents.tools import Complete, PaperSearch
from paperqa.utils import get_year
from paperqa_server.agents.env import WorldModelPQAEnvironment
from paperqa_server.agents.tools import (
    GatherEvidenceTool,
    GenerateAnswerTool,
    PaperSearchTool,
    PlannerTool,
)
from paperqa_server.callback import MockWebsocket
from paperqa_server.models import (
    DEFAULT_WORLD_MODEL,
    WORLD_MODEL_AGENT_PROMPT,
    WORLD_MODEL_ITERATION_PLAN_PROMPT,
    WORLD_MODEL_PLAN_PROMPT,
    WORLD_MODEL_QA_PROMPT,
    AgentSettings,
    AnswerSettings,
    ParsingConfiguration,
    PromptSettings,
    QueryRequest,
    QuerySettings,
)
from paperqa_server.settings import INTERNAL_CASCADING_MODEL_ORDER

from ether0.task import make_parsing_and_index_settings


class Ether0RepoWorldModelPQAEnvironment(WorldModelPQAEnvironment):
    @classmethod
    def from_task(  # type: ignore[override]
        cls,
        task: str,
        world_model: str | UUID | None = None,
        trajectory_id: str | UUID | None = None,
        continued_trajectory_id: str | UUID | None = None,
        start_db_on_reset: bool = True,
    ) -> Self:
        """Query a planning PaperQA to update a world model."""
        # Kind of indirection, but we put it here
        # so we can put it static into QA prompt

        world_model_id: UUID | None = None
        if world_model is None:
            world_model: WorldModel | None = WorldModel(  # type: ignore[no-redef]
                content=DEFAULT_WORLD_MODEL.format(task=task)
            )
        elif isinstance(world_model, str):
            world_model_id = UUID(world_model)
            world_model = None
        elif isinstance(world_model, UUID):
            world_model_id = world_model

        parsing_settings, index_settings = make_parsing_and_index_settings()
        return cls(
            world_model=world_model,  # type: ignore[arg-type]
            world_model_id=world_model_id,
            query=None,  # Pull from QueryRequest
            settings=None,  # Pull from QueryRequest
            docs=Docs(),
            websocket=MockWebsocket(),
            query_request=QueryRequest(
                id=trajectory_id or uuid4(),
                query=task,
                settings=QuerySettings(
                    override_model_selection_using_rate_limits=True,
                    agent=AgentSettings(
                        agent_prompt=WORLD_MODEL_AGENT_PROMPT,
                        gather_evidence_early_stop_count=4,
                        tool_names=[
                            PaperSearch.TOOL_FN_NAME,
                            GatherEvidenceTool.TOOL_FN_NAME,
                            GenerateAnswerTool.TOOL_FN_NAME,
                            PlannerTool.TOOL_FN_NAME,
                            Complete.TOOL_FN_NAME,
                        ],
                        index=index_settings,
                    ),
                    answer=AnswerSettings(
                        answer_length="250000",
                        evidence_summary_length="about 200",  # we do way more of these, they can be shorter  # noqa: E501
                        use_critic_in_gen_answer=True,
                        gen_answer_is_world_diff=True,
                    ),
                    parsing=ParsingConfiguration(**parsing_settings.model_dump()),
                    prompts=PromptSettings(
                        qa=WORLD_MODEL_QA_PROMPT,
                        plan=WORLD_MODEL_PLAN_PROMPT,
                        iteration_plan=WORLD_MODEL_ITERATION_PLAN_PROMPT,
                    ),
                    # note we set the qa prompt later
                ),
            ),
            start_db_on_reset=start_db_on_reset,
            continued_trajectory_id=continued_trajectory_id,
            use_internal_tools=True,
            model_cascade=INTERNAL_CASCADING_MODEL_ORDER,
        )

    def make_tools(self) -> list[Tool]:
        tools = super().make_tools()
        search_tool = Tool.from_function(
            PaperSearch(
                settings=self._settings,
                embedding_model=self._settings.get_embedding_model(),
            ).paper_search
        )
        for pname in ("min_year", "max_year"):
            search_tool.info.get_properties()[pname]["description"] = cast(
                "str", search_tool.info.get_properties()[pname]["description"]
            ).format(current_year=get_year())
        prior_search_tool_index = [
            i
            for i, tool in enumerate(tools)
            if tool.info.name == PaperSearchTool.TOOL_FN_NAME
        ]
        if len(prior_search_tool_index) != 1:
            raise NotImplementedError(
                f"Expected one search tool, found {len(prior_search_tool_index)} in"
                f" tool names { [t.info.name for t in tools] }."
            )
        tools[prior_search_tool_index[0]] = search_tool
        return tools
