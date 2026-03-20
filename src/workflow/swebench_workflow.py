# %%
import os
from typing import Literal
import uuid

from datasets import load_dataset
import dotenv
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
import pandas as pd
from string import Template
from typing_extensions import TypedDict


from src.agent.analysis_agent import analysis_agent
from src.agent.fixing_agent import fixing_agent
from src.agent.knowledge_agent import (
    knowledge_analysis_agent,
    knowledge_summary_agent,
)
from src.agent.planning_agent import planning_agent
from src.graph.state import CustomState
from src.runtime.runtime_config import RuntimeConfig
from src.workflow.analysis_workflow import analysis_workflow
from src.workflow.plan_and_fix_workflow import plan_and_fix_workflow


rc = RuntimeConfig()

dotenv.load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env",
    )
)

members = ["analysis_workflow", "plan_and_fix_workflow"]
options = members + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next_agent: Literal[*options]
    thought: str


def input_handler_node(state: CustomState) -> Command[Literal["analysis_workflow"]]:
    """in issue solving, input handler will take input of
    1.swe-bench id,
    2.issue link and setup the env accordingly"""
    user_input = state["messages"][0].content
    if "/issues/" in user_input:
        # the input are github link
        rc.load_from_github_issue_url(user_input)
    else:
        rc.load_from_swe_rex_docker_instance(user_input)

    issue_description = rc.issue_desc
    return Command(
        update={
            "messages": [
                RemoveMessage(id=state["messages"][0].id),
                HumanMessage(content=issue_description),
            ],
            "last_agent": "input_handler",
        },
        goto="analysis_workflow",
    )


swebench_workflow_builder = StateGraph(CustomState)
swebench_workflow_builder.add_edge(START, "input_handler")
swebench_workflow_builder.add_node(
    "input_handler",
    input_handler_node,
    destinations=({"analysis_workflow": "input_handler-analysis_workflow"}),
)
swebench_workflow_builder.add_node(
    "analysis_workflow",
    analysis_workflow,
    destinations=({"plan_and_fix_workflow": "analysis_workflow-plan_and_fix_workflow"}),
)
swebench_workflow_builder.add_node(
    "plan_and_fix_workflow",
    plan_and_fix_workflow,
    destinations=({END: "END"}),
)

swebench_workflow_builder.add_edge("analysis_workflow", "plan_and_fix_workflow")
swebench_resolve_graph = swebench_workflow_builder.compile()


# # %%
if __name__ == "__main__":
    # set os env of LANGSMITH_TRACING to true
    rc = RuntimeConfig()

    # when using input_handler_node, no need to initialized
    os.environ["LANGSMITH_TRACING"] = "true"
    thread = {
        "recursion_limit": 100,
        "run_id": uuid.uuid4(),
        "tags": ["interrupt"],
        "configurable": {"thread_id": "1"},
    }
    preset = "astropy__astropy-12907"
    initial_input = {
        "messages": [HumanMessage(content=preset)],
        "preset": preset,
        "human_in_the_loop": False,
    }

    for chunk in swebench_resolve_graph.stream(
        initial_input, config=thread, stream_mode="values"
    ):
        if "messages" in chunk and len(chunk["messages"]) > 0:
            chunk["messages"][-1].pretty_print()
