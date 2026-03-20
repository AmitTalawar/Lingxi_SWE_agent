# %%
import os
from typing import Literal
import uuid

import dotenv
from datasets import load_dataset
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
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
from src.runtime.runtime_config import RuntimeConfig
from src.graph.state import CustomState
from src.prompts.knowledge_agent import (
    KNOWLEDGE_AGENT_ANALYSIS_USER_PROMPT,
    KNOWLEDGE_AGENT_SUMMARY_USER_PROMPT,
)
from src.tools.utils import get_runtime_config

rc = RuntimeConfig()

dotenv.load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env",
    )
)

members = ["knowledge_agent", "analysis_agent", "planning_agent", "fixing_agent"]
options = members + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next_agent: Literal[*options]
    thought: str


def input_handler_node(state: CustomState) -> Command[Literal["analysis_agent"]]:
    """in issue solving, input handler will take input of
    1.swe-bench id,
    2.issue link and setup the env accordingly"""

    if not rc.initialized:
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
        goto="planning_agent",
    )


def planning_agent_node(state: CustomState) -> Command[Literal["fixing_agent"]]:
    result = planning_agent.invoke(state)

    new_messages = result["messages"][len(state["messages"]) :]

    for msg in new_messages:
        if isinstance(msg, AIMessage):
            msg.name = "planning_agent"

    return Command(
        update={"messages": new_messages, "last_agent": "planning_agent"},
        goto="fixing_agent",
    )


def fixing_agent_node(state: CustomState) -> Command[END]:
    result = fixing_agent.invoke(state)
    new_messages = result["messages"][len(state["messages"]) :]

    for msg in new_messages:
        if isinstance(msg, AIMessage):
            msg.name = "fixing_agent"

    return Command(
        update={"messages": new_messages, "last_agent": "fixing_agent"},
        goto=END,
    )


plan_and_fix_workflow_builder = StateGraph(CustomState)
plan_and_fix_workflow_builder.add_edge(START, "input_handler")
plan_and_fix_workflow_builder.add_node(
    "input_handler",
    input_handler_node,
    destinations=({"planning_agent": "input_handler-planning_agent"}),
)
plan_and_fix_workflow_builder.add_node(
    "planning_agent",
    planning_agent_node,
    destinations=({"fixing_agent": "planning_agent-fixing_agent"}),
)
plan_and_fix_workflow_builder.add_node(
    "fixing_agent",
    fixing_agent_node,
    destinations=({END: "END"}),
)

plan_and_fix_workflow = plan_and_fix_workflow_builder.compile()


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
    preset = "astropy__astropy-12907"  # "https://github.com/gitpython-developers/GitPython/issues/1413"
    initial_input = {
        "messages": [HumanMessage(content=preset)],
        "preset": preset,
        "human_in_the_loop": False,
    }

    for chunk in plan_and_fix_workflow.stream(
        initial_input, config=thread, stream_mode="values"
    ):
        if "messages" in chunk and len(chunk["messages"]) > 0:
            chunk["messages"][-1].pretty_print()
