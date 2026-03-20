# %%
import os
from typing import Literal
import uuid

import dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from typing_extensions import TypedDict

from src.agent.analysis_agent import analysis_agent
from src.runtime.runtime_config import RuntimeConfig
from src.graph.state import CustomState

rc = RuntimeConfig()

dotenv.load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env",
    )
)

members = ["analysis_agent"]
options = members + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next_agent: Literal[*options]
    thought: str


def input_handler_node(state: CustomState) -> Command[Literal["analysis_agent"]]:
    """in issue solving, input handler will take input of
    1.swe-bench id,
    2.issue link and setup the env accordingly"""

    rc = RuntimeConfig()
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
        goto="analysis_agent",
    )


def analysis_agent_node(state: CustomState) -> Command[END]:
    result = analysis_agent.invoke(state)
    new_messages = result["messages"][len(state["messages"]) :]

    for msg in new_messages:
        if isinstance(msg, AIMessage):
            msg.name = "analysis_agent"

    return Command(
        update={"messages": new_messages, "last_agent": "analysis_agent"},
        goto=END,
    )


analysis_agent_builder = StateGraph(CustomState)
analysis_agent_builder.add_edge(START, "input_handler")
analysis_agent_builder.add_node(
    "input_handler",
    input_handler_node,
    destinations=({"analysis_agent": "input_handler-analysis_agent"}),
)
analysis_agent_builder.add_node(
    "analysis_agent",
    analysis_agent_node,
    destinations=({END: "END"}),
)


analysis_workflow = analysis_agent_builder.compile()


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
    initial_input = {
        "messages": [
            HumanMessage(
                content="https://github.com/gitpython-developers/GitPython/issues/1413"
            )
        ],
        "preset": "https://github.com/gitpython-developers/GitPython/issues/1413",
        "human_in_the_loop": False,
    }

    for chunk in analysis_workflow.stream(
        initial_input, config=thread, stream_mode="values"
    ):
        if "messages" in chunk and len(chunk["messages"]) > 0:
            chunk["messages"][-1].pretty_print()
