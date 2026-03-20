# %%
import os
from typing import Literal
import uuid

import dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
import pandas as pd
from string import Template
from typing_extensions import TypedDict

from src.agent.knowledge_agent import (
    knowledge_analysis_agent,
    knowledge_summary_agent,
)
from src.runtime.runtime_config import RuntimeConfig
from src.graph.state import CustomState
from src.prompts.knowledge_agent import (
    KNOWLEDGE_AGENT_ANALYSIS_USER_PROMPT,
    KNOWLEDGE_AGENT_SUMMARY_USER_PROMPT,
)

rc = RuntimeConfig()

dotenv.load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env",
    )
)

members = ["knowledge_agent"]
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
        goto="historic_issue_patch_pair_retriever",
    )


def historic_issue_patch_pair_retriever_node(state: CustomState):
    """The knowledge construction workflow expects issue description and patch as input.
    As outlined in the paper, the corresponding issue should be a historic one, defined
    as semantically similar respective to the current SWEBench instance. The provided
    file, "20250917-Historic-Issue-Patch-Pairs-astropy__astropy.jsonl", contains samples
    of real-world examples for what such issues are expected to look like for the repo
    astropy/astropy."""
    instance_id = state["preset"]
    historic_issue_patch_pairs = pd.read_json(
        os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "data",
            "20250917-Historic-Issue-Patch-Pairs-astropy__astropy.jsonl",
        ),
        lines=True,
    )  # Load the sample data for historic issue patch pairs
    pair = None
    if "/issue/" not in state["preset"]:
        samples = historic_issue_patch_pairs[
            historic_issue_patch_pairs.instance_id == instance_id
        ]
        if samples.shape[0] > 0:
            pair = samples.sample(n=1).iloc[0]
    if pair is None:
        raise ValueError(
            f"No historic issue patch pair was found for SWEbench instance {state["preset"]}.\
                Please check the provided instance_id or historic data pairs."
        )

    historic_issue_description = pair.retrieved_issue_desc
    historic_issue_patch = pair.retrieved_patch

    print(
        f"Historic info. Issue: {historic_issue_description}. Patch: {historic_issue_patch[:100]}"
    )
    return Command(
        update={
            "historic_issue_description": historic_issue_description,
            "historic_issue_patch": historic_issue_patch,
        },
        goto="knowledge_agent",
    )


def knowledge_agent_node(state: CustomState) -> Command[END]:
    # Analysis
    custom_state_analysis = {"messages": []}
    custom_state_analysis["messages"] += [
        HumanMessage(
            content=Template(KNOWLEDGE_AGENT_ANALYSIS_USER_PROMPT).substitute(
                {
                    "ISSUE_DESCRIPTION": state["historic_issue_description"],
                    "PATCH": state["historic_issue_patch"],
                }
            )
        )
    ]
    print("Start knowledge agent analysis")
    result = knowledge_analysis_agent.invoke(custom_state_analysis)
    new_messages_analysis = result["messages"][len(custom_state_analysis["messages"]) :]
    analysis_result = new_messages_analysis[-1]

    # Summary
    custom_state_summary = {"messages": []}
    custom_state_summary["messages"] += [
        HumanMessage(
            content=Template(KNOWLEDGE_AGENT_SUMMARY_USER_PROMPT).substitute(
                {
                    "HISTORIC_ISSUE_ANALYSIS": analysis_result.content,
                    "PATCH": state["historic_issue_patch"],
                }
            )
        )
    ]
    print("Start knowledge agent summary")
    result = knowledge_summary_agent.invoke(custom_state_summary)
    new_messages_summary = result["messages"][len(custom_state_summary["messages"]) :]
    summary_result = new_messages_summary[-1]

    # Merge results and proceed
    new_messages = [
        AIMessage(content=f"{analysis_result.content}\n\n{summary_result.content}")
    ]

    for msg in new_messages:
        if isinstance(msg, AIMessage):
            msg.name = "knowledge_agent"

    return Command(
        update={"messages": new_messages, "last_agent": "knowledge_agent"},
        goto=END,
    )


knowledge_agent_builder = StateGraph(CustomState)
knowledge_agent_builder.add_edge(START, "input_handler")
knowledge_agent_builder.add_node(
    "input_handler",
    input_handler_node,
    destinations=(
        {
            "historic_issue_patch_pair_retriever": "input_handler-historic_issue_patch_pair_retriever"
        }
    ),
)
knowledge_agent_builder.add_node(
    "historic_issue_patch_pair_retriever",
    historic_issue_patch_pair_retriever_node,
    destinations=(
        {"knowledge_agent": "historic_issue_patch_pair_retriever-knowledge_agent"}
    ),
)
knowledge_agent_builder.add_node(
    "knowledge_agent",
    knowledge_agent_node,
    destinations=({END: "END"}),
)


issue_resolve_graph = knowledge_agent_builder.compile()


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
        "messages": [HumanMessage(content="astropy__astropy-12907")],
        "preset": "astropy__astropy-12907",
        "human_in_the_loop": False,
    }

    for chunk in issue_resolve_graph.stream(
        initial_input, config=thread, stream_mode="values"
    ):
        if "messages" in chunk and len(chunk["messages"]) > 0:
            chunk["messages"][-1].pretty_print()
