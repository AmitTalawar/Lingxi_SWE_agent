from langgraph.prebuilt import create_react_agent

from src.agent.llm import llm
from src.prompts import ISSUE_RESOLVE_PLANNING_AGENT_SYSTEM_PROMPT
from src.tools.sepl_tools import (
    view_directory,
    search_files_by_keywords,
    view_file_content,
)

planning_agent_tools = [view_directory, search_files_by_keywords, view_file_content]
planning_agent = create_react_agent(
    llm,
    tools=planning_agent_tools,
    prompt=ISSUE_RESOLVE_PLANNING_AGENT_SYSTEM_PROMPT,
)
