from langgraph.prebuilt import create_react_agent

from src.agent.llm import llm
from src.prompts import ISSUE_RESOLVE_FIXING_AGENT_SYSTEM_PROMPT
from src.tools.edit_tool import str_replace_editor
from src.tools.sepl_tools import (
    view_directory,
    search_files_by_keywords,
    view_file_content,
)

fixing_agent_tools = [
    view_directory,
    search_files_by_keywords,
    view_file_content,
    str_replace_editor,
]
fixing_agent = create_react_agent(
    llm,
    tools=fixing_agent_tools,
    prompt=ISSUE_RESOLVE_FIXING_AGENT_SYSTEM_PROMPT,
)
