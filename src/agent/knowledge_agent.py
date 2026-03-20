from langgraph.prebuilt import create_react_agent

from src.agent.llm import llm
from src.prompts.knowledge_agent import (
    KNOWLEDGE_AGENT_ANALYSIS_SYSTEM_PROMPT,
    KNOWLEDGE_AGENT_ANALYSIS_USER_PROMPT,
    KNOWLEDGE_AGENT_SUMMARY_SYSTEM_PROMPT,
    KNOWLEDGE_AGENT_SUMMARY_USER_PROMPT,
)
from src.tools.sepl_tools import (
    view_directory,
    search_files_by_keywords,
    view_file_content,
)

knowledge_agent_tools = [
    view_directory,
    search_files_by_keywords,
    view_file_content,
]
knowledge_analysis_agent = create_react_agent(
    llm,
    tools=knowledge_agent_tools,
    prompt=KNOWLEDGE_AGENT_ANALYSIS_SYSTEM_PROMPT,
)
knowledge_summary_agent = create_react_agent(
    llm,
    tools=knowledge_agent_tools,
    prompt=KNOWLEDGE_AGENT_SUMMARY_SYSTEM_PROMPT,
)
