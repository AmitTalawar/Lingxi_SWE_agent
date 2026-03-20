from typing import Annotated, List, Optional
import logging
import time

from langchain_core.tools import tool

from src.runtime import runtime_config
from src.tools.oheditor import CLIResult, OHEditor
from langchain_core.runnables import RunnableConfig
from src.tools.utils import get_runtime_config
from src.agent.logging_config import get_logger

# Setup logging
logger = get_logger(__name__)

_GLOBAL_EDITOR = OHEditor()


def _make_cli_result(tool_result: CLIResult) -> str:
    """Convert an CLIResult to an API ToolResultBlockParam."""
    if tool_result.error:
        return f"ERROR:\n{tool_result.error}"

    assert tool_result.output, "Expected output in file_editor."
    return tool_result.output


@tool("str_replace_editor")
def str_replace_editor(
    command: Annotated[
        str, "The command to be executed (view, create, str_replace, insert)"
    ],
    path: Annotated[
        str,
        "Relative path from root of the repository to file or directory, e.g., 'file.py' or 'workspace'",
    ],
    config: RunnableConfig = None,
    file_text: Optional[str] = None,
    old_str: Optional[str] = None,
    new_str: Optional[str] = None,
    insert_line: Optional[int] = None,
    view_range: Optional[List[int]] = None,
):
    """
    Custom editing tool for viewing, creating and editing files in plain-text format
    * State is persistent across command calls and discussions with the user
    * If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
    * The `create` command cannot be used if the specified `path` already exists as a file
    * If a `command` generates a long output, it will be truncated and marked with `<response clipped>`


    Before using this tool to edit a file:
    1. Use the `view` command to understand the file's contents and context
    2. Verify the directory path is correct (only applicable when creating new files):
        - Use the `view` command to verify the parent directory exists and is the correct location

    When making edits:
        - Ensure the edit results in idiomatic, correct code
        - Do not leave the code in a broken state
        - Always use relative file paths (starting with ./)

    CRITICAL REQUIREMENTS FOR USING THIS TOOL:

    1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file, including all whitespace and indentation. The tool will fail if `old_str` matches multiple locations or doesn't match exactly with the file content.

    2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:
        - Include sufficient context before and after the change point (3-5 lines recommended)
        - If not unique, the replacement will not be performed

    3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. Both strings must be different.

    Remember: when making multiple file edits in a row to the same file, you should prefer to send all edits in a single message with multiple calls to this tool, rather than multiple messages with a single call each.


    Args:
        command (str): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`.
        path (str): Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`.
        file_text (Optional[str]): Required parameter of `create` command, with the content of the file to be created.
        old_str (Optional[str]): Required parameter of `str_replace` command containing the string in `path` to replace.
        new_str (Optional[str]): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
        insert_line (Optional[int]): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
        view_range (Optional[List[int]]): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [100, 600] will show content between line 100 and 600. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file. Unless you are sure about the line numbers, otherwise, try to view the whole file for better understanding and do not set this parameter.

    """
    # Get runtime config and project path
    log_output = []
    if config:
        agent_name = config.get("configurable", {}).get("agent_name")
        log_output.append(f"{agent_name}:")
    else:
        agent_name = None

    # Get runtime config to access project path
    runtime_obj = get_runtime_config(config)
    proj_path = runtime_obj.proj_path
    log_output.append(f"--str_replace_editor")

    log_output.append(f"----command: {command}")
    log_params = []
    if path is not None:
        log_params.append(f"path: {path}")
    if file_text is not None:
        log_params.append(f"file_text: {file_text}")
    if view_range is not None:
        log_params.append(f"view_range: {view_range}")
    if old_str is not None:
        log_params.append(f"old_str: {old_str}")
    if new_str is not None:
        log_params.append(f"new_str: {new_str}")
    if insert_line is not None:
        log_params.append(f"insert_line: {insert_line}")
    log_output.append(f"----{' '.join(log_params)}")
    result = _GLOBAL_EDITOR(
        command=command,
        path=path,
        file_text=file_text,
        view_range=view_range,
        old_str=old_str,
        new_str=new_str,
        insert_line=insert_line,
        proj_path=proj_path,
    )
    # print(result)
    if result.error:
        log_output.append(f"--str_replace_editor return ERROR: {result.error}")

    logger.info("\n".join(log_output))
    cli_result = _make_cli_result(result)
    logger.info(f"----result (first 80 chars): {cli_result[:80] if cli_result else ''}")
    return cli_result


@tool("str_replace_based_edit_tool")
def str_replace_based_edit_tool(
    command: Annotated[
        str, "The command to be executed (view, create, str_replace, insert)"
    ],
    path: Annotated[
        str,
        "Relative path from root of the repository to file or directory, e.g., 'file.py' or 'workspace'",
    ],
    config: RunnableConfig = None,
    file_text: Optional[str] = None,
    old_str: Optional[str] = None,
    new_str: Optional[str] = None,
    insert_line: Optional[int] = None,
    view_range: Optional[List[int]] = None,
):
    """
    Custom editing tool for viewing, creating and editing files in plain-text format
    * State is persistent across command calls and discussions with the user
    * If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
    * The `create` command cannot be used if the specified `path` already exists as a file
    * If a `command` generates a long output, it will be truncated and marked with `<response clipped>`


    Before using this tool to edit a file:
    1. Use the `view` command to understand the file's contents and context
    2. Verify the directory path is correct (only applicable when creating new files):
        - Use the `view` command to verify the parent directory exists and is the correct location

    When making edits:
        - Ensure the edit results in idiomatic, correct code
        - Do not leave the code in a broken state
        - Always use relative file paths (starting with ./)

    CRITICAL REQUIREMENTS FOR USING THIS TOOL:

    1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file, including all whitespace and indentation. The tool will fail if `old_str` matches multiple locations or doesn't match exactly with the file content.

    2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:
        - Include sufficient context before and after the change point (3-5 lines recommended)
        - If not unique, the replacement will not be performed

    3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. Both strings must be different.

    Remember: when making multiple file edits in a row to the same file, you should prefer to send all edits in a single message with multiple calls to this tool, rather than multiple messages with a single call each.


    Args:
        command (str): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`.
        path (str): Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`.
        file_text (Optional[str]): Required parameter of `create` command, with the content of the file to be created.
        old_str (Optional[str]): Required parameter of `str_replace` command containing the string in `path` to replace.
        new_str (Optional[str]): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
        insert_line (Optional[int]): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
        view_range (Optional[List[int]]): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [100, 600] will show content between line 100 and 600. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file. Unless you are sure about the line numbers, otherwise, try to view the whole file for better understanding and do not set this parameter.

    """
    # Get runtime config and project path
    log_output = []
    if config:
        agent_name = config.get("configurable", {}).get("agent_name")
        log_output.append(f"{agent_name}:")
    else:
        agent_name = None

    # Get runtime config to access project path
    runtime_obj = get_runtime_config(config)
    proj_path = runtime_obj.proj_path
    log_output.append(f"--str_replace_editor(runtime proj_path: {proj_path})")

    log_output.append(f"----command: {command}")
    log_params = []
    if path is not None:
        log_params.append(f"path: {path}")
    if file_text is not None:
        log_params.append(f"file_text: {file_text}")
    if view_range is not None:
        log_params.append(f"view_range: {view_range}")
    if old_str is not None:
        log_params.append(f"old_str: {old_str}")
    if new_str is not None:
        log_params.append(f"new_str: {new_str}")
    if insert_line is not None:
        log_params.append(f"insert_line: {insert_line}")
    log_output.append(f"----{' '.join(log_params)}")
    result = _GLOBAL_EDITOR(
        command=command,
        path=path,
        file_text=file_text,
        view_range=view_range,
        old_str=old_str,
        new_str=new_str,
        insert_line=insert_line,
        proj_path=proj_path,
    )
    # print(result)
    if result.error:
        log_output.append(f"--str_replace_editor return ERROR: {result.error}")

    logger.info("\n".join(log_output))
    cli_result = _make_cli_result(result)
    logger.info(f"----result (first 80 chars): {cli_result[:80] if cli_result else ''}")
    return cli_result


if __name__ == "__main__":
    # Set up logging for standalone execution
    from src.agent.logging_config import configure_logging

    configure_logging(level=logging.INFO, log_file=f"edit_tool_{int(time.time())}.log")

    rc = runtime_config.RuntimeConfig()
    # rc.load_from_preset("gitpython-developers+GitPython@1413.yaml")
    # rc.load_from_swe_docker_instance("sympy__sympy-16792")  # from one of the easy instances
    rc.load_from_swe_rex_docker_instance("astropy__astropy-14096")
    logger.info("=" * 50)
    rc.pretty_print_runtime()
    logger.info("=" * 50)

    # Test AUTO_EXTEND feature
    logger.info("=" * 50)
    logger.info("Testing AUTO_EXTEND feature")
    logger.info("=" * 50)

    # First, create a test Python file with a class and methods
    test_code = '''class TestClass:
    """Test class for AUTO_EXTEND feature."""
    
    def method_one(self):
        """First method."""
        x = 1
        y = 2
        return x + y
    
    def method_two(self):
        """Second method."""
        result = []
        for i in range(5):
            result.append(i * 2)
        return result
    
    def method_three(self):
        """Third method."""
        def nested_func():
            return "nested"
        return nested_func()

def standalone_function():
    """A standalone function."""
    return 42
'''

    # Create the test file
    create_result = str_replace_editor.invoke(
        {"command": "create", "path": "./test_auto_extend.py", "file_text": test_code}
    )
    logger.info(f"Created test file: {create_result}")

    # Test 1: View inside a single method (should extend to method only)
    logger.info("\nTest 1: View lines 5-6 (inside method_one)")
    logger.info("Expected: Should extend to complete method_one")
    view_result = str_replace_editor.invoke(
        {"command": "view", "path": "./test_auto_extend.py", "view_range": [5, 6]}
    )
    logger.info(f"Result:\n{view_result}")

    # Test 2: View across two methods (should extend to class)
    logger.info("\nTest 2: View lines 7-11 (spanning method_one and method_two)")
    logger.info("Expected: Should extend to entire TestClass")
    view_result = str_replace_editor.invoke(
        {"command": "view", "path": "./test_auto_extend.py", "view_range": [7, 11]}
    )
    logger.info(f"Result:\n{view_result}")

    # Test 3: View nested function (should extend to containing method)
    logger.info("\nTest 3: View lines 18-19 (inside nested_func within method_three)")
    logger.info("Expected: Should extend to method_three")
    view_result = str_replace_editor.invoke(
        {"command": "view", "path": "./test_auto_extend.py", "view_range": [18, 19]}
    )
    logger.info(f"Result:\n{view_result}")

    # Test 4: View standalone function (should extend to just that function)
    logger.info("\nTest 4: View line 24 (inside standalone_function)")
    logger.info("Expected: Should extend to standalone_function only")
    view_result = str_replace_editor.invoke(
        {"command": "view", "path": "./test_auto_extend.py", "view_range": [24, 24]}
    )
    logger.info(f"Result:\n{view_result}")

    # Test 5: Toggle AUTO_EXTEND off and view exact lines
    logger.info("\nTest 5: Testing with AUTO_EXTEND disabled")
    logger.info("Setting _GLOBAL_EDITOR.AUTO_EXTEND = False")
    _GLOBAL_EDITOR.AUTO_EXTEND = False

    view_result = str_replace_editor.invoke(
        {"command": "view", "path": "./test_auto_extend.py", "view_range": [5, 6]}
    )
    logger.info(f"Result (should show only lines 5-6):\n{view_result}")

    # Re-enable AUTO_EXTEND
    _GLOBAL_EDITOR.AUTO_EXTEND = True
    logger.info("\nAUTO_EXTEND re-enabled")

    logger.info("=" * 50)
    logger.info("AUTO_EXTEND tests completed!")
    logger.info("=" * 50)

    input("Press Enter to continue...")
