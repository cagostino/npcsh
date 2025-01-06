import pexpect
import sys
import time
import re
import pytest
from npcsh.command_history import CommandHistory
from npcsh.llm_funcs import (
    get_llm_response,
    execute_llm_command,
    generate_image,
    check_llm_command,
)
from npcsh.npc_compiler import NPCCompiler
from npcsh.helpers import (
    setup_npcsh_config,
    initialize_base_npcs_if_needed,
    load_all_files,
    is_npcsh_initialized,
)


def test_npcsh():
    # Start the npcsh process
    npcsh = pexpect.spawn("npcsh", encoding="utf-8", timeout=30)
    npcsh.logfile = sys.stdout  # Log output to console for visibility

    # Wait for the prompt
    npcsh.expect("npcsh>")

    # Test 1: Compile the foreman NPC
    npcsh.sendline("/compile foreman.npc")
    npcsh.expect("Compiled NPC profile:")
    npcsh.expect("npcsh>")

    # Test 2: Switch to foreman NPC
    npcsh.sendline("/foreman")
    npcsh.expect("Switched to NPC: foreman")
    npcsh.expect("foreman>")

    # Test 3: Test weather_tool
    npcsh.sendline("What's the weather in Tokyo?")
    # Expect the assistant to provide a weather update
    npcsh.expect("The weather in .* is", timeout=60)
    npcsh.expect("foreman>")
    print("Test 3 passed: weather_tool executed successfully.")
    time.sleep(1)

    # Test 4: Test calculator tool
    npcsh.sendline("Calculate the sum of 2 and 3.")
    npcsh.expect("The result of .* is 5", timeout=30)
    npcsh.expect("foreman>")
    print("Test 4 passed: calculator tool executed successfully.")
    time.sleep(1)

    # Test 5: Test database_query tool
    npcsh.sendline("Find all users with the role 'admin'.")
    npcsh.expect("Here are the results:", timeout=30)
    npcsh.expect("foreman>")
    print("Test 5 passed: database_query tool executed successfully.")
    time.sleep(1)

    # Exit npcsh
    npcsh.sendline("/exit")
    npcsh.expect(pexpect.EOF)


def test_command_history():
    db_path = ":memory:"
    command_history = CommandHistory(db_path)
    command_history.add_command(
        "test_command", "test_subcommands", "test_output", "test_location"
    )
    history = command_history.get_all()
    assert len(history) == 1
    assert history[0][2] == "test_command"
    command_history.close()


def test_llm_functions():
    response = get_llm_response("Hello, how are you?")
    assert "response" in response

    command_history = CommandHistory(":memory:")
    result = execute_llm_command("echo Hello", command_history)
    assert "output" in result

    image_path = generate_image("A sunny day in the park", "dall-e-2", "openai")
    assert image_path is not None

    check_result = check_llm_command("echo Hello", command_history)
    assert "output" in check_result


def test_npc_compilation():
    npc_compiler = NPCCompiler("~/.npcsh/npc_team", ":memory:")
    compiled_npc = npc_compiler.compile("foreman.npc")
    assert "name" in compiled_npc


def test_command_history_search():
    db_path = ":memory:"
    command_history = CommandHistory(db_path)
    command_history.add_command(
        "test_command", "test_subcommands", "test_output", "test_location"
    )
    search_results = command_history.search("test_command")
    assert len(search_results) == 1
    assert search_results[0][2] == "test_command"
    command_history.close()


def test_npcsh_command_history():
    npcsh = pexpect.spawn("npcsh", encoding="utf-8", timeout=30)
    npcsh.logfile = sys.stdout  # Log output to console for visibility

    # Wait for the prompt
    npcsh.expect("npcsh>")

    # Test command history
    npcsh.sendline("echo Hello")
    npcsh.expect("Hello")
    npcsh.expect("npcsh>")

    npcsh.sendline("/history")
    npcsh.expect("1. .* echo Hello")
    npcsh.expect("npcsh>")

    # Exit npcsh
    npcsh.sendline("/exit")
    npcsh.expect(pexpect.EOF)


def test_npcsh_llm_functions():
    npcsh = pexpect.spawn("npcsh", encoding="utf-8", timeout=30)
    npcsh.logfile = sys.stdout  # Log output to console for visibility

    # Wait for the prompt
    npcsh.expect("npcsh>")

    # Test LLM command execution
    npcsh.sendline("/cmd echo Hello")
    npcsh.expect("Hello")
    npcsh.expect("npcsh>")

    # Test LLM question execution
    npcsh.sendline("/question What is the capital of France?")
    npcsh.expect("The capital of France is Paris.")
    npcsh.expect("npcsh>")

    # Exit npcsh
    npcsh.sendline("/exit")
    npcsh.expect(pexpect.EOF)


def test_npcsh_npc_compilation():
    npcsh = pexpect.spawn("npcsh", encoding="utf-8", timeout=30)
    npcsh.logfile = sys.stdout  # Log output to console for visibility

    # Wait for the prompt
    npcsh.expect("npcsh>")

    # Test NPC compilation
    npcsh.sendline("/compile foreman.npc")
    npcsh.expect("Compiled NPC profile:")
    npcsh.expect("npcsh>")

    # Test tool execution
    npcsh.sendline("/foreman")
    npcsh.expect("Switched to NPC: foreman")
    npcsh.expect("foreman>")

    npcsh.sendline("Calculate the sum of 2 and 3.")
    npcsh.expect("The result of .* is 5", timeout=30)
    npcsh.expect("foreman>")

    # Exit npcsh
    npcsh.sendline("/exit")
    npcsh.expect(pexpect.EOF)
