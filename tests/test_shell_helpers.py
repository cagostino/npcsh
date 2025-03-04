import pytest
import os
import sqlite3
import tempfile
from pathlib import Path
from npcsh.shell_helpers import execute_command
from npcsh.command_history import CommandHistory
from npcsh.npc_compiler import NPCCompiler
from npcsh.npc_sysenv import (
    get_system_message,
    lookup_provider,
    NPCSH_STREAM_OUTPUT,
    get_available_tables,
)


@pytest.fixture
def test_db():
    """Create a test database with all required tables"""
    # Create temp file that persists during tests
    temp_db = tempfile.NamedTemporaryFile(delete=False)
    db_path = temp_db.name

    db = sqlite3.connect(db_path)
    cursor = db.cursor()

    # Create your tables
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS command_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            command TEXT,
            path TEXT,
            output TEXT
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS compiled_npcs (
            name TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            compiled_content TEXT
        )
    """
    )

    # Insert test data
    cursor.execute(
        """
        INSERT INTO compiled_npcs (name, source_path, compiled_content)
        VALUES (?, ?, ?)
        """,
        (
            "sibiji",
            os.path.abspath("../npcsh/npc_team/sibiji.npc"),
            "name: sibiji\nprimary_directive: You are a helpful assistant.\nmodel: gpt-4o-mini\nprovider: openai\n",
        ),
    )

    db.commit()
    db.close()

    yield db_path  # Return the path to the temp file

    # Clean up after tests
    os.unlink(db_path)


@pytest.fixture
def command_history():
    return CommandHistory()


@pytest.fixture
def npc_compiler():
    # Get the absolute path to the npc_team directory
    npc_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "npcsh", "npc_team")
    )
    if not os.path.exists(npc_dir):
        os.makedirs(npc_dir)
    return NPCCompiler(npc_dir, ":memory:")


def test_execute_slash_commands(command_history, npc_compiler, test_db):
    """Test various slash commands"""

    result = execute_command("/help", command_history, test_db, npc_compiler)
    assert "Available Commands" in result["output"]


def test_execute_command_with_model_override(command_history, npc_compiler, test_db):
    """Test command execution with model override"""
    result = execute_command(
        "@gpt-4o-mini What is 2+2?",
        command_history,
        test_db,
        npc_compiler,
    )
    assert result["output"] is not None


def test_execute_command_who_was_simon_bolivar(command_history, npc_compiler, test_db):
    """Test the command for querying information about Simón Bolívar."""
    result = execute_command(
        "What country was Simon Bolivar born in?",
        command_history,
        test_db,
        npc_compiler,
    )
    assert "venezuela" in str(result["output"]).lower()


def test_execute_command_capital_of_france(command_history, npc_compiler, test_db):
    """Test the command for querying the capital of France."""
    result = execute_command(
        "What is the capital of France?", command_history, test_db, npc_compiler
    )
    assert "paris" in str(result["output"]).lower()


def test_execute_command_weather_info(command_history, npc_compiler, test_db):
    """Test the command for getting weather information."""
    result = execute_command(
        "what is the weather in Tokyo?", command_history, test_db, npc_compiler
    )
    print(result)  # Add print for debugging
    assert "tokyo" in str(result["output"]).lower()


def test_execute_command_linked_list_implementation(
    command_history, npc_compiler, test_db
):
    """Test the command for querying linked list implementation in Python."""
    result = execute_command(
        " Tell me a way to implement a linked list in Python?",
        command_history,
        test_db,
        npc_compiler,
    )
    assert "class Node:" in str(result["output"])
    assert "class LinkedList:" in str(result["output"])


def test_execute_command_inquiry_with_npcs(command_history, npc_compiler, test_db):
    """Test inquiry using NPCs."""
    result = execute_command(
        "/search -p duckduckgo who is the current us president",
        command_history,
        test_db,
        npc_compiler,
    )
    assert "President" in result["output"]  # Check for presence of expected output


def test_execute_command_rag_search(command_history, npc_compiler, test_db):
    """Test the command for a RAG search."""
    result = execute_command(
        "/rag -f dummy_linked_list.py linked list",
        command_history,
        test_db,
        npc_compiler,
    )

    print(result)  # Print the result for debugging visibility
    # Instead of specific class search, check if it includes any relevant text
    assert (
        "Found similar texts:" in result["output"]
    )  # Check for invocation acknowledgement
    assert "linked" in result["output"].lower()  # Check for mention of linked list
