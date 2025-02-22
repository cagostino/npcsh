import pytest
import os
import sqlite3
import tempfile
from pathlib import Path
from npcsh.shell_helpers import execute_command
from npcsh.command_history import CommandHistory
from npcsh.npc_compiler import NPCCompiler
from npcsh.npc_sysenv import get_system_message, lookup_provider, NPCSH_STREAM_OUTPUT


@pytest.fixture
def test_db():
    """Create a test database with all required tables"""
    db = sqlite3.connect(":memory:")
    cursor = db.cursor()

    # Create all necessary tables
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

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS morning_routine (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            description TEXT,
            schedule TEXT
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pipeline_name TEXT,
            step_name TEXT,
            output TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Insert default NPC
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
    return ":memory:"


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


def test_execute_command_real_cd(command_history, npc_compiler, test_db):
    """Test actual directory changes"""
    original_dir = os.getcwd()
    home_dir = os.path.expanduser("~")

    result = execute_command("cd ~", command_history, test_db, npc_compiler)
    assert os.getcwd() == home_dir
    assert "Changed directory" in result["output"]

    test_dir = tempfile.mkdtemp()
    result = execute_command(f"cd {test_dir}", command_history, test_db, npc_compiler)
    assert os.getcwd() == test_dir

    os.chdir(original_dir)
    os.rmdir(test_dir)


def test_execute_command_real_file_operations(command_history, npc_compiler, test_db):
    """Test actual file operations"""
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)

        result = execute_command(
            "echo 'test content' > test.txt", command_history, test_db, npc_compiler
        )
        assert os.path.exists("test.txt")
        with open("test.txt") as f:
            assert "test content" in f.read()

        result = execute_command("cat test.txt", command_history, test_db, npc_compiler)
        assert "test content" in result["output"]

        result = execute_command("rm test.txt", command_history, test_db, npc_compiler)
        assert not os.path.exists("test.txt")


def test_execute_command_with_pipes(command_history, npc_compiler, test_db):
    """Test command piping functionality"""
    result = execute_command(
        "echo 'hello world' | grep 'hello'", command_history, test_db, npc_compiler
    )
    assert "hello" in result["output"]


def test_execute_command_error_handling(command_history, npc_compiler, test_db):
    """Test error handling in commands"""
    result = execute_command(
        "cd /thisdirectorydoesnotexist", command_history, test_db, npc_compiler
    )
    assert "not found" in result["output"]


def test_execute_slash_commands(command_history, npc_compiler, test_db):
    """Test various slash commands"""
    result = execute_command("/help", command_history, test_db, npc_compiler)
    assert "Available Commands" in result["output"]


def test_execute_command_with_model_override(command_history, npc_compiler, test_db):
    """Test command execution with model override"""
    result = execute_command(
        "@gpt-4 What is 2+2?", command_history, test_db, npc_compiler
    )
    assert result["output"] is not None
