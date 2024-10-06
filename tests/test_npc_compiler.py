import pytest
import os
import tempfile
import sqlite3
from unittest.mock import patch, MagicMock
from npcsh.npc_compiler import NPCCompiler, NPC

@pytest.fixture
def temp_npc_directory():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def mock_db_conn():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [('table1',), ('table2',)]
    return conn

def create_npc_file(directory, filename, content):
    with open(os.path.join(directory, filename), 'w') as f:
        f.write(content)

def test_npc_compiler_init(temp_npc_directory):
    compiler = NPCCompiler(temp_npc_directory)
    assert compiler.npc_directory == temp_npc_directory
    assert isinstance(compiler.jinja_env, Environment)
    assert compiler.npc_cache == {}
    assert compiler.resolved_npcs == {}

def test_npc_compiler_compile_invalid_extension(temp_npc_directory):
    compiler = NPCCompiler(temp_npc_directory)
    with pytest.raises(ValueError, match="File must have .npc extension"):
        compiler.compile("invalid.txt")

def test_npc_compiler_compile_valid(temp_npc_directory):
    compiler = NPCCompiler(temp_npc_directory)
    npc_content = """
    name: Test NPC
    primary_directive: Test directive
    suggested_tools_to_use: [tool1, tool2]
    restrictions: []
    model: gpt-3.5-turbo
    """
    create_npc_file(temp_npc_directory, "test.npc", npc_content)
    result = compiler.compile("test.npc")
    assert result['name'] == "Test NPC"
    assert result['primary_directive'] == "Test directive"
    assert result['suggested_tools_to_use'] == ['tool1', 'tool2']
    assert result['restrictions'] == []
    assert result['model'] == "gpt-3.5-turbo"

def test_npc_compiler_compile_with_inheritance(temp_npc_directory):
    compiler = NPCCompiler(temp_npc_directory)
    parent_content = """
    name: Parent NPC
    primary_directive: Parent directive
    suggested_tools_to_use: [tool1]
    restrictions: []
    model: gpt-3.5-turbo
    """
    child_content = """
    inherits_from: parent
    name: Child NPC
    suggested_tools_to_use: [tool2]
    """
    create_npc_file(temp_npc_directory, "parent.npc", parent_content)
    create_npc_file(temp_npc_directory, "child.npc", child_content)
    result = compiler.compile("child.npc")
    assert result['name'] == "Child NPC"
    assert result['primary_directive'] == "Parent directive"
    assert result['suggested_tools_to_use'] == ['tool1', 'tool2']
    assert result['restrictions'] == []
    assert result['model'] == "gpt-3.5-turbo"

def test_npc_compiler_compile_missing_required_key(temp_npc_directory):
    compiler = NPCCompiler(temp_npc_directory)
    npc_content = """
    name: Test NPC
    primary_directive: Test directive
    suggested_tools_to_use: [tool1, tool2]
    restrictions: []
    """
    create_npc_file(temp_npc_directory, "test.npc", npc_content)
    with pytest.raises(ValueError, match="Missing required key in NPC profile: model"):
        compiler.compile("test.npc")

def test_npc_init(mock_db_conn):
    npc = NPC(
        name="Test NPC",
        primary_directive="Test directive",
        suggested_tools_to_use=["tool1", "tool2"],
        restrictions=[],
        model="gpt-3.5-turbo",
        provider="openai",
        db_conn=mock_db_conn
    )
    assert npc.name == "Test NPC"
    assert npc.primary_directive == "Test directive"
    assert npc.suggested_tools_to_use == ["tool1", "tool2"]
    assert npc.restrictions == []
    assert npc.model == "gpt-3.5-turbo"
    assert npc.provider == "openai"
    assert npc.db_conn == mock_db_conn
    assert npc.tables == [('table1',), ('table2',)]

def test_npc_str(mock_db_conn):
    npc = NPC(
        name="Test NPC",
        primary_directive="Test directive",
        suggested_tools_to_use=["tool1", "tool2"],
        restrictions=[],
        model="gpt-3.5-turbo",
        provider="openai",
        db_conn=mock_db_conn
    )
    assert str(npc) == "NPC: Test NPC\nDirective: Test directive\nModel: gpt-3.5-turbo"

@patch('npc_compiler.get_data_response')
def test_npc_get_data_response(mock_get_data_response, mock_db_conn):
    npc = NPC(
        name="Test NPC",
        primary_directive="Test directive",
        suggested_tools_to_use=["tool1", "tool2"],
        restrictions=[],
        model="gpt-3.5-turbo",
        provider="openai",
        db_conn=mock_db_conn
    )
    mock_get_data_response.return_value = "Data response"
    result = npc.get_data_response("Test request")
    assert result == "Data response"
    mock_get_data_response.assert_called_once_with("Test request", mock_db_conn, [('table1',), ('table2',)])

@patch('npc_compiler.get_llm_response')
def test_npc_get_llm_response(mock_get_llm_response, mock_db_conn):
    npc = NPC(
        name="Test NPC",
        primary_directive="Test directive",
        suggested_tools_to_use=["tool1", "tool2"],
        restrictions=[],
        model="gpt-3.5-turbo",
        provider="openai",
        db_conn=mock_db_conn
    )
    mock_get_llm_response.return_value = "LLM response"
    result = npc.get_llm_response("Test request", temperature=0.7)
    assert result == "LLM response"
    mock_get_llm_response.assert_called_once_with("Test request", "gpt-3.5-turbo", "openai", temperature=0.7)