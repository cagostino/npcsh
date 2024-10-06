import pytest
import json
import sqlite3
import pandas as pd
from unittest.mock import patch, MagicMock
from llm_funcs import *

@pytest.fixture
def mock_ollama_response():
    return {
        "message": {
            "role": "assistant",
            "content": "This is a test response."
        }
    }

@pytest.fixture
def mock_db_connection():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE test_table (id INTEGER, name TEXT)''')
    cursor.executemany('INSERT INTO test_table VALUES (?, ?)', [(1, 'Alice'), (2, 'Bob')])
    conn.commit()
    return conn

def test_get_ollama_conversation(mock_ollama_response):
    with patch('ollama.chat', return_value=mock_ollama_response):
        messages = [{"role": "user", "content": "Hello"}]
        result = get_ollama_conversation(messages, "test_model")
        assert len(result) == 2
        assert result[1]['role'] == 'assistant'
        assert result[1]['content'] == "This is a test response."

def test_debug_loop():
    with patch('llm_funcs.get_ollama_response', side_effect=["Error occurred", "No error"]):
        assert debug_loop("test prompt", "Error", "test_model") == True
        assert debug_loop("test prompt", "Error", "test_model") == False

def test_get_data_response(mock_db_connection):
    with patch('llm_funcs.get_llm_response', return_value={"query": "SELECT * FROM test_table", "choice": 1}):
        result = get_data_response("Get all data", mock_db_connection)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

def test_check_output_sufficient():
    sample_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    with patch('llm_funcs.get_llm_response', return_value={"IS_SUFFICIENT": True}):
        result = check_output_sufficient("test request", sample_df, "SELECT * FROM table")
        assert result.equals(sample_df)

def test_process_data_output(mock_db_connection):
    with patch('llm_funcs.check_output_sufficient', return_value=True):
        result = process_data_output({"choice": 1, "query": "SELECT * FROM test_table"}, mock_db_connection, "test request")
        assert result['code'] == 200
        assert isinstance(result['response'], pd.DataFrame)

def test_get_ollama_response():
    with patch('requests.post') as mock_post:
        mock_post.return_value.text = json.dumps({"response": "Test response"})
        result = get_ollama_response("Test prompt", "test_model")
        assert result == "Test response"

def test_get_llm_response():
    with patch('llm_funcs.get_ollama_response', return_value="Ollama response"):
        result = get_llm_response("Test prompt", provider="ollama", model="test_model")
        assert result == "Ollama response"

@pytest.fixture
def mock_command_history():
    return MagicMock()

def test_execute_data_operations(mock_command_history):
    with patch('llm_funcs.get_llm_response', return_value={"engine": "SQL", "data_operation": "SELECT * FROM test_table"}), \
         patch('sqlite3.connect') as mock_connect:
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [(1, 'Alice'), (2, 'Bob')]
        result = execute_data_operations("Get all data", mock_command_history)
        assert result["engine"] == "SQL"
        assert result["data_operation"] == "SELECT * FROM test_table"

def test_execute_llm_command(mock_command_history):
    with patch('llm_funcs.get_llm_response', return_value={"bash_command": "echo 'Hello'"}), \
         patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Hello")
        result = execute_llm_command("Print Hello", mock_command_history)
        assert "Hello" in result

def test_check_llm_command(mock_command_history):
    with patch('llm_funcs.get_llm_response', return_value={"is_command": "yes", "explanation": "This is a command"}), \
         patch('llm_funcs.execute_llm_command', return_value="Command executed"):
        result = check_llm_command("ls -l", mock_command_history)
        assert result == "Command executed"

def test_execute_llm_question(mock_command_history):
    with patch('llm_funcs.get_llm_response', return_value={"response": "This is an answer"}):
        result = execute_llm_question("What is the capital of France?", mock_command_history)
        assert result == "This is an answer"