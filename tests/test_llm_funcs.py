import pytest
import json
import sqlite3
import pandas as pd


import pytest
import json
from npcsh.llm_funcs import (
    get_openai_response,
    get_anthropic_response,
    get_openai_like_response,
)


def test_get_openai_like_response():
    prompt = "What is the capital of France?"
    response = get_openai_like_response(
        prompt, "llama3.2", api_url="http://localhost:11434/v1/chat/completions"
    )
    assert "Paris" in response, "OpenAI response should contain 'Paris'"


def test_get_openai_response():
    prompt = """Generate a short JSON object with a
    'greeting' key and a 'farewell' key.
    Respond only with the formatted JSON object. 
    Do not include any extra text or markdown formatting.
    """
    response = get_openai_response(prompt, "gpt-4o-mini")

    # Check if the response is a valid JSON
    try:
        json_response = json.loads(response)
        assert isinstance(json_response, dict)
        assert "greeting" in json_response
        assert "farewell" in json_response
    except json.JSONDecodeError:
        pytest.fail("OpenAI response is not a valid JSON")


def test_openai_response_content():
    prompt = "What is the capital of France?"
    response = get_openai_response(prompt)
    assert "Paris" in response, "OpenAI response should contain 'Paris'"


def test_anthropic_response_content():
    prompt = "What is the capital of Japan?"
    response = get_anthropic_response(prompt, "claude-3-haiku-20240307")
    assert "Tokyo" in response, "Anthropic response should contain 'Tokyo'"


def test_openai_response_length():
    prompt = "Write a short paragraph about artificial intelligence."
    response = get_openai_response(prompt)
    words = response.split()
    assert 20 <= len(words) <= 100, "OpenAI response should be between 20 and 100 words"


def test_anthropic_response_length():
    prompt = "Write a short paragraph about machine learning."
    response = get_anthropic_response(prompt)
    words = response.split()
    assert (
        20 <= len(words) <= 100
    ), "Anthropic response should be between 20 and 100 words"


@pytest.fixture
def mock_ollama_response():
    return {"message": {"role": "assistant", "content": "This is a test response."}}


@pytest.fixture
def mock_db_connection():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE test_table (id INTEGER, name TEXT)""")
    cursor.executemany(
        "INSERT INTO test_table VALUES (?, ?)", [(1, "Alice"), (2, "Bob")]
    )
    conn.commit()
    return conn


def test_get_ollama_conversation(mock_ollama_response):
    with patch("ollama.chat", return_value=mock_ollama_response):
        messages = [{"role": "user", "content": "Hello"}]
        result = get_ollama_conversation(messages, "test_model")
        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "This is a test response."


def test_get_ollama_response():
    prompt = "This is a test prompt."
    model = "phi3"
    response = get_ollama_response(prompt, model)
    print(response)
    assert response is not None
    assert isinstance(response, str)

    prompt = """A user submitted this query: "SELECT * FROM table_name". 
    You need to generate a script that will accomplish the user\'s intent. 
    Respond ONLY with the procedure that should be executed. Place it in a JSON object with the key 
    "script_to_test".
    The format and requiremrents of the output are as follows:
    {
    "script_to_test": {"type": "string", 
    "description": "a valid SQL query that will accomplish the task"}
    }

    """
    model = "phi3"

    response = get_ollama_response(prompt, model, format="json")
    print(response)
    assert response is not None
    assert isinstance(response, dict)
    assert "script_to_test" in response


def test_debug_loop():
    with patch(
        "llm_funcs.get_ollama_response", side_effect=["Error occurred", "No error"]
    ):
        assert debug_loop("test prompt", "Error", "test_model") == True
        assert debug_loop("test prompt", "Error", "test_model") == False


def test_get_data_response(mock_db_connection):
    with patch(
        "llm_funcs.get_llm_response",
        return_value={"query": "SELECT * FROM test_table", "choice": 1},
    ):
        result = get_data_response("Get all data", mock_db_connection)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


def test_check_output_sufficient():
    sample_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    with patch("llm_funcs.get_llm_response", return_value={"IS_SUFFICIENT": True}):
        result = check_output_sufficient(
            "test request", sample_df, "SELECT * FROM table"
        )
        assert result.equals(sample_df)


def test_process_data_output(mock_db_connection):
    with patch("llm_funcs.check_output_sufficient", return_value=True):
        result = process_data_output(
            {"choice": 1, "query": "SELECT * FROM test_table"},
            mock_db_connection,
            "test request",
        )
        assert result["code"] == 200
        assert isinstance(result["response"], pd.DataFrame)


def test_get_ollama_response():
    with patch("requests.post") as mock_post:
        mock_post.return_value.text = json.dumps({"response": "Test response"})
        result = get_ollama_response("Test prompt", "test_model")
        assert result == "Test response"


def test_get_llm_response():
    with patch("llm_funcs.get_ollama_response", return_value="Ollama response"):
        result = get_llm_response("Test prompt", provider="ollama", model="test_model")
        assert result == "Ollama response"


@pytest.fixture
def mock_command_history():
    return MagicMock()


def test_execute_data_operations(mock_command_history):
    with patch(
        "llm_funcs.get_llm_response",
        return_value={"engine": "SQL", "data_operation": "SELECT * FROM test_table"},
    ), patch("sqlite3.connect") as mock_connect:
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value.cursor.return_value = (
            mock_cursor
        )
        mock_cursor.fetchall.return_value = [(1, "Alice"), (2, "Bob")]
        result = execute_data_operations("Get all data", mock_command_history)
        assert result["engine"] == "SQL"
        assert result["data_operation"] == "SELECT * FROM test_table"


def test_execute_llm_command(mock_command_history):
    with patch(
        "llm_funcs.get_llm_response", return_value={"bash_command": "echo 'Hello'"}
    ), patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Hello")
        result = execute_llm_command("Print Hello", mock_command_history)
        assert "Hello" in result


def test_check_llm_command(mock_command_history):
    with patch(
        "llm_funcs.get_llm_response",
        return_value={"is_command": "yes", "explanation": "This is a command"},
    ), patch("llm_funcs.execute_llm_command", return_value="Command executed"):
        result = check_llm_command("ls -l", mock_command_history)
        assert result == "Command executed"


def test_execute_llm_question(mock_command_history):
    with patch(
        "llm_funcs.get_llm_response", return_value={"response": "This is an answer"}
    ):
        result = execute_llm_question(
            "What is the capital of France?", mock_command_history
        )
        assert result == "This is an answer"
