import pytest
import json
import sqlite3
import pandas as pd


import pytest
import json
from npcsh.llm_funcs import (
    get_stream,
    generate_image_gemini,
    generate_image_openai,
    get_llm_response,
    execute_llm_command,
    check_llm_command,
)
import pytest
from npcsh.llm_funcs import get_stream  # Adjust with your actual import path

# You can define global variables or fixtures if needed


def test_generate_single_image_gemini():
    prompt = "A drummer fading into cookies and cream pudding"
    images = generate_image_gemini(prompt, number_of_images=1)
    print(images)  # Output: ["generated_image_1.jpg"]


def test_generate_multiple_images_gemini():
    prompt = """A plate that has been shattered in half.
    half of a cheesecake.
    Both sit on top of a table that has been cut in half."""
    images = generate_image_gemini(prompt, number_of_images=3, aspect_ratio="16:9")
    print(
        images
    )  # Output: ["generated_image_1.jpg", "generated_image_2.jpg", "generated_image_3.jpg"]


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


def test_check_llm_command():
    with patch(
        "llm_funcs.get_llm_response",
        return_value={"is_command": "yes", "explanation": "This is a command"},
    ), patch("llm_funcs.execute_llm_command", return_value="Command executed"):
        result = check_llm_command("ls -l")
        assert result == "Command executed"


def test_execute_llm_question():
    with patch(
        "llm_funcs.get_llm_response", return_value={"response": "This is an answer"}
    ):
        result = execute_llm_question(
            "What is the capital of France?",
        )
        assert result == "This is an answer"


import pytest
from pydantic import ValidationError
from npcsh.llm_funcs import get_ollama_response, get_openai_response

from pydantic import BaseModel


class Country(BaseModel):
    name: str
    capital: str
    languages: list[str]


# Test for get_ollama_response
def test_get_response():
    # Given a prompt that is expected to return a structured response
    prompt = "Tell me about Canada."

    # When calling get_ollama_response with the schema
    response = get_response(prompt, model="llama3.2", provider="ollama", format=Country)

    # Then we verify that the response matches our expected structure
    assert isinstance(response, Country)
    assert response.name == "Canada"
    assert response.capital == "Ottawa"
    assert "English" in response.languages
    assert "French" in response.languages


def test_get_stream():
    # Given a prompt that is expected to return a structured response
    prompt = "Tell me about Canada."

    # When calling get_stream with the schema
    stream = get_stream(prompt, model="llama3.2", provider="ollama", format=Country)

    # Then we verify that the response matches our expected structure
    assert isinstance(stream, list)
    assert len(stream) > 0
    for response in stream:
        assert isinstance(response, Country)
        assert response.name == "Canada"
        assert response.capital == "Ottawa"
        assert "English" in response.languages
        assert "French" in response.languages
