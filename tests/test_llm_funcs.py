import pytest
import json
import sqlite3
import pandas as pd


import pytest
import json
from npcsh.llm_funcs import (
    get_stream,
    get_anthropic_stream,
    get_anthropic_response,
    generate_image_anthropic,
    get_deepseek_response,
    get_deepseek_stream,
    get_gemini_response,
    generate_image_gemini,
    get_ollama_response,
    get_ollama_stream,
    generate_image_ollama,
    get_openai_response,
    get_openai_stream,
    get_gemini_stream,
    get_openai_like_stream,
    get_openai_like_response,
    generate_image_openai,
    generate_image_openai_like,
    get_llm_response,
    execute_llm_command,
    check_llm_command,
)
import pytest
from npcsh.llm_funcs import get_stream  # Adjust with your actual import path

# You can define global variables or fixtures if needed


def test_get_anthropic_stream():
    messages = [
        {
            "role": "user",
            "content": "Can you write a script that counts the number of windows in new york city",
        }
    ]
    model = "claude-3-5-haiku-latest"  # Replace with your actual model name
    response = get_stream(messages, model=model, provider="anthropic")
    for resp in response:
        if hasattr(resp, "delta") and hasattr(resp.delta, "text"):
            print(resp.delta.text, end="", flush=True)


def test_get_openai_stream():
    messages = [{"role": "user", "content": "what is the meaning of a star being red."}]
    model = "gpt-4o-mini"
    response = get_stream(messages, model=model, provider="openai")
    for resp in response:
        if resp.choices:
            for choice in resp.choices:
                if choice.delta.content:
                    print(choice.delta.content, end="", flush=True)


def test_get_ollama_stream():
    # Define your messages
    messages = [{"role": "user", "content": "What is the current weather?"}]
    model = "llama3.2"  # Replace with your actual model name

    # Call the function and check the type of response
    response = get_stream(messages, model=model, provider="ollama")


def test_get_gemini_response():
    prompt = "What is the capital of France?"
    response = get_gemini_response(prompt, "gemini-1.5-flash")
    assert "Paris" in response, "Gemini response should contain 'Paris'"


def test_get_gemini_json_response():
    response = get_gemini_response(
        "Describe this image in JSON format, including objects and their colors.",
        model="gemini-1.5-flash",
        images=[{"file_path": "path/to/image.png"}],
        format="json",
    )
    print(response["response"])


def test_gemini_pydantic_response():
    from pydantic import BaseModel
    from typing import List

    class Person(BaseModel):
        name: str
        age: int
        favorite_foods: List[str]

    response = get_gemini_response(
        """Return a JSON object with a person's name, age, and a list of their favorite foods.
        Do not include any additional markdown formatting.""",
        model="gemini-1.5-flash",
        format=Person,
    )
    print(response["response"])


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


def test_generate_image_gemini():
    prompt = "the boys getting back into town"
    images = generate_image_gemini(
        prompt,
        safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
        person_generation="ALLOW_ADULT",
    )
    print(images)  # Output: ["generated_image_1.jpg"]


def test_get_gemini_messages():
    # Start a conversation
    messages = [
        {"role": "user", "parts": ["My name is Alice."]},
        {"role": "model", "parts": ["Nice to meet you, Alice! How can I help you?"]},
    ]

    # First user message
    response = get_gemini_response(
        "What's the deal with airline food?",
        model="gemini-1.5-flash",
        messages=messages,
    )
    print(response["response"])


def test_get_gemini_image_analysis():
    response = get_gemini_response(
        "What is in this image?", images=[{"file_path": "path/to/image.png"}]
    )
    print(response["response"])


def test_get_deepseek_response():
    prompt = "What is the capital of France?"
    response = get_deepseek_response(prompt, "deepseek-chat")
    assert "Paris" in response, "DeepSeek response should contain 'Paris'"


# Sample example for `generate_image_ollama`
def test_generate_image_ollama():
    prompt = "A scenic landscape"
    model = "latest-model"
    image_url = generate_image_ollama(prompt, model)
    assert isinstance(image_url, str) and image_url.startswith("http")  #


# Sample example for `generate_image_openai`
def test_generate_image_openai():
    prompt = "A futuristic cityscape"

    image_url = generate_image_openai(prompt)
    assert isinstance(image_url, str) and image_url.startswith("http")  #


# Sample example for `generate_image_anthropic`
def test_generate_image_anthropic():
    prompt = "An underwater scene"
    model = "anthropic-image-model"
    api_key = "your_anthropic_api_key"
    image_url = generate_image_anthropic(prompt, model, api_key)
    assert isinstance(image_url, str) and image_url.startswith("http")  #


# Sample example for `generate_image_openai_like`
def test_generate_image_openai_like():
    prompt = "A beautiful mountain"
    model = "image-beta-002"
    api_url = "https://api.example.com"
    api_key = "your_custom_api_key"
    image_url = generate_image_openai_like(prompt, model, api_url, api_key)
    assert isinstance(image_url, str) and image_url.startswith("http")  #


def test_ollama_image():
    image_path = "/home/caug/.npcsh/screenshots/screenshot_1728963234.png"
    prompt = "What do you see in this image?"
    image_data = {"file_path": image_path}

    response = get_ollama_response(prompt=prompt, model="llava-phi3", image=image_data)

    assert isinstance(response, dict), "Response should be a dictionary"
    assert "response" in response, "Response should contain 'response' key"
    assert len(response["response"]) > 0, "Response should not be empty"
    assert isinstance(response["response"], str), "Response content should be a string"


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


import pytest
from pydantic import ValidationError
from npcsh.llm_funcs import get_ollama_response, get_openai_response

from pydantic import BaseModel


class Country(BaseModel):
    name: str
    capital: str
    languages: list[str]


# Test for get_ollama_response
def test_get_ollama_response():
    # Given a prompt that is expected to return a structured response
    prompt = "Tell me about Canada."

    # When calling get_ollama_response with the schema
    response = get_ollama_response(prompt, model="llama3.2", format=Country)

    # Then we verify that the response matches our expected structure
    assert isinstance(response, Country)
    assert response.name == "Canada"
    assert response.capital == "Ottawa"
    assert "English" in response.languages
    assert "French" in response.languages


# Test for get_openai_response
def test_get_openai_response():
    # Given a prompt that is expected to return a structured response
    prompt = "Tell me about Canada."

    # When calling get_openai_response with the schema
    response = get_openai_response(prompt, model="gpt-4o", format=Country)

    # Then we verify that the response matches our expected structure
    assert isinstance(response, Country)
    assert response.name == "Canada"
    assert response.capital == "Ottawa"
    assert "English" in response.languages
    assert "French" in response.languages


# Test for invalid response
def test_get_ollama_response_invalid():
    # Given a prompt that does not match the schema
    prompt = "Give me a country."

    # When calling get_ollama_response, we expect a ValidationError
    with pytest.raises(ValidationError):
        get_ollama_response(prompt, model="llama3.1", output_schema=Country)


def test_get_openai_response_invalid():
    # Given a prompt that does not match the schema
    prompt = "Provide a random fact."

    # When calling get_openai_response, we expect a ValidationError
    with pytest.raises(ValidationError):
        get_openai_response(prompt, model="gpt-4o", output_schema=Country)
