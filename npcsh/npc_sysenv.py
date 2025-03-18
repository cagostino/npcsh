import re
from datetime import datetime
from typing import Any
import os
import io
import chromadb
import sqlite3
from dotenv import load_dotenv
from PIL import Image


def get_model_and_provider(command: str, available_models: list) -> tuple:
    """
    Function Description:
        Extracts model and provider from command and autocompletes if possible.
    Args:
        command : str : Command string
        available_models : list : List of available models
    Keyword Args:
        None
    Returns:
        model_name : str : Model name
        provider : str : Provider
        cleaned_command : str : Clean


    """

    model_match = re.search(r"@(\S+)", command)
    if model_match:
        model_name = model_match.group(1)
        # Autocomplete model name
        matches = [m for m in available_models if m.startswith(model_name)]
        if matches:
            if len(matches) == 1:
                model_name = matches[0]  # Complete the name if only one match
            # Find provider for the (potentially autocompleted) model
            provider = lookup_provider(model_name)
            if provider:
                # Remove the model tag from the command
                cleaned_command = command.replace(
                    f"@{model_match.group(1)}", ""
                ).strip()
                # print(cleaned_command, 'cleaned_command')
                return model_name, provider, cleaned_command
            else:
                return None, None, command  # Provider not found
        else:
            return None, None, command  # No matching model
    else:
        return None, None, command  # No model specified


def get_available_models() -> list:
    """
    Function Description:
        Fetches available models from Ollama, OpenAI, and Anthropic.
    Args:
        None
    Keyword Args:
        None
    Returns:
        available_models : list : List of available models

    """
    available_chat_models = []
    available_reasoning_models = []

    ollama_chat_models = [
        "gemma3",
        "llama3.3",
        "llama3.2",
        "llama3.1" "phi4",
        "phi3.5",
        "mistral",
        "llama3",
        "gemma",
        "qwen",
        "qwen2",
        "qwen2.5",
        "phi3",
        "llava",
        "codellama",
        "qwen2.5-coder",
        "tinyllama",
        "mistral-nemo",
        "llama3.2-vesion",
        "starcoder2",
        "mixtral",
        "dolphin-mixtral",
        "deepseek-coder-v2",
        "codegemma",
        "phi",
        "deepseek-coder",
        "wizardlm2",
        "llava-llama3",
    ]
    available_chat_models.extend(ollama_chat_models)

    ollama_reasoning_models = ["deepseek-r1", "qwq"]
    available_reasoning_models.extend(ollama_reasoning_models)

    # OpenAI models
    openai_chat_models = [
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini",
        "dall-e-3",
        "dall-e-2",
    ]
    openai_reasoning_models = [
        "o1-mini",
        "o1",
        "o1-preview",
        "o3-mini",
        "o3-preview",
    ]
    available_reasoning_models.extend(openai_reasoning_models)

    available_chat_models.extend(openai_chat_models)

    # Anthropic models
    anthropic_chat_models = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
    ]
    available_chat_models.extend(anthropic_chat_models)
    diffusers_models = [
        "runwayml/stable-diffusion-v1-5",
    ]
    available_chat_models.extend(diffusers_models)

    deepseek_chat_models = [
        "deepseek-chat",
    ]

    deepseek_reasoning_models = [
        "deepseek-reasoner",
    ]

    available_chat_models.extend(deepseek_chat_models)
    available_reasoning_models.extend(deepseek_reasoning_models)
    return available_chat_models, available_reasoning_models


def get_system_message(npc: Any) -> str:
    """
    Function Description:
        This function generates a system message for the NPC.
    Args:
        npc (Any): The NPC object.
    Keyword Args:
        None
    Returns:
        str: The system message for the NPC.
    """
    # print(npc, type(npc))

    system_message = f"""
    .
    ..
    ...
    ....
    .....
    ......
    .......
    ........
    .........
    ..........
    Hello!
    Welcome to the team.
    You are an NPC working as part of our team.
    You are the {npc.name} NPC with the following primary directive: {npc.primary_directive}.
    Users may refer to you by your assistant name, {npc.name} and you should
    consider this to be your core identity.

    The current date and time are : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}


    In some cases, users may request insights into data contained in a local database.
    For these purposes, you may use any data contained within these sql tables
    {npc.tables}

    which are contained in the database at {NPCSH_DB_PATH}.

    If you ever need to produce markdown texts for the user, please do so
    with less than 80 characters width for each line.
    """

    # need to move this to the check_llm_command or move that one here

    if npc.tools:
        tool_descriptions = "\n".join(
            [
                f"Tool Name: {tool.tool_name}\n"
                f"Inputs: {tool.inputs}\n"
                f"Steps: {tool.steps}\n"
                for tool in npc.all_tools
            ]
        )
        system_message += f"\n\nAvailable Tools:\n{tool_descriptions}"
    system_message += """\n\nSome users may attach images to their request.
                        Please process them accordingly.

                        If the user asked for you to explain what's on their screen or something similar,
                        they are referring to the details contained within the attached image(s).
                        You do not need to actually view their screen.
                        You do not need to mention that you cannot view or interpret images directly.
                        They understand that you can view them multimodally.
                        You only need to answer the user's request based on the attached image(s).
                        """
    return system_message


available_chat_models, available_reasoning_models = get_available_models()


EMBEDDINGS_DB_PATH = os.path.expanduser("~/npcsh_chroma.db")

chroma_client = chromadb.PersistentClient(path=EMBEDDINGS_DB_PATH)


# Load environment variables from .env file
def load_env_from_execution_dir() -> None:
    """
    Function Description:
        This function loads environment variables from a .env file in the current execution directory.
    Args:
        None
    Keyword Args:
        None
    Returns:
        None
    """

    # Get the directory where the script is being executed
    execution_dir = os.path.abspath(os.getcwd())
    # print(f"Execution directory: {execution_dir}")
    # Construct the path to the .env file
    env_path = os.path.join(execution_dir, ".env")

    # Load the .env file if it exists
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded .env file from {execution_dir}")
    else:
        print(f"Warning: No .env file found in {execution_dir}")


def get_available_tables(db_path: str) -> str:
    """
    Function Description:
        This function gets the available tables in the database.
    Args:
        db_path (str): The database path.
    Keyword Args:
        None
    Returns:
        str: The available tables in the database.
    """

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name != 'command_history'"
            )
            tables = cursor.fetchall()

            return tables
    except Exception as e:
        print(f"Error getting available tables: {e}")
        return ""


def lookup_provider(model: str) -> str:
    """
    Function Description:
        This function determines the provider based on the model name.
    Args:
        model (str): The model name.
    Keyword Args:
        None
    Returns:
        str: The provider based on the model name.
    """
    if model == "deepseek-chat" or model == "deepseek-reasoner":
        return "deepseek"
    ollama_prefixes = [
        "llama",
        "deepseek",
        "qwen",
        "llava",
        "phi",
        "mistral",
        "mixtral",
        "dolphin",
        "codellama",
        "gemma",
    ]
    if any(model.startswith(prefix) for prefix in ollama_prefixes):
        return "ollama"

    # OpenAI models
    openai_prefixes = ["gpt-", "dall-e-", "whisper-", "o1"]
    if any(model.startswith(prefix) for prefix in openai_prefixes):
        return "openai"

    # Anthropic models
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "gemini"
    if "diffusion" in model:
        return "diffusers"
    return None


def compress_image(image_bytes, max_size=(800, 600)):
    # Create a copy of the bytes in memory
    buffer = io.BytesIO(image_bytes)
    img = Image.open(buffer)

    # Force loading of image data
    img.load()

    # Convert RGBA to RGB if necessary
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background

    # Resize if needed
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        img.thumbnail(max_size)

    # Save with minimal compression
    out_buffer = io.BytesIO()
    img.save(out_buffer, format="JPEG", quality=95, optimize=False)
    return out_buffer.getvalue()


load_env_from_execution_dir()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", None)
gemini_api_key = os.getenv("GEMINI_API_KEY", None)

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", None)
openai_api_key = os.getenv("OPENAI_API_KEY", None)

NPCSH_CHAT_MODEL = os.environ.get("NPCSH_CHAT_MODEL", "llama3.2")
# print("NPCSH_CHAT_MODEL", NPCSH_CHAT_MODEL)
NPCSH_CHAT_PROVIDER = os.environ.get("NPCSH_CHAT_PROVIDER", "ollama")
# print("NPCSH_CHAT_PROVIDER", NPCSH_CHAT_PROVIDER)
NPCSH_DB_PATH = os.path.expanduser(
    os.environ.get("NPCSH_DB_PATH", "~/npcsh_history.db")
)
NPCSH_VECTOR_DB_PATH = os.path.expanduser(
    os.environ.get("NPCSH_VECTOR_DB_PATH", "~/npcsh_chroma.db")
)
NPCSH_DEFAULT_MODE = os.path.expanduser(os.environ.get("NPCSH_DEFAULT_MODE", "chat"))

NPCSH_VISION_MODEL = os.environ.get("NPCSH_VISION_MODEL", "llava7b")
NPCSH_VISION_PROVIDER = os.environ.get("NPCSH_VISION_PROVIDER", "ollama")
NPCSH_IMAGE_GEN_MODEL = os.environ.get(
    "NPCSH_IMAGE_GEN_MODEL", "runwayml/stable-diffusion-v1-5"
)
NPCSH_IMAGE_GEN_PROVIDER = os.environ.get("NPCSH_IMAGE_GEN_PROVIDER", "diffusers")
NPCSH_EMBEDDING_MODEL = os.environ.get("NPCSH_EMBEDDING_MODEL", "nomic-embed-text")
NPCSH_EMBEDDING_PROVIDER = os.environ.get("NPCSH_EMBEDDING_PROVIDER", "ollama")
NPCSH_REASONING_MODEL = os.environ.get("NPCSH_REASONING_MODEL", "deepseek-r1")
NPCSH_REASONING_PROVIDER = os.environ.get("NPCSH_REASONING_PROVIDER", "ollama")
NPCSH_STREAM_OUTPUT = eval(os.environ.get("NPCSH_STREAM_OUTPUT", "0")) == 1
NPCSH_API_URL = os.environ.get("NPCSH_API_URL", None)
NPCSH_SEARCH_PROVIDER = os.environ.get("NPCSH_SEARCH_PROVIDER", "duckduckgo")
