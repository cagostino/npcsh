########
########
########
########
######## STREAM
########
########

from .npc_sysenv import get_system_message
from typing import Any, Dict, Generator, List
import os
import anthropic
import ollama  # Add to setup.py if missing
from openai import OpenAI
from diffusers import StableDiffusionPipeline
from google.generativeai import types
import google.generativeai as genai
import base64


def get_anthropic_stream(
    messages,
    model: str,
    npc: Any = None,
    tools: list = None,
    images: List[Dict[str, str]] = None,
    api_key: str = None,
    **kwargs,
) -> Generator:
    """Streams responses from Anthropic, supporting images and yielding raw text chunks."""

    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    for message in messages:
        if isinstance(message["content"], str):
            message["content"] = [{"type": "text", "text": message["content"]}]
    if images:
        for img in images:
            with open(img["file_path"], "rb") as image_file:
                img["data"] = base64.b64encode(image_file.read()).decode("utf-8")
                img["media_type"] = "image/jpeg"
                messages[-1]["content"].append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img["media_type"],
                            "data": img["data"],
                        },
                    }
                )

    response = client.messages.create(
        model=model, messages=messages, max_tokens=8192, stream=True
    )

    for chunk in response:
        yield chunk


def get_ollama_stream(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
    images: list = None,
    tools: list = None,
    **kwargs,
) -> Generator:
    """Streams responses from Ollama, yielding raw text chunks."""
    messages_copy = messages.copy()
    if images:
        messages[-1]["images"] = [image["file_path"] for image in images]

    if messages_copy[0]["role"] != "system":
        if npc is not None:
            system_message = get_system_message(npc)
            messages_copy.insert(0, {"role": "system", "content": system_message})

    for chunk in ollama.chat(
        model=model, messages=messages_copy, stream=True, **kwargs
    ):
        yield chunk


def get_openai_stream(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
    tools: list = None,
    images: List[Dict[str, str]] = None,
    api_key: str = None,
    **kwargs,
) -> Generator:
    """Streams responses from OpenAI, supporting images and yielding raw text chunks."""

    if api_key is None:
        api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)

    system_message = get_system_message(npc) if npc else "You are a helpful assistant."

    if not messages:
        messages = [{"role": "system", "content": system_message}]

    # Add images if provided
    if images:
        last_user_message = (
            messages[-1]
            if messages and messages[-1]["role"] == "user"
            else {"role": "user", "content": []}
        )

        if isinstance(last_user_message["content"], str):
            last_user_message["content"] = [
                {"type": "text", "text": last_user_message["content"]}
            ]

        for image in images:
            with open(image["file_path"], "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                last_user_message["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    }
                )

        if last_user_message not in messages:
            messages.append(last_user_message)

    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True, **kwargs
    )

    for chunk in stream:
        yield chunk


def get_openai_like_stream(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
    tools: list = None,
    api_key: str = None,
    api_url: str = None,
) -> List[Dict[str, str]]:
    """
    Function Description:
        This function generates a conversation using the OpenAI API.
    Args:
        messages (List[Dict[str, str]]): The list of messages in the conversation.
        model (str): The model to use for the conversation.
    Keyword Args:
        npc (Any): The NPC object.
        api_key (str): The API key for accessing the OpenAI API.
    Returns:
        List[Dict[str, str]]: The list of messages in the conversation.
    """

    client = OpenAI(api_key=api_key, base_url=api_url)

    system_message = get_system_message(npc) if npc else "You are a helpful assistant."

    if messages is None:
        messages = []

    # Ensure the system message is at the beginning
    if not any(msg["role"] == "system" for msg in messages):
        messages.insert(0, {"role": "system", "content": system_message})

    # messages should already include the user's latest message

    # Make the API call with the messages including the latest user input
    completion = client.chat.completions.create(
        model=model, messages=messages, stream=True, **kwargs
    )

    for chunk in completion:
        yield chunk


def get_deepseek_stream(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
    tools: list = None,
    api_key: str = None,
    **kwargs,
) -> List[Dict[str, str]]:
    """
    Function Description:
        This function generates a conversation using the Deepseek API.
    Args:
        messages (List[Dict[str, str]]): The list of messages in the conversation.
        model (str): The model to use for the conversation.
    Keyword Args:
        npc (Any): The NPC object.
        api_key (str): The API key for accessing the Deepseek API.
    Returns:
        List[Dict[str, str]]: The list of messages in the conversation.
    """
    if api_key is None:
        api_key = os.environ["DEEPSEEK_API_KEY"]
    client = deepseek.Deepseek(api_key=api_key)

    system_message = get_system_message(npc) if npc else ""

    messages_copy = messages.copy()
    if messages_copy[0]["role"] != "system":
        messages_copy.insert(0, {"role": "system", "content": system_message})

    completion = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=messages_copy,
        stream=True,
        **kwargs,  # Include any additional keyword arguments
    )

    for response in completion:
        yield response


def get_gemini_stream(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
    tools: list = None,
    api_key: str = None,
    **kwargs,
) -> List[Dict[str, str]]:
    """
    Function Description:
        This function generates a conversation using the Gemini API.
    Args:
        messages (List[Dict[str, str]]): The list of messages in the conversation.
        model (str): The model to use for the conversation.
    Keyword Args:
        npc (Any): The NPC object.
        api_key (str): The API key for accessing the Gemini API.
    Returns:
        List[Dict[str, str]]: The list of messages in the conversation.

    """

    return
