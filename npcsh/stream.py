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


def get_anthropic_stream(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
    api_key: str = None,
    **kwargs,
) -> Generator[str, None, None]:
    """Streams responses from Anthropic, yielding raw text chunks."""
    messages_copy = messages.copy()

    system_message = get_system_message(npc) if npc else "You are a helpful assistant."
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        system=system_message,
        messages=messages,
        max_tokens=8192,
        stream=True,
    )
    for chunk in response:
        if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
            yield chunk.delta.text  # Extracts raw text


def get_ollama_stream(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
    **kwargs,
) -> Generator[str, None, None]:
    """Streams responses from Ollama, yielding raw text chunks."""
    messages_copy = messages.copy()
    if messages_copy[0]["role"] != "system":
        if npc is not None:
            system_message = get_system_message(npc)
            messages_copy.insert(0, {"role": "system", "content": system_message})

    response = ollama.chat(model=model, messages=messages_copy, stream=True, **kwargs)

    for chunk in response:
        if isinstance(chunk, dict) and "message" in chunk:
            yield chunk["message"]  # Extracts raw text


def get_openai_stream(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
    api_key: str = None,
    **kwargs,
) -> Generator[str, None, None]:
    """Streams responses from OpenAI, yielding raw text chunks."""
    if api_key is None:
        api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
    system_message = get_system_message(npc) if npc else "You are a helpful assistant."
    if messages is None or len(messages) == 0:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

    completion = client.chat.completions.create(
        model=model, messages=messages, stream=True, **kwargs
    )

    for chunk in completion:
        if chunk.choices:
            for choice in chunk.choices:
                if choice.delta.content:
                    yield choice.delta.content  # Extracts raw text


def get_openai_like_stream(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
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
