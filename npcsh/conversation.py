########
########
########
########
######## CONVERSATION
########
from typing import Any, Dict, Generator, List
import os
import anthropic
import ollama  # Add to setup.py if missing
from openai import OpenAI
from diffusers import StableDiffusionPipeline
from google.generativeai import types
import google.generativeai as genai
from .npc_sysenv import get_system_message


def get_ollama_conversation(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
    tools: list = None,
    images=None,
    **kwargs,
) -> List[Dict[str, str]]:
    """
    Function Description:
        This function generates a conversation using the Ollama API.
    Args:
        messages (List[Dict[str, str]]): The list of messages in the conversation.
        model (str): The model to use for the conversation.
    Keyword Args:
        npc (Any): The NPC object.
    Returns:
        List[Dict[str, str]]: The list of messages in the conversation.
    """

    messages_copy = messages.copy()
    if messages_copy[0]["role"] != "system":
        if npc is not None:
            system_message = get_system_message(npc)
            messages_copy.insert(0, {"role": "system", "content": system_message})

    response = ollama.chat(model=model, messages=messages_copy)
    messages_copy.append(response["message"])
    return messages_copy


def get_openai_conversation(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
    tools: list = None,
    api_key: str = None,
    images=None,
    **kwargs,
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

    try:
        if api_key is None:
            api_key = os.environ["OPENAI_API_KEY"]
        client = OpenAI(api_key=api_key)

        system_message = (
            get_system_message(npc) if npc else "You are a helpful assistant."
        )

        if messages is None:
            messages = []

        # Ensure the system message is at the beginning
        if not any(msg["role"] == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": system_message})

        # messages should already include the user's latest message

        # Make the API call with the messages including the latest user input
        completion = client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

        response_message = completion.choices[0].message
        messages.append({"role": "assistant", "content": response_message.content})

        return messages

    except Exception as e:
        return f"Error interacting with OpenAI: {e}"


def get_openai_like_conversation(
    messages: List[Dict[str, str]],
    model: str,
    api_url: str,
    npc: Any = None,
    images=None,
    tools: list = None,
    api_key: str = None,
    **kwargs,
) -> List[Dict[str, str]]:
    """
    Function Description:
        This function generates a conversation using an OpenAI-like API.
    Args:
        messages (List[Dict[str, str]]): The list of messages in the conversation.
        model (str): The model to use for the conversation.
    Keyword Args:
        npc (Any): The NPC object.
        api_url (str): The URL of the API endpoint.
        api_key (str): The API key for accessing the API.
    Returns:
        List[Dict[str, str]]: The list of messages in the conversation.
    """

    if api_url is None:
        raise ValueError("api_url is required for openai-like provider")
    if api_key is None:
        api_key = "dummy_api_key"
    try:
        client = OpenAI(api_key=api_key, base_url=api_url)

        system_message = (
            get_system_message(npc) if npc else "You are a helpful assistant."
        )

        if messages is None:
            messages = []

        # Ensure the system message is at the beginning
        if not any(msg["role"] == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": system_message})

        # messages should already include the user's latest message

        # Make the API call with the messages including the latest user input

        completion = client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        response_message = completion.choices[0].message
        messages.append({"role": "assistant", "content": response_message.content})

        return messages

    except Exception as e:
        return f"Error interacting with OpenAI: {e}"

    return messages


def get_anthropic_conversation(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
    tools: list = None,
    images=None,
    api_key: str = None,
    **kwargs,
) -> List[Dict[str, str]]:
    """
    Function Description:
        This function generates a conversation using the Anthropic API.
    Args:
        messages (List[Dict[str, str]]): The list of messages in the conversation.
        model (str): The model to use for the conversation.
    Keyword Args:
        npc (Any): The NPC object.
        api_key (str): The API key for accessing the Anthropic API.
    Returns:
        List[Dict[str, str]]: The list of messages in the conversation.
    """

    try:
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY", None)
        system_message = get_system_message(npc) if npc else ""
        client = anthropic.Anthropic(api_key=api_key)
        last_user_message = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break

        if last_user_message is None:
            raise ValueError("No user message found in the conversation history.")

        # if a sys message is in messages, remove it
        if messages[0]["role"] == "system":
            messages.pop(0)

        message = client.messages.create(
            model=model,
            system=system_message,  # Include system message in each turn for Anthropic
            messages=messages,  # Send only the last user message
            max_tokens=8192,
            **kwargs,
        )

        messages.append({"role": "assistant", "content": message.content[0].text})

        return messages

    except Exception as e:
        return f"Error interacting with Anthropic conversations: {e}"


def get_gemini_conversation(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
    tools: list = None,
    api_key: str = None,
) -> List[Dict[str, str]]:
    """
    Function Description:
        This function generates a conversation using the Gemini API.
    Args:
        messages (List[Dict[str, str]]): The list of messages in the conversation.
        model (str): The model to use for the conversation.
    Keyword Args:
        npc (Any): The NPC object.
    Returns:
        List[Dict[str, str]]: The list of messages in the conversation.
    """
    # Make the API call to Gemini

    # print(messages)
    response = get_gemini_response(
        messages[-1]["content"], model, messages=messages[1:], npc=npc
    )
    # print(response)
    return response.get("messages", [])


def get_deepseek_conversation(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
    tools: list = None,
    api_key: str = None,
) -> List[Dict[str, str]]:
    """
    Function Description:
        This function generates a conversation using the DeepSeek API.
    Args:
        messages (List[Dict[str, str]]): The list of messages in the conversation.
        model (str): The model to use for the conversation.
    Keyword Args:
        npc (Any): The NPC object.
    Returns:
        List[Dict[str, str]]: The list of messages in the conversation.
    """

    system_message = get_system_message(npc) if npc else "You are a helpful assistant."

    # Prepare the messages list
    if messages is None or len(messages) == 0:
        messages = [{"role": "system", "content": system_message}]
    elif not any(msg["role"] == "system" for msg in messages):
        messages.insert(0, {"role": "system", "content": system_message})

    # Make the API call to DeepSeek
    try:
        response = get_deepseek_response(
            messages[-1]["content"], model, messages=messages, npc=npc
        )
        messages.append(
            {"role": "assistant", "content": response.get("response", "No response")}
        )

    except Exception as e:
        messages.append(
            {
                "role": "assistant",
                "content": f"Error interacting with DeepSeek: {str(e)}",
            }
        )

    return messages
