########
########
########
########
######## CONVERSATION
########
from typing import Any, Dict, Generator, List
import os

from litellm import completion
from npcsh.npc_sysenv import get_system_message


def get_litellm_conversation(
    messages: List[Dict[str, str]],
    model: str,
    provider: str,
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


    Examples:
        >>> messages = [    {"role": "user", "content": "Hello, how are you?"}]
        >>> model = 'openai/gpt-4o-mini'
        >>> response = get_litellm_conversation(messages, model)
    """
    system_message = get_system_message(npc) if npc else "You are a helpful assistant."
    if messages is None:
        messages = []

    if not any(msg["role"] == "system" for msg in messages):
        messages.insert(0, {"role": "system", "content": system_message})

    resp = completion(model=f"{provider}/{model}", messages=messages)

    response_message = resp.choices[0].message
    messages.append({"role": "assistant", "content": response_message.content})

    return messages
