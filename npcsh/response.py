import json
import requests
import base64
import os
from PIL import Image
from typing import Any, Dict, Generator, List, Union

from pydantic import BaseModel
from npcsh.npc_sysenv import (
    get_system_message,
    compress_image,
    available_chat_models,
    available_reasoning_models,
)

from litellm import completion

import litellm

litellm._turn_on_debug()

try:
    import ollama
except:
    pass


def get_ollama_response(
    prompt: str,
    model: str,
    images: List[Dict[str, str]] = None,
    npc: Any = None,
    tools: list = None,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generates a response using the Ollama API.

    Args:
        prompt (str): Prompt for generating the response.
        model (str): Model to use for generating the response.
        images (List[Dict[str, str]], optional): List of image data. Defaults to None.
        npc (Any, optional): Optional NPC object. Defaults to None.
        format (Union[str, BaseModel], optional): Response format or schema. Defaults to None.
        messages (List[Dict[str, str]], optional): Existing messages to append responses. Defaults to None.

    Returns:
        Dict[str, Any]: The response, optionally including updated messages.
    """
    import ollama

    # try:
    # Prepare the message payload
    system_message = get_system_message(npc) if npc else "You are a helpful assistant."
    if messages is None or len(messages) == 0:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

    if images:
        messages[-1]["images"] = [image["file_path"] for image in images]

    # Prepare format
    if isinstance(format, type):
        schema = format.model_json_schema()
        res = ollama.chat(model=model, messages=messages, format=schema)

    elif isinstance(format, str):
        if format == "json":
            res = ollama.chat(model=model, messages=messages, format=format)
        else:
            res = ollama.chat(model=model, messages=messages)
    else:
        res = ollama.chat(model=model, messages=messages)
    response_content = res.get("message", {}).get("content")

    # Prepare the return dictionary
    result = {"response": response_content}

    # Append response to messages if provided
    if messages is not None:
        messages.append({"role": "assistant", "content": response_content})
        result["messages"] = messages

    # Handle JSON format if specified
    if format == "json":
        if model in available_reasoning_models:
            raise NotImplementedError("Reasoning models do not support JSON output.")
        try:
            if isinstance(response_content, str):
                if response_content.startswith("```json"):
                    response_content = (
                        response_content.replace("```json", "")
                        .replace("```", "")
                        .strip()
                    )
                response_content = json.loads(response_content)
            # print(response_content, type(response_content))
            result["response"] = response_content
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON response: {response_content}"}

    return result


def get_litellm_response(
    prompt: str,
    model: str,
    provider: str = None,
    images: List[Dict[str, str]] = None,
    npc: Any = None,
    tools: list = None,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    api_key: str = None,
    api_url: str = None,
    tool_choice: Dict = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Improved version with consistent JSON parsing
    """
    if provider == "ollama":
        return get_ollama_response(
            prompt, model, images, npc, tools, format, messages, **kwargs
        )

    system_message = get_system_message(npc) if npc else "You are a helpful assistant."
    if format == "json":
        prompt += """If you are a returning a json object, begin directly with the opening {.
            If you are returning a json array, begin directly with the opening [.
            Do not include any additional markdown formatting or leading
            ```json tags in your response. The item keys should be based on the ones provided
            by the user. Do not invent new ones.

        """
    if messages is None or len(messages) == 0:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

    if images:
        for image in images:
            with open(image["file_path"], "rb") as image_file:
                image_data = base64.b64encode(compress_image(image_file.read())).decode(
                    "utf-8"
                )
                messages[-1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    }
                )
    api_params = {
        "messages": messages,
    }
    if provider is None:
        split = model.split("/")
        if len(split) == 2:
            provider = split[0]
        # if provider == "ollama":
        # uncomment the two lines below once litellm works better with ollama
        # litellm works better with ollama_chat
        # api_params["api_base"] = "http://localhost:11434"
        # provider = "ollama_chat"
        api_params["format"] = format

    # else:
    if api_url is not None:
        # the default api_url is for npcsh's NPCSH_API_URL
        # for an openai-like provider.
        # so the proviuder should only ever be openai-like
        if provider == "openai-like":
            api_params["api_base"] = api_url

    if format == "json":
        api_params["response_format"] = {"type": "json_object"}
    elif format is not None:
        # pydantic model
        api_params["response_format"] = format

    if "/" not in model:  # litellm expects provder/model so let ppl provide like that
        model_str = f"{provider}/{model}"
    else:
        model_str = model
    api_params["model"] = model_str
    if api_key is not None:
        api_params["api_key"] = api_key
    # Add tools if provided
    if tools:
        api_params["tools"] = tools
    # Add tool choice if specified
    if tool_choice:
        api_params["tool_choice"] = tool_choice
    if kwargs:
        for key, value in kwargs.items():
            # minimum parameter set for anthropic to work
            if key in [
                "stream",
                "stop",
                "temperature",
                "top_p",
                "max_tokens",
                "max_completion_tokens",
                "tools",
                "tool_choice",
                "extra_headers",
                "parallel_tool_calls",
                "response_format",
                "user",
            ]:
                api_params[key] = value

    try:
        print(api_params)
        # litellm completion appears to have some
        # ollama issues, so will default to our
        # custom implementation until we can revisit
        # when its likely more better supported
        resp = completion(
            **api_params,
        )

        # Get the raw response content
        llm_response = resp.choices[0].message.content

        # Prepare return dict
        items_to_return = {
            "response": llm_response,
            "messages": messages,
            "raw_response": resp,  # Include the full response for debugging
        }

        # Handle JSON format requests
        print(format)
        if format == "json":
            try:
                if isinstance(llm_response, str):
                    print("converting the json")
                    loaded = json.loads(llm_response)
                else:
                    loaded = llm_response  # Assume it's already parsed
                if "json" in loaded:
                    items_to_return["response"] = loaded["json"]
                else:
                    items_to_return["response"] = loaded

            except (json.JSONDecodeError, TypeError) as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Raw response: {llm_response}")
                items_to_return["error"] = "Invalid JSON response"
                return items_to_return

        # Add assistant response to message history
        items_to_return["messages"].append(
            {
                "role": "assistant",
                "content": (
                    llm_response if isinstance(llm_response, str) else str(llm_response)
                ),
            }
        )

        return items_to_return

    except Exception as e:
        print(f"Error in get_litellm_response: {str(e)}")
        return {"error": str(e), "messages": messages, "response": None}
