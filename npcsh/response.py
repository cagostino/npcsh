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


def get_litellm_response(
    prompt: str,
    model: str,
    provider: str,
    images: List[Dict[str, str]] = None,
    npc: Any = None,
    tools: list = None,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    api_key: str = None,
    api_url: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Function Description:
        This function generates a response using the Lite LLM API.
    Args:

        prompt (str): The prompt for generating the response.
        model (str): The model to use for generating the response.
    Keyword Args:
        images (List[Dict[str, str]]): The list of images.
        npc (Any): The NPC object.
        format (str): The format of the response.
        messages (List[Dict[str, str]]): The list of messages.
    Returns:
        Dict
    """

    system_message = get_system_message(npc) if npc else "You are a helpful assistant."
    if messages is None or len(messages) == 0:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
    if images:
        for image in images:
            # print(f"Image file exists: {os.path.exists(image['file_path'])}")

            with open(image["file_path"], "rb") as image_file:
                image_data = base64.b64encode(compress_image(image_file.read())).decode(
                    "utf-8"
                )
                messages[-1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                        },
                    }
                )
    response_format = None if format == "json" else format
    if response_format is None:
        resp = completion(model=f"{provider}/{model}", messages=messages)

        llm_response = resp.choices[0].message.content
        items_to_return = {"response": llm_response}

        items_to_return["messages"] = messages
        # print(llm_response, model)
        if format == "json":
            try:
                items_to_return["response"] = json.loads(llm_response)

                return items_to_return
            except json.JSONDecodeError:
                print(f"Warning: Expected JSON response, but received: {llm_response}")
                return {"error": "Invalid JSON response"}
        else:
            items_to_return["messages"].append(
                {"role": "assistant", "content": llm_response}
            )
            return items_to_return

    else:
        if model in available_reasoning_models:
            raise NotImplementedError("Reasoning models do not support JSON output.")
        try:
            resp = completion(
                model=f"{provider}/{model}",
                messages=messages,
                response_format=response_format,
            )
            items_to_return = {"response": resp.choices[0].message.parsed.dict()}
            items_to_return["messages"] = messages

            items_to_return["messages"].append(
                {"role": "assistant", "content": completion.choices[0].message.parsed}
            )
            return items_to_return
        except Exception as e:
            print("pydantic outputs not yet implemented with deepseek?")
