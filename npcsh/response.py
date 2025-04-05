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
    Improved version with consistent JSON parsing
    """
    system_message = get_system_message(npc) if npc else "You are a helpful assistant."

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

    try:
        # Make the API call
        resp = completion(
            model=f"{provider}/{model}",
            messages=messages,
            response_format={"type": "json_object"} if format == "json" else None,
            **kwargs,
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
        if format == "json":
            try:
                if isinstance(llm_response, str):
                    items_to_return["response"] = json.loads(llm_response)
                else:
                    items_to_return["response"] = (
                        llm_response  # Assume it's already parsed
                    )
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
