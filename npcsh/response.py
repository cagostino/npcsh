from typing import Any, Dict, Generator, List, Union
from pydantic import BaseModel
import os
import anthropic
from openai import OpenAI
from google.generativeai import types
from google import genai

import google.generativeai as genai
from npcsh.npc_sysenv import (
    get_system_message,
    compress_image,
    available_chat_models,
    available_reasoning_models,
)

import json
import requests
import base64
from PIL import Image


def get_deepseek_response(
    prompt: str,
    model: str,
    images: List[Dict[str, str]] = None,
    npc: Any = None,
    tools: list = None,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    api_key: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Function Description:
        This function generates a response using the DeepSeek API.
    Args:
        prompt (str): The prompt for generating the response.
        model (str): The model to use for generating the response.
    Keyword Args:
        images (List[Dict[str, str]]): The list of images.
        npc (Any): The NPC object.
        format (str): The format of the response.
        messages (List[Dict[str, str]]): The list of messages.
    Returns:
        Any: The response generated by the DeepSeek API.


    """
    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY", None)
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # print(client)

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
    # print(messages)
    # print(model)
    response_format = None if format == "json" else format
    if response_format is None:
        completion = client.chat.completions.create(model=model, messages=messages)
        llm_response = completion.choices[0].message.content
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
            completion = client.beta.chat.completions.parse(
                model=model, messages=messages, response_format=response_format
            )
            items_to_return = {"response": completion.choices[0].message.parsed.dict()}
            items_to_return["messages"] = messages

            items_to_return["messages"].append(
                {"role": "assistant", "content": completion.choices[0].message.parsed}
            )
            return items_to_return
        except Exception as e:
            print("pydantic outputs not yet implemented with deepseek?")


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

    # except Exception as e:
    #    return {"error": f"Exception occurred: {e}"}


def get_openai_response(
    prompt: str,
    model: str,
    images: List[Dict[str, str]] = None,
    npc: Any = None,
    tools: list = None,
    format: Union[str, BaseModel] = None,
    api_key: str = None,
    messages: List[Dict[str, str]] = None,
    **kwargs,
):
    """
    Function Description:
        This function generates a response using the OpenAI API.
    Args:
        prompt (str): The prompt for generating the response.
        model (str): The model to use for generating the response.
    Keyword Args:
        images (List[Dict[str, str]]): The list of images.
        npc (Any): The NPC object.
        format (str): The format of the response.
        api_key (str): The API key for accessing the OpenAI API.
        messages (List[Dict[str, str]]): The list of messages.
    Returns:
        Any: The response generated by the OpenAI API.
    """

    # try:
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if len(api_key) == 0:
            raise ValueError("API key not found.")
    client = OpenAI(api_key=api_key)
    # print(npc)

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
    # print(model)
    response_format = None if format == "json" else format
    if response_format is None:
        completion = client.chat.completions.create(model=model, messages=messages)
        llm_response = completion.choices[0].message.content
        items_to_return = {"response": llm_response}

        items_to_return["messages"] = messages
        # print(llm_response, model)
        if format == "json":
            if model in available_reasoning_models:
                raise NotImplementedError(
                    "Reasoning models do not support JSON output."
                )
            try:
                if isinstance(llm_response, str):
                    if llm_response.startswith("```json"):
                        llm_response = (
                            llm_response.replace("```json", "")
                            .replace("```", "")
                            .strip()
                        )
                    llm_response = json.loads(llm_response)
                items_to_return["response"] = llm_response
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
        completion = client.beta.chat.completions.parse(
            model=model, messages=messages, response_format=response_format
        )
        items_to_return = {"response": completion.choices[0].message.parsed.dict()}
        items_to_return["messages"] = messages

        items_to_return["messages"].append(
            {"role": "assistant", "content": completion.choices[0].message.parsed}
        )
        return items_to_return
    # except Exception as e:
    #    print("openai api key", api_key)
    #    print(f"Error interacting with OpenAI: {e}")
    #    return f"Error interacting with OpenAI: {e}"


def get_anthropic_response(
    prompt: str,
    model: str,
    images: List[Dict[str, str]] = None,
    npc: Any = None,
    tools: list = None,
    format: str = None,
    api_key: str = None,
    messages: List[Dict[str, str]] = None,
    **kwargs,
):
    """
    Function Description:
        This function generates a response using the Anthropic API.
    Args:
        prompt (str): The prompt for generating the response.
        model (str): The model to use for generating the response.
    Keyword Args:
        images (List[Dict[str, str]]): The list of images.
        npc (Any): The NPC object.
        format (str): The format of the response.
        api_key (str): The API key for accessing the Anthropic API.
        messages (List[Dict[str, str]]): The list of messages.
    Returns:
        Any: The response generated by the Anthropic API.
    """

    try:
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)

        if messages[0]["role"] == "system":
            system_message = messages[0]
            messages = messages[1:]
        elif npc is not None:
            system_message = get_system_message(npc)

        # Preprocess messages to ensure content is a list of dicts
        for message in messages:
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
        # Add images if provided
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

        # Prepare API call parameters

        api_params = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 8192),
            "stream": False,
            "system": system_message,
        }

        # Add tools if provided
        if tools:
            api_params["tools"] = tools

        # Add tool choice if specified
        if tool_choice:
            api_params["tool_choice"] = tool_choice

        # Make the API call
        response = client.messages.create(**api_params)

        llm_response = message.content[0].text
        items_to_return = {"response": llm_response}
        messages.append(
            {"role": "assistant", "content": {"type": "text", "text": llm_response}}
        )
        items_to_return["messages"] = messages

        # Handle JSON format if requested
        if format == "json":
            try:
                if isinstance(llm_response, str):
                    if llm_response.startswith("```json"):
                        llm_response = (
                            llm_response.replace("```json", "")
                            .replace("```", "")
                            .strip()
                        )
                    llm_response = json.loads(llm_response)
                items_to_return["response"] = llm_response
                return items_to_return
            except json.JSONDecodeError:
                print(f"Warning: Expected JSON response, but received: {llm_response}")
                return {"response": llm_response, "error": "Invalid JSON response"}
        else:
            # only append to messages if the response is not json
            messages.append({"role": "assistant", "content": llm_response})
        # print("teststea")
        return items_to_return

    except Exception as e:
        return f"Error interacting with Anthropic llm response: {e}"


def get_openai_like_response(
    prompt: str,
    model: str,
    api_url: str,
    api_key: str = None,
    npc: Any = None,
    tools: list = None,
    images: list = None,
    messages: list = None,
    format=None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Function Description:
        This function generates a response using  API.
        penai-like
    Args:
        prompt (str): The prompt for generating the response.
        model (str): The model to use for generating the response.
    Keyword Args:
        images (List[Dict[str, str]]): The list of images.
        npc (Any): The NPC object.
        format (str): The format of the response.
        messages (List[Dict[str, str]]): The list of messages.
    Returns:
        Any: The response generated by the DeepSeek API.


    """
    if api_key is None:
        api_key = "dummy_api_key"
    client = OpenAI(api_key=api_key, base_url=api_url)
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
        completion = client.chat.completions.create(model=model, messages=messages)
        llm_response = completion.choices[0].message.content
        items_to_return = {"response": llm_response}

        items_to_return["messages"] = messages
        # print(llm_response, model)
        if format == "json":
            if model in available_reasoning_models:
                raise NotImplementedError(
                    "Reasoning models do not support JSON output."
                )
            try:
                if isinstance(llm_response, str):
                    if llm_response.startswith("```json"):
                        llm_response = (
                            llm_response.replace("```json", "")
                            .replace("```", "")
                            .strip()
                        )
                    # print(llm_response)
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

        completion = client.beta.chat.completions.parse(
            model=model, messages=messages, response_format=response_format
        )

        items_to_return = {"response": completion.choices[0].message.parsed.dict()}
        items_to_return["messages"] = messages

        items_to_return["messages"].append(
            {"role": "assistant", "content": completion.choices[0].message.parsed}
        )
        return items_to_return


def get_gemini_response(
    prompt: str,
    model: str,
    images: List[Dict[str, str]] = None,
    npc: Any = None,
    tools: list = None,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    api_key: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generates a response using the Gemini API.
    """
    # Configure the Gemini API
    if api_key is None:
        genai.configure(api_key=gemini_api_key)

    # Prepare the system message
    system_message = get_system_message(npc) if npc else "You are a helpful assistant."
    model = genai.GenerativeModel(model, system_instruction=system_message)

    # Extract just the content to send to the model
    if messages is None or len(messages) == 0:
        content_to_send = prompt
    else:
        # Get the latest message's content
        latest_message = messages[-1]
        content_to_send = (
            latest_message["parts"][0]
            if "parts" in latest_message
            else latest_message.get("content", prompt)
        )
    history = []
    if messages:
        for msg in messages:
            if "content" in msg:
                # Convert content to parts format
                history.append({"role": msg["role"], "parts": [msg["content"]]})
            else:
                # Already in parts format
                history.append(msg)
    # If no history, create a new message list
    if not history:
        history = [{"role": "user", "parts": [prompt]}]
    elif isinstance(prompt, str):  # Add new prompt to existing history
        history.append({"role": "user", "parts": [prompt]})

    # Handle images if provided
    # Handle images by adding them to the last message's parts
    if images:
        for image in images:
            with open(image["file_path"], "rb") as image_file:
                img = Image.open(image_file)
                history[-1]["parts"].append(img)
    # Generate the response
    # try:
    # Send the entire conversation history to maintain context
    response = model.generate_content(history)
    llm_response = response.text

    # Filter out empty parts
    if isinstance(llm_response, list):
        llm_response = " ".join([part for part in llm_response if part.strip()])
    elif not llm_response.strip():
        llm_response = ""

    # Prepare the return dictionary
    items_to_return = {"response": llm_response, "messages": history}
    # print(llm_response, type(llm_response))

    # Handle JSON format if specified
    if format == "json":
        if isinstance(llm_response, str):
            if llm_response.startswith("```json"):
                llm_response = (
                    llm_response.replace("```json", "").replace("```", "").strip()
                )

        try:
            items_to_return["response"] = json.loads(llm_response)
        except json.JSONDecodeError:
            print(f"Warning: Expected JSON response, but received: {llm_response}")
            return {"error": "Invalid JSON response"}
    else:
        # Append the model's response to the messages
        history.append({"role": "model", "parts": [llm_response]})
        items_to_return["messages"] = history

    return items_to_return

    # except Exception as e:
    #    return {"error": f"Error generating response: {str(e)}"}
