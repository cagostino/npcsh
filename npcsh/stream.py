########
########
########
########
######## STREAM
########
########

from npcsh.npc_sysenv import get_system_message
from typing import Any, Dict, Generator, List
import os
import anthropic
import ollama  # Add to setup.py if missing
from openai import OpenAI
from diffusers import StableDiffusionPipeline
from google import genai

from google.generativeai import types
import google.generativeai as genai
import base64
import json


def get_anthropic_stream(
    messages,
    model: str,
    npc: Any = None,
    tools: list = None,
    images: List[Dict[str, str]] = None,
    api_key: str = None,
    tool_choice: Dict = None,
    **kwargs,
) -> Generator:
    """
    Streams responses from Anthropic, supporting images, tools, and yielding raw text chunks.

    Args:
        messages: List of conversation messages
        model: Anthropic model to use
        npc: Optional NPC context
        tools: Optional list of tools to provide to Claude
        images: Optional list of images to include
        api_key: Anthropic API key
        tool_choice: Optional tool choice configuration
        **kwargs: Additional arguments for the API call
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    if messages[0]["role"] == "system":
        system_message = messages[0]["content"]
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
        "stream": True,
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

    for chunk in response:
        yield chunk


def process_anthropic_tool_stream(
    stream, tool_map: Dict[str, callable], messages: List[Dict] = None
) -> List[Dict]:
    """
    Process the Anthropic tool use stream
    """
    tool_results = []
    current_tool = None
    current_input = ""
    context = messages[-1]["content"] if messages else ""

    for chunk in stream:
        # Look for tool use blocks
        if (
            chunk.type == "content_block_start"
            and getattr(chunk, "content_block", None)
            and chunk.content_block.type == "tool_use"
        ):
            current_tool = {
                "id": chunk.content_block.id,
                "name": chunk.content_block.name,
            }
            current_input = ""

        # Collect input JSON deltas
        if chunk.type == "content_block_delta" and hasattr(chunk.delta, "partial_json"):
            current_input += chunk.delta.partial_json

        # When tool input is complete
        if chunk.type == "content_block_stop" and current_tool:
            try:
                # Parse the complete input
                tool_input = json.loads(current_input) if current_input.strip() else {}

                # Add context to tool input
                tool_input["context"] = context

                # Execute the tool
                tool_func = tool_map.get(current_tool["name"])
                if tool_func:
                    result = tool_func(tool_input)
                    tool_results.append(
                        {
                            "tool_name": current_tool["name"],
                            "tool_input": tool_input,
                            "tool_result": result,
                        }
                    )
                else:
                    tool_results.append(
                        {
                            "tool_name": current_tool["name"],
                            "tool_input": tool_input,
                            "error": f"Tool {current_tool['name']} not found",
                        }
                    )

            except Exception as e:
                tool_results.append(
                    {
                        "tool_name": current_tool["name"],
                        "tool_input": current_input,
                        "error": str(e),
                    }
                )

            # Reset current tool
            current_tool = None
            current_input = ""

    return tool_results


from typing import List, Dict, Any, Literal

ProviderType = Literal[
    "openai", "anthropic", "ollama", "gemini", "deepseek", "openai-like"
]


def generate_tool_schema(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    provider: ProviderType,
    required: List[str] = None,
) -> Dict[str, Any]:
    """
    Generate provider-specific function/tool schema from common parameters

    Args:
        name: Name of the function
        description: Description of what the function does
        parameters: Dict of parameter names and their properties
        provider: Which provider to generate schema for
        required: List of required parameter names
    """
    if required is None:
        required = []

    if provider == "openai":
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required,
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    elif provider == "anthropic":
        return {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        }

    elif provider == "ollama":
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required,
                },
            },
        }
    elif provider == "gemini":
        # Convert our generic tool schema to a Gemini function declaration
        function = {
            "name": name,
            "description": description,
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    k: {
                        "type": v.get("type", "STRING").upper(),
                        "description": v.get("description", ""),
                        "enum": v.get("enum", None),
                    }
                    for k, v in parameters.items()
                },
                "required": required or [],
            },
        }

        # Create a Tool object as shown in the example
        return types.Tool(function_declarations=[function])

    raise ValueError(f"Unknown provider: {provider}")


def get_ollama_stream(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
    tools: list = None,
    images: list = None,
    tool_choice: Dict = None,
    **kwargs,
) -> Generator:
    """Streams responses from Ollama, supporting images and tools."""
    messages_copy = messages.copy()

    # Handle images if provided
    if images:
        messages[-1]["images"] = [image["file_path"] for image in images]

    # Add system message if not present
    if messages_copy[0]["role"] != "system":
        if npc is not None:
            system_message = get_system_message(npc)
            messages_copy.insert(0, {"role": "system", "content": system_message})

    # Prepare API call parameters
    api_params = {
        "model": model,
        "messages": messages_copy,
        "stream": True,
    }

    # Add tools if provided
    if tools:
        api_params["tools"] = tools

    # Make the API call
    for chunk in ollama.chat(**api_params):
        yield chunk


def process_ollama_tool_stream(
    stream, tool_map: Dict[str, callable], tools: List[Dict]
) -> List[Dict]:
    """Process the Ollama tool use stream"""
    tool_results = []
    content = ""

    # Build tool schema map
    tool_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"] for tool in tools
    }

    def convert_params(tool_name: str, params: dict) -> dict:
        """Convert parameters to the correct type based on schema"""
        schema = tool_schemas.get(tool_name, {})
        properties = schema.get("properties", {})

        converted = {}
        for key, value in params.items():
            prop_schema = properties.get(key, {})
            prop_type = prop_schema.get("type")

            if prop_type == "integer" and isinstance(value, str):
                try:
                    converted[key] = int(value)
                except (ValueError, TypeError):
                    converted[key] = 0
            else:
                converted[key] = value

        return converted

    # Accumulate content
    for chunk in stream:
        if chunk.message and chunk.message.content:
            content += chunk.message.content

    # Process complete JSON objects when done
    try:
        # Find all JSON objects in the content
        json_objects = []
        current = ""
        for char in content:
            current += char
            if current.count("{") == current.count("}") and current.strip().startswith(
                "{"
            ):
                json_objects.append(current.strip())
                current = ""

        # Process each JSON object
        for json_str in json_objects:
            try:
                tool_call = json.loads(json_str)
                tool_name = tool_call.get("name")
                tool_params = tool_call.get("parameters", {})

                if tool_name in tool_map:
                    # Convert parameters to correct types
                    converted_params = convert_params(tool_name, tool_params)
                    result = tool_map[tool_name](converted_params)
                    tool_results.append(
                        {
                            "tool_name": tool_name,
                            "tool_input": converted_params,
                            "tool_result": result,
                        }
                    )
            except Exception as e:
                tool_results.append({"error": str(e), "partial_json": json_str})

    except Exception as e:
        tool_results.append({"error": str(e), "content": content})

    return tool_results


def get_openai_stream(
    messages: List[Dict[str, str]],
    model: str,
    npc: Any = None,
    tools: list = None,
    images: List[Dict[str, str]] = None,
    api_key: str = None,
    tool_choice: Dict = None,
    **kwargs,
) -> Generator:
    """Streams responses from OpenAI, supporting images, tools and yielding raw text chunks."""

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

    # Prepare API call parameters
    api_params = {"model": model, "messages": messages, "stream": True, **kwargs}

    # Add tools if provided
    if tools:
        api_params["tools"] = tools

    # Add tool choice if specified
    if tool_choice:
        api_params["tool_choice"] = tool_choice

    stream = client.chat.completions.create(**api_params)

    for chunk in stream:
        yield chunk


def process_openai_tool_stream(stream, tool_map: Dict[str, callable]) -> List[Dict]:
    """
    Process the OpenAI tool use stream
    """
    final_tool_calls = {}
    tool_results = []

    for chunk in stream:
        delta = chunk.choices[0].delta

        # Process tool calls if present
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                index = tool_call.index

                # Initialize tool call if new
                if index not in final_tool_calls:
                    final_tool_calls[index] = {
                        "id": tool_call.id,
                        "name": tool_call.function.name if tool_call.function else None,
                        "arguments": (
                            tool_call.function.arguments if tool_call.function else ""
                        ),
                    }
                # Append arguments if continuing
                elif tool_call.function and tool_call.function.arguments:
                    final_tool_calls[index]["arguments"] += tool_call.function.arguments

    # Process all complete tool calls
    for tool_call in final_tool_calls.values():
        try:
            # Parse the arguments
            tool_input = (
                json.loads(tool_call["arguments"])
                if tool_call["arguments"].strip()
                else {}
            )

            # Execute the tool
            tool_func = tool_map.get(tool_call["name"])
            if tool_func:
                result = tool_func(tool_input)
                tool_results.append(
                    {
                        "tool_name": tool_call["name"],
                        "tool_input": tool_input,
                        "tool_result": result,
                    }
                )
            else:
                tool_results.append(
                    {
                        "tool_name": tool_call["name"],
                        "tool_input": tool_input,
                        "error": f"Tool {tool_call['name']} not found",
                    }
                )

        except Exception as e:
            tool_results.append(
                {
                    "tool_name": tool_call["name"],
                    "tool_input": tool_call["arguments"],
                    "error": str(e),
                }
            )

    return tool_results


def get_openai_like_stream(
    messages: List[Dict[str, str]],
    model: str,
    api_url: str,
    npc: Any = None,
    images: List[Dict[str, str]] = None,
    tools: list = None,
    api_key: str = None,
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
    if api_key is None:
        api_key = "dummy"
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
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    system_message = get_system_message(npc) if npc else ""

    messages_copy = messages.copy()
    if messages_copy[0]["role"] != "system":
        messages_copy.insert(0, {"role": "system", "content": system_message})

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
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
) -> Generator:
    """Streams responses from Gemini, supporting tools and yielding chunks."""
    import google.generativeai as genai

    if api_key is None:
        api_key = os.environ["GEMINI_API_KEY"]

    # Configure the Gemini API
    genai.configure(api_key=api_key)

    # Create model instance
    model = genai.GenerativeModel(model_name=model)

    # Convert all messages to contents list
    contents = []
    for msg in messages:
        if msg["role"] != "system":
            contents.append(
                {
                    "role": "user" if msg["role"] == "user" else "model",
                    "parts": [{"text": msg["content"]}],
                }
            )

    try:
        # Generate streaming response with full history
        response = model.generate_content(
            contents=contents, tools=tools if tools else None, stream=True, **kwargs
        )

        for chunk in response:
            yield chunk

    except Exception as e:
        print(f"Error in Gemini stream: {str(e)}")
        yield None


def process_gemini_tool_stream(
    stream, tool_map: Dict[str, callable], tools: List[Dict]
) -> List[Dict]:
    """Process the Gemini tool stream and execute tools."""
    tool_results = []

    try:
        for response in stream:
            if not response.candidates:
                continue

            for candidate in response.candidates:
                if not candidate.content or not candidate.content.parts:
                    continue

                for part in candidate.content.parts:
                    if hasattr(part, "function_call"):
                        try:
                            tool_name = part.function_call.name
                            # Convert MapComposite to dict
                            tool_args = dict(part.function_call.args)

                            if tool_name in tool_map:
                                result = tool_map[tool_name](tool_args)
                                tool_results.append(
                                    {
                                        "tool_name": tool_name,
                                        "tool_input": tool_args,
                                        "tool_result": result,
                                    }
                                )
                        except Exception as e:
                            tool_results.append(
                                {
                                    "error": str(e),
                                    "tool_name": (
                                        tool_name
                                        if "tool_name" in locals()
                                        else "unknown"
                                    ),
                                    "tool_input": (
                                        tool_args if "tool_args" in locals() else None
                                    ),
                                }
                            )

    except Exception as e:
        tool_results.append({"error": f"Stream processing error: {str(e)}"})

    return tool_results
