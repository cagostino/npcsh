from typing import Generator, Dict, List, Any, Optional
import re

from npcsh.llm_funcs import get_stream, get_llm_response


def enter_human_in_the_loop(
    messages: List[Dict[str, str]],
    model: str = "deepseek-r1",
    provider: str = "ollama",
    npc: Any = None,
) -> Generator[str, None, None]:
    """
    Stream responses while checking for think tokens and handling human input when needed.

    Args:
        messages: List of conversation messages
        model: LLM model to use
        provider: Model provider
        npc: NPC instance if applicable

    Yields:
        Streamed response chunks
    """
    # Get the initial stream
    response_stream = get_stream(messages, model=model, provider=provider, npc=npc)

    thoughts = []
    response_chunks = []
    in_think_block = False

    for chunk in response_stream:
        # Extract content based on provider/model type
        if provider == "ollama":
            chunk_content = chunk.get("message", {}).get("content", "")
        elif provider == "openai" or provider == "deepseek":
            chunk_content = "".join(
                choice.delta.content
                for choice in chunk.choices
                if choice.delta.content is not None
            )
        elif provider == "anthropic":
            if chunk.type == "content_block_delta":
                chunk_content = chunk.delta.text
            else:
                chunk_content = ""
        else:
            # Default extraction
            chunk_content = str(chunk)

        # Always yield the chunk whether in think block or not
        response_chunks.append(chunk_content)
        # Track think block state and accumulate thoughts
        if "<th" in "".join(response_chunks) and "/th" not in "".join(response_chunks):
            in_think_block = True
            print("Think block started")
        if in_think_block:
            thoughts.append(chunk_content)
        print("/th" in chunk_content)

        if "</th" in "".join(response_chunks):
            print("Think block ended")

            in_think_block = False

            # Analyze thoughts to determine if user input is needed
            thought_text = "".join(thoughts)

            # Check if additional input is needed based on thoughts
            input_needed = analyze_thoughts_for_input(thought_text, messages)

            if input_needed:
                # Get clarification from user
                user_input = request_user_input(input_needed)

                # Add user input to messages and restart stream
                messages.append({"role": "user", "content": user_input})

                # Recursively call with updated messages
                yield from enter_human_in_the_loop(messages, model, provider, npc)
                return

        yield chunk


def analyze_thoughts_for_input(
    thought_text: str, messages: List[Dict[str, str]]
) -> Optional[Dict[str, str]]:
    """
    Analyze accumulated thoughts to determine if user input is needed.

    Args:
        thought_text: Accumulated text from think block
        messages: Conversation history

    Returns:
        Dict with input request details if needed, None otherwise
    """

    messages_for_analysis = [
        {
            "role": "user",
            "content": f"""
         Analyze these thoughts:
         {thought_text}
         and determine if additional user input would be helpful.
        Return a JSON object with:"""
            + """
        {
            "input_needed": boolean,
            "request_reason": string explaining why input is needed,
            "request_prompt": string to show user if input needed
        }
        Consider things like:
        - Ambiguity in the user's request
        - Missing context that would help provide a better response
        - Clarification needed about user preferences/requirements
        Only request input if it would meaningfully improve the response.""",
        },
    ]

    response = get_llm_response(
        messages[-1]["content"],
        model="gpt-4o-mini",
        provider="openai",
        messages=messages,
    )
    print(response)
    try:
        result = response.get("response", {})
        if isinstance(result, str):
            result = json.loads(result)

        if result.get("input_needed"):
            return {
                "reason": result["request_reason"],
                "prompt": result["request_prompt"],
            }
    except:
        pass

    return None


def request_user_input(input_request: Dict[str, str]) -> str:
    """
    Request and get input from user.

    Args:
        input_request: Dict with reason and prompt for input

    Returns:
        User's input text
    """
    print(f"\nAdditional input needed: {input_request['reason']}")
    return input(f"{input_request['prompt']}: ")


if __name__ == "__main__":
    # Example usage
    messages = [
        {
            "role": "user",
            "content": "Tell me a joke. Think first though and use <think> tags.",
        },
    ]

    for chunk in enter_human_in_the_loop(messages, "deepseek-reasoner", "deepseek"):
        chunk_content = "".join(
            choice.delta.content
            for choice in chunk.choices
            if choice.delta.content is not None
        )
        print(chunk_content, end="")
