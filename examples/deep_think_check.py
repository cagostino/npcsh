from typing import Generator, Dict, List, Any, Optional
import re

from npcsh.llm_funcs import enter_reasoning_human_in_the_loop

if __name__ == "__main__":
    # Example usage
    messages = [
        {
            "role": "user",
            "content": "Tell me a joke about my favorite animal and my favorite city",
        },
    ]

    for chunk in enter_reasoning_human_in_the_loop(
        messages, "deepseek-reasoner", "deepseek"
    ):
        chunk_content = "".join(
            choice.delta.content
            for choice in chunk.choices
            if choice.delta.content is not None
        )
        print(chunk_content, end="")
