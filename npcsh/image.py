# image.py
# import os
import time
import platform
import subprocess
from typing import Dict, Any

from .llm_funcs import get_llm_response


def capture_screenshot(npc: Any = None) -> Dict[str, str]:
    """
    Function Description:
        This function captures a screenshot of the current screen and saves it to a file.
    Args:
        npc: The NPC object representing the current NPC.
    Keyword Args:
        None
    Returns:
        A dictionary containing the filename, file path, and model kwargs.
    """
    # Ensure the directory exists
    directory = os.path.expanduser("~/.npcsh/screenshots")
    os.makedirs(directory, exist_ok=True)

    # Generate a unique filename
    filename = f"screenshot_{int(time.time())}.png"
    file_path = os.path.join(directory, filename)

    system = platform.system()
    model_kwargs = {}

    if npc is not None:
        if npc.provider is not None:
            model_kwargs["provider"] = npc.provider

        if npc.model is not None:
            model_kwargs["model"] = npc.model

    if system == "Darwin":  # macOS
        subprocess.run(["screencapture", "-i", file_path])  # This waits on macOS
    elif system == "Linux":
        # Use a loop to check for the file's existence
        if (
            subprocess.run(
                ["which", "gnome-screenshot"], capture_output=True
            ).returncode
            == 0
        ):
            subprocess.Popen(
                ["gnome-screenshot", "-a", "-f", file_path]
            )  # Use Popen for non-blocking
            while not os.path.exists(file_path):  # Wait for file to exist
                time.sleep(0.1)  # Check every 100ms
        elif subprocess.run(["which", "scrot"], capture_output=True).returncode == 0:
            subprocess.Popen(["scrot", "-s", file_path])  # Use Popen for non-blocking
            while not os.path.exists(file_path):  # Wait for file to exist
                time.sleep(0.1)  # Check every 100ms

        else:
            print(
                "No supported screenshot tool found. Please install gnome-screenshot or scrot."
            )
            return None
    else:
        print(f"Unsupported operating system: {system}")
        return None

    print(f"Screenshot saved to: {file_path}")
    return {"filename": filename, "file_path": file_path, "model_kwargs": model_kwargs}


def analyze_image_base(
    user_prompt: str, file_path: str, filename: str, npc: Any = None
) -> Dict[str, str]:
    """
    Function Description:
        This function analyzes an image using the LLM model and returns the response.
    Args:
        user_prompt: The user prompt to provide to the LLM model.
        file_path: The path to the image file.
        filename: The name of the image file.
    Keyword Args:
        npc: The NPC object representing the current NPC.
    Returns:
        The response from the LLM model

    """

    if os.path.exists(file_path):
        image_info = {"filename": filename, "file_path": file_path}

        if user_prompt:
            # try:
            response = get_llm_response(user_prompt, images=[image_info], npc=npc)

            # Add to command history *inside* the try block

            # print(response["response"])  # Print response after adding to history
            return response

            # except Exception as e:
            # error_message = f"Error during LLM processing: {e}"
            # print(error_message)
            # return error_message

        else:  # This part needs to be inside the outer 'if os.path.exists...' block
            print("Skipping LLM processing.")
            return image_info  # Return image info if no prompt is given
    else:  # This else also needs to be part of the outer 'if os.path.exists...' block
        print("Screenshot capture failed or was cancelled.")
        return None


def analyze_image(
    command_history: Any,
    user_prompt: str,
    file_path: str,
    filename: str,
    npc: Any = None,
    **model_kwargs,
) -> Dict[str, str]:
    """
    Function Description:
        This function captures a screenshot, analyzes it using the LLM model, and returns the response.
    Args:
        command_history: The command history object to add the command to.
        user_prompt: The user prompt to provide to the LLM model.
        file_path: The path to the image file.
        filename: The name of the image file.
    Keyword Args:
        npc: The NPC object representing the current NPC.
        model_kwargs: Additional keyword arguments for the LLM model.
    Returns:
        The response from the LLM model.
    """

    if os.path.exists(file_path):
        image_info = {"filename": filename, "file_path": file_path}

        if user_prompt:
            try:
                response = get_llm_response(
                    user_prompt, images=[image_info], npc=npc, **model_kwargs
                )

                # Add to command history *inside* the try block
                command_history.add(
                    f"screenshot with prompt: {user_prompt}",
                    ["screenshot", npc.name if npc else ""],
                    response,
                    os.getcwd(),
                )
                # import pdb
                # pdb.set_trace()
                print(response["response"])  # Print response after adding to history
                return response

            except Exception as e:
                error_message = f"Error during LLM processing: {e}"
                print(error_message)
                return error_message

        else:  # This part needs to be inside the outer 'if os.path.exists...' block
            print("Skipping LLM processing.")
            return image_info  # Return image info if no prompt is given
    else:  # This else also needs to be part of the outer 'if os.path.exists...' block
        print("Screenshot capture failed or was cancelled.")
        return None
