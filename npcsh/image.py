# image.py
# import os
import time
import platform
import subprocess
from typing import Dict, Any
from PIL import ImageGrab  # Import ImageGrab from Pillow

from .npc_sysenv import NPCSH_VISION_MODEL, NPCSH_VISION_PROVIDER, NPCSH_API_URL
from .llm_funcs import get_llm_response, get_stream
import os


def _windows_snip_to_file(file_path: str) -> bool:
    """Helper function to trigger Windows snipping and save to file."""
    try:
        # Import Windows-specific modules only when needed
        import win32clipboard
        from PIL import ImageGrab
        from ctypes import windll

        # Simulate Windows + Shift + S
        windll.user32.keybd_event(0x5B, 0, 0, 0)  # WIN down
        windll.user32.keybd_event(0x10, 0, 0, 0)  # SHIFT down
        windll.user32.keybd_event(0x53, 0, 0, 0)  # S down
        windll.user32.keybd_event(0x53, 0, 0x0002, 0)  # S up
        windll.user32.keybd_event(0x10, 0, 0x0002, 0)  # SHIFT up
        windll.user32.keybd_event(0x5B, 0, 0x0002, 0)  # WIN up

        # Wait for user to complete the snip
        print("Please select an area to capture...")
        time.sleep(1)  # Give a moment for snipping tool to start

        # Keep checking clipboard for new image
        max_wait = 30  # Maximum seconds to wait
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                image = ImageGrab.grabclipboard()
                if image:
                    image.save(file_path, "PNG")
                    return True
            except Exception:
                pass
            time.sleep(0.5)

        return False

    except ImportError:
        print("Required packages not found. Please install: pip install pywin32 Pillow")
        return False


def capture_screenshot(npc: Any = None, full=False) -> Dict[str, str]:
    """
    Function Description:
        This function captures a screenshot of the current screen and saves it to a file.
    Args:
        npc: The NPC object representing the current NPC.
        full: Boolean to determine if full screen capture is needed
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

    if full:
        if system == "Darwin":
            subprocess.run(["screencapture", file_path])
        elif system == "Linux":
            if (
                subprocess.run(
                    ["which", "gnome-screenshot"], capture_output=True
                ).returncode
                == 0
            ):
                subprocess.Popen(["gnome-screenshot", "-f", file_path])
                while not os.path.exists(file_path):
                    time.sleep(0.5)
            elif (
                subprocess.run(["which", "scrot"], capture_output=True).returncode == 0
            ):
                subprocess.Popen(["scrot", file_path])
                while not os.path.exists(file_path):
                    time.sleep(0.5)
            else:
                print(
                    "No supported screenshot tool found. Please install gnome-screenshot or scrot."
                )
                return None
        elif system == "Windows":
            # For full screen on Windows, we'll use a different approach
            try:
                import win32gui
                import win32ui
                import win32con
                from PIL import Image

                # Get screen dimensions
                width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
                height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)

                # Create device context
                hdesktop = win32gui.GetDesktopWindow()
                desktop_dc = win32gui.GetWindowDC(hdesktop)
                img_dc = win32ui.CreateDCFromHandle(desktop_dc)
                mem_dc = img_dc.CreateCompatibleDC()

                # Create bitmap
                screenshot = win32ui.CreateBitmap()
                screenshot.CreateCompatibleBitmap(img_dc, width, height)
                mem_dc.SelectObject(screenshot)
                mem_dc.BitBlt((0, 0), (width, height), img_dc, (0, 0), win32con.SRCCOPY)

                # Save
                screenshot.SaveBitmapFile(mem_dc, file_path)

                # Cleanup
                mem_dc.DeleteDC()
                win32gui.DeleteObject(screenshot.GetHandle())

            except ImportError:
                print(
                    "Required packages not found. Please install: pip install pywin32"
                )
                return None
        else:
            print(f"Unsupported operating system: {system}")
            return None
    else:
        if system == "Darwin":
            subprocess.run(["screencapture", "-i", file_path])
        elif system == "Linux":
            if (
                subprocess.run(
                    ["which", "gnome-screenshot"], capture_output=True
                ).returncode
                == 0
            ):
                subprocess.Popen(["gnome-screenshot", "-a", "-f", file_path])
                while not os.path.exists(file_path):
                    time.sleep(0.5)
            elif (
                subprocess.run(["which", "scrot"], capture_output=True).returncode == 0
            ):
                subprocess.Popen(["scrot", "-s", file_path])
                while not os.path.exists(file_path):
                    time.sleep(0.5)
            else:
                print(
                    "No supported screenshot tool found. Please install gnome-screenshot or scrot."
                )
                return None
        elif system == "Windows":
            success = _windows_snip_to_file(file_path)
            if not success:
                print("Screenshot capture failed or timed out.")
                return None
        else:
            print(f"Unsupported operating system: {system}")
            return None

    # Check if screenshot was successfully saved
    if os.path.exists(file_path):
        print(f"Screenshot saved to: {file_path}")
        return {
            "filename": filename,
            "file_path": file_path,
            "model_kwargs": model_kwargs,
        }
    else:
        print("Screenshot capture failed or was cancelled.")
        return None


def analyze_image_base(
    user_prompt: str,
    file_path: str,
    filename: str,
    npc: Any = None,
    stream: bool = False,
    **model_kwargs,
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
            try:
                response = get_llm_response(
                    user_prompt, images=[image_info], npc=npc, **model_kwargs
                )
                return response
            except Exception as e:
                error_message = f"Error during LLM processing: {e}"
                print(error_message)
                return {"response": error_message}
        else:
            print("Skipping LLM processing.")
            return {"response": str(image_info)}
    else:
        print("Screenshot capture failed or was cancelled.")
        return {"response": "Screenshot capture failed or was cancelled."}


def analyze_image(
    user_prompt: str,
    file_path: str,
    filename: str,
    npc: Any = None,
    stream: bool = False,
    messages: list = None,
    model: str = NPCSH_VISION_MODEL,
    provider: str = NPCSH_VISION_PROVIDER,
    api_key: str = None,
    api_url: str = NPCSH_API_URL,
) -> Dict[str, str]:
    """
    Function Description:
        This function captures a screenshot, analyzes it using the LLM model, and returns the response.
    Args:

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
                # print("Analyzing image...")
                # print(model_kwargs)
                # print("stream", stream)
                if stream:
                    # print("going to stream")
                    return get_stream(
                        messages, images=[image_info], npc=npc, **model_kwargs
                    )

                else:
                    response = get_llm_response(
                        user_prompt,
                        images=[image_info],
                        npc=npc,
                        model=model,
                        provider=provider,
                        api_url=api_url,
                        api_key=api_key,
                    )

                    print(response)
                    # Add to command history *inside* the try block
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
