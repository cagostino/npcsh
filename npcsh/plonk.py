import json
import time

try:
    import pyautogui
except KeyError as e:
    print(f"Could not load pyautogui due to the following error: {e}")

from .image import capture_screenshot
from .llm_funcs import get_llm_response

import subprocess
import os


from typing import Any

action_space = {
    "hotkey": {"key": "string"},  # For pressing hotkeys
    "click": {
        "x": "int between 0 and 100",
        "y": "int between 0 and 100",
    },  # For clicking
    "drag": {
        "x": "int between 0 and 100",
        "y": "int between 0 and 100",
        "duration": "int",
    },  # For dragging
    "wait": {"duration": "int"},  # For waiting
    "type": {"text": "string"},
    "right_click": {"x": "int between 0 and 100", "y": "int between 0 and 100"},
    "double_click": {"x": "int between 0 and 100", "y": "int between 0 and 100"},
    "bash": {"command": "string"},
}


def perform_action(action):
    """
    Execute different types of actions using PyAutoGUI
    """
    try:
        pyautogui.PAUSE = 1  # Add a small pause between actions
        pyautogui.FAILSAFE = (
            True  # Enable fail-safe to stop script by moving mouse to corner
        )

        print(f"Action received: {action}")  # Debug print

        if action["type"] == "click":
            pyautogui.click(x=action.get("x"), y=action.get("y"))

        elif action["type"] == "double_click":
            pyautogui.doubleClick(x=action.get("x"), y=action.get("y"))

        elif action["type"] == "right_click":
            pyautogui.rightClick(x=action.get("x"), y=action.get("y"))

        elif action["type"] == "drag":
            pyautogui.dragTo(
                x=action.get("x"), y=action.get("y"), duration=action.get("duration", 1)
            )

        elif action["type"] == "type":
            text = action.get("text", "")
            if isinstance(text, dict):
                text = text.get("text", "")
            pyautogui.typewrite(text)

        elif action["type"] == "hotkey":
            keys = action.get("text", "")
            print(f"Hotkey action: {keys}")  # Debug print
            if isinstance(keys, str):
                keys = [keys]
            elif isinstance(keys, dict):
                keys = [keys.get("key", "")]
            pyautogui.hotkey(*keys)

        elif action["type"] == "wait":
            time.sleep(action.get("duration", 1))  # Wait for the given time in seconds

        elif action["type"] == "bash":
            command = action.get("command", "")
            print(f"Running bash command: {command}")  # Debug print
            subprocess.Popen(
                command, shell=True
            )  # Run the command without waiting for it to complete
            print(f"Bash Command Output: {result.stdout.decode()}")  # Debug output
            print(f"Bash Command Error: {result.stderr.decode()}")  # Debug error

        return {"status": "success"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def plonk(request, action_space, model=None, provider=None, npc=None):
    """
    Main interaction loop with LLM for action determination

    Args:
        request (str): The task to be performed
        action_space (dict): Available action types and the inputs they require
        npc (optional): NPC object for context and screenshot
        **kwargs: Additional arguments for LLM response
    """
    prompt = f"""
    Here is a request from a user:
    {request}

    Your job is to choose certain actions, take screenshots,
    and evaluate what the next step is to complete the task.

    You can choose from the following action types:
    {json.dumps(action_space)}



    Attached to the message is a screenshot of the current screen.

    Please use that information to determine the next steps.
    Your response must be a JSON with an 'actions' key containing a list of actions.
    Each action should have a 'type' and any necessary parameters.https://www.reddit.com


    For example:
        Your response should look like:

        {{
            "actions": [
                {{"type":"bash", "command":"firefox &"}},
                {{"type": "click", "x": 5, "y": 5}},
                {{'type': 'type', 'text': 'https://www.google.com'}}
                ]
                }}

    IF you have to type something, ensure that it iis first opened and selected. Do not
    begin with typing immediately.
    If you have to click, the numbers should range from 0 to 100 in x and y  with 0,0 being in the upper left.


    IF you have accomplished the task, return an empty list.
    Do not include any additional markdown formatting.
    """

    while True:
        # Capture screenshot using NPC-based method
        screenshot = capture_screenshot(npc=npc, full=True)

        # Ensure screenshot was captured successfully
        if not screenshot:
            print("Screenshot capture failed")
            return None

        # Get LLM response
        response = get_llm_response(
            prompt,
            images=[screenshot],
            model=model,
            provider=provider,
            npc=npc,
            format="json",
        )
        # print("LLM Response:", response, type(response))
        # Check if task is complete
        print(response["response"])
        if not response["response"].get("actions", []):
            return response

        # Execute actions
        for action in response["response"]["actions"]:
            print("Performing action:", action)
            action_result = perform_action(action)
            perform_action({"type": "wait", "duration": 5})

            # Optional: Add error handling or logging
            if action_result.get("status") == "error":
                print(f"Error performing action: {action_result.get('message')}")

        # Small delay between action batches
        time.sleep(1)


def test_open_reddit(npc: Any = None):
    """
    Test function to open a web browser and navigate to Reddit using plonk
    """
    # Define the action space for web navigation

    # Request to navigate to Reddit
    request = "Open a web browser and go to reddit.com"

    # Determine the browser launch hotkey based on the operating system
    import platform

    system = platform.system()

    if system == "Darwin":  # macOS
        browser_launch_keys = ["command", "space"]
        browser_search = "chrome"
    elif system == "Windows":
        browser_launch_keys = ["win", "r"]
        browser_search = "chrome"
    else:  # Linux or other
        browser_launch_keys = ["alt", "f2"]
        browser_search = "firefox"

    # Perform the task using plonk
    result = plonk(
        request,
        action_space,
        model="gpt-4o-mini",
        provider="openai",
    )

    # Optionally, you can add assertions or print results
    print("Reddit navigation test result:", result)

    return result


def generate_plonk(
    request,
):
    prompt = f"""

    A user asked the following question: {request}

    You are in charge of creating a plonk plan that will handle their request.
    This plonk plan will be a series of steps that you will write that will be
    used to generate a fully functioning system that will accomplish the user's request.
    your plonk plan should be a python script that generates LLM prompts
    that will be used to generate the distinct pieces of software.

    The goal here is modularization, abstraction, separation of scales.
    A careful set of instructions can pave the way for a system that can be iterated on
    and improved with successive steps.

    Here is an example of a question and answer that you might generate:

    Question: "Set up an automation system that will open a web browser every morning
        and go to my bank account and export my transactions."

    Answer:
    "{{'plonk plan': ```

from npcsh.llm_funcs import get_llm_response

automation_script = get_llm_response( '''
    Write a python script that will request input from a user about what bank they use. Then use selenium to open the browser and navigate to the bank's website.
    Get the user's username and password and log in, also through raw input.
    Then navigate to the transactions page and export the transactions. Ensure you are sufficiently logging information at each step of the way so that the results can be
    debugged efficiently.
    Return the script without any additional comment or Markdown formatting. It is imperative that you do not include any additional text.
''')
# write the automation script to a file
automation_script_file = open('automation_script.py', 'w')
automation_script_file.write(automation_script)
automation_script_file.close()


scheduling_script = get_llm_response( f'''
    Write a bash script that will set up an OS scheduler to run the automation script every morning at 8 am.
    The automation script is located at ./automation_script.py.
    You'll need to ensure that the full path is used in the scheduling script.
    Return the script without any additional comment or Markdown formatting.
    It is imperative that you do not include any additional text.
    Do not leave any placeholder paths or variables in the script.
    They must be able to execute without
    any further modification by you or the user.
    ''')
# write the scheduling script to a file
scheduling_script_file = open('scheduling_script.sh', 'w')
scheduling_script_file.write(scheduling_script)

scheduling_script_file.close()
# attempt to run the scheduling script
import subprocess
subprocess.run(['bash', 'scheduling_script.sh'])
```}}

    In this example, we have set up a plan that will require multiple other LLM calls to generate the necessary items to
    accomplish the user's request.

    """

    return get_llm_response(prompt)


# Optional: If you want to run this as a standalone script
if __name__ == "__main__":
    test_open_reddit()
