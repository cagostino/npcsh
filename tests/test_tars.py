import os
import base64
import requests
import json
import torch
import pyautogui  # For taking screenshots
import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from screeninfo import get_monitors
from npcsh.llm_funcs import get_openai_response


def get_screen_resolution():
    monitor = get_monitors()[0]  # Get primary monitor
    return monitor.width, monitor.height


from PIL import Image


def capture_screenshot() -> str:
    """Captures a screenshot and saves it to a specified path."""
    screenshot_path = "screenshot.png"
    screenshot = pyautogui.screenshot()

    # Resize screenshot to fit model's pixel range
    desired_width = 1280  # Adjust as needed based on max_pixels range
    desired_height = int(
        (desired_width * screenshot.height) / screenshot.width
    )  # Maintain aspect ratio
    screenshot = screenshot.resize((desired_width, desired_height))

    screenshot.save(screenshot_path)
    return screenshot_path


# Adjust processor for specific pixel range
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28


def encode_image_to_base64(image_path: str) -> str:
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded_string}"


def get_tars_response(command: str, model_name: str) -> str:
    """Generates a response from the UI-TARS model based on the command and screenshot image."""
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #    model_name, torch_dtype="auto", device_map="auto"
    # )
    # processor = AutoProcessor.from_pretrained(model_name)

    # capture the current screen
    im = capture_screenshot()
    image_data = encode_image_to_base64(im)
    prompt = (
        f"""You are a GUI agent. You are given a task and your action history,
    with screenshots.     You need to perform the next action or set of actions to complete the task.
    here is the task you must complete: {command}
        """
        + r"""
    click(start_box='<|box_start|>(x1,y1)<|box_end|>')
    left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
    right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
    drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
    hotkey(key='')
    type(content='') #If you want to submit your input, use "\
    " at the end of `content`.
    scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
    wait() #Sleep for 5s and take a screenshot to check for any changes.
    finished()
    call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.

    your response should be a list of actions to perform in the order they should be performed.
    Provide a single json object with the following format:
    { "actions":  ['action1', 'action2', 'action3']    }
    Do not provide any additional text or markdown formatting.
    """
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data},
                },
                {"type": "text", "text": command},
            ],
        }
    ]

    # tars:
    """text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text, padding=True, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return output_text[0]
    """
    gpt4o_response = get_openai_response(
        prompt, model="gpt-4o-mini", messages=messages, format="json"
    )

    return gpt4o_response


def execute_actions(actions: list):
    """Executes the actions received from the model using pyautogui."""
    for action in actions:
        if action.startswith("click"):
            x, y = map(int, action[action.find("(") + 1 : action.find(")")].split(","))
            pyautogui.click(x, y)
        elif action.startswith("left_double"):
            x, y = map(int, action[action.find("(") + 1 : action.find(")")].split(","))
            pyautogui.doubleClick(x, y)
        elif action.startswith("right_single"):
            x, y = map(int, action[action.find("(") + 1 : action.find(")")].split(","))
            pyautogui.rightClick(x, y)
        elif action.startswith("drag"):
            coords = list(
                map(
                    int,
                    action[action.find("(") + 1 : action.find(")")]
                    .replace("(", "")
                    .replace(")", "")
                    .split(","),
                )
            )
            pyautogui.moveTo(coords[0], coords[1])
            pyautogui.dragTo(coords[2], coords[3], duration=0.5)
        elif action.startswith("type"):
            text = action.split("('")[1].split("')")[0]
            pyautogui.write(text, interval=0.05)
        elif action.startswith("hotkey"):
            key = action.split("('")[1].split("')")[0]
            pyautogui.hotkey(key)
        elif action.startswith("scroll"):
            direction = action.split("('")[1].split("')")[0]
            amount = -100 if direction == "down" else 100
            pyautogui.scroll(amount)
        elif action.startswith("wait"):
            time.sleep(5)
        elif action.startswith("finished"):
            print("Task completed.")


def ui_tars_control_loop(model_name: str):
    """Main loop for interacting with the user and executing commands via UI-TARS."""
    print("UI-TARS Control Loop Started.")
    screen_width, screen_height = get_screen_resolution()
    print(f"Screen resolution: {screen_width}x{screen_height}")

    while True:
        command = input("Enter your command (or type 'exit' to quit): ")
        if command.lower() == "exit":
            print("Exiting UI-TARS Control Loop.")
            break

        screenshot_path = capture_screenshot()
        tars_result = get_tars_response(command, screenshot_path, model_name)
        print(f"UI-TARS Response: {tars_result}")

        try:
            actions = json.loads(tars_result).get("actions", [])
            execute_actions(actions)
        except json.JSONDecodeError:
            print("Error parsing actions from UI-TARS response.")


if __name__ == "__main__":
    MODEL_NAME = "ui-tars-7B"  # Replace with your actual UI-TARS model name
    ui_tars_control_loop(MODEL_NAME)
