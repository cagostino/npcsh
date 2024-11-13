tool_name: "screen_capture_analysis_tool"
inputs:
  - "prompt"
preprocess:
  - engine: "python"
    code: |
      # Capture the screen
      import pyautogui
      import datetime
      import os
      from PIL import Image
      from npcsh.helpers import analyze_image_base

      # Generate filename
      filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
      screenshot = pyautogui.screenshot()
      screenshot.save(filename)
      print(f"Screenshot saved as {filename}")

      # Load image
      image = Image.open(filename)
      
      # Full file path
      file_path = os.path.abspath('./'+filename)
      # Analyze the image

      llm_output = analyze_image_base(inputs['prompt']+ '\n\n attached is a screenshot of my screen currently.', file_path, filename)
prompt:
  engine: "plain_english"
  code: ""
postprocess:
  - engine: "plain_english"
    code: |
      Screenshot captured and saved as {{ filename }}.
      Analysis Result: {{ llm_output }}