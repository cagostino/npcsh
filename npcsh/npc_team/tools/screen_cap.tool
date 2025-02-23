tool_name: "screen_capture_analysis_tool"
description: Captures the whole screen and sends the image for analysis
inputs:
  - "prompt"
steps:
  - engine: "python"
    code: |
      # Capture the screen
      import pyautogui
      import datetime
      import os
      from PIL import Image
      import time
      from npcsh.image import analyze_image_base, capture_screenshot

      out = capture_screenshot(npc = npc, full = True)

      llm_response = analyze_image_base( '{{prompt}}' + "\n\nAttached is a screenshot of my screen currently. Please use this to evaluate the situation. If the user asked for you to explain what's on their screen or something similar, they are referring to the details contained within the attached image. You do not need to actually view their screen. You do not need to mention that you cannot view or interpret images directly. You only need to answer the user's request based on the attached screenshot!",
                                        out['file_path'],
                                        out['filename'],
                                        npc=npc,
                                        **out['model_kwargs'])
      # To this:
      if isinstance(llm_response, dict):
          llm_response = llm_response.get('response', 'No response from image analysis')
      else:
          llm_response = 'No response from image analysis'