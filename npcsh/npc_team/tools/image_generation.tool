tool_name: "image_generation_tool"
description: |
  Generates images based on a text prompt.
inputs:
  - prompt
  - model: 'runwayml/stable-diffusion-v1-5'
  - provider: 'diffusers'

steps:
  - engine: "python"
    code: |
      image_prompt = '{{prompt}}'.strip()

      # Generate the image
      filename = generate_image(
          image_prompt,
          npc=npc,
          model='{{model}}',  # You can adjust the model as needed
          provider='{{provider}}'
      )
      if filename:
          image_generated = True
      else:
          image_generated = False

