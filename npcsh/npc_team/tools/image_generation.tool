tool_name: "image_generation_tool"
description: |
  Generates images based on a text prompt.
inputs:
  - "prompt"
steps:
  - engine: "python"
    code: |
      # Clean and prepare the prompt
      image_prompt = '{{prompt}}'.strip()
      #print(f"Prompt: {image_prompt}")

      # Generate the image
      filename = generate_image(
          image_prompt,
          npc=npc,
          model='dall-e-2',  # You can adjust the model as needed
          provider='openai'
      )
      if filename:
          image_generated = True
      else:
          image_generated = False
  - engine: "natural"
    code: |
      {% if image_generated %}
      The image has been saved as {{ filename }}.
      ![Generated Image]({{ filename }})
      {% else %}
      Failed to generate image.
      {% endif %}