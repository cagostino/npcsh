tool_name: "image_generation_tool"
inputs:
  - "prompt"
preprocess:
  - engine: "python"
    code: |
      # Clean and prepare the prompt
      image_prompt = inputs['prompt'].strip()
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
prompt:
  engine: "plain_english"
  code: ""
postprocess:
  - engine: "plain_english"
    code: |
      {% if image_generated %}
      The image has been saved as {{ filename }}.
      ![Generated Image]({{ filename }})
      {% else %}
      Failed to generate image.
      {% endif %}