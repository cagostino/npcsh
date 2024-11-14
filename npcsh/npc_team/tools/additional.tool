tool_name: additional_tool
inputs:
  - data
preprocess:
  - "Received data: {{ inputs['data'] }}"
prompt: "Enhance the following data: {{ inputs['data'] }}"
postprocess:
  - "Enhanced data: {{ llm_response }}"