tool_name: "example_tool"
description: "An example tool for demonstration purposes"
inputs:
  - "input_param"
preprocess:
  - engine: "python"
    code: "processed_input = inputs['input_param'].strip().lower()"
prompt:
  engine: "natural"
  code: "Please process the following input: {{ processed_input }}"
postprocess:
  - engine: "python"
    code: "final_output = llm_response.upper()"