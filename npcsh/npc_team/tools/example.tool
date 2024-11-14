tool_name: "example_tool"
inputs:
  - "input_param"
preprocess:
  - engine: "python"
    code: "processed_input = inputs['input_param'].strip().lower()"
prompt:
  engine: "plain_english"
  code: "Please process the following input: {{ processed_input }}"
postprocess:
  - engine: "python"
    code: "final_output = llm_response.upper()"