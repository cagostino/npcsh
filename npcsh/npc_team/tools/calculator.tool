tool_name: "calculator"
inputs:
  - "expression"
preprocess:
  - engine: "plain_english"
    code: "Simplify the following mathematical expression: {{ inputs['expression'] }}"
  - engine: "python"
    code: "simplified_expression = llm_response.strip()"
prompt:
  engine: "python"
  code: "result = eval(simplified_expression)"
postprocess:
  - engine: "plain_english"
    code: "The result of {{ inputs['expression'] }} is {{ result }}."