tool_name: "calculator"
inputs:
  - "expression"
preprocess:
  - engine: "natural"
    code: "Simplify the following mathematical expression: {{ inputs['expression'] }}"
  - engine: "python"
    code: "simplified_expression = llm_response.strip()"
prompt:
  engine: "python"
  code: "result = eval(simplified_expression)"
postprocess:
  - engine: "natural"
    code: "The result of {{ inputs['expression'] }} is {{ result }}."