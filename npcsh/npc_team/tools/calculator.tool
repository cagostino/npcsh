tool_name: "calculator"
description: "A tool to simplify and evaluate mathematical expressions"
inputs:
  - "expression"
preprocess:
  - engine: python
    code: |
      output = eval(inputs['expression'])


prompt:
  engine: "natural"
  code: ""

postprocess:
  - engine: "natural"
    code: ""