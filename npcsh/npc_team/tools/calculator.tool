tool_name: "calculator"
description: "A tool to simplify and evaluate mathematical expressions"
inputs:
  - expression
steps:
  - engine: python
    code: |
      output = eval('{{ expression }}')
