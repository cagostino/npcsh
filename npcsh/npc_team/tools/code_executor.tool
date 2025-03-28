tool_name: code_executor
description: Execute scripts with a specified language. choose from python, bash, R, or javascript. Set the ultimate result as the "output" variable. It must be a string. Do not add unnecessary print statements.
inputs:
  - code
  - language
steps:
  - engine: '{{ language }}'
    code: |
      {{code}}
  - engine: natural
    code: |
      Here is the result of the code execution that an agent ran.
      ```
      {{ output }}
      ```
      please provide a response accordingly.