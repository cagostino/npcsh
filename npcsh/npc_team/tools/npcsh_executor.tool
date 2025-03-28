tool_name: npcsh_executor
description: Execute npcsh commands. Use the macro commands.
inputs:
  - code
  - language
steps:
  - engine: "{{language}}"
    code: |
      {{code}}
