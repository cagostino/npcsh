tool_name: bash_executor
description: Execute bash queries.
inputs:
  - bash_command
  - user_request
steps:
  - engine: python
    code: |
      import subprocess
      import os
      cmd = '{{bash_command}}'  # Properly quote the command input
      def run_command(cmd):
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if stderr:
          print(f"Error: {stderr.decode('utf-8')}")
          return stderr
        return stdout
      result = run_command(cmd)
      output = result.decode('utf-8')

  - engine: natural
    code: |

      Here is the result of the bash command:
      ```
      {{ output }}
      ```
      This was the original user request: {{ user_request }}

      Please provide a response accordingly.

