tool_name: sql_executor
description: Execute SQL queries on the ~/npcsh_history.db and display the result. The database contains only information about conversations and other user-provided data. It does not store any information about individual files.
inputs:
  - sql_query
  - interpret: false  # Note that this is not a boolean, but a string

steps:
  - engine: python
    code: |
      import pandas as pd
      df = pd.read_sql_query('{{sql_query}}', npc.db_conn)

      output = df.to_string()

  - engine: natural
    code: |
      {% if summarize %}
      Here is the result of the SQL query:
      ```
      {{ df.to_string() }}  # Convert DataFrame to string for a nicer display
      ```
      {% endif %}