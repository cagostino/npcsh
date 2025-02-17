tool_name: sql_executor
description: Execute SQL queries and display the result
inputs:
  - sql_query
  - interpret: "false"  # Note that this is not a boolean, but a string

steps:
  - engine: python
    code: |
      import pandas as pd
      df = pd.read_sql_query(inputs['sql_query'], npc.db_conn)
      output = df.to_string()

  - engine: natural
    code: |
      {% if inputs['interpret'] == "true" %}
      Here is the result of the SQL query:
      ```
      {{ df.to_string() }}  # Convert DataFrame to string for a nicer display
      ```
      {% endif %}