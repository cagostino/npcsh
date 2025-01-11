tool_name: sql_executor
description: Execute SQL queries and display the result
inputs:
  - sql_query
preprocess:
  - engine: python
    code: |
      import pandas as pd
      df = pd.read_sql_query(inputs['sql_query'], npc.db_conn)
      context['df'] = df
prompt:
  engine: natural
  code: |
    The result of your SQL query is:
    ```
    {{ df }}
    ```
postprocess:
  - engine: natural
    code: |
      {{ llm_response }}