# data_loader.tool

tool_name: data_loader
inputs:
  - file_path
  - table_name
preprocess:
  - engine: python
    code: |
      import pandas as pd
      # Read data from CSV
      df = pd.read_csv(inputs['file_path'])
      # Write data to the database
      df.to_sql(inputs['table_name'], npc.db_conn, if_exists='replace', index=False)
      output = f"Data from '{inputs['file_path']}' loaded into database table '{inputs['table_name']}'."
prompt:
  engine: plain_english
  code: ""
postprocess:
  - engine: plain_english
    code: ""