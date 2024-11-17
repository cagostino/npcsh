# database_query.tool

tool_name: database_query
inputs:
  - query_text
preprocess:
  - engine: python
    code: |
      import pandas as pd
      import sqlite3
      import re
      from tabulate import tabulate
      from npcsh.llm_funcs import render_markdown

      query_text = inputs['query_text'].lower()
      
      output = None
      
      # Initialize dataframes dict in context if it doesn't exist
      if 'dataframes' not in context:
          context['dataframes'] = {}
      
      # Handle "load table" requests
      if 'load' in query_text and ('table' in query_text or 'data' in query_text):
          # Extract table name - assume it's the last word
          table_name = query_text.split()[-1]
          try:
              # Try to read the table directly
              query = f"SELECT * FROM {table_name}"
              print('Trying to load table through SQL \n : query = ', query)
              df = pd.read_sql_query(query, npc.db_conn)
              print(f'{table_name} loaded')
              print(df)
              
              # Store in context
              context['dataframes'][table_name] = df
              
              output = df
          except Exception as e:
              # If fails, try to find CSV with similar name
              try:
                  print('Trying to load table through CSV')
                  df = pd.read_csv(f"{table_name}.csv")
                  print(f'{table_name} loaded')
                  
                  # Store in context
                  context['dataframes'][table_name] = df
                  
                  output = df
              except Exception as e2:
                  output = f"Failed to load table or file: {str(e2)}"
      else:
          # Handle regular SQL queries
          try:
              df = pd.read_sql_query(inputs['query_text'], npc.db_conn)
              output = df
          except Exception as e:
              output = f"Query execution failed: {str(e)}"

      print("Available dataframes:", list(context['dataframes'].keys()))


prompt:
  engine: plain_english
  code: ""
postprocess:
  - engine: python
    code: context['output'] = f'{{ output}} '