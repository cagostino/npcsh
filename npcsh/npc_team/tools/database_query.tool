tool_name: "database_query"
inputs:
  - "query_params"
preprocess:
  - engine: "plain_english"
    code: |
      Based on the following parameters, generate an SQL query:
      Parameters: {{ inputs['query_params'] }}
  - engine: "python"
    code: "sql_query = llm_response.strip()"
prompt:
  engine: "sql"
  code: "{{ sql_query }}"
postprocess:
  - engine: "plain_english"
    code: "Here are the results:\n{{ llm_response }}"