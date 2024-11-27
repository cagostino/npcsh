tool_name: "generic_search_tool"
inputs:
  - "query"
preprocess:
  - engine: "python"
    code: |
      from npcsh.helpers import search_web
      query = inputs['query'].strip().title()
      # Perform the web search
      results = search_web(query, num_results=5)
prompt:
  engine: "plain_english"
  code: |
    Using the following information extracted from the web:

    {{ results }}

    Answer the users question: {{ inputs['query'] }}
