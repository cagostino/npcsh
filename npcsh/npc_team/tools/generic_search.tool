tool_name: "internet_search"
description: Searches the web for information based on a query in order to verify timiely details (e.g. current events) or to corroborate information in uncertain situations.
inputs:
  - query
steps:
  - engine: "python"
    code: |
      from npcsh.search import search_web
      query = "{{ query }}"
      print('QUERY in tool', query)
      results = search_web(query, num_results=5)
      print('RESULTS in tool', results)
  - engine: "natural"
    code: |
      Using the following information extracted from the web:

      {{ results }}

      Answer the users question: {{ query }}
