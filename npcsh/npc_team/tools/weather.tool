tool_name: "weather_tool"
description: "Summarize the weather forecast for a location in November 2024"
inputs:
  - "location"
preprocess:
  - engine: "python"
    code: |
      from npcsh.search import search_web
      formatted_location = inputs['location'].strip().title()
      search_query = f"temperature in {formatted_location} in November 2024"
      # Perform the web search
      results = search_web(search_query, num_results=1)
      if results:
          # Extract text from the search result
          web_content = results[0]['body']
      else:
          web_content = "Weather information not found."
prompt:
  engine: "natural"
  code: |
    Using the following information extracted from the web:

    {{ web_content }}

    Summarize the weather forecast for {{ formatted_location }} in November 2024.
postprocess:
  - engine: "natural"
    code: "The weather in {{ formatted_location }} in November 2024 is: {{ llm_response }}"