tool_name: deck_creator
description: |
  A tool to create a presentation deck based on the analysis provided.
inputs:
  - analysis
preprocess:
  - engine: python
    code: |
      # Code to create a presentation deck from the analysis
      def create_deck(analysis):
          # Placeholder for creating a deck
          return f"Creating a presentation deck based on the following analysis: {analysis}"

      context['presentation_deck'] = create_deck(inputs['analysis'])

prompt:
  engine: natural
  code: |
    Based on the analysis provided:
    {{ analysis }},
    please create a presentation deck outlining the key points and conclusions.

postprocess:
  - engine: natural
    code: ""