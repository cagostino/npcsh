tool_name: pref_analyzer
description: |
  A tool to analyze feedback and determine user preferences.
inputs:
  - feedback
preprocess:
  - engine: python
    code: |
      # Code to analyze feedback and determine user preferences
      def analyze_feedback(feedback):
          # Placeholder for analysis logic, can be complex processing
          return f"Analyzing the following feedback: {feedback}"

      context['analysis'] = analyze_feedback(inputs['feedback'])

prompt:
  engine: natural
  code: |
    Based on the following feedback: {{ inputs['feedback'] }},
    what can we conclude about user preferences? {{ analysis }}

postprocess:
  - engine: natural
    code: ""