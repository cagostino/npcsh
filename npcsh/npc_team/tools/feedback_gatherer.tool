tool_name: feedback_gatherer
description: |
  A tool to gather feedback for a given user ID and summarize it.
inputs:
  - id
preprocess:
  - engine: python
    code: |
      # Code to process and gather feedback for a given user ID
      def gather_feedback(user_id):
          # Placeholder for the actual feedback gathering logic
          return f"Gathering feedback for user ID: {user_id}"

      context['feedback'] = gather_feedback(inputs['id'])

prompt:
  engine: natural
  code: |
    Please summarize the feedback gathered from user ID {{ inputs['id'] }}: {{ feedback }}

postprocess:
  - engine: natural
    code: ""