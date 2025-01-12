tool_name: pandas_executor
description: Execute pandas code and display the result
inputs:
  - code
preprocess:
  - engine: python
    code: |
      exec_globals = {'pd': pd, 'np': np, 'plt': plt, **context.get('dataframes', {})}
      exec(inputs['code'], exec_globals)
      context['dataframes'] = {k: v for k, v in exec_globals.items() if isinstance(v, pd.DataFrame)}
prompt:
  engine: natural
  code: |
    Executed the provided pandas code successfully.
postprocess:
  - engine: natural
    code: |
      {{ llm_response }}