# npcsh/npc_team/tools/pandas_executor.tool

tool_name: pandas_executor
inputs:
  - code
preprocess:
  - engine: python
    code: |
      exec_globals = {'pd': pd, 'np': np, 'plt': plt, **context.get('dataframes', {})}
      exec(inputs['code'], exec_globals)
      context['dataframes'] = {k: v for k, v in exec_globals.items() if isinstance(v, pd.DataFrame)}
prompt:
  engine: plain_english
  code: |
    Executed the provided pandas code successfully.
postprocess:
  - engine: plain_english
    code: |
      {{ llm_response }}