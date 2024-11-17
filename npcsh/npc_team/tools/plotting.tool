# npcsh/npc_team/tools/plotting_tool.tool

tool_name: plotting_tool
inputs:
  - code
preprocess:
  - engine: python
    code: |
      exec_globals = {'pd': pd, 'np': np, 'plt': plt, **context.get('dataframes', {})}
      exec(inputs['code'], exec_globals)
      plt.savefig('plot.png')
      context['plot_path'] = 'plot.png'
prompt:
  engine: plain_english
  code: |
    Generated plot saved to {{ plot_path }}.
postprocess:
  - engine: plain_english
    code: |
      {{ llm_response }}
      ![Plot]({{ plot_path }})