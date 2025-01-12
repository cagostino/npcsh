# npcsh/npc_team/tools/plotting_tool.tool
description: Generate and display a plot based on the provided code.
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
  engine: natural
  code: |
    Generated plot saved to {{ plot_path }}.
postprocess:
  - engine: natural
    code: |
      {{ llm_response }}
      ![Plot]({{ plot_path }})