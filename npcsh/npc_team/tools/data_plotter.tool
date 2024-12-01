tool_name: data_plotter
inputs:
  - table_name
  - plot_type
  - columns
preprocess:
  - engine: python
    code: |
      import matplotlib.pyplot as plt
      df = context['dataframes'][inputs['table_name']]
      if inputs['plot_type'] == 'hist':
          df[inputs['columns']].hist()
      elif inputs['plot_type'] == 'line':
          df[inputs['columns']].plot()
      elif inputs['plot_type'] == 'scatter':
          plt.scatter(df[inputs['columns'][0]], df[inputs['columns'][1]])
      plt.savefig('plot.png')
prompt:
  engine: natural
  code: |
    Generated {{ inputs['plot_type'] }} plot for columns {{ inputs['columns'] }} from table '{{ inputs['table_name'] }}'. The plot is saved as 'plot.png'.
postprocess:
  - engine: natural
    code: |
      {{ llm_response }}
      ![Plot](plot.png)