tool_name: stats_calculator
inputs:
  - table_name   
  - operation    
  - filters      
preprocess:
  - engine: python
    code: |

      import pandas as pd
      import operator  # Add this line

      df = context['dataframes'][inputs['table_name']]
            
      def fix_inputs(error_msg, current_inputs, df):
        #render_markdown("FIXING INPUTS")
        #render_markdown(f"Error was: {error_msg}")
        #render_markdown(f"Current inputs: {current_inputs}")
        #render_markdown(f"DataFrame:")
        #render_markdown(df.head().to_string())

        prompt = f"""
        Error: {error_msg}

        Current inputs: {current_inputs}

        DataFrame info:
        Columns: {df.columns.tolist()}

        Sample data:
        {df.head().to_string()}

        Please fix the inputs to work with this data structure so that the code that follows will be able to execute:

        ```
        # Filtering logic
        ops = {{
            '==': 'equal to',
            '!=': 'not equal to',
            '>': 'greater than',
            '<': 'less than',
            '>=': 'greater than or equal to',
            '<=': 'less than or equal to',
            'between': 'between two values',
            'in': 'values in a list',
        }}

        # Example of filters:
        'filters': {{
            'Date': {{
                'operator': 'between',
                'start': '2023-01-01',
                'end': '2023-01-31'
            }},
            'Units_Sold': {{
                'operator': '>=',
                'value': 50
            }}
        }}
        ```

        You may need to adjust the filters or operation in order to accomplish the user's command. This includes changing which filters are used.

        Remember that the column names are case-sensitive.

        Return **only** valid JSON with double quotes for both keys and string values.

        The JSON should include the keys: "table_name", "operation", "operation_column", and "filters".

        Do not include any extra information or markdown formatting.
        """

        response = get_llm_response(prompt, format="json")
        #render_markdown('response : '+str(response))

        if 'error' in response:
            print(f"Error in LLM response: {response['error']}")
            raise ValueError("LLM failed to return valid JSON.")

        fixed = response['response']
        return fixed

      max_retries = 3
      attempt = 0


      while attempt < max_retries:
          try:
              # render_markdown(f"\nATTEMPT {attempt + 1}")
              # render_markdown(f"Current inputs: {inputs}")

              # render_markdown("\nDataFrame:")
              # render_markdown(df.head().to_string())

              # Define operator mappings
              ops = {
                  '==': operator.eq,
                  '!=': operator.ne,
                  '>': operator.gt,
                  '<': operator.lt,
                  '>=': operator.ge,
                  '<=': operator.le,
                  'between': lambda col, start, end: col.between(start, end),
                  'in': lambda col, values: col.isin(values),
              }

              filtered_df = df.copy()
              for col, condition in inputs['filters'].items():
                  print(f"\nApplying filter on column: {col}")
                  op = condition.get('operator')
                  if op not in ops:
                      raise ValueError(f"Unsupported operator: {op}")

                  if op == 'between':
                      start = condition.get('start')
                      end = condition.get('end')
                      if start is None or end is None:
                          raise ValueError("Between operator requires 'start' and 'end' values.")
                      # Ensure date columns are datetime
                      if filtered_df[col].dtype == 'object':
                          filtered_df[col] = pd.to_datetime(filtered_df[col])
                      mask = ops[op](filtered_df[col], start, end)
                      filtered_df = filtered_df[mask]
                  elif op == 'in':
                      values = condition.get('values')
                      if values is None:
                          raise ValueError(f"'in' operator requires a 'values' list.")
                      mask = ops[op](filtered_df[col], values)
                      filtered_df = filtered_df[mask]
                  else:
                      value = condition.get('value')
                      if value is None:
                          raise ValueError(f"Operator '{op}' requires 'value'.")
                      # Corrected line here:
                      mask = ops[op](filtered_df[col], value)
                      filtered_df = filtered_df[mask]
                  #render_markdown("After filter:")
                  #render_markdown(filtered_df.head().to_string())

              # Specify the operation column
              operation_column = inputs.get('operation_column')  
              result = getattr(filtered_df[operation_column], inputs['operation'])()
              #render_markdown(f"\nResult: {result}")
              context['stats'] = str(result)
              break

          except Exception as e:
              print(f"\nERROR CAUGHT: {str(e)}")
              attempt += 1
              if attempt >= max_retries:
                  print("Maximum retries reached. Exiting.")
                  raise  # Re-raise the exception after max retries
              inputs = fix_inputs(str(e), inputs, df)

prompt:
  engine: natural
  code: |

    A user made this request:
    {{command}}

    Based on the calculation results:
    {{stats}}

    Please write a response to the user. Make it succinct and clear.

postprocess:
  - engine: natural
    code: ""
      