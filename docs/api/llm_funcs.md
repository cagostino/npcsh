# LLM Functions

::: npcsh.llm_funcs
    options:
      show_source: true
      members: true
      filters:
        - "!^_"          # Hide private members
        - "!^[A-Z]{2,}"  # Hide constants (all-caps)
        - "!test_"       # Hide test functions
      inherited_members: false