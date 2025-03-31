# LLM Functions

::: npcsh.llm_funcs
    options:
      show_source: true
      members:
        - "!^_"          # Exclude private members
        - "!^[A-Z_]+$"   # Exclude ALL_CAPS constants but keep functions
      filters:
        - "!^__"         # Exclude dunders
      inherited_members: false
      show_root_heading: false
      show_if_no_docstring: true