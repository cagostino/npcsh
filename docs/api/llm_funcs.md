# LLM Functions

::: npcsh.llm_funcs
    options:
      show_source: true
      members:
        - "!^_"          # Excludes all private members
        - "!^[A-Z]"      # Excludes ALL_CAPS constants
        - "!^[A-Z].*$"   # Additional pattern for module-level attributes
      filters:
        - "!^_"          # Double protection against private members
        - "!^[A-Z]"      # Double protection against constants
      inherited_members: false
      show_if_no_docstring: true
      show_root_heading: false