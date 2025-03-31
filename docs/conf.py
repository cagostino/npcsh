import os
import sys
from pathlib import Path

# Add both project root and package directory to path
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../npcsh"))

project = "npcsh"
copyright = "2025, Christopher Agostino"
author = "Christopher Agostino"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

# Mock entire problematic packages, not just classes
autodoc_mock_imports = [
    "openai",
    "anthropic",
    "google",
    "ollama",
    "pyautogui",
    "sentence_transformers",
    "gtts",
    # Add any other problematic packages
]

# Simplify autodoc configuration
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}

# Disable the ExternalDocumenter - the mocking handles this now
# Keep your other settings...

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "imported-members": False,
    "ignore-module-all": True,  # Ignore __all__ definitions
    "member-order": "groupwise",  # Group by type instead of alphabetically
}

# Show full module paths
add_module_names = True

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False  # Don't include imported members in summaries

# General settings
templates_path = ["_templates"]
exclude_patterns = ["_build", "**/.ipynb_checkpoints", "**/__pycache__"]
html_theme = "alabaster"
html_static_path = ["_static"]

# Module discovery
import npcsh
import pkgutil

all_modules = [name for _, name, _ in pkgutil.iter_modules(npcsh.__path__, "npcsh.")]
