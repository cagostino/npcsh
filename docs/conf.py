import os
import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(".."))

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

# Show full module paths
add_module_names = True

# Autosummary settings
autosummary_generate = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "imported-members": True,
}

# General settings
templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "alabaster"
html_static_path = ["_static"]

# This is important - it ensures all modules are importable
import npcsh
import pkgutil

all_modules = [name for _, name, _ in pkgutil.iter_modules(npcsh.__path__, "npcsh.")]
