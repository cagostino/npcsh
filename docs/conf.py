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

# Add these near the top
import sphinx.ext.autodoc

# Prevent Sphinx from trying to resolve/import certain things
nitpicky = False  # Don't warn about missing references
autodoc_typehints = "none"  # Don't try to document type hints
autodoc_mock_imports = ["openai", "anthropic", "google.generativeai", "ollama"]


class ExternalDocumenter(sphinx.ext.autodoc.ClassDocumenter):
    """Don't try to build documentation for external classes"""

    priority = 10 + sphinx.ext.autodoc.ClassDocumenter.priority  # Higher priority

    def import_object(self, raiseerror=False):
        """Called by Sphinx. Return True if you want to process this object."""
        if self.get_attr(self.object, "__module__", None) in [
            "openai",
            "anthropic",
            "google.generativeai",
            "ollama",
            "typing",  # Also ignore typing stuff
            "types",
        ]:
            return False
        return super().import_object(raiseerror)


def setup(app):
    app.add_autodocumenter(ExternalDocumenter)


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
