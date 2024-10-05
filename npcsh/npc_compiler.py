import os
import subprocess
import sqlite3
import yaml
from jinja2 import Environment, FileSystemLoader, Template

import os
import yaml
from jinja2 import Environment, FileSystemLoader, TemplateError, Template, Undefined

import pandas as pd
from .llm_funcs import get_llm_response, process_data_output, get_data_response


class SilentUndefined(Undefined):
    def _fail_with_undefined_error(self, *args, **kwargs):
        return ""


class NPCCompiler:
    def __init__(self, npc_directory):
        self.npc_directory = npc_directory
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.npc_directory), undefined=SilentUndefined
        )
        self.npc_cache = {}
        self.resolved_npcs = {}

    def compile(self, npc_file: str):
        if not npc_file.endswith(".npc"):
            raise ValueError("File must have .npc extension")

        try:
            # First pass: parse all NPC files without resolving Jinja templates
            self.parse_all_npcs()

            # Second pass: resolve Jinja templates and merge inherited properties
            self.resolve_all_npcs()

            # Final pass: resolve any remaining references
            parsed_content = self.finalize_npc_profile(npc_file)
            return parsed_content
        except Exception as e:
            raise

    def parse_all_npcs(self):
        for filename in os.listdir(self.npc_directory):
            if filename.endswith(".npc"):
                self.parse_npc_file(filename)

    def parse_npc_file(self, npc_file: str):
        if npc_file in self.npc_cache:
            return self.npc_cache[npc_file]

        try:
            with open(os.path.join(self.npc_directory, npc_file), "r") as f:
                npc_content = f.read()

            # Parse YAML without resolving Jinja templates
            profile = yaml.safe_load(npc_content)
            self.npc_cache[npc_file] = profile

            return profile
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in NPC profile {npc_file}: {str(e)}")

    def resolve_all_npcs(self):
        for npc_file in self.npc_cache:
            self.resolve_npc_profile(npc_file)

    def resolve_npc_profile(self, npc_file: str):
        if npc_file in self.resolved_npcs:
            return self.resolved_npcs[npc_file]

        profile = self.npc_cache[npc_file].copy()

        # Resolve Jinja templates
        for key, value in profile.items():
            if isinstance(value, str):
                template = self.jinja_env.from_string(value)
                profile[key] = template.render(self.npc_cache)

        # Handle inheritance
        if "inherits_from" in profile:
            parent_profile = self.resolve_npc_profile(profile["inherits_from"] + ".npc")
            profile = self.merge_profiles(parent_profile, profile)

        self.resolved_npcs[npc_file] = profile
        return profile

    def finalize_npc_profile(self, npc_file: str):
        profile = self.resolved_npcs[npc_file].copy()

        # Resolve any remaining references
        for key, value in profile.items():
            if isinstance(value, str):
                template = self.jinja_env.from_string(value)
                profile[key] = template.render(self.resolved_npcs)

        required_keys = [
            "name",
            "primary_directive",
            "suggested_tools_to_use",
            "restrictions",
            "model",
        ]
        for key in required_keys:
            if key not in profile:
                raise ValueError(f"Missing required key in NPC profile: {key}")

        return profile

    def merge_profiles(self, parent, child):
        merged = parent.copy()
        for key, value in child.items():
            if isinstance(value, list) and key in merged:
                merged[key] = merged[key] + value
            elif isinstance(value, dict) and key in merged:
                merged[key] = self.merge_profiles(merged[key], value)
            else:
                merged[key] = value
        return merged


# Usage:
# compiler = NPCCompiler('/path/to/npc/directory')
# compiled_script = compiler.compile('your_npc_file.npc')


class NPC:
    def __init__(
        self,
        name,
        primary_directive,
        suggested_tools_to_use,
        restrictions,
        model,
        provider,
        db_conn,
    ):
        self.name = name
        self.primary_directive = primary_directive
        self.suggested_tools_to_use = suggested_tools_to_use
        self.restrictions = restrictions
        self.model = model
        self.db_conn = db_conn
        self.tables = self.db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
        self.provider = provider

    def __str__(self):
        return f"NPC: {self.name}\nDirective: {self.primary_directive}\nModel: {self.model}"

    def get_data_response(self, request):
        return get_data_response(request, self.db_conn, self.tables)

    def get_llm_response(self, request, **kwargs):
        return get_llm_response(request, self.model, self.provider, **kwargs)
