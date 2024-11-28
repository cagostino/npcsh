import subprocess
import sqlite3

import numpy as np
import os
import yaml
from jinja2 import Environment, FileSystemLoader, TemplateError, Template, Undefined

import pandas as pd
from .llm_funcs import (
    get_llm_response,
    process_data_output,
    get_data_response,
    npcsh_db_path,
    generate_image,
    render_markdown, 
    
)
import pathlib
import fnmatch
import matplotlib.pyplot as plt
#plt.ion()
 
import json
from .helpers import search_web, get_npc_path
import sklearn.feature_extraction.text
import sklearn.metrics.pairwise
import numpy as np

import sklearn
class SilentUndefined(Undefined):
    def _fail_with_undefined_error(self, *args, **kwargs):
        return ""


class NPC:
    def __init__(
        self,
        name: str,
        db_conn: sqlite3.Connection,
        primary_directive: str = None,
        suggested_tools_to_use: str = None,
        restrictions: list = None,
        model: str = None,
        provider: str = None,
        api_url: str = None,
        tools: list = None,
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
        self.api_url = api_url
        self.tools = tools or []
        self.tools_dict = {tool.tool_name: tool for tool in self.tools}
        self.shared_context = {
            'dataframes': {},
            'current_data': None,
            'computation_results': {}
        }

    def __str__(self):
        return f"NPC: {self.name}\nDirective: {self.primary_directive}\nModel: {self.model}"

    def get_data_response(self, request):
        return get_data_response(request, self.db_conn, self.tables)

    def get_llm_response(self, request, **kwargs):
        print(self.model, self.provider)
        return get_llm_response(request, self.model, self.provider, npc=self, **kwargs)

class NPCCompiler:
    def __init__(self, npc_directory, db_path):
        self.npc_directory = npc_directory  # Global NPC directory (`~/.npcsh/npc_team`)
        self.project_npc_directory = os.path.abspath("./npc_team")  # Project directory
        self.db_path = db_path
        self.npc_cache = {}
        self.resolved_npcs = {}
        self.pipe_cache = {}

        # Set tools directories
        self.global_tools_directory = os.path.join(self.npc_directory, "tools")
        self.project_tools_directory = os.path.join(self.project_npc_directory, "tools")

        # Initialize Jinja environment with multiple loaders
        self.jinja_env = Environment(
            loader=FileSystemLoader(
                [self.npc_directory, self.project_npc_directory]
            ),
            undefined=SilentUndefined
        )
    def generate_tool_script(self, tool):
        script_content = f"""
    # Auto-generated script for tool: {tool.tool_name}

    def {tool.tool_name}_execute(inputs):
        # Preprocess steps
    """
        # Add preprocess steps
        for step in tool.preprocess:
            script_content += f"    # Preprocess: {step}\n"

        # Add prompt rendering
        script_content += f"""
        # Render prompt
        prompt = '''{tool.prompt}'''
        # You might need to render the prompt with inputs

        # Call the LLM (this is simplified)
        llm_response = get_llm_response(prompt)
        
        # Postprocess steps
    """
        for step in tool.postprocess:
            script_content += f"    # Postprocess: {step}\n"

        script_content += f"    return llm_response\n"

        # Write the script to a file
        script_filename = f"{tool.tool_name}_script.py"
        with open(script_filename, "w") as script_file:
            script_file.write(script_content)
    def compile(self, npc_file: str):
        self.npc_cache.clear()  # Clear the cache
        self.resolved_npcs.clear()

        if isinstance(npc_file, NPC):
            npc_file = npc_file.name + ".npc"
        if not npc_file.endswith(".npc"):
            raise ValueError("File must have .npc extension")

        try:
            # Parse NPCs from both global and project directories
            self.parse_all_npcs()

            # Resolve NPCs
            self.resolve_all_npcs()

            # Finalize NPC profile
            parsed_content = self.finalize_npc_profile(npc_file)

            # Load tools from both global and project directories
            tools = self.load_tools()
            parsed_content["tools"] = [tool.to_dict() for tool in tools]

            self.update_compiled_npcs_table(npc_file, parsed_content)
            return parsed_content
        except Exception as e:
            raise e  # Re-raise exception for debugging
    def load_tools(self):
        tools = []
        # Load tools from global and project directories
        tool_paths = []

        if os.path.exists(self.global_tools_directory):
            for filename in os.listdir(self.global_tools_directory):
                if filename.endswith(".tool"):
                    tool_paths.append(os.path.join(self.global_tools_directory, filename))

        if os.path.exists(self.project_tools_directory):
            for filename in os.listdir(self.project_tools_directory):
                if filename.endswith(".tool"):
                    tool_paths.append(os.path.join(self.project_tools_directory, filename))

        tool_dict = {}
        for tool_path in tool_paths:
            tool_data = self.load_tool_from_file(tool_path)
            if tool_data:
                tool = Tool(tool_data)
                # Project tools override global tools
                tool_dict[tool.tool_name] = tool

        return list(tool_dict.values())

    def load_tool_from_file(self, tool_path):
        try:
            with open(tool_path, "r") as f:
                tool_content = f.read()
            if not tool_content.strip():
                print(f"Tool file {tool_path} is empty. Skipping.")
                return None
            tool_data = yaml.safe_load(tool_content)
            if tool_data is None:
                print(f"Tool file {tool_path} is invalid or empty. Skipping.")
                return None
            return tool_data
        except yaml.YAMLError as e:
            print(f"Error parsing tool {tool_path}: {e}")
            return None
        except Exception as e:
            print(f"Error loading tool {tool_path}: {e}")
            return None
    def parse_all_npcs(self):
        directories = [self.npc_directory]
        if os.path.exists(self.project_npc_directory):
            directories.append(self.project_npc_directory)
        for directory in directories:
            for filename in os.listdir(directory):
                if filename.endswith(".npc"):
                    npc_path = os.path.join(directory, filename)
                    self.parse_npc_file(npc_path)
    def parse_npc_file(self, npc_file_path: str):
        npc_file = os.path.basename(npc_file_path)
        if npc_file in self.npc_cache:
            # Project NPCs override global NPCs
            if npc_file_path.startswith(self.project_npc_directory):
                print(f"Overriding NPC {npc_file} with project version.")
            else:
                # Skip if already loaded from project directory
                return self.npc_cache[npc_file]

        try:
            with open(npc_file_path, "r") as f:
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
        profile = self.resolved_npcs.get(npc_file)
        if not profile:
            raise ValueError(f"NPC {npc_file} has not been resolved.")

        # Resolve any remaining references
        for key, value in profile.items():
            if isinstance(value, str):
                template = self.jinja_env.from_string(value)
                profile[key] = template.render(self.resolved_npcs)

        required_keys = ["name", "primary_directive"]
        for key in required_keys:
            if key not in profile:
                raise ValueError(f"Missing required key in NPC profile: {key}")

        return profile

    def compile_pipe(self, pipe_file: str, initial_input=None):
        if pipe_file in self.pipe_cache:
            return self.pipe_cache[pipe_file]

        if not pipe_file.endswith(".pipe"):
            raise ValueError("Pipeline file must have .pipe extension")

        try:
            with open(os.path.join(self.npc_directory, pipe_file), "r") as f:
                pipeline_data = yaml.safe_load(f)

            context = {"input": initial_input}
            results = []
            jinja_env = Environment(
                loader=FileSystemLoader("."), undefined=SilentUndefined
            )

            for step in pipeline_data["steps"]:
                npc_name = step["npc"]
                prompt_template = step["prompt"]

                # Load the NPC
                npc_path = get_npc_path(npc_name, self.db_path)
                npc = load_npc_from_file(
                    npc_path, sqlite3.connect(self.db_path)
                )  # Create a new connection for each NPC

                # Render the prompt using Jinja2
                prompt_template = jinja_env.from_string(prompt_template)
                prompt = prompt_template.render(context)

                response = npc.get_llm_response(prompt)
                print(response)
                results.append({"npc": npc_name, "response": response})

                # Update context with the response for the next step
                context["input"] = response
                context[npc_name] = response  # Update context with NPC's response

                # Pass information to the next NPC if specified
                pass_to = step.get("pass_to")
                if pass_to:
                    context["pass_to"] = pass_to

            self.pipe_cache[pipe_file] = results  # Cache the results
            return results

        except FileNotFoundError:
            raise ValueError(f"NPC file not found: {npc_path}")
        except yaml.YAMLError as e:
            raise ValueError(
                f"Error parsing YAML in pipeline file {pipe_file}: {str(e)}"
            )
        except Exception as e:
            raise ValueError(f"Error compiling pipeline {pipe_file}: {str(e)}")

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

    def update_compiled_npcs_table(self, npc_file, parsed_content):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                npc_name = parsed_content["name"]
                source_path = os.path.join(
                    self.npc_directory, npc_file
                )  # Use source_path
                cursor.execute(
                    "INSERT OR REPLACE INTO compiled_npcs (name, source_path, compiled_content) VALUES (?, ?, ?)",  # Correct column name
                    (npc_name, source_path, yaml.dump(parsed_content)),
                )
                conn.commit()
        except Exception as e:
            print(
                f"Error updating compiled_npcs table: {str(e)}"
            )  # Print the full error

class Tool:
    def __init__(self, tool_data):
        if not tool_data or not isinstance(tool_data, dict):
            raise ValueError("Invalid tool data provided.")
        if "tool_name" not in tool_data:
            raise KeyError("Missing 'tool_name' in tool definition.")
        self.tool_name = tool_data.get("tool_name")
        self.inputs = tool_data.get("inputs", [])

        # Parse steps with engines
        self.preprocess = self.parse_steps(tool_data.get("preprocess", []))
        self.prompt = self.parse_step(tool_data.get("prompt", {}))
        self.postprocess = self.parse_steps(tool_data.get("postprocess", []))

    def parse_step(self, step):
        if isinstance(step, dict):
            return {
                "engine": step.get("engine", "plain_english"),
                "code": step.get("code", ""),
            }
        elif isinstance(step, str):  # For backward compatibility
            return {"engine": "plain_english", "code": step}
        else:
            raise ValueError("Invalid step format")

    def parse_steps(self, steps):
        return [self.parse_step(step) for step in steps]



    def execute(self, input_values, tools_dict, jinja_env, command, npc=None):
        context = npc.shared_context
        context.update({
            "inputs": input_values,
            "tools": tools_dict,
            "llm_response": None,
            "output": None, 
            "command": command
        })
        # Process Preprocess Steps
        for step in self.preprocess:
            context = self.execute_step(step, context, jinja_env, npc=npc)

        # Process Prompt
        context = self.execute_step(self.prompt, context, jinja_env, npc=npc)

        # Process Postprocess Steps
        for step in self.postprocess:
            context = self.execute_step(step, context, jinja_env, npc=npc)

        # Return the final output
        if context.get('output') is not None:
            return context.get('output')
        elif context.get('llm_response') is not None:
            return context.get('llm_response')
            

    def execute_step(self, step, context, jinja_env, npc=None):
        engine = step.get("engine", "plain_english")
        code = step.get("code", "")

        if engine == "plain_english":
            
            # Create template with debugging
            from jinja2 import Environment, DebugUndefined
            debug_env = Environment(undefined=DebugUndefined)
            template = debug_env.from_string(code)
            

            rendered_text = template.render(**context)  # Unpack context as kwargs
            #print(len(rendered_text.strip()))
            if len(rendered_text.strip()) > 0:
                llm_response = get_llm_response(rendered_text, npc=npc)
                #print(llm_response)
                if context.get("llm_response") is None:
                    # This is the prompt step
                    context["llm_response"] = llm_response.get("response", "")
                else:
                    # This is a postprocess step; set output
                    context["output"] = llm_response.get("response", "")                

        
        elif engine == "python":
            # Execute Python code
            exec_globals = {
                "__builtins__": __builtins__,
                "npc": npc,  # Pass npc to the execution environment
                "context": context,
                # Include necessary imports
                "pd": pd,
                "plt":plt, 
                "np":np,
                "os": os,
                "get_llm_response": get_llm_response,
                "generate_image": generate_image,
                "render_markdown": render_markdown,
                "search_web": search_web,
                "json": json,
                "sklearn": __import__('sklearn'),
                "TfidfVectorizer": __import__('sklearn.feature_extraction.text').feature_extraction.text.TfidfVectorizer,
                "cosine_similarity": __import__('sklearn.metrics.pairwise').metrics.pairwise.cosine_similarity,
                "Path": __import__('pathlib').Path,
            
                
                "fnmatch": fnmatch,
                "pathlib": pathlib,
                "subprocess": subprocess,
                
            }
            exec_env = context.copy()
            exec(code, exec_globals, exec_env)
            context.update(exec_env)
            #import pdb 
            #pdb.set_trace()
        else:
            raise ValueError(f"Unsupported engine '{engine}'")

        return context

    def to_dict(self):
        return {
            "tool_name": self.tool_name,
            "inputs": self.inputs,
            "preprocess": [self.step_to_dict(step) for step in self.preprocess],
            "prompt": self.step_to_dict(self.prompt),
            "postprocess": [self.step_to_dict(step) for step in self.postprocess],
        }

    def step_to_dict(self, step):
        return {
            "engine": step.get("engine"),
            "code": step.get("code"),
        }

# npc_compiler.py

def load_npc_from_file(npc_file: str, db_conn: sqlite3.Connection) -> NPC:
    try:
        with open(npc_file, "r") as f:
            npc_data = yaml.safe_load(f)

        # Extract fields from YAML
        name = npc_data["name"]
        primary_directive = npc_data.get("primary_directive")
        suggested_tools_to_use = npc_data.get("suggested_tools_to_use")
        restrictions = npc_data.get("restrictions", [])
        model = npc_data.get("model", os.environ.get("NPCSH_MODEL", "llama3.2"))
        provider = npc_data.get("provider", os.environ.get("NPCSH_PROVIDER", "ollama"))
        api_url = npc_data.get("api_url", os.environ.get("NPCSH_API_URL", None))

        # Load tools from global and project-specific directories
        tools = []
        # 1. Load tools defined within the NPC profile
        if "tools" in npc_data:
            for tool_data in npc_data["tools"]:
                tool = Tool(tool_data)
                tools.append(tool)
        # 2. Load global tools from ~/.npcsh/npc_team/tools
        user_home = os.path.expanduser("~")
        global_tools_directory = os.path.join(user_home, ".npcsh", "npc_team", "tools")
        tools.extend(load_tools_from_directory(global_tools_directory))
        # 3. Load project-specific tools from ./npc_team/tools
        project_tools_directory = os.path.abspath("./npc_team/tools")
        tools.extend(load_tools_from_directory(project_tools_directory))

        # Remove duplicates, giving precedence to project-specific tools
        tool_dict = {}
        for tool in tools:
            tool_dict[tool.tool_name] = tool  # Project tools overwrite global tools

        tools = list(tool_dict.values())

        # Initialize and return the NPC object
        return NPC(
            name,
            db_conn,
            primary_directive=primary_directive,
            suggested_tools_to_use=suggested_tools_to_use,
            restrictions=restrictions,
            model=model,
            provider=provider,
            api_url=api_url,
            tools=tools,  # Pass the tools
        )

    except FileNotFoundError:
        raise ValueError(f"NPC file not found: {npc_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML in NPC file {npc_file}: {str(e)}")
    except KeyError as e:
        raise ValueError(f"Missing required key in NPC file {npc_file}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading NPC from file {npc_file}: {str(e)}")
def load_tools_from_directory(directory):
    tools = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".tool"):
                full_path = os.path.join(directory, filename)
                with open(full_path, "r") as f:
                    tool_content = f.read()
                    try:
                        if not tool_content.strip():
                            print(f"Tool file {filename} is empty. Skipping.")
                            continue
                        tool_data = yaml.safe_load(tool_content)
                        if tool_data is None:
                            print(f"Tool file {filename} is invalid or empty. Skipping.")
                            continue
                        tool = Tool(tool_data)
                        tools.append(tool)
                    except yaml.YAMLError as e:
                        print(f"Error parsing tool {filename}: {e}")
    #else:
        #print(f"Tools directory not found: {directory}")
    return tools