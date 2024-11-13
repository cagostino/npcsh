import subprocess
import sqlite3


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
)
from .helpers import search_web, get_npc_path


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

    def __str__(self):
        return f"NPC: {self.name}\nDirective: {self.primary_directive}\nModel: {self.model}"

    def get_data_response(self, request):
        return get_data_response(request, self.db_conn, self.tables)

    def get_llm_response(self, request, **kwargs):
        print(self.model, self.provider)
        return get_llm_response(request, self.model, self.provider, npc=self, **kwargs)


class NPCCompiler:
    def __init__(self, npc_directory, db_path, tools_directory=None):
        self.npc_directory = npc_directory
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.npc_directory), undefined=SilentUndefined
        )
        self.npc_cache = {}
        self.resolved_npcs = {}
        self.db_path = db_path
        self.tools_directory = tools_directory
        self.pipe_cache = {}

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
        self.npc_cache.clear()  # Clear the cache at the start of each compilation
        self.resolved_npcs.clear()
        if isinstance(npc_file, NPC):
            npc_file = npc_file.name + ".npc"
        if not npc_file.endswith(".npc"):
            raise ValueError("File must have .npc extension")

        if not npc_file.endswith(".npc"):
            raise ValueError("File must have .npc extension")

        try:
            # First pass: parse all NPC files without resolving Jinja templates
            self.parse_all_npcs()

            # Second pass: resolve Jinja templates and merge inherited properties
            self.resolve_all_npcs()

            # Final pass: resolve any remaining references
            parsed_content = self.finalize_npc_profile(npc_file)

            if self.tools_directory is not None and os.path.exists(
                self.tools_directory
            ):
                tools = self.load_tools(self.tools_directory)
                parsed_content["tools"] = [
                    tool.to_dict() for tool in tools
                ]  # Add tools to profile
                if parsed_content.get("tools"):
                    for tool_dict in parsed_content["tools"]:
                        tool = Tool(tool_dict)
                        if tool.engine == "plain_english":
                            self.generate_tool_script(tool)

            self.update_compiled_npcs_table(npc_file, parsed_content)
            return parsed_content
        except Exception as e:
            raise

    def load_tools(self, tools_directory):
        tools = []
        for filename in os.listdir(tools_directory):
            if filename.endswith(".tool"):
                with open(os.path.join(tools_directory, filename), "r") as f:
                    tool_content = f.read()
                    try:
                        tool_data = yaml.safe_load(tool_content)
                        tool = Tool(tool_data)
                        tools.append(tool)
                    except yaml.YAMLError as e:
                        print(f"Error parsing tool {filename}: {e}")
        return tools

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
        ]
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
        parsed_steps = []
        for step in steps:
            parsed_steps.append(self.parse_step(step))
        return parsed_steps

    def execute(self, input_values, tools_dict, jinja_env, npc=None):
        context = {
            "inputs": input_values,
            "tools": tools_dict,
            "llm_response": None,
            "output": None,
        }

        # Process Preprocess Steps
        for step in self.preprocess:
            context = self.execute_step(step, context, jinja_env, npc=npc)

        # Process Prompt
        context = self.execute_step(self.prompt, context, jinja_env, npc=npc)

        # Process Postprocess Steps
        for step in self.postprocess:
            context = self.execute_step(step, context, jinja_env, npc=npc)

        # Return the final output
        return context.get("output") or context.get("llm_response")

    def execute_step(self, step, context, jinja_env, npc=None):
        engine = step["engine"]
        code = step["code"]
        if engine == "plain_english":
            # Render the prompt or postprocess template
            template = jinja_env.from_string(code)
            rendered_text = template.render(context)
            llm_response = get_llm_response(rendered_text)
            print(llm_response)
            if context.get("llm_response") is None and code.strip():
                # This is the prompt step, but since the code is empty, we skip calling the LLM
                # If the prompt step is empty, we move directly to the postprocess
                context["llm_response"] = llm_response
            else:
                # This is a postprocess step; set output
                context["output"] = llm_response
        elif engine == "python":
            # Execute Python code
            exec_env = context.copy()
            # Ensure necessary imports and functions are available
            exec_globals = {
                "__builtins__": __builtins__,
                "generate_image": generate_image,
                "npc": npc,  # Pass the NPC object
                # Include other necessary global variables or functions
            }
            exec(code, exec_globals, exec_env)
            context.update(exec_env)
        else:
            raise ValueError(f"Unsupported engine '{engine}'")
        return context


# compiler = NPCCompiler('/path/to/npc/directory')
# compiled_script = compiler.compile('your_npc_file.npc')


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

        # Load tools if present
        tools = []
        if "tools" in npc_data:
            for tool_data in npc_data["tools"]:
                tool = Tool(tool_data)
                tools.append(tool)
        else:
            # Load tools from 'npc_team/tools' directory if exists
            tools_directory = os.path.abspath("./npc_team/tools")
            if os.path.exists(tools_directory):
                for filename in os.listdir(tools_directory):
                    if filename.endswith(".tool"):
                        with open(os.path.join(tools_directory, filename), "r") as f:
                            tool_content = f.read()
                            try:
                                tool_data = yaml.safe_load(tool_content)
                                tool = Tool(tool_data)
                                tools.append(tool)
                            except yaml.YAMLError as e:
                                print(f"Error parsing tool {filename}: {e}")

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
