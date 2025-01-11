import subprocess
import sqlite3
import numpy as np
import os
import yaml
from jinja2 import Environment, FileSystemLoader, Template, Undefined
import pandas as pd
import pathlib
from typing import Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import json
import pathlib
import fnmatch
import matplotlib.pyplot as plt
import re
import random

# plt.ion()
from datetime import datetime
import json
import hashlib

import sklearn.feature_extraction.text
import sklearn.metrics.pairwise
import numpy as np

import sklearn

import os
import re
import sqlite3
import pandas as pd
from typing import Dict, List, Set, Union
from pathlib import Path
from collections import defaultdict, deque


from .llm_funcs import (
    get_llm_response,
    process_data_output,
    get_data_response,
    generate_image,
)

from .helpers import get_npc_path

from .search import search_web, rag_search
from .image import capture_screenshot, analyze_image_base

import sqlite3
import pandas as pd


import subprocess
import sqlite3
import numpy as np
import os
import yaml
from jinja2 import Environment, FileSystemLoader, Template, Undefined
import pandas as pd
import pathlib
from typing import Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import json
import pathlib
import fnmatch
import matplotlib.pyplot as plt

# plt.ion()

import sklearn.feature_extraction.text
import sklearn.metrics.pairwise
import numpy as np

import sklearn


from .llm_funcs import (
    get_llm_response,
    process_data_output,
    get_data_response,
    generate_image,
)

from .helpers import get_npc_path

from .search import search_web, rag_search


def create_or_replace_table(db_path, table_name, data):
    """
    Creates or replaces a table in the SQLite database.

    :param db_path: Path to the SQLite database.
    :param table_name: Name of the table to create/replace.
    :param data: Pandas DataFrame containing the data to insert.
    """
    conn = sqlite3.connect(db_path)
    try:
        # Replace the table with new data
        data.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Table '{table_name}' created/replaced successfully.")
    except Exception as e:
        print(f"Error creating/replacing table '{table_name}': {e}")
    finally:
        conn.close()


def init_pipeline_runs():
    # need to move elsewhere....
    db_path = "~/npcsh_history.db"

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Add pipeline_runs table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_name TEXT,
                step_name TEXT,
                output TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()


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
        # print(self.tables)
        self.provider = provider
        self.api_url = api_url
        self.tools = tools or []
        # for npcs, if a set of tools is provided in the NPC file, we need to ensure they are loaded and refed
        # well also need to introduce a "default" tool set that will be available in addition to the manual tools
        # this will allow for a more flexible tooling system


        self.tools_dict = {tool.tool_name: tool for tool in self.tools}
        self.shared_context = {
            "dataframes": {},
            "current_data": None,
            "computation_results": {},
        }

    def __str__(self):
        return f"NPC: {self.name}\nDirective: {self.primary_directive}\nModel: {self.model}"

    def analyze_db_data(self, request: str):
        # print("npc")
        # print(self.db_conn, self.tables, self.model, self.provider)
        return get_data_response(
            request,
            self.db_conn,
            self.tables,
            model=self.model,
            provider=self.provider,
            npc=self,
        )

    def get_llm_response(self, request: str, **kwargs):
        # print(self.model, self.provider)
        return get_llm_response(
            request, model=self.model, provider=self.provider, npc=self, **kwargs
        )


class SilentUndefined(Undefined):
    def _fail_with_undefined_error(self, *args, **kwargs):
        return ""


class Tool:
    def __init__(self, tool_data: dict):
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

    def parse_step(self, step: Union[dict, str]) -> dict:
        if isinstance(step, dict):
            return {
                "engine": step.get("engine", "natural"),
                "code": step.get("code", ""),
            }
        elif isinstance(step, str):  # For backward compatibility
            return {"engine": "natural", "code": step}
        else:
            raise ValueError("Invalid step format")

    def parse_steps(self, steps: list) -> list:
        return [self.parse_step(step) for step in steps]

    def execute(
        self,
        input_values: dict,
        tools_dict: dict,
        jinja_env: Environment,
        command: str,
        npc=None,
    ):
        context = npc.shared_context
        context.update(
            {
                "inputs": input_values,
                "tools": tools_dict,
                "llm_response": None,
                "output": None,
                "command": command,
            }
        )
        # print(tools_dict)
        # Process Preprocess Steps
        for step in self.preprocess:
            context = self.execute_step(step, context, jinja_env, npc=npc)

        # Process Prompt
        context = self.execute_step(self.prompt, context, jinja_env, npc=npc)

        # Process Postprocess Steps
        for step in self.postprocess:
            context = self.execute_step(step, context, jinja_env, npc=npc)

        # Return the final output
        if context.get("output") is not None:
            return context.get("output")
        elif context.get("llm_response") is not None:
            return context.get("llm_response")

    def execute_step(
        self, step: dict, context: dict, jinja_env: Environment, npc: NPC = None
    ):
        engine = step.get("engine", "natural")
        code = step.get("code", "")
        # print(step)

        if engine == "natural":
            # Create template with debugging
            from jinja2 import Environment, DebugUndefined

            debug_env = Environment(undefined=DebugUndefined)
            template = debug_env.from_string(code)

            rendered_text = template.render(**context)  # Unpack context as kwargs
            # print(len(rendered_text.strip()))
            if len(rendered_text.strip()) > 0:
                llm_response = get_llm_response(rendered_text, npc=npc)
                # print(llm_response)
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
                "plt": plt,
                "np": np,
                "os": os,
                "get_llm_response": get_llm_response,
                "generate_image": generate_image,
                "search_web": search_web,
                "json": json,
                "sklearn": __import__("sklearn"),
                "TfidfVectorizer": __import__(
                    "sklearn.feature_extraction.text"
                ).feature_extraction.text.TfidfVectorizer,
                "cosine_similarity": __import__(
                    "sklearn.metrics.pairwise"
                ).metrics.pairwise.cosine_similarity,
                "Path": __import__("pathlib").Path,
                "fnmatch": fnmatch,
                "pathlib": pathlib,
                "subprocess": subprocess,
            }
            exec_env = context.copy()
            exec(code, exec_globals, exec_env)
            context.update(exec_env)
            # import pdb
            # pdb.set_trace()
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
            loader=FileSystemLoader([self.npc_directory, self.project_npc_directory]),
            undefined=SilentUndefined,
        )

    def generate_tool_script(self, tool: Tool):
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
        # get the absolute path
        npc_file = os.path.abspath(npc_file)

        try:
            # Parse NPCs from both global and project directories
            self.parse_all_npcs()

            # Resolve NPCs
            self.resolve_all_npcs()

            # Finalize NPC profile
            # print(npc_file)
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
                    tool_paths.append(
                        os.path.join(self.global_tools_directory, filename)
                    )

        if os.path.exists(self.project_tools_directory):
            for filename in os.listdir(self.project_tools_directory):
                if filename.endswith(".tool"):
                    tool_paths.append(
                        os.path.join(self.project_tools_directory, filename)
                    )

        tool_dict = {}
        for tool_path in tool_paths:
            tool_data = self.load_tool_from_file(tool_path)
            if tool_data:
                tool = Tool(tool_data)
                # Project tools override global tools
                tool_dict[tool.tool_name] = tool

        return list(tool_dict.values())

    def load_tool_from_file(self, tool_path: str) -> Union[dict, None]:
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

    def parse_all_npcs(self) -> None:
        directories = [self.npc_directory]
        if os.path.exists(self.project_npc_directory):
            directories.append(self.project_npc_directory)
        for directory in directories:
            for filename in os.listdir(directory):
                if filename.endswith(".npc"):
                    npc_path = os.path.join(directory, filename)
                    self.parse_npc_file(npc_path)

    def parse_npc_file(self, npc_file_path: str) -> dict:
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
            npc = self.resolve_npc_profile(npc_file)
            # print(npc)

    def resolve_npc_profile(self, npc_file: str) -> dict:
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

    def finalize_npc_profile(self, npc_file: str) -> dict:
        profile = self.resolved_npcs.get(os.path.basename(npc_file))
        if not profile:
            raise ValueError(f"NPC {npc_file} has not been resolved.")

        # Resolve any remaining references
        # Log the profile content before processing
        # print(f"Initial profile for {npc_file}: {profile}")

        for key, value in profile.items():
            if isinstance(value, str):
                template = self.jinja_env.from_string(value)
                profile[key] = template.render(self.resolved_npcs)

        required_keys = ["name", "primary_directive"]
        for key in required_keys:
            if key not in profile:
                raise ValueError(f"Missing required key in NPC profile: {key}")

        return profile

    def execute_stage(self, stage, context, jinja_env):
        step_name = stage["step_name"]
        npc_name = stage["npc"]
        npc_name = jinja_env.from_string(npc_name).render(context)
        # print("npc name: ", npc_name)
        npc_path = get_npc_path(npc_name, self.db_path)
        # print("npc path: ", npc_path)
        prompt_template = stage["task"]
        num_samples = stage.get("num_samples", 1)

        step_results = []
        for sample_index in range(num_samples):
            # Load the NPC
            npc = load_npc_from_file(npc_path, sqlite3.connect(self.db_path))

            # Render the prompt using Jinja2
            prompt_template = jinja_env.from_string(prompt_template)
            prompt = prompt_template.render(context, sample_index=sample_index)

            response = npc.get_llm_response(prompt)
            # print(response)
            step_results.append({"npc": npc_name, "response": response["response"]})

            # Update context with the response for the next step
            context[f"{step_name}_{sample_index}"] = response[
                "response"
            ]  # Update context with step's response

        return step_results

    def aggregate_step_results(self, step_results, aggregation_strategy):
        responses = [result["response"] for result in step_results]
        if len(responses) == 1:
            return responses[0]
        if aggregation_strategy == "concat":
            return "\n".join(responses)
        elif aggregation_strategy == "summary":
            # Use the LLM to generate a summary of the responses
            response_text = "\n".join(responses)
            summary_prompt = (
                f"Please provide a concise summary of the following responses: "
                + response_text
            )

            summary = self.get_llm_response(summary_prompt)["response"]
            return summary
        elif aggregation_strategy == "pessimistic_critique":
            # Use the LLM to provide a pessimistic critique of the responses
            response_text = "\n".join(responses)
            critique_prompt = f"Please provide a pessimistic critique of the following responses:\n\n{response_text}"

            critique = self.get_llm_response(critique_prompt)["response"]
            return critique
        elif aggregation_strategy == "optimistic_view":
            # Use the LLM to provide an optimistic view of the responses
            response_text = "\n".join(responses)
            optimistic_prompt = f"Please provide an optimistic view of the following responses:\n\n{response_text}"
            optimistic_view = self.get_llm_response(optimistic_prompt)["response"]
            return optimistic_view
        elif aggregation_strategy == "balanced_analysis":
            # Use the LLM to provide a balanced analysis of the responses
            response = "\n".join(responses)
            analysis_prompt = f"Please provide a balanced analysis of the following responses:\n\n{response}"

            balanced_analysis = self.get_llm_response(analysis_prompt)["response"]
            return balanced_analysis
        elif aggregation_strategy == "first":
            return responses[0]
        elif aggregation_strategy == "last":
            return responses[-1]
        else:
            raise ValueError(f"Invalid aggregation strategy: {aggregation_strategy}")

    def compile_pipe(self, pipe_file: str, initial_input=None) -> dict:
        if pipe_file in self.pipe_cache:
            return self.pipe_cache[pipe_file]

        if not pipe_file.endswith(".pipe"):
            raise ValueError("Pipeline file must have .pipe extension")

        # print(pipe_file)

        with open(pipe_file, "r") as f:
            pipeline_data = yaml.safe_load(f)

        final_output = {}
        jinja_env = Environment(loader=FileSystemLoader("."), undefined=SilentUndefined)

        context = {"input": initial_input, **self.npc_cache}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            pipeline_name = os.path.basename(pipe_file).replace(".pipe", "")

            for stage in pipeline_data["steps"]:
                step_results = self.execute_stage(stage, context, jinja_env)
                aggregated_result = self.aggregate_step_results(
                    step_results, stage.get("aggregation_strategy", "first")
                )

                # Store in database
                cursor.execute(
                    "INSERT INTO pipeline_runs (pipeline_name, step_name, output) VALUES (?, ?, ?)",
                    (pipeline_name, stage["step_name"], str(aggregated_result)),
                )

                final_output[stage["step_name"]] = aggregated_result
                context[stage["step_name"]] = aggregated_result

            conn.commit()

        self.pipe_cache[pipe_file] = final_output  # Cache the results

        return final_output

    def merge_profiles(self, parent, child) -> dict:
        merged = parent.copy()
        for key, value in child.items():
            if isinstance(value, list) and key in merged:
                merged[key] = merged[key] + value
            elif isinstance(value, dict) and key in merged:
                merged[key] = self.merge_profiles(merged[key], value)
            else:
                merged[key] = value
        return merged

    def update_compiled_npcs_table(self, npc_file, parsed_content) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                npc_name = parsed_content["name"]
                source_path = npc_file

                cursor.execute(
                    "INSERT OR REPLACE INTO compiled_npcs (name, source_path, compiled_content) VALUES (?, ?, ?)",  # Correct column name
                    (npc_name, source_path, yaml.dump(parsed_content)),
                )
                conn.commit()
        except Exception as e:
            print(
                f"Error updating compiled_npcs table: {str(e)}"
            )  # Print the full error


import os
import yaml
import hashlib
import sqlite3
from sqlalchemy import create_engine
import pandas as pd
import json
from datetime import datetime
from jinja2 import Template
import re


class PipelineRunner:
    def __init__(
        self,
        pipeline_file: str,
        db_path: str = "~/npcsh_history.db",
        npc_root_dir: str = "../",
    ):
        self.pipeline_file = pipeline_file
        self.pipeline_data = self.load_pipeline()
        self.db_path = os.path.expanduser(db_path)
        self.npc_root_dir = npc_root_dir
        self.npc_cache = {}
        self.db_engine = create_engine(f"sqlite:///{self.db_path}")

    def load_pipeline(self):
        with open(self.pipeline_file, "r") as f:
            return yaml.safe_load(f)

    def compute_pipeline_hash(self):
        with open(self.pipeline_file, "r") as f:
            content = f.read()
        return hashlib.sha256(content.encode()).hexdigest()

    def execute_pipeline(self):
        context = {
            "npc": self.npc_ref,
            "ref": lambda step_name: step_name,  # Directly use step name
            "source": self.fetch_data_from_source,
        }

        pipeline_hash = self.compute_pipeline_hash()
        pipeline_name = os.path.splitext(os.path.basename(self.pipeline_file))[0]
        results_table_name = f"{pipeline_name}_results"
        self.ensure_tables_exist(results_table_name)
        run_id = self.create_run_entry(pipeline_hash)

        for step in self.pipeline_data["steps"]:
            self.execute_step(step, context, run_id, results_table_name)

    def npc_ref(self, npc_name: str):
        clean_name = npc_name.replace("MISSING_REF_", "")
        try:
            npc_path = self.find_npc_path(clean_name)
            return clean_name if npc_path else f"MISSING_REF_{clean_name}"
        except Exception:
            return f"MISSING_REF_{clean_name}"

    def execute_step(
        self, step: dict, context: dict, run_id: int, results_table_name: str
    ):
        """Execute pipeline step and store results in the database."""
        print("\nStarting step execution...")

        mixa = step.get("mixa", False)
        mixa_turns = step.get("mixa_turns", 5 if mixa else None)

        npc_name = Template(step.get("npc", "")).render(context)
        npc = self.load_npc(npc_name)
        model = step.get("model", npc.model)
        provider = step.get("provider", npc.provider)

        response_text = ""

        if mixa:
            print("Executing mixture of agents strategy...")
            response_text = self.execute_mixture_of_agents(
                step,
                context,
                run_id,
                results_table_name,
                npc,
                model,
                provider,
                mixa_turns,
            )
        else:
            source_matches = re.findall(
                r"{{\s*source\('([^']+)'\)\s*}}", step.get("task", "")
            )
            print(f"Found source matches: {source_matches}")

            if not source_matches:
                rendered_task = Template(step.get("task", "")).render(context)
                response = get_llm_response(
                    rendered_task, model=model, provider=provider, npc=npc
                )
                response_text = response.get("response", "")
            else:
                table_name = source_matches[0]
                df = pd.read_sql(f"SELECT * FROM {table_name}", self.db_engine)
                print(f"\nQuerying table: {table_name}")
                print(f"Found {len(df)} rows")

                if step.get("batch_mode", False):
                    data_str = df.to_json(orient="records")
                    rendered_task = step.get("task", "").replace(
                        f"{{{{ source('{table_name}') }}}}", data_str
                    )
                    rendered_task = Template(rendered_task).render(context)

                    response = get_llm_response(
                        rendered_task, model=model, provider=provider, npc=npc
                    )
                    response_text = response.get("response", "")
                else:
                    all_responses = []
                    for idx, row in df.iterrows():
                        row_data = json.dumps(row.to_dict())
                        row_task = step.get("task", "").replace(
                            f"{{{{ source('{table_name}') }}}}", row_data
                        )
                        rendered_task = Template(row_task).render(context)

                        response = get_llm_response(
                            rendered_task, model=model, provider=provider, npc=npc
                        )
                        result = response.get("response", "")
                        all_responses.append(result)

                    response_text = all_responses

        # Storing the final result in the database
        self.store_result(
            run_id,
            step["step_name"],
            npc_name,
            model,
            provider,
            {"response": response_text},
            response_text,
            results_table_name,
        )

        context[step["step_name"]] = response_text
        print(f"\nStep complete. Response stored in context[{step['step_name']}]")
        return response_text

    def store_result(
        self,
        run_id,
        task_name,
        npc_name,
        model,
        provider,
        inputs,
        outputs,
        results_table_name,
    ):
        """Store results into the specified results table in the database."""
        cleaned_inputs = self.clean_for_json(inputs)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                f"""
                INSERT INTO {results_table_name} (run_id, task_name, npc_name,
                model, provider, inputs, outputs) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    task_name,
                    npc_name,
                    model,
                    provider,
                    json.dumps(cleaned_inputs),
                    json.dumps(outputs),
                ),
            )
            conn.commit()
        except Exception as e:
            print(f"Error storing result: {e}")
        finally:
            conn.close()

    def execute_mixture_of_agents(
        self,
        step,
        context,
        run_id,
        results_table_name,
        npc,
        model,
        provider,
        mixa_turns,
    ):
        """Facilitates multi-agent decision-making with feedback for refinement."""

        # Read agent counts from the step configuration
        num_generating_agents = len(step.get("mixa_agents", []))
        num_voting_agents = len(step.get("mixa_voters", []))
        num_voters = step.get("mixa_voter_count", num_voting_agents)

        # Step 1: Initial Response Generation
        round_responses = []
        print("\nInitial responses generation:")
        for agent_index in range(num_generating_agents):
            task_template = Template(step.get("task", "")).render(context)
            response = get_llm_response(
                task_template, model=model, provider=provider, npc=npc
            )
            round_responses.append(response.get("response", ""))
            print(
                f"Agent {agent_index + 1} generated: " f"{response.get('response', '')}"
            )

        # Loop for each round of voting and refining
        for turn in range(1, mixa_turns + 1):
            print(f"\n--- Round {turn}/{mixa_turns} ---")

            # Step 2: Voting Logic by voting agents
            votes = self.conduct_voting(round_responses, num_voters)

            # Step 3: Report results to generating agents
            print("\nVoting Results:")
            for idx, response in enumerate(round_responses):
                print(f"Response {idx + 1} received {votes[idx]} votes.")

            # Provide feedback on the responses
            feedback_message = "Responses and their votes:\n" + "\n".join(
                f"Response {i + 1}: {resp} - Votes: {votes[i]} "
                for i, resp in enumerate(round_responses)
            )

            # Step 4: Refinement feedback to each agent
            refined_responses = []
            for agent_index in range(num_generating_agents):
                refined_task = (
                    feedback_message
                    + f"\nRefine your response: {round_responses[agent_index]}"
                )
                response = get_llm_response(
                    refined_task, model=model, provider=provider, npc=npc
                )
                refined_responses.append(response.get("response", ""))
                print(
                    f"Agent {agent_index + 1} refined response: "
                    f"{response.get('response', '')}"
                )

            # Update responses for the next round
            round_responses = refined_responses

        # Step 5: Final synthesis using the LLM
        final_synthesis_input = (
            "Synthesize the following refined responses into a coherent answer:\n"
            + "\n".join(round_responses)
        )
        final_synthesis = get_llm_response(
            final_synthesis_input, model=model, provider=provider, npc=npc
        )

        return final_synthesis  # Return synthesized response based on LLM output

    def conduct_voting(self, responses, num_voting_agents):
        """Conducts voting among agents on the given responses."""
        votes = [0] * len(responses)
        for _ in range(num_voting_agents):
            voted_index = random.choice(range(len(responses)))  # Randomly vote
            votes[voted_index] += 1
        return votes

    def synthesize_responses(self, votes):
        """Synthesizes the responses based on votes."""
        # Example: Choose the highest voted response
        max_votes = max(votes)
        chosen_idx = votes.index(max_votes)
        return f"Synthesized response based on votes from agents: " f"{chosen_idx + 1}"

    def resolve_sources_in_task(self, task: str, context: dict) -> str:
        # Use Jinja2 template rendering directly for simplicity
        template = Template(task)
        return template.render(context)

    def fetch_data_from_source(self, table_name):
        query = f"SELECT * FROM {table_name}"
        try:
            df = pd.read_sql(query, con=self.db_engine)
        except Exception as e:
            raise RuntimeError(f"Error fetching data from '{table_name}': {e}")
        return self.format_data_as_string(df)

    def format_data_as_string(self, df):
        return df.to_json(orient="records", lines=True, indent=2)

    def ensure_tables_exist(self, results_table_name):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS pipeline_runs ("
                "run_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "pipeline_hash TEXT, timestamp DATETIME)"
            )
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {results_table_name} ("
                "result_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "run_id INTEGER, task_name TEXT, npc_name TEXT, "
                "model TEXT, provider TEXT, inputs JSON, "
                "outputs JSON, FOREIGN KEY(run_id) "
                "REFERENCES pipeline_runs(run_id))"
            )
            conn.commit()
        finally:
            conn.close()

    def create_run_entry(self, pipeline_hash):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO pipeline_runs (pipeline_hash, timestamp) VALUES (?, ?)",
                (pipeline_hash, datetime.now()),
            )
            conn.commit()
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        finally:
            conn.close()

    def clean_for_json(self, obj):
        if isinstance(obj, dict):
            return {
                k: self.clean_for_json(v)
                for k, v in obj.items()
                if not k.startswith("_") and not callable(v)
            }
        elif isinstance(obj, list):
            return [self.clean_for_json(i) for i in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    def load_npc(self, npc_name: str):
        if npc_name in self.npc_cache:
            return self.npc_cache[npc_name]

        npc_path = self.find_npc_path(npc_name)
        try:
            if npc_path:
                connection = sqlite3.connect(self.db_path)
                npc = load_npc_from_file(npc_path, db_conn=connection)
                self.npc_cache[npc_name] = npc
                return npc
            else:
                raise FileNotFoundError(f"NPC file not found for {npc_name}")
        except Exception as e:
            raise RuntimeError(f"Error loading NPC {npc_name}: {e}")

    def find_npc_path(self, npc_name: str) -> str:
        for root, _, files in os.walk(self.npc_root_dir):
            print(f"Checking in directory: {root}")  # Debug output
            for file in files:
                if file.startswith(npc_name) and file.endswith(".npc"):
                    print(f"Found NPC file: {file} at {root}")  # Debug output
                    return os.path.join(root, file)
        print(f"NPC file not found for: {npc_name}")  # Debug output
        return None


def load_npc_from_file(npc_file: str, db_conn: sqlite3.Connection) -> NPC:
    # print(npc_file)
    if not npc_file.endswith(".npc"):
        # append it just incase
        name += ".npc"

    try:
        if not os.path.isabs(npc_file):
            npc_file = os.path.abspath(npc_file)

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


def load_tools_from_directory(directory) -> list:
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
                            print(
                                f"Tool file {filename} is invalid or empty. Skipping."
                            )
                            continue
                        tool = Tool(tool_data)
                        tools.append(tool)
                    except yaml.YAMLError as e:
                        print(f"Error parsing tool {filename}: {e}")
    # else:
    # print(f"Tools directory not found: {directory}")
    return tools


import pandas as pd
import yaml
from typing import List, Dict, Any, Union


class NPCSQLOperations(NPCCompiler):
    def __init__(self, npc_directory, db_path):
        super().__init__(npc_directory, db_path)

    def _get_context(
        self, df: pd.DataFrame, context: Union[str, Dict, List[str]]
    ) -> str:
        """Resolve context from different sources"""
        if isinstance(context, str):
            # Check if it's a column reference
            if context in df.columns:
                return df[context].to_string()
            # Assume it's static text
            return context
        elif isinstance(context, list):
            # List of column names to include
            return " ".join(df[col].to_string() for col in context if col in df.columns)
        elif isinstance(context, dict):
            # YAML-style context
            return yaml.dump(context)
        return ""

    # SINGLE PROMPT OPERATIONS
    def synthesize(
        self,
        query,
        df: pd.DataFrame,
        columns: List[str],
        npc: str,
        context: Union[str, Dict, List[str]],
        framework: str,
    ) -> pd.Series:
        context_text = self._get_context(df, context)

        def apply_synthesis(row):
            # we have f strings from the query, we want to fill those back in in the request
            request = query.format(**row[columns])
            prompt = f"""Framework: {framework}
                        Context: {context_text}
                        Text to synthesize: {request}
                        Synthesize the above text."""

            result = self.execute_stage(
                {"step_name": "synthesize", "npc": npc, "task": prompt},
                {},
                self.jinja_env,
            )

            return result[0]["response"]

        # columns a list
        columns_str = "_".join(columns)
        df_out = df[columns].apply(apply_synthesis, axis=1)
        return df_out

    # MULTI-PROMPT/PARALLEL OPERATIONS
    def spread_and_sync(
        self,
        df: pd.DataFrame,
        column: str,
        npc: str,
        variations: List[str],
        sync_strategy: str,
        context: Union[str, Dict, List[str]],
    ) -> pd.Series:
        context_text = self._get_context(df, context)

        def apply_spread_sync(text):
            results = []
            for variation in variations:
                prompt = f"""Variation: {variation}
                            Context: {context_text}
                            Text to analyze: {text}
                            Analyze the above text with {variation} perspective."""

                result = self.execute_stage(
                    {"step_name": f"spread_{variation}", "npc": npc, "task": prompt},
                    {},
                    self.jinja_env,
                )

                results.append(result[0]["response"])

            # Sync results
            sync_result = self.aggregate_step_results(
                [{"response": r} for r in results], sync_strategy
            )

            return sync_result

        return df[column].apply(apply_spread_sync)
        # COMPARISON OPERATIONS

    def contrast(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str,
        npc: str,
        context: Union[str, Dict, List[str]],
        comparison_framework: str,
    ) -> pd.Series:
        context_text = self._get_context(df, context)

        def apply_contrast(row):
            prompt = f"""Framework: {comparison_framework}
                        Context: {context_text}
                        Text 1: {row[col1]}
                        Text 2: {row[col2]}
                        Compare and contrast the above texts."""

            result = self.execute_stage(
                {"step_name": "contrast", "npc": npc, "task": prompt},
                {},
                self.jinja_env,
            )

            return result[0]["response"]

        return df.apply(apply_contrast, axis=1)

    def sql_operations(self, sql: str) -> pd.DataFrame:
        # Execute the SQL query

        """
        1. delegate(COLUMN, npc, query, context, tools, reviewers)
        2. dilate(COLUMN, npc, query, context, scope, reviewers)
        3. erode(COLUMN, npc, query, context, scope, reviewers)
        4. strategize(COLUMN, npc, query, context, timeline, constraints)
        5. validate(COLUMN, npc, query, context, criteria)
        6. synthesize(COLUMN, npc, query, context, framework)
        7. decompose(COLUMN, npc, query, context, granularity)
        8. criticize(COLUMN, npc, query, context, framework)
        9. summarize(COLUMN, npc, query, context, style)
        10. advocate(COLUMN, npc, query, context, perspective)

        MULTI-PROMPT/PARALLEL OPERATIONS
        11. spread_and_sync(COLUMN, npc, query, variations, sync_strategy, context)
        12. bootstrap(COLUMN, npc, query, sample_params, sync_strategy, context)
        13. resample(COLUMN, npc, query, variation_strategy, sync_strategy, context)

        COMPARISON OPERATIONS
        14. mediate(COL1, COL2, npc, query, context, resolution_strategy)
        15. contrast(COL1, COL2, npc, query, context, comparison_framework)
        16. reconcile(COL1, COL2, npc, query, context, alignment_strategy)

        MULTI-COLUMN INTEGRATION
        17. integrate(COLS[], npc, query, context, integration_method)
        18. harmonize(COLS[], npc, query, context, harmony_rules)
        19. orchestrate(COLS[], npc, query, context, workflow)
        """

    # Example usage in SQL-like syntax:
    """
    def execute_sql(self, sql: str) -> pd.DataFrame:
        # This would be implemented to parse and execute SQL with our custom functions
        # Example SQL:
        '''
        SELECT
            customer_id,
            synthesize(feedback_text,
                      npc='analyst',
                      context=customer_segment,
                      framework='satisfaction') as analysis,
            spread_and_sync(price_sensitivity,
                          npc='pricing_agent',
                          variations=['conservative', 'aggressive'],
                          sync_strategy='balanced_analysis',
                          context=market_context) as price_strategy
        FROM customer_data
        '''
        pass
    """


class NPCDBTAdapter:
    def __init__(self, npc_sql: NPCSQLOperations):
        self.npc_sql = npc_sql
        self.models = {}

    def ref(self, model_name: str) -> pd.DataFrame:
        # Implementation for model referencing
        return self.models.get(model_name)

    def parse_model(self, model_sql: str) -> pd.DataFrame:
        # Parse the SQL model and execute with our custom functions
        pass


class AIFunctionParser:
    """Handles parsing and extraction of AI function calls from SQL"""

    @staticmethod
    def extract_function_params(sql: str) -> Dict[str, Dict]:
        """Extract AI function parameters from SQL"""
        ai_functions = {}

        pattern = r"(\w+)\s*\(((?:[^()]*|\([^()]*\))*)\)"
        matches = re.finditer(pattern, sql)

        for match in matches:
            func_name = match.group(1)
            if func_name in ["synthesize", "spread_and_sync"]:
                params = match.group(2).split(",")
                ai_functions[func_name] = {
                    "query": params[0].strip().strip("\"'"),
                    "npc": params[1].strip().strip("\"'"),
                    "context": params[2].strip().strip("\"'"),
                }

        return ai_functions


class SQLModel:
    def __init__(self, name: str, content: str, path: str, npc_directory: str):
        self.name = name
        self.content = content
        self.path = path
        self.npc_directory = npc_directory  # This sets the npc_directory attribute

        self.dependencies = self._extract_dependencies()
        self.has_ai_function = self._check_ai_functions()
        self.ai_functions = self._extract_ai_functions()
        print(f"Initializing SQLModel with NPC directory: {npc_directory}")

    def _extract_dependencies(self) -> Set[str]:
        """Extract model dependencies using ref() calls"""
        pattern = r"\{\{\s*ref\(['\"]([^'\"]+)['\"]\)\s*\}\}"
        return set(re.findall(pattern, self.content))

    def _check_ai_functions(self) -> bool:
        """Check if the model contains AI function calls"""
        ai_functions = [
            "synthesize",
            "spread_and_sync",
            "delegate",
            "dilate",
            "erode",
            "strategize",
            "validate",
            "decompose",
            "criticize",
            "summarize",
            "advocate",
            "bootstrap",
            "resample",
            "mediate",
            "contrast",
            "reconcile",
            "integrate",
            "harmonize",
            "orchestrate",
        ]
        return any(func in self.content for func in ai_functions)

    def _extract_ai_functions(self) -> Dict[str, Dict]:
        """Extract all AI functions and their parameters from the SQL content."""
        ai_functions = {}
        pattern = r"(\w+)\s*\(((?:[^()]*|\([^()]*\))*)\)"
        matches = re.finditer(pattern, self.content)

        for match in matches:
            func_name = match.group(1)
            if func_name in [
                "synthesize",
                "spread_and_sync",
                "delegate",
                "dilate",
                "erode",
                "strategize",
                "validate",
                "decompose",
                "criticize",
                "summarize",
                "advocate",
                "bootstrap",
                "resample",
                "mediate",
                "contrast",
                "reconcile",
                "integrate",
                "harmonize",
                "orchestrate",
            ]:
                params = [
                    param.strip().strip("\"'") for param in match.group(2).split(",")
                ]
                npc = params[1]
                if not npc.endswith(".npc"):
                    npc = npc.replace(".npc", "")
                if self.npc_directory in npc:
                    npc = npc.replace(self.npc_directory, "")

                # print(npc)
                ai_functions[func_name] = {
                    "column": params[0],
                    "npc": npc,
                    "query": params[2],
                    "context": params[3] if len(params) > 3 else None,
                }
        return ai_functions


class ModelCompiler:
    def __init__(self, models_dir: str, db_path: str, npc_directory: str):
        self.models_dir = Path(models_dir)
        self.db_path = db_path
        self.models: Dict[str, SQLModel] = {}
        self.npc_operations = NPCSQLOperations(npc_directory, db_path)
        self.npc_directory = npc_directory

    def discover_models(self):
        """Discover all SQL models in the models directory"""
        self.models = {}
        for sql_file in self.models_dir.glob("**/*.sql"):
            model_name = sql_file.stem
            with open(sql_file, "r") as f:
                content = f.read()
            self.models[model_name] = SQLModel(
                model_name, content, str(sql_file), self.npc_directory
            )
            print(f"Discovered model: {model_name}")
        return self.models

    def build_dag(self) -> Dict[str, Set[str]]:
        """Build dependency graph"""
        dag = {}
        for model_name, model in self.models.items():
            dag[model_name] = model.dependencies
        print(f"Built DAG: {dag}")
        return dag

    def topological_sort(self) -> List[str]:
        """Generate execution order using topological sort"""
        dag = self.build_dag()
        in_degree = defaultdict(int)

        for node, deps in dag.items():
            for dep in deps:
                in_degree[dep] += 1
                if dep not in dag:
                    dag[dep] = set()

        queue = deque([node for node in dag.keys() if len(dag[node]) == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for dependent, deps in dag.items():
                if node in deps:
                    deps.remove(node)
                    if len(deps) == 0:
                        queue.append(dependent)

        if len(result) != len(dag):
            raise ValueError("Circular dependency detected")

        print(f"Execution order: {result}")
        return result

    def _replace_model_references(self, sql: str) -> str:
        ref_pattern = r"\{\{\s*ref\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}"

        def replace_ref(match):
            model_name = match.group(1)
            if model_name not in self.models:
                raise ValueError(
                    f"Model '{model_name}' not found during ref replacement."
                )
            return model_name

        replaced_sql = re.sub(ref_pattern, replace_ref, sql)
        return replaced_sql

    def compile_model(self, model_name: str) -> str:
        """Compile a single model, resolving refs."""
        model = self.models[model_name]
        compiled_sql = model.content
        compiled_sql = self._replace_model_references(compiled_sql)
        print(f"Compiled SQL for {model_name}:\n{compiled_sql}")
        return compiled_sql

    def _extract_base_query(self, sql: str) -> str:
        for dep in self.models[self.current_model].dependencies:
            sql = sql.replace(f"{{{{ ref('{dep}') }}}}", dep)

        parts = sql.split("FROM", 1)
        if len(parts) != 2:
            raise ValueError("Invalid SQL syntax")

        select_part = parts[0].replace("SELECT", "").strip()
        from_part = "FROM" + parts[1]

        columns = re.split(r",\s*(?![^()]*\))", select_part.strip())

        final_columns = []
        for col in columns:
            if "synthesize(" not in col:
                final_columns.append(col)
            else:
                alias_match = re.search(r"as\s+(\w+)\s*$", col, re.IGNORECASE)
                if alias_match:
                    final_columns.append(f"NULL as {alias_match.group(1)}")

        final_sql = f"SELECT {', '.join(final_columns)} {from_part}"
        print(f"Extracted base query:\n{final_sql}")

        return final_sql

    def execute_model(self, model_name: str) -> pd.DataFrame:
        """Execute a model and materialize it to the database"""
        self.current_model = model_name
        model = self.models[model_name]
        compiled_sql = self.compile_model(model_name)

        try:
            if model.has_ai_function:
                df = self._execute_ai_model(compiled_sql, model)
            else:
                df = self._execute_standard_sql(compiled_sql)

            self._materialize_to_db(model_name, df)
            return df

        except Exception as e:
            print(f"Error executing model {model_name}: {str(e)}")
            raise

    def _execute_standard_sql(self, sql: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            try:
                sql = re.sub(r"--.*?\n", "\n", sql)
                sql = re.sub(r"\s+", " ", sql).strip()
                return pd.read_sql(sql, conn)
            except Exception as e:
                print(f"Failed to execute SQL: {sql}")
                print(f"Error: {str(e)}")
                raise

    def execute_ai_function(self, query, npc, column_value, context):
        """Execute a specific AI function logic - placeholder"""
        print(f"Executing AI function on value: {column_value}")
        synthesized_value = (
            f"Processed({query}): {column_value} in context {context} with npc {npc}"
        )
        return synthesized_value

    def _execute_ai_model(self, sql: str, model: SQLModel) -> pd.DataFrame:
        try:
            base_sql = self._extract_base_query(sql)
            print(f"Executing base SQL:\n{base_sql}")
            df = self._execute_standard_sql(base_sql)

            # extract the columns they are between {} pairs
            columns = re.findall(r"\{([^}]+)\}", sql)

            # Handle AI function a
            for func_name, params in model.ai_functions.items():
                if func_name == "synthesize":
                    query_template = params["query"]

                    npc = params["npc"]
                    # only take the after the split "/"
                    npc = npc.split("/")[-1]
                    context = params["context"]
                    # Call the synthesize method using DataFrame directly
                    synthesized_df = self.npc_operations.synthesize(
                        query=query_template,  # The raw query to format
                        df=df,  # The DataFrame containing the data
                        columns=columns,  # The column(s) used to format the query
                        npc=npc,  # NPC parameter
                        context=context,  # Context parameter
                        framework="default_framework",  # Adjust this as per your needs
                    )

                    # Optionally pull the synthesized data into a new column
                    df[
                        "ai_analysis"
                    ] = synthesized_df  # Adjust as per what synthesize returns

            return df

        except Exception as e:
            print(f"Error in AI model execution: {str(e)}")
            raise

    def _materialize_to_db(self, model_name: str, df: pd.DataFrame):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {model_name}")
            df.to_sql(model_name, conn, index=False)
            print(f"Materialized model {model_name} to database")

    def _table_exists(self, table_name: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name=?;
            """,
                (table_name,),
            )
            return cursor.fetchone() is not None

    def run_all_models(self):
        """Execute all models in dependency order"""
        self.discover_models()
        execution_order = self.topological_sort()
        print(f"Running models in order: {execution_order}")

        results = {}
        for model_name in execution_order:
            print(f"\nExecuting model: {model_name}")

            model = self.models[model_name]
            for dep in model.dependencies:
                if not self._table_exists(dep):
                    raise ValueError(
                        f"Dependency {dep} not found in database for model {model_name}"
                    )

            results[model_name] = self.execute_model(model_name)

        return results


def create_example_models(
    models_dir: str = os.path.abspath("./npc_team/factory/models/"),
    db_path: str = "~/npcsh_history.db",
    npc_directory: str = "./npc_team/",
):
    """Create example SQL model files"""
    os.makedirs(os.path.abspath("./npc_team/factory/"), exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    db_path = os.path.expanduser(db_path)
    conn = sqlite3.connect(db_path)
    df = pd.DataFrame(
        {
            "feedback": ["Great product!", "Could be better", "Amazing service"],
            "customer_id": [1, 2, 3],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )

    df.to_sql("raw_customer_feedback", conn, index=False, if_exists="replace")
    print("Created raw_customer_feedback table")

    compiler = ModelCompiler(models_dir, db_path, npc_directory)
    results = compiler.run_all_models()

    for model_name, df in results.items():
        print(f"\nResults for {model_name}:")
        print(df.head())

    customer_feedback = """
    SELECT
        feedback,
        customer_id,
        timestamp
    FROM raw_customer_feedback
    WHERE LENGTH(feedback) > 10;
    """

    customer_insights = """
    SELECT
        customer_id,
        feedback,
        timestamp,
        synthesize(
            "feedback text: {feedback}",
            "analyst",
            "feedback_analysis"
        ) as ai_analysis
    FROM {{ ref('customer_feedback') }};
    """

    models = {
        "customer_feedback.sql": customer_feedback,
        "customer_insights.sql": customer_insights,
    }

    for name, content in models.items():
        path = os.path.join(models_dir, name)
        with open(path, "w") as f:
            f.write(content)
        print(f"Created model: {name}")
