import subprocess
import sqlite3
import numpy as np
import os
import yaml
from jinja2 import Environment, FileSystemLoader, Template, Undefined
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Set
import matplotlib.pyplot as plt
import json
import pathlib
import fnmatch
import re
import ast
import random
from datetime import datetime
import hashlib
from collections import defaultdict, deque

# Importing functions
from .llm_funcs import (
    get_llm_response,
    get_stream,
    process_data_output,
    get_data_response,
    generate_image,
    check_llm_command,
    handle_tool_call,
    execute_llm_command,
)
from .helpers import get_npc_path
from .search import search_web, rag_search
from .image import capture_screenshot, analyze_image_base


def create_or_replace_table(db_path: str, table_name: str, data: pd.DataFrame):
    """
    Creates or replaces a table in the SQLite database.

    :param db_path: Path to the SQLite database.
    :param table_name: Name of the table to create/replace.
    :param data: Pandas DataFrame containing the data to insert.
    """
    conn = sqlite3.connect(db_path)
    try:
        data.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Table '{table_name}' created/replaced successfully.")
    except Exception as e:
        print(f"Error creating/replacing table '{table_name}': {e}")
    finally:
        conn.close()


def load_npc_team(template_path):
    """
    Load an NPC team from a template directory.

    Args:
        template_path: Path to the NPC team template directory

    Returns:
        A dictionary containing the NPC team definition with loaded NPCs and tools
    """
    template_path = os.path.expanduser(template_path)

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template directory not found: {template_path}")

    # Initialize team structure
    npc_team = {
        "name": os.path.basename(template_path),
        "npcs": [],
        "tools": [],
        "assembly_lines": [],
        "sql_models": [],
        "jobs": [],
    }

    # Load NPCs
    npc_objects = {}
    db_conn = sqlite3.connect(os.path.expanduser("~/npcsh_history.db"))

    for filename in os.listdir(template_path):
        if filename.endswith(".npc"):
            npc_path = os.path.join(template_path, filename)

            with open(npc_path, "r") as f:
                npc_content = f.read()
                npc_data = yaml.safe_load(npc_content)
                npc_team["npcs"].append(npc_data)

                # Load as NPC object

                npc_obj = load_npc_from_file(npc_path, db_conn)
                npc_name = npc_data.get("name", os.path.splitext(filename)[0])
                npc_objects[npc_name] = npc_obj

    # Load tools
    tools_dir = os.path.join(template_path, "tools")
    tool_objects = {}

    if os.path.exists(tools_dir):
        for filename in os.listdir(tools_dir):
            if filename.endswith(".tool"):
                tool_path = os.path.join(tools_dir, filename)
                with open(tool_path, "r") as f:
                    tool_content = f.read()
                    tool_data = yaml.safe_load(tool_content)
                    npc_team["tools"].append(tool_data)

                    # Load as Tool object
                    try:
                        tool_obj = Tool(tool_data)
                        tool_name = tool_data.get(
                            "tool_name", os.path.splitext(filename)[0]
                        )
                        tool_objects[tool_name] = tool_obj
                    except Exception as e:
                        print(f"Warning: Could not load tool {filename}: {str(e)}")

    # Load assembly lines
    assembly_lines_dir = os.path.join(template_path, "assembly_lines")
    if os.path.exists(assembly_lines_dir):
        for filename in os.listdir(assembly_lines_dir):
            if filename.endswith(".pipe"):
                pipe_path = os.path.join(assembly_lines_dir, filename)
                with open(pipe_path, "r") as f:
                    pipe_content = f.read()
                    pipe_data = yaml.safe_load(pipe_content)
                    npc_team["assembly_lines"].append(pipe_data)

    # Load SQL models
    sql_models_dir = os.path.join(template_path, "sql_models")
    if os.path.exists(sql_models_dir):
        for filename in os.listdir(sql_models_dir):
            if filename.endswith(".sql"):
                sql_path = os.path.join(sql_models_dir, filename)
                with open(sql_path, "r") as f:
                    sql_content = f.read()
                    npc_team["sql_models"].append(
                        {"name": os.path.basename(sql_path), "content": sql_content}
                    )

    # Load jobs
    jobs_dir = os.path.join(template_path, "jobs")
    if os.path.exists(jobs_dir):
        for filename in os.listdir(jobs_dir):
            if filename.endswith(".job"):
                job_path = os.path.join(jobs_dir, filename)
                with open(job_path, "r") as f:
                    job_content = f.read()
                    job_data = yaml.safe_load(job_content)
                    npc_team["jobs"].append(job_data)

    # Add loaded objects to the team structure
    npc_team["npc_objects"] = npc_objects
    npc_team["tool_objects"] = tool_objects
    npc_team["template_path"] = template_path

    return npc_team


def get_template_npc_team(template, template_dir="~/.npcsh/npc_team/templates/"):

    # get the working directory where the

    npc_team = load_npc_team(template_dir + template)
    return npc_team


def generate_npcs_from_area_of_expertise(
    areas_of_expertise,
    context,
    templates: list = None,
    model=None,
    provider=None,
    npc=None,
):

    prompt = f"""
    Here are the areas of expertise that a user requires a team of agents to be developed for.

    {areas_of_expertise}

    Here is some additional context that may be useful:
    {context}

    """
    # print(templates)
    if templates is not None:
        prompt += "the user has also provided the following templates to use as a base for the NPC team:\n"
        for template in templates:
            prompt += f"{template}\n"
        prompt += "your output should use these templates and modify them accordingly. Your response must contain the specific named NPCs included in these templates, with their primary directives adjusted accordingly based on the context and the areas of expertise. any other new npcs should complement these template ones and should not overlap."

    prompt += """
    Now, generate a set of 2-5 NPCs that cover the required areas of expertise and adequatetly incorporate the context provided.
    according to the following framework and return a json response
    {"npc_team":    [
    {
        "name":"name of npc1",
        "primary_directive": "a 2-3 sentence description of the NPCs duties and responsibilities in the second person"
    },
    {
        "name":"name of npc2",
        "primary_directive": "a 2-3 sentence description of the NPCs duties and responsibilities in the second person"
    }
    ]}

    Each npc's name should be one word.
    The npc's primary directive must be essentially an assistant system message, so ensure that when you
    write it, you are writing it in that way.
    For example, here is an npc named 'sibiji' with a primary directive:
    {
        "name":"sibiji",
        "primary_directive": "You are sibiji, the foreman of an NPC team. You are a foundational AI assistant. Your role is to provide basic support and information. Respond to queries concisely and accurately."
    }
    When writing out your response, you must ensure that the agents have distinct areas of
    expertise such that they are not redundant in their abilities. Keeping the agent team
    small is important and we do not wwish to clutter the team with agents that have overlapping
    areas of expertise or responsibilities that make it difficult to know which agent should be
    called upon in a specific situation.


    do not include any additional markdown formatting or leading ```json tags.
    """

    response = get_llm_response(
        prompt, model=model, provider=provider, npc=npc, format="json"
    )
    response = response.get("response").get("npc_team")
    return response


def edit_areas(areas):
    for i, area in enumerate(areas):
        print(f"{i+1}. {area}")

    index = input("Which area would you like to edit? (number or 'c' to continue):   ")
    if index.lower() in ["c", "continue"]:
        return areas
    else:
        index = int(index)
    if 0 <= index < len(areas):
        new_value = input(f"Current value: {areas[index]}. Enter new value: ")
        areas[index] = new_value
    else:
        print("invalid index, please try again")
    return edit_areas(areas)


def delete_areas(areas):
    for i, area in enumerate(areas):
        print(f"{i+1}. {area}")

    index = (
        int(input("Which area would you like to delete? (number or 'c' to continue): "))
        - 1
    )

    if index.lower() in ["c", "continue"]:
        return areas
    if 0 <= index < len(areas):
        del areas[index]

    return delete_areas(areas)


def conjure_team(
    context,
    templates,
    npc=None,
    model=None,
    provider=None,
):
    """
    Function to generate an NPC team using existing templates and identifying additional areas of expertise.

    Args:
        templates: List of template names to use as a base
        context: Description of the project and what the team should do
        npc: The NPC to use for generating the areas (optional)
        model: The model to use for generation (optional)
        provider: The provider to use for generation (optional)

    Returns:
        Dictionary with identified areas of expertise
    """
    teams = []
    for team in templates:
        npc_team = get_template_npc_team(team)
        teams.append(npc_team)

    # Extract existing areas of expertise from templates
    prompt = f"""
                The user has provided the following context:

                {context}
                """

    if templates is not None:
        prompt += f"""
        The user has requested to generate an NPC team using the following templates:

        {templates}

        """

    prompt += """
    Now what is important in generating an NPC team is to ensure that the NPCs are balanced and distinctly necessary.
    Each NPC should essentially focus on a single area of expertise. This does not mean that they should only focus on a
    single function, but rather that they have a specific purview.

    To first figure out what NPCs would be necessary in addition to the templates given the combination of the templates
    and the user-provided context, we will need to generate a list of the abstract areas that the user requires in an NPC team.
    Now, given that information, consider whether other potential areas of expertise would complement the provided templates and the user context?
    Try to think carefully about this in a way to determine what other potential issues might arise for a team like this to anticipate whether it may be
    necessary to cover additional areas of expertise.

    Now, generate a list of 3-5 abstract areas explicitly required.
    It is actually quite important that you consolidate and abstract away various areas
        into general forms. Agents will be generated based on these descriptions, and an agentic team is more
        useful when it is as small as reasonably possible.

    Similarly, generate a list of 2-3 suggested areas of expertise that would complement the existing templates and the user context.

    This will be provided to the user for confirmation and adjustment before the NPC team is generated.

    Return a json response with two lists. It should be formatted like so:

    {
        "explicit_areas": ["area 1", "area 2"],
        "suggested_areas": ["area 3", "area 4"]
    }

    Do not include any additional markdown formatting or leading ```json tags.

    """

    response = get_llm_response(
        prompt, model=model, provider=provider, npc=npc, format="json"
    )

    response = response.get("response")
    explicit_areas = response.get("explicit_areas", [])
    suggested_areas = response.get("suggested_areas", [])
    combined_areas = explicit_areas + suggested_areas
    print("\nExplicit areas of expertise:")
    for i, area in enumerate(explicit_areas):
        print(f"{i+1}. {area}")

    print("\nSuggested areas of expertise:")
    for i, area in enumerate(suggested_areas):
        print(f"{i+1}. {area}")

    user_input = input(
        """\n\n
Above is the generated list of areas of expertise.

Would you like to edit the suggestions, delete any of them, or regenerate the team with revised context?
Type '(e)dit', '(d)elete', or '(r)egenerate' or '(a)ccept': """
    )
    if user_input.lower() in ["e", "edit"]:
        revised_areas = edit_areas(combined_areas)
    elif user_input.lower() in ["d", "delete"]:
        revised_areas = delete_areas(combined_areas)
    elif user_input.lower() in ["r", "regenerate"]:
        updated_context = input(
            f"Here is the context you provided: {context}\nPlease provide a fully revised version: "
        )
        print("Beginning again with updated context")
        return conjure_team(
            updated_context,
            templates=templates,
            npc=npc,
            model=model,
            provider=provider,
        )

    elif user_input.lower() in ["a", "accept"]:
        # Return the finalized areas of expertise
        revised_areas = combined_areas

    # proceed now with generation of npc for each revised area
    npc_out = generate_npcs_from_area_of_expertise(
        revised_areas,
        context,
        templates=[team["npcs"] for team in teams],
        model=model,
        provider=provider,
        npc=npc,
    )
    # print(npc_out)
    # now save all of the npcs to the ./npc_team directory

    for npc in npc_out:
        # make the npc team dir if not existst

        if isinstance(npc, str):
            npc = ast.literal_eval(npc)

        npc_team_dir = os.path.join(os.getcwd(), "npc_team")
        os.makedirs(npc_team_dir, exist_ok=True)
        # print(npc, type(npc))
        npc_path = os.path.join(os.getcwd(), "npc_team", f"{npc['name']}.npc")
        with open(npc_path, "w") as f:
            f.write(yaml.dump(npc))

    return {
        "templates": templates,
        "context": context,
        "expertise_areas": response,
        "npcs": npc_out,
    }


def initialize_npc_project(
    directory=None,
    templates=None,
    context=None,
    model=None,
    provider=None,
) -> str:
    """
    Function Description:
        This function initializes an NPC project in the current directory.
    Args:
        None
    Keyword Args:
        None
    Returns:
        A message indicating the success or failure of the operation.
    """
    if directory is None:
        directory = os.getcwd()

    # Create 'npc_team' folder in current directory
    npc_team_dir = os.path.join(directory, "npc_team")
    os.makedirs(npc_team_dir, exist_ok=True)

    # Create 'foreman.npc' file in 'npc_team' directory
    foreman_npc_path = os.path.join(npc_team_dir, "sibiji.npc")
    if context is not None:
        team = conjure_team(
            context, templates=templates, model=model, provider=provider
        )

    if not os.path.exists(foreman_npc_path):
        foreman_npc_content = """name: sibiji
primary_directive: "You are sibiji, the foreman of an NPC team. You are a foundational AI assistant. Your role is to provide basic support and information. Respond to queries concisely and accurately."
model: llama3.2
provider: ollama
"""
        with open(foreman_npc_path, "w") as f:
            f.write(foreman_npc_content)
    else:
        print(f"{foreman_npc_path} already exists.")

    # Create 'tools' folder within 'npc_team' directory
    tools_dir = os.path.join(npc_team_dir, "tools")
    os.makedirs(tools_dir, exist_ok=True)

    # assembly_lines
    assembly_lines_dir = os.path.join(npc_team_dir, "assembly_lines")
    os.makedirs(assembly_lines_dir, exist_ok=True)
    # sql models
    sql_models_dir = os.path.join(npc_team_dir, "sql_models")
    os.makedirs(sql_models_dir, exist_ok=True)
    # jobs
    jobs_dir = os.path.join(npc_team_dir, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)

    # just copy all the base npcsh tools and npcs.
    return f"NPC project initialized in {npc_team_dir}"


def init_pipeline_runs(db_path: str = "~/npcsh_history.db"):
    """
    Initialize the pipeline runs table in the database.
    """
    with sqlite3.connect(os.path.expanduser(db_path)) as conn:
        cursor = conn.cursor()
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


# SilentUndefined handles undefined behavior in Jinja2
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
        self.description = tool_data.get("description", "")
        self.steps = self.parse_steps(tool_data.get("steps", []))

    def parse_step(self, step: Union[dict, str]) -> dict:
        if isinstance(step, dict):
            return {
                "engine": step.get("engine", None),
                "code": step.get("code", ""),
            }
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
        model: str = None,
        provider: str = None,
        npc=None,
        stream: bool = False,
        messages: List[Dict[str, str]] = None,
    ):
        # Create the context with input values at top level for Jinja access
        context = npc.shared_context.copy() if npc else {}
        context.update(input_values)  # Spread input values directly in context
        context.update(
            {
                "tools": tools_dict,
                "llm_response": None,
                "output": None,
                "command": command,
            }
        )

        # Process Steps
        for i, step in enumerate(self.steps):
            context = self.execute_step(
                step,
                context,
                jinja_env,
                model=model,
                provider=provider,
                npc=npc,
                stream=stream,
                messages=messages,
            )
            # if i is the last step and the user has reuqested a streaming output
            # then we should return the stream
            if i == len(self.steps) - 1 and stream:  # this was causing the big issue X:
                return context
        # Return the final output
        if context.get("output") is not None:
            return context.get("output")
        elif context.get("llm_response") is not None:
            return context.get("llm_response")

    def execute_step(
        self,
        step: dict,
        context: dict,
        jinja_env: Environment,
        npc: Any = None,
        model: str = None,
        provider: str = None,
        stream: bool = False,
        messages: List[Dict[str, str]] = None,
    ):
        engine = step.get("engine", "natural")
        code = step.get("code", "")

        # Render template with all context variables
        try:
            template = jinja_env.from_string(code)
            rendered_code = template.render(**context)
        except Exception as e:
            print(f"Error rendering template: {e}")
            rendered_code = code

        if engine == "natural":
            if len(rendered_code.strip()) > 0:
                # print(f"Executing natural language step: {rendered_code}")
                if stream:
                    messages = messages.copy() if messages else []
                    messages.append({"role": "user", "content": rendered_code})
                    return get_stream(messages, model=model, provider=provider, npc=npc)

                else:
                    llm_response = get_llm_response(
                        rendered_code, model=model, provider=provider, npc=npc
                    )
                    response_text = llm_response.get("response", "")
                    # Store both in context for reference
                    context["llm_response"] = response_text
                    context["results"] = response_text

        elif engine == "python":
            exec_globals = {
                "__builtins__": __builtins__,
                "npc": npc,
                "context": context,
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
            new_locals = {}
            exec_env = context.copy()
            try:
                exec(rendered_code, exec_globals, new_locals)
                exec_env.update(new_locals)
                context.update(exec_env)
                # If output is set, also set it as results
                if "output" in exec_env:
                    if exec_env["output"] is not None:
                        context["results"] = exec_env["output"]
            except NameError as e:
                print(f"NameError: {e} , on the following tool code: ", rendered_code)
            except SyntaxError as e:
                print(f"SyntaxError: {e} , on the following tool code: ", rendered_code)
            except Exception as e:
                print(f"Error executing Python code: {e}")

        return context

    def to_dict(self):
        return {
            "tool_name": self.tool_name,
            "description": self.description,
            "inputs": self.inputs,
            "steps": [self.step_to_dict(step) for step in self.steps],
        }

    def step_to_dict(self, step):
        return {
            "engine": step.get("engine"),
            "code": step.get("code"),
        }


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
    return tools


class NPC:
    def __init__(
        self,
        db_conn: sqlite3.Connection,
        name: str,
        primary_directive: str = None,
        tools: list = None,  # from the npc profile
        model: str = None,
        provider: str = None,
        api_url: str = None,
        all_tools: list = None,  # all available tools in global and project, this is an anti pattern i need to solve eventually but for now it works
        use_global_tools: bool = True,
        use_npc_network: bool = True,
        global_npc_directory: str = None,
        project_npc_directory: str = None,
        global_tools_directory: str = None,
        project_tools_directory: str = None,
    ):
        # 2. Load global tools from ~/.npcsh/npc_team/tools
        if global_tools_directory is None:
            user_home = os.path.expanduser("~")
            self.global_tools_directory = os.path.join(
                user_home, ".npcsh", "npc_team", "tools"
            )
        else:
            self.global_tools_directory = global_tools_directory

        if project_tools_directory is None:
            self.project_tools_directory = os.path.abspath("./npc_team/tools")
        else:
            self.project_tools_directory = project_tools_directory

        if global_npc_directory is None:
            self.global_npc_directory = os.path.join(user_home, ".npcsh", "npc_team")
        else:
            self.global_npc_directory = global_npc_directory

        if project_npc_directory is None:
            self.project_npc_directory = os.path.abspath("./npc_team")

        self.jinja_env = Environment(
            loader=FileSystemLoader(
                [
                    self.project_npc_directory,
                    self.global_npc_directory,
                    self.global_tools_directory,
                    self.project_tools_directory,
                ]
            ),
            undefined=SilentUndefined,
        )

        self.name = name
        self.primary_directive = primary_directive
        self.tools = tools or []

        self.model = model
        self.db_conn = db_conn
        self.tables = self.db_conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table';"
        ).fetchall()

        self.provider = provider
        self.api_url = api_url
        self.all_tools = all_tools or []
        self.all_tools_dict = {tool.tool_name: tool for tool in self.all_tools}
        if self.tools:
            self.tools = self.load_suggested_tools(
                tools,
                self.global_tools_directory,
                self.project_tools_directory,
            )
        self.shared_context = {
            "dataframes": {},
            "current_data": None,
            "computation_results": {},
        }
        self.use_global_tools = use_global_tools
        self.use_npc_network = use_npc_network

        # Load tools if flag is set
        if self.use_global_tools:
            self.default_tools = self.load_tools()
        else:
            self.default_tools = []
        self.npc_cache = {}

        self.resolved_npcs = {}

        # Load NPC dependencies if flag is set
        if self.use_npc_network:
            self.parsed_npcs = self.parse_all_npcs()
            self.resolved_npcs = self.resolve_all_npcs()
        else:
            self.parsed_npcs = []

    def get_memory(self):
        return

    def to_dict(self):
        return {
            "name": self.name,
            "primary_directive": self.primary_directive,
            "model": self.model,
            "provider": self.provider,
            "tools": [tool.to_dict() for tool in self.tools],
            "use_global_tools": self.use_global_tools,
            "api_url": self.api_url,
        }

    def _check_llm_command(
        self,
        command,
        retrieved_docs=None,
        messages=None,
        n_docs=5,
    ):
        return check_llm_command(
            command,
            model=self.model,
            provider=self.provider,
            npc=self,
            retrieved_docs=retrieved_docs,
            messages=messages,
            n_docs=n_docs,
        )

    def handle_agent_pass(
        self,
        npc_to_pass: Any,
        command: str,
        messages: List[Dict[str, str]] = None,
        retrieved_docs=None,
        n_docs: int = 5,
    ) -> Union[str, Dict[str, Any]]:
        """
        Function Description:
            This function handles an agent pass.
        Args:
            command (str): The command.

        Keyword Args:
            model (str): The model to use for handling the agent pass.
            provider (str): The provider to use for handling the agent pass.
            messages (List[Dict[str, str]]): The list of messages.
            npc (Any): The NPC object.
            retrieved_docs (Any): The retrieved documents.
            n_docs (int): The number of documents.
        Returns:
            Union[str, Dict[str, Any]]: The result of handling the agent pass.
        """
        # print(npc_to_pass, command)

        target_npc = self.get_npc(npc_to_pass)
        if target_npc is None:
            return "NPC not found."

        # initialize them as an actual NPC
        npc_to_pass_init = NPC(self.db_conn, **target_npc)
        # print(npc_to_pass_init, command)
        updated_command = (
            command
            + "/n"
            + f"""

            NOTE: THIS COMMAND HAS ALREADY BEEN PASSED FROM ANOTHER NPC
            TO YOU, {npc_to_pass}.

            THUS YOU WILL LIKELY NOT NEED TO PASS IT AGAIN TO YOURSELF
            OR TO ANOTHER NPC. pLEASE CHOOSE ONE OF THE OTHER OPTIONS WHEN
            RESPONDING.


        """
        )
        return npc_to_pass_init._check_llm_command(
            updated_command,
            retrieved_docs=retrieved_docs,
            messages=messages,
            n_docs=n_docs,
        )

    def get_npc(self, npc_name: str):
        if npc_name + ".npc" in self.npc_cache:
            return self.npc_cache[npc_name + ".npc"]

    def load_suggested_tools(
        self,
        tools: list,
        global_tools_directory: str,
        project_tools_directory: str,
    ) -> List[Tool]:
        suggested_tools = []
        for tool_name in tools:
            # load tool from file
            if not tool_name.endswith(".tool"):
                tool_name += ".tool"
            if (
                global_tools_directory not in tool_name
                and project_tools_directory not in tool_name
            ):
                # try to load from global tools directory
                try:
                    tool_data = self.load_tool_from_file(
                        os.path.join(global_tools_directory, tool_name)
                    )
                    if tool_data is None:
                        raise ValueError(f"Tool {tool_name} not found.")

                    print(f"Tool {tool_name} loaded from global directory.")

                except ValueError as e:
                    print(f"Error loading tool from global directory: {e}")
                    # trying to load from project tools directory
                    try:
                        tool_data = self.load_tool_from_file(
                            os.path.join(project_tools_directory, tool_name)
                        )
                        if tool_data is None:
                            raise ValueError(f"Tool {tool_name} not found.")
                        print(f"Tool {tool_name} loaded from project directory.")
                    except ValueError as e:
                        print(f"Error loading tool from project directory: {e}")
                        continue

            # print(tool_name)
            # print(tool_data)
            tool = Tool(tool_data)
            self.all_tools.append(tool)
            self.all_tools_dict[tool.tool_name] = tool
            suggested_tools.append(tool)
        return suggested_tools

    def __str__(self):
        return f"NPC: {self.name}\nDirective: {self.primary_directive}\nModel: {self.model}"

    def analyze_db_data(self, request: str):
        return get_data_response(
            request,
            self.db_conn,
            self.tables,
            model=self.model,
            provider=self.provider,
            npc=self,
        )

    def get_llm_response(self, request: str, **kwargs):
        return get_llm_response(
            request, model=self.model, provider=self.provider, npc=self, **kwargs
        )

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

    def parse_all_npcs(self) -> None:
        directories = [self.global_npc_directory, self.project_npc_directory]
        for directory in directories:
            if os.path.exists(directory):
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
        resolved_npcs = []
        for npc_file in self.npc_cache:
            npc = self.resolve_npc_profile(npc_file)
            resolved_npcs.append(npc)
            # print(npc)
        return resolved_npcs

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
            # try to resolve it with load_npc_from_file
            profile = load_npc_from_file(npc_file, self.db_conn).to_dict()

        #    raise ValueError(f"NPC {npc_file} has not been resolved.")

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


class SilentUndefined(Undefined):
    def _fail_with_undefined_error(self, *args, **kwargs):
        return ""


# perhaps the npc compiling is more than just for jinja reasons.
# we can turn each agent into a referenceable program executable.


# finish testing out a python based version rather than jinja only
class NPCCompiler:
    def __init__(
        self,
        npc_directory,
        db_path,
    ):
        self.npc_directory = npc_directory
        self.dirs = [self.npc_directory]
        # import pdb
        self.is_global_dir = self.npc_directory == os.path.expanduser(
            "~/.npcsh/npc_team/"
        )

        # pdb.set_trace()
        if self.is_global_dir:
            self.project_npc_directory = None
            self.project_tools_directory = None
        else:
            self.project_npc_directory = npc_directory
            self.project_tools_directory = os.path.join(
                self.project_npc_directory, "tools"
            )
            self.dirs.append(self.project_npc_directory)

        self.db_path = db_path
        self.npc_cache = {}
        self.resolved_npcs = {}
        self.pipe_cache = {}

        # Set tools directories
        self.global_tools_directory = os.path.join(
            os.path.expanduser("~/.npcsh/npc_team/"), "tools"
        )

        # Initialize Jinja environment with multiple loaders
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.dirs),
            undefined=SilentUndefined,
        )

        self.all_tools_dict = self.load_tools()
        self.all_tools = list(self.all_tools_dict.values())

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

        self.parse_all_npcs()
        # Resolve NPCs
        self.resolve_all_npcs()

        # Finalize NPC profile
        # print(npc_file)
        # print(npc_file, "npc_file")
        parsed_content = self.finalize_npc_profile(npc_file)

        # Load tools from both global and project directories
        parsed_content["tools"] = [tool.to_dict() for tool in self.all_tools]

        self.update_compiled_npcs_table(npc_file, parsed_content)
        return parsed_content

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
        if self.project_tools_directory is not None:
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

        return tool_dict

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
        # print(self.dirs)
        for directory in self.dirs:
            if os.path.exists(directory):

                for filename in os.listdir(directory):
                    if filename.endswith(".npc"):
                        npc_path = os.path.join(directory, filename)
                        self.parse_npc_file(npc_path)

    def parse_npc_file(self, npc_file_path: str) -> dict:
        npc_file = os.path.basename(npc_file_path)
        if npc_file in self.npc_cache:
            # Project NPCs override global NPCs
            if self.project_npc_directory is not None:
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
            # try to resolve it with load_npc_from_file
            profile = load_npc_from_file(
                npc_file, sqlite3.connect(self.db_path)
            ).to_dict()

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


def load_npc_from_file(npc_file: str, db_conn: sqlite3.Connection) -> NPC:
    if not npc_file.endswith(".npc"):
        # append it just incase
        name += ".npc"

    try:
        if "~" in npc_file:
            npc_file = os.path.expanduser(npc_file)
        if not os.path.isabs(npc_file):
            npc_file = os.path.abspath(npc_file)

        with open(npc_file, "r") as f:
            npc_data = yaml.safe_load(f)

        # Extract fields from YAML
        name = npc_data["name"]

        primary_directive = npc_data.get("primary_directive")
        tools = npc_data.get("tools")
        model = npc_data.get("model", os.environ.get("NPCSH_CHAT_MODEL", "llama3.2"))
        provider = npc_data.get(
            "provider", os.environ.get("NPCSH_CHAT_PROVIDER", "ollama")
        )
        api_url = npc_data.get("api_url", os.environ.get("NPCSH_API_URL", None))
        use_global_tools = npc_data.get("use_global_tools", True)
        # print(use_global_tools)
        # Load tools from global and project-specific directories
        all_tools = []
        # 1. Load tools defined within the NPC profile
        if "tools" in npc_data:
            for tool_data in npc_data["tools"]:
                tool = Tool(tool_data)
                tools.append(tool)
        # 2. Load global tools from ~/.npcsh/npc_team/tools
        user_home = os.path.expanduser("~")
        global_tools_directory = os.path.join(user_home, ".npcsh", "npc_team", "tools")
        all_tools.extend(load_tools_from_directory(global_tools_directory))
        # 3. Load project-specific tools from ./npc_team/tools
        project_tools_directory = os.path.abspath("./npc_team/tools")
        all_tools.extend(load_tools_from_directory(project_tools_directory))

        # Remove duplicates, giving precedence to project-specific tools
        tool_dict = {}
        for tool in all_tools:
            tool_dict[tool.tool_name] = tool  # Project tools overwrite global tools

        all_tools = list(tool_dict.values())

        # Initialize and return the NPC object
        return NPC(
            db_conn,
            name,
            primary_directive=primary_directive,
            tools=tools,
            use_global_tools=use_global_tools,
            model=model,
            provider=provider,
            api_url=api_url,
            all_tools=all_tools,  # Pass the tools
        )

    except FileNotFoundError:
        raise ValueError(f"NPC file not found: {npc_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML in NPC file {npc_file}: {str(e)}")
    except KeyError as e:
        raise ValueError(f"Missing required key in NPC file {npc_file}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading NPC from file {npc_file}: {str(e)}")


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


###
###
###
###
### What is a pipeline file?
"""

steps:
  - step_name: "step_name"
    npc: npc_name
    task: "task"
    tools: ['tool1', 'tool2']


# results within the pipeline need to be referenceable by the shared context through the step name
#
# so if step name is review_email and a tool is called we can refer to the intermediate objects
# as review_email['tool1']['{var_name_in_tool_definition'}]

so in step 2 i can do in the task
 task: "sort the emails by tone by reviewing the outputs from the email review tool: {{ review_email['email_review']['tone'] }}"
"""


"""
adding in context and fabs
"""


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
                    df["ai_analysis"] = (
                        synthesized_df  # Adjust as per what synthesize returns
                    )

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
