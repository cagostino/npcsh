import logging
from typing import List, Dict, Any, Optional
import os
import sqlite3
import subprocess
import platform
import yaml
import nltk
import numpy as np

import filecmp

import shutil
import tempfile
import pandas as pd

try:
    from sentence_transformers import util
except Exception as e:
    print(f"Error importing sentence_transformers: {e}")


def get_shell_config_file() -> str:
    """

    Function Description:
        This function returns the path to the shell configuration file.
    Args:
        None
    Keyword Args:
        None
    Returns:
        The path to the shell configuration file.
    """
    # Check the current shell
    shell = os.environ.get("SHELL", "")

    if "zsh" in shell:
        return os.path.expanduser("~/.zshrc")
    elif "bash" in shell:
        # On macOS, use .bash_profile for login shells
        if platform.system() == "Darwin":
            return os.path.expanduser("~/.bash_profile")
        else:
            return os.path.expanduser("~/.bashrc")
    else:
        # Default to .bashrc if we can't determine the shell
        return os.path.expanduser("~/.bashrc")


def initial_table_print(cursor: sqlite3.Cursor) -> None:
    """
    Function Description:
        This function is used to print the initial table.
    Args:
        cursor : sqlite3.Cursor : The SQLite cursor.
    Keyword Args:
        None
    Returns:
        None
    """

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name != 'command_history'"
    )
    tables = cursor.fetchall()

    print("\nAvailable tables:")
    for i, table in enumerate(tables, 1):
        print(f"{i}. {table[0]}")


def ensure_npcshrc_exists() -> str:
    """
    Function Description:
        This function ensures that the .npcshrc file exists in the user's home directory.
    Args:
        None
    Keyword Args:
        None
    Returns:
        The path to the .npcshrc file.
    """

    npcshrc_path = os.path.expanduser("~/.npcshrc")
    if not os.path.exists(npcshrc_path):
        with open(npcshrc_path, "w") as npcshrc:
            npcshrc.write("# NPCSH Configuration File\n")
            npcshrc.write("export NPCSH_INITIALIZED=0\n")
            npcshrc.write("export NPCSH_DEFAULT_MODE='chat'\n")
            npcshrc.write("export NPCSH_CHAT_PROVIDER='ollama'\n")
            npcshrc.write("export NPCSH_CHAT_MODEL='llama3.2'\n")
            npcshrc.write("export NPCSH_REASONING_PROVIDER='ollama'\n")
            npcshrc.write("export NPCSH_REASONING_MODEL='deepseek-r1'\n")

            npcshrc.write("export NPCSH_EMBEDDING_PROVIDER='ollama'\n")
            npcshrc.write("export NPCSH_EMBEDDING_MODEL='nomic-embed-text'\n")
            npcshrc.write("export NPCSH_VISION_PROVIDER='ollama'\n")
            npcshrc.write("export NPCSH_VISION_MODEL='llava7b'\n")
            npcshrc.write(
                "export NPCSH_IMAGE_GEN_MODEL='runwayml/stable-diffusion-v1-5'\n"
            )

            npcshrc.write("export NPCSH_IMAGE_GEN_PROVIDER='diffusers'\n")

            npcshrc.write("export NPCSH_API_URL=''\n")
            npcshrc.write("export NPCSH_DB_PATH='~/npcsh_history.db'\n")
            npcshrc.write("export NPCSH_VECTOR_DB_PATH='~/npcsh_chroma.db'\n")
            npcshrc.write("export NPCSH_STREAM_OUTPUT=0")
    return npcshrc_path


# Function to check and download NLTK data if necessary
def ensure_nltk_punkt() -> None:
    """
    Function Description:
        This function ensures that the NLTK 'punkt' tokenizer is downloaded.
    Args:
        None
    Keyword Args:
        None
    Returns:
        None
    """

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download("punkt")


def load_all_files(
    directory: str, extensions: List[str] = None, depth: int = 1
) -> Dict[str, str]:
    """
    Function Description:
        This function loads all text files in a directory and its subdirectories.
    Args:
        directory: The directory to search.
    Keyword Args:
        extensions: A list of file extensions to include.
        depth: The depth of subdirectories to search.
    Returns:
        A dictionary with file paths as keys and file contents as values.
    """
    text_data = {}
    if depth < 1:
        return text_data  # Reached the specified depth, stop recursion.

    if extensions is None:
        # Default to common text file extensions
        extensions = [
            ".txt",
            ".md",
            ".py",
            ".java",
            ".c",
            ".cpp",
            ".html",
            ".css",
            ".js",
            ".ts",
            ".tsx",
            ".npc",
            # Add more extensions if needed
        ]

    try:
        # List all entries in the directory
        entries = os.listdir(directory)
    except Exception as e:
        print(f"Could not list directory {directory}: {e}")
        return text_data

    for entry in entries:
        path = os.path.join(directory, entry)
        if os.path.isfile(path):
            if any(path.endswith(ext) for ext in extensions):
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as file:
                        text_data[path] = file.read()
                except Exception as e:
                    print(f"Could not read file {path}: {e}")
        elif os.path.isdir(path):
            # Recurse into subdirectories, decreasing depth by 1
            subdir_data = load_all_files(path, extensions, depth=depth - 1)
            text_data.update(subdir_data)

    return text_data


def add_npcshrc_to_shell_config() -> None:
    """
    Function Description:
        This function adds the sourcing of the .npcshrc file to the user's shell configuration file.
    Args:
        None
    Keyword Args:
        None
    Returns:
        None
    """

    if os.getenv("NPCSH_INITIALIZED") is not None:
        return
    config_file = get_shell_config_file()
    npcshrc_line = "\n# Source NPCSH configuration\nif [ -f ~/.npcshrc ]; then\n    . ~/.npcshrc\nfi\n"

    with open(config_file, "a+") as shell_config:
        shell_config.seek(0)
        content = shell_config.read()
        if "source ~/.npcshrc" not in content and ". ~/.npcshrc" not in content:
            shell_config.write(npcshrc_line)
            print(f"Added .npcshrc sourcing to {config_file}")
        else:
            print(f".npcshrc already sourced in {config_file}")


def setup_npcsh_config() -> None:
    """
    Function Description:
        This function initializes the NPCSH configuration.
    Args:
        None
    Keyword Args:
        None
    Returns:
        None
    """

    ensure_npcshrc_exists()
    add_npcshrc_to_shell_config()


def is_npcsh_initialized() -> bool:
    """
    Function Description:
        This function checks if the NPCSH initialization flag is set.
    Args:
        None
    Keyword Args:
        None
    Returns:
        A boolean indicating whether NPCSH is initialized.
    """

    return os.environ.get("NPCSH_INITIALIZED", None) == "1"


def set_npcsh_initialized() -> None:
    """
    Function Description:
        This function sets the NPCSH initialization flag in the .npcshrc file.
    Args:
        None
    Keyword Args:
        None
    Returns:

        None
    """

    npcshrc_path = ensure_npcshrc_exists()

    with open(npcshrc_path, "r+") as npcshrc:
        content = npcshrc.read()
        if "export NPCSH_INITIALIZED=0" in content:
            content = content.replace(
                "export NPCSH_INITIALIZED=0", "export NPCSH_INITIALIZED=1"
            )
            npcshrc.seek(0)
            npcshrc.write(content)
            npcshrc.truncate()

    # Also set it for the current session
    os.environ["NPCSH_INITIALIZED"] = "1"
    print("NPCSH initialization flag set in .npcshrc")


def get_directory_npcs(directory: str = None) -> List[str]:
    """
    Function Description:
        This function retrieves a list of valid NPCs from the database.
    Args:
        db_path: The path to the database file.
    Keyword Args:
        None
    Returns:
        A list of valid NPCs.
    """
    if directory is None:
        directory = os.path.expanduser("./npc_team")
    npcs = []
    for filename in os.listdir(directory):
        if filename.endswith(".npc"):
            npcs.append(filename[:-4])
    return npcs


def get_db_npcs(db_path: str) -> List[str]:
    """
    Function Description:
        This function retrieves a list of valid NPCs from the database.
    Args:
        db_path: The path to the database file.
    Keyword Args:
        None
    Returns:
        A list of valid NPCs.
    """
    if "~" in db_path:
        db_path = os.path.expanduser(db_path)
    db_conn = sqlite3.connect(db_path)
    cursor = db_conn.cursor()
    cursor.execute("SELECT name FROM compiled_npcs")
    npcs = [row[0] for row in cursor.fetchall()]
    db_conn.close()
    return npcs


def get_npc_path(npc_name: str, db_path: str) -> str:
    # First, check in project npc_team directory
    project_npc_team_dir = os.path.abspath("./npc_team")
    project_npc_path = os.path.join(project_npc_team_dir, f"{npc_name}.npc")

    # Then, check in global npc_team directory
    user_npc_team_dir = os.path.expanduser("~/.npcsh/npc_team")
    global_npc_path = os.path.join(user_npc_team_dir, f"{npc_name}.npc")

    # Check database for compiled NPCs
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            query = f"SELECT source_path FROM compiled_npcs WHERE name = '{npc_name}'"
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                return result[0]

    except Exception as e:
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                query = f"SELECT source_path FROM compiled_npcs WHERE name = {npc_name}"
                cursor.execute(query)
                result = cursor.fetchone()
                if result:
                    return result[0]
        except Exception as e:
            print(f"Database query error: {e}")

    # Fallback to file paths
    if os.path.exists(project_npc_path):
        return project_npc_path

    if os.path.exists(global_npc_path):
        return global_npc_path

    raise ValueError(f"NPC file not found: {npc_name}")


def initialize_base_npcs_if_needed(db_path: str) -> None:
    """
    Function Description:
        This function initializes the base NPCs if they are not already in the database.
    Args:
        db_path: The path to the database file.
    Keyword Args:

        None
    Returns:
        None
    """

    if is_npcsh_initialized():
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the compiled_npcs table if it doesn't exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS compiled_npcs (
            name TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            compiled_content TEXT
        )
        """
    )

    # Get the path to the npc_team directory in the package
    package_dir = os.path.dirname(__file__)
    package_npc_team_dir = os.path.join(package_dir, "npc_team")

    # User's global npc_team directory
    user_npc_team_dir = os.path.expanduser("~/.npcsh/npc_team")

    user_tools_dir = os.path.join(user_npc_team_dir, "tools")
    user_templates_dir = os.path.join(user_npc_team_dir, "templates")
    os.makedirs(user_npc_team_dir, exist_ok=True)
    os.makedirs(user_tools_dir, exist_ok=True)
    os.makedirs(user_templates_dir, exist_ok=True)
    # Copy NPCs from package to user directory
    for filename in os.listdir(package_npc_team_dir):
        if filename.endswith(".npc"):
            source_path = os.path.join(package_npc_team_dir, filename)
            destination_path = os.path.join(user_npc_team_dir, filename)
            if not os.path.exists(destination_path) or file_has_changed(
                source_path, destination_path
            ):
                shutil.copy2(source_path, destination_path)

    # Copy tools from package to user directory
    package_tools_dir = os.path.join(package_npc_team_dir, "tools")
    if os.path.exists(package_tools_dir):
        for filename in os.listdir(package_tools_dir):
            if filename.endswith(".tool"):
                source_tool_path = os.path.join(package_tools_dir, filename)
                destination_tool_path = os.path.join(user_tools_dir, filename)
                if (not os.path.exists(destination_tool_path)) or file_has_changed(
                    source_tool_path, destination_tool_path
                ):
                    shutil.copy2(source_tool_path, destination_tool_path)
                    print(f"Copied tool {filename} to {destination_tool_path}")

    templates = os.path.join(package_npc_team_dir, "templates")
    if os.path.exists(templates):
        for folder in os.listdir(templates):
            os.makedirs(os.path.join(user_templates_dir, folder), exist_ok=True)
            for file in os.listdir(os.path.join(templates, folder)):
                if file.endswith(".npc"):
                    source_template_path = os.path.join(templates, folder, file)

                    destination_template_path = os.path.join(
                        user_templates_dir, folder, file
                    )
                    if not os.path.exists(
                        destination_template_path
                    ) or file_has_changed(
                        source_template_path, destination_template_path
                    ):
                        shutil.copy2(source_template_path, destination_template_path)
                        print(f"Copied template {file} to {destination_template_path}")
    conn.commit()
    conn.close()
    set_npcsh_initialized()
    add_npcshrc_to_shell_config()


def file_has_changed(source_path: str, destination_path: str) -> bool:
    """
    Function Description:
        This function compares two files to determine if they are different.
    Args:
        source_path: The path to the source file.
        destination_path: The path to the destination file.
    Keyword Args:
        None
    Returns:
        A boolean indicating whether the files are different
    """

    # Compare file modification times or contents to decide whether to update the file
    return not filecmp.cmp(source_path, destination_path, shallow=False)


def is_valid_npc(npc: str, db_path: str) -> bool:
    """
    Function Description:
        This function checks if an NPC is valid based on the database.
    Args:
        npc: The name of the NPC.
        db_path: The path to the database file.
    Keyword Args:
        None
    Returns:
        A boolean indicating whether the NPC is valid.
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM compiled_npcs WHERE name = ?", (npc,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def execute_python(code: str) -> str:
    """
    Function Description:
        This function executes Python code and returns the output.
    Args:
        code: The Python code to execute.
    Keyword Args:
        None
    Returns:
        The output of the code execution.
    """

    try:
        result = subprocess.run(
            ["python", "-c", code], capture_output=True, text=True, timeout=30
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out"


def execute_r(code: str) -> str:
    """
    Function Description:
        This function executes R code and returns the output.
    Args:
        code: The R code to execute.
    Keyword Args:
        None
    Returns:
        The output of the code execution.
    """

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".R", delete=False
        ) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        result = subprocess.run(
            ["Rscript", temp_file_path], capture_output=True, text=True, timeout=30
        )
        os.unlink(temp_file_path)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        os.unlink(temp_file_path)
        return "Error: Execution timed out"


def execute_sql(code: str) -> str:
    """
    Function Description:
        This function executes SQL code and returns the output.
    Args:
        code: The SQL code to execute.
    Keyword Args:
        None
    Returns:
        result: The output of the code execution.
    """
    # use pandas to run the sql
    try:
        result = pd.read_sql_query(code, con=sqlite3.connect("npcsh_history.db"))
        return result
    except Exception as e:
        return f"Error: {e}"


def list_directory(args: List[str]) -> None:
    """
    Function Description:
        This function lists the contents of a directory.
    Args:
        args: The command arguments.
    Keyword Args:
        None
    Returns:
        None
    """
    directory = args[0] if args else "."
    try:
        files = os.listdir(directory)
        for f in files:
            print(f)
    except Exception as e:
        print(f"Error listing directory: {e}")


def read_file(args: List[str]) -> None:
    """
    Function Description:
        This function reads the contents of a file.
    Args:
        args: The command arguments.
    Keyword Args:
        None
    Returns:
        None
    """

    if not args:
        print("Usage: /read <filename>")
        return
    filename = args[0]
    try:
        with open(filename, "r") as file:
            content = file.read()
            print(content)
    except Exception as e:
        print(f"Error reading file: {e}")


import os
import json
from pathlib import Path


def get_npcshrc_path_windows():
    return Path.home() / ".npcshrc"


def read_rc_file_windows(path):
    """Read shell-style rc file"""
    config = {}
    if not path.exists():
        return config

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Match KEY='value' or KEY="value" format
                match = re.match(r'^([A-Z_]+)\s*=\s*[\'"](.*?)[\'"]$', line)
                if match:
                    key, value = match.groups()
                    config[key] = value
    return config


def get_setting_windows(key, default=None):
    # Try environment variable first
    if env_value := os.getenv(key):
        return env_value

    # Fall back to .npcshrc file
    config = read_rc_file_windows(get_npcshrc_path())
    return config.get(key, default)
