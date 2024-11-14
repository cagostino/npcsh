# helpers.py
import logging

logging.basicConfig(
    filename=".npcsh.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
import os
import sqlite3
import subprocess
import platform
import yaml
import nltk
import numpy as np
import sqlite3
import time
import numpy as np
import wave
import tempfile

try:
    from sentence_transformers import util
except Exception as e:
    print(f"Error importing sentence_transformers: {e}")
try:
    import whisper
    from playsound import playsound
    from gtts import gTTS
    import pyaudio
except Exception as e:
    print(f"Error importing whisper: {e}")


from .llm_funcs import get_llm_response


BASH_COMMANDS = [
    "open",
    "alias",
    "bg",
    "bind",
    "break",
    "builtin",
    "case",
    "command",
    "compgen",
    "complete",
    "continue",
    "declare",
    "dirs",
    "disown",
    "echo",
    "enable",
    "eval",
    "exec",
    "exit",
    "export",
    "fc",
    "fg",
    "getopts",
    "hash",
    "help",
    "history",
    "if",
    "jobs",
    "kill",
    "let",
    "local",
    "logout",
    "popd",
    "printf",
    "pushd",
    "pwd",
    "read",
    "readonly",
    "return",
    "set",
    "shift",
    "shopt",
    "source",
    "suspend",
    "test",
    "times",
    "trap",
    "type",
    "typeset",
    "ulimit",
    "umask",
    "unalias",
    "unset",
    "until",
    "wait",
    "while",
    # Common Unix commands
    "ls",
    "cp",
    "mv",
    "rm",
    "mkdir",
    "rmdir",
    "touch",
    "cat",
    "less",
    "more",
    "head",
    "tail",
    "grep",
    "find",
    "sed",
    "awk",
    "sort",
    "uniq",
    "wc",
    "diff",
    "chmod",
    "chown",
    "chgrp",
    "ln",
    "tar",
    "gzip",
    "gunzip",
    "zip",
    "unzip",
    "ssh",
    "scp",
    "rsync",
    "wget",
    "curl",
    "ping",
    "netstat",
    "ifconfig",
    "route",
    "traceroute",
    "ps",
    "top",
    "htop",
    "kill",
    "killall",
    "su",
    "sudo",
    "whoami",
    "who",
    "w",
    "last",
    "finger",
    "uptime",
    "free",
    "df",
    "du",
    "mount",
    "umount",
    "fdisk",
    "mkfs",
    "fsck",
    "dd",
    "cron",
    "at",
    "systemctl",
    "service",
    "journalctl",
    "man",
    "info",
    "whatis",
    "whereis",
    "which",
    "date",
    "cal",
    "bc",
    "expr",
    "screen",
    "tmux",
    "git",
    "vim",
    "emacs",
    "nano",
    "pip",
]


TERMINAL_EDITORS = ["vim", "emacs", "nano"]


def capture_screenshot(npc=None):
    # Ensure the directory exists
    directory = os.path.expanduser("~/.npcsh/screenshots")
    os.makedirs(directory, exist_ok=True)

    # Generate a unique filename
    filename = f"screenshot_{int(time.time())}.png"
    file_path = os.path.join(directory, filename)

    system = platform.system()
    model_kwargs = {}

    if npc is not None:
        if npc.provider is not None:
            model_kwargs["provider"] = npc.provider

        if npc.model is not None:
            model_kwargs["model"] = npc.model

    if system == "Darwin":  # macOS
        subprocess.run(["screencapture", "-i", file_path])  # This waits on macOS
    elif system == "Linux":
        # Use a loop to check for the file's existence
        if (
            subprocess.run(
                ["which", "gnome-screenshot"], capture_output=True
            ).returncode
            == 0
        ):
            subprocess.Popen(
                ["gnome-screenshot", "-a", "-f", file_path]
            )  # Use Popen for non-blocking
            while not os.path.exists(file_path):  # Wait for file to exist
                time.sleep(0.1)  # Check every 100ms
        elif subprocess.run(["which", "scrot"], capture_output=True).returncode == 0:
            subprocess.Popen(["scrot", "-s", file_path])  # Use Popen for non-blocking
            while not os.path.exists(file_path):  # Wait for file to exist
                time.sleep(0.1)  # Check every 100ms

        else:
            print(
                "No supported screenshot tool found. Please install gnome-screenshot or scrot."
            )
            return None
    else:
        print(f"Unsupported operating system: {system}")
        return None

    print(f"Screenshot saved to: {file_path}")
    return {"filename": filename, "file_path": file_path, "model_kwargs": model_kwargs}


def analyze_image_base(user_prompt, file_path, filename, npc=None):
    if os.path.exists(file_path):
        image_info = {"filename": filename, "file_path": file_path}

        if user_prompt:
            # try:
            response = get_llm_response(user_prompt, images=[image_info], npc=npc)

            # Add to command history *inside* the try block

            print(response["response"])  # Print response after adding to history
            return response

            # except Exception as e:
            # error_message = f"Error during LLM processing: {e}"
            # print(error_message)
            # return error_message

        else:  # This part needs to be inside the outer 'if os.path.exists...' block
            print("Skipping LLM processing.")
            return image_info  # Return image info if no prompt is given
    else:  # This else also needs to be part of the outer 'if os.path.exists...' block
        print("Screenshot capture failed or was cancelled.")
        return None


def analyze_image(
    command_history, user_prompt, file_path, filename, npc=None, **model_kwargs
):
    if os.path.exists(file_path):
        image_info = {"filename": filename, "file_path": file_path}

        if user_prompt:
            try:
                response = get_llm_response(
                    user_prompt, image=image_info, npc=npc, **model_kwargs
                )

                # Add to command history *inside* the try block
                command_history.add(
                    f"screenshot with prompt: {user_prompt}",
                    ["screenshot", npc.name if npc else ""],
                    response,
                    os.getcwd(),
                )

                print(response["response"])  # Print response after adding to history
                return response

            except Exception as e:
                error_message = f"Error during LLM processing: {e}"
                print(error_message)
                return error_message

        else:  # This part needs to be inside the outer 'if os.path.exists...' block
            print("Skipping LLM processing.")
            return image_info  # Return image info if no prompt is given
    else:  # This else also needs to be part of the outer 'if os.path.exists...' block
        print("Screenshot capture failed or was cancelled.")
        return None


def execute_set_command(command, value):
    config_path = os.path.expanduser("~/.npcshrc")

    # Map command to environment variable name
    var_map = {
        "model": "NPCSH_MODEL",
        "provider": "NPCSH_PROVIDER",
        "db_path": "NPCSH_DB_PATH",
    }

    if command not in var_map:
        return f"Unknown setting: {command}"

    env_var = var_map[command]

    # Read the current configuration
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            lines = f.readlines()
    else:
        lines = []

    # Check if the property exists and update it, or add it if it doesn't exist
    property_exists = False
    for i, line in enumerate(lines):
        if line.startswith(f"export {env_var}="):
            lines[i] = f"export {env_var}='{value}'\n"
            property_exists = True
            break

    if not property_exists:
        lines.append(f"export {env_var}='{value}'\n")

    # Save the updated configuration
    with open(config_path, "w") as f:
        f.writelines(lines)

    return f"{command.capitalize()} has been set to: {value}"


def get_shell_config_file():
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


def ensure_npcshrc_exists():
    npcshrc_path = os.path.expanduser("~/.npcshrc")
    if not os.path.exists(npcshrc_path):
        with open(npcshrc_path, "w") as npcshrc:
            npcshrc.write("# NPCSH Configuration File\n")
            npcshrc.write("export NPCSH_INITIALIZED=0\n")
            npcshrc.write("export NPCSH_PROVIDER='ollama'\n")
            npcshrc.write("export NPCSH_MODEL='llama3.2'\n")
            npcshrc.write("export NPCSH_API_URL=''")
            npcshrc.write("export NPCSH_DB_PATH='~/npcsh_history.db'\n")
    return npcshrc_path


# Function to check and download NLTK data if necessary
def ensure_nltk_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download("punkt")


def load_all_files(directory, extensions=None, depth=1):
    """
    Recursively load all files from the given directory up to a specified depth.

    :param directory: The root directory to start searching from.
    :param extensions: A list of file extensions to include. If None, all files are included.
    :param depth: The maximum depth to recurse into subdirectories.
    :return: A dictionary with file paths as keys and file contents as values.
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


import requests
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import urllib.parse
from duckduckgo_search import DDGS


def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using DuckDuckGo
    Returns: List of dicts with 'title', 'link', and 'snippet' keys
    """
    # Encode the query for URL
    encoded_query = urllib.parse.quote(query)

    # HTML search which gives better results than their API
    results = DDGS().text(query, max_results=num_results)

    return results


def rag_search(
    query, text_data, embedding_model, text_data_embedded=None, similarity_threshold=0.2
):
    """
    Retrieve lines from documents that are relevant to the query.

    :param query: The query string.
    :param text_data: A dictionary with file paths as keys and file contents as values.
    :param similarity_threshold: The similarity threshold for considering a line relevant.
    :return: A list of tuples (filename, snippet) of relevant snippets.
    """

    results = []

    # Compute the embedding of the query
    query_embedding = embedding_model.encode(
        query, convert_to_tensor=True, show_progress_bar=False
    )

    for filename, content in text_data.items():
        # Split content into lines
        lines = content.split("\n")
        if not lines:
            continue
        # Compute embeddings for each line
        if text_data_embedded is None:
            line_embeddings = embedding_model.encode(lines, convert_to_tensor=True)
        else:
            line_embeddings = text_data_embedded[filename]
        # Compute cosine similarities
        cosine_scores = util.cos_sim(query_embedding, line_embeddings)[0].cpu().numpy()

        # Find indices of lines above the similarity threshold
        ##print("most similar", np.max(cosine_scores))
        ##print("most similar doc", lines[np.argmax(cosine_scores)])
        relevant_line_indices = np.where(cosine_scores >= similarity_threshold)[0]

        for idx in relevant_line_indices:
            idx = int(idx)  # Ensure idx is an integer
            # Get context lines (±10 lines)
            start_idx = max(0, idx - 10)
            end_idx = min(len(lines), idx + 11)  # +11 because end index is exclusive
            snippet = "\n".join(lines[start_idx:end_idx])
            results.append((filename, snippet))
    # print("results", results)
    return results


def add_npcshrc_to_shell_config():
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


def setup_npcsh_config():
    ensure_npcshrc_exists()
    add_npcshrc_to_shell_config()


def initialize_npc_project():
    import os

    # Get the current directory
    current_directory = os.getcwd()

    # Create 'npc_team' folder in current directory
    npc_team_dir = os.path.join(current_directory, "npc_team")
    os.makedirs(npc_team_dir, exist_ok=True)

    # Create 'foreman.npc' file in 'npc_team' directory
    foreman_npc_path = os.path.join(npc_team_dir, "foreman.npc")
    if not os.path.exists(foreman_npc_path):
        # Create initial content for 'foreman.npc'
        foreman_npc_content = """name: foreman
primary_directive: "You are the foreman of an NPC team."
"""
        with open(foreman_npc_path, "w") as f:
            f.write(foreman_npc_content)
    else:
        print(f"{foreman_npc_path} already exists.")

    # Create 'tools' folder within 'npc_team' directory
    tools_dir = os.path.join(npc_team_dir, "tools")
    os.makedirs(tools_dir, exist_ok=True)

    # Create 'example.tool' file in 'tools' folder
    example_tool_path = os.path.join(tools_dir, "example.tool")
    if not os.path.exists(example_tool_path):
        # Create initial content for 'example.tool'
        example_tool_content = """tool_name: example
inputs: []
preprocess: ""
prompt: ""
postprocess: ""
"""
        with open(example_tool_path, "w") as f:
            f.write(example_tool_content)
    else:
        print(f"{example_tool_path} already exists.")

    return f"NPC project initialized in {npc_team_dir}"


def is_npcsh_initialized():
    return os.environ.get("NPCSH_INITIALIZED", None) == "1"


def set_npcsh_initialized():
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


def get_valid_npcs(db_path):
    db_conn = sqlite3.connect(db_path)
    cursor = db_conn.cursor()
    cursor.execute("SELECT name FROM compiled_npcs")
    npcs = [row[0] for row in cursor.fetchall()]
    db_conn.close()
    return npcs
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
    else:
        print(f"Tools directory not found: {directory}")
    return tools

def get_npc_from_command(command):
    parts = command.split()
    npc = None
    for part in parts:
        if part.startswith("npc="):
            npc = part.split("=")[1]
            break
    return npc
def get_npc_path(npc_name, db_path):
    # First, check in project npc_team directory
    project_npc_team_dir = os.path.abspath("./npc_team")
    npc_path = os.path.join(project_npc_team_dir, f"{npc_name}.npc")
    if os.path.exists(npc_path):
        return npc_path

    # Then, check in global npc_team directory
    user_npc_team_dir = os.path.expanduser("~/.npcsh/npc_team")
    npc_path = os.path.join(user_npc_team_dir, f"{npc_name}.npc")
    if os.path.exists(npc_path):
        return npc_path
    else:
        print(f"NPC file not found: {npc_name}")
        return None
        
# helpers.py

import os
import shutil
import filecmp
import sqlite3



def initialize_base_npcs_if_needed(db_path):
    if is_npcsh_initialized():
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the compiled_npcs table if it doesn't exist
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS compiled_npcs (
            name TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            compiled_content TEXT
        )
        '''
    )

    # Get the path to the npc_team directory in the package
    package_dir = os.path.dirname(__file__)
    package_npc_team_dir = os.path.join(package_dir, "npc_team")

    # User's global npc_team directory
    user_npc_team_dir = os.path.expanduser("~/.npcsh/npc_team")
    user_tools_dir = os.path.join(user_npc_team_dir, "tools")
    os.makedirs(user_npc_team_dir, exist_ok=True)
    os.makedirs(user_tools_dir, exist_ok=True)

    # Copy NPCs from package to user directory
    for filename in os.listdir(package_npc_team_dir):
        if filename.endswith(".npc"):
            source_path = os.path.join(package_npc_team_dir, filename)
            destination_path = os.path.join(user_npc_team_dir, filename)
            if not os.path.exists(destination_path) or file_has_changed(source_path, destination_path):
                shutil.copy2(source_path, destination_path)
                print(f"Copied {filename} to {destination_path}")

    # Copy tools from package to user directory
    package_tools_dir = os.path.join(package_npc_team_dir, "tools")
    if os.path.exists(package_tools_dir):
        for filename in os.listdir(package_tools_dir):
            if filename.endswith(".tool"):
                source_tool_path = os.path.join(package_tools_dir, filename)
                destination_tool_path = os.path.join(user_tools_dir, filename)
                if (not os.path.exists(destination_tool_path)) or file_has_changed(source_tool_path, destination_tool_path):
                    shutil.copy2(source_tool_path, destination_tool_path)
                    print(f"Copied tool {filename} to {destination_tool_path}")

    conn.commit()
    conn.close()
    set_npcsh_initialized()
    add_npcshrc_to_shell_config()

def file_has_changed(source_path, destination_path):
    # Compare file modification times or contents to decide whether to update the file
    return not filecmp.cmp(source_path, destination_path, shallow=False)

def is_valid_npc(npc, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM compiled_npcs WHERE name = ?", (npc,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def get_audio_level(audio_data):
    return np.max(np.abs(np.frombuffer(audio_data, dtype=np.int16)))


def calibrate_silence(sample_rate=16000, duration=2):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024,
    )

    print("Calibrating silence level. Please remain quiet...")
    levels = []
    for _ in range(int(sample_rate * duration / 1024)):
        data = stream.read(1024)
        levels.append(get_audio_level(data))

    stream.stop_stream()
    stream.close()
    p.terminate()

    avg_level = np.mean(levels)
    silence_threshold = avg_level * 1.5  # Set threshold slightly above average
    print(f"Silence threshold set to: {silence_threshold}")
    return silence_threshold


def is_silent(audio_data, threshold):
    return get_audio_level(audio_data) < threshold


def record_audio(sample_rate=16000, max_duration=10, silence_threshold=None):
    if silence_threshold is None:
        silence_threshold = calibrate_silence()

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024,
    )

    print("Listening... (speak now)")
    frames = []
    silent_chunks = 0
    has_speech = False
    max_silent_chunks = int(sample_rate * 1.0 / 1024)  # 1.0 seconds of silence
    max_chunks = int(sample_rate * max_duration / 1024)  # Maximum duration in chunks

    start_time = time.time()
    for _ in range(max_chunks):
        data = stream.read(1024)
        frames.append(data)

        if is_silent(data, silence_threshold):
            silent_chunks += 1
            if has_speech and silent_chunks > max_silent_chunks:
                break
        else:
            silent_chunks = 0
            has_speech = True

        if len(frames) % 10 == 0:  # Print a dot every ~0.5 seconds
            print(".", end="", flush=True)

        if time.time() - start_time > max_duration:
            print("\nMax duration reached.")
            break

    print("\nProcessing...")

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b"".join(frames)


def speak_text(text):
    try:
        tts = gTTS(text=text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            playsound(fp.name)
        os.unlink(fp.name)
    except Exception as e:
        print(f"Text-to-speech error: {e}")


def open_terminal_editor(command):
    try:
        os.system(command)
    except Exception as e:
        print(f"Error opening terminal editor: {e}")


# Helper functions for language-specific execution remain the same


def execute_python(code):
    try:
        result = subprocess.run(
            ["python", "-c", code], capture_output=True, text=True, timeout=30
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out"


def execute_r(code):
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


def execute_sql(code):
    # This is a placeholder. You'll need to implement SQL execution based on your database setup.
    return "SQL execution not implemented yet."


def execute_scala(code):
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".scala", delete=False
        ) as temp_file:
            temp_file.write(
                f"object TempObject {{ def main(args: Array[String]): Unit = {{ {code} }} }}"
            )
            temp_file_path = temp_file.name

        compile_result = subprocess.run(
            ["scalac", temp_file_path], capture_output=True, text=True, timeout=30
        )
        if compile_result.returncode != 0:
            os.unlink(temp_file_path)
            return f"Compilation Error: {compile_result.stderr}"

        run_result = subprocess.run(
            ["scala", "TempObject"], capture_output=True, text=True, timeout=30
        )
        os.unlink(temp_file_path)
        return (
            run_result.stdout
            if run_result.returncode == 0
            else f"Runtime Error: {run_result.stderr}"
        )
    except subprocess.TimeoutExpired:
        os.unlink(temp_file_path)
        return "Error: Execution timed out"


def execute_java(code):
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".java", delete=False
        ) as temp_file:
            temp_file.write(
                f"public class TempClass {{ public static void main(String[] args) {{ {code} }} }}"
            )
            temp_file_path = temp_file.name

        compile_result = subprocess.run(
            ["javac", temp_file_path], capture_output=True, text=True, timeout=30
        )
        if compile_result.returncode != 0:
            os.unlink(temp_file_path)
            return f"Compilation Error: {compile_result.stderr}"

        run_result = subprocess.run(
            ["java", "TempClass"], capture_output=True, text=True, timeout=30
        )
        os.unlink(temp_file_path)
        return (
            run_result.stdout
            if run_result.returncode == 0
            else f"Runtime Error: {run_result.stderr}"
        )
    except subprocess.TimeoutExpired:
        os.unlink(temp_file_path)
        return "Error: Execution timed out"


def list_directory(args):
    directory = args[0] if args else "."
    try:
        files = os.listdir(directory)
        for f in files:
            print(f)
    except Exception as e:
        print(f"Error listing directory: {e}")


def read_file(args):
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


def log_action(action, detail=""):
    logging.info(f"{action}: {detail}")
