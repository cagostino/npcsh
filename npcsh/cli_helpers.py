import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Union
import numpy as np
import readline
from colorama import Fore, Back, Style
import re
import tempfile
import sqlite3
import wave
import datetime

import shlex
import logging
import textwrap
import subprocess
from termcolor import colored
import sys
import termios
import tty
import pty
import select
import signal
import time

import whisper

try:
    from sentence_transformers import SentenceTransformer
except:
    print("Could not load the sentence-transformers package.")

from .llm_funcs import (
    get_available_models,
    get_model_and_provider,
    execute_llm_command,
    execute_llm_question,
    get_conversation,
    get_system_message,
    check_llm_command,
)
from .helpers import get_valid_npcs, get_npc_path
from .npc_compiler import NPCCompiler, NPC, load_npc_from_file
from .search import rag_search
from .audio import calibrate_silence, record_audio, speak_text
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from .command_history import CommandHistory


interactive_commands = {
    "ipython": ["ipython"],
    "python": ["python", "-i"],
    "sqlite3": ["sqlite3"],
    "r": ["R", "--interactive"],
}
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


def preprocess_code_block(code_text):
    """
    Preprocess code block text to remove leading spaces.
    """
    lines = code_text.split("\n")
    return "\n".join(line.lstrip() for line in lines)


def render_code_block(code_text):
    """
    Render code block with no leading spaces.
    """
    processed_code = preprocess_code_block(code_text)
    console = Console()
    console.print(processed_code, style="")


def preprocess_markdown(md_text):
    """
    Preprocess markdown text to handle code blocks separately.
    """
    lines = md_text.split("\n")
    processed_lines = []

    inside_code_block = False
    current_code_block = []

    for line in lines:
        if line.startswith("```"):  # Toggle code block
            if inside_code_block:
                # Close code block, unindent, and append
                processed_lines.append("```")
                processed_lines.extend(
                    textwrap.dedent("\n".join(current_code_block)).split("\n")
                )
                processed_lines.append("```")
                current_code_block = []
            inside_code_block = not inside_code_block
        elif inside_code_block:
            current_code_block.append(line)
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)


def render_markdown(text: str) -> None:
    """
    Renders markdown text, but handles code blocks as plain syntax-highlighted text.
    """
    lines = text.split("\n")
    console = Console()

    inside_code_block = False
    code_lines = []
    lang = None

    for line in lines:
        if line.startswith("```"):
            if inside_code_block:
                # End of code block - render the collected code
                code = "\n".join(code_lines)
                if code.strip():
                    syntax = Syntax(
                        code, lang or "python", theme="monokai", line_numbers=False
                    )
                    console.print(syntax)
                code_lines = []
            else:
                # Start of code block - get language if specified
                lang = line[3:].strip() or None
            inside_code_block = not inside_code_block
        elif inside_code_block:
            code_lines.append(line)
        else:
            # Regular markdown
            console.print(Markdown(line))


def change_directory(command_parts: list, messages: list) -> dict:
    """
    Function Description:
        Changes the current directory.
    Args:
        command_parts : list : Command parts
        messages : list : Messages
    Keyword Args:
        None
    Returns:
        dict : dict : Dictionary

    """

    try:
        if len(command_parts) > 1:
            new_dir = os.path.expanduser(command_parts[1])
        else:
            new_dir = os.path.expanduser("~")
        os.chdir(new_dir)
        return {
            "messages": messages,
            "output": f"Changed directory to {os.getcwd()}",
        }
    except FileNotFoundError:
        return {
            "messages": messages,
            "output": f"Directory not found: {new_dir}",
        }
    except PermissionError:
        return {"messages": messages, "output": f"Permission denied: {new_dir}"}


def log_action(action: str, detail: str = "") -> None:
    """
    Function Description:
        This function logs an action with optional detail.
    Args:
        action: The action to log.
        detail: Additional detail to log.
    Keyword Args:
        None
    Returns:
        None
    """
    logging.info(f"{action}: {detail}")


TERMINAL_EDITORS = ["vim", "emacs", "nano"]


def complete(text: str, state: int) -> str:
    """
    Function Description:
        Handles autocompletion for the npcsh shell.
    Args:
        text : str : Text to autocomplete
        state : int : State
    Keyword Args:
        None
    Returns:
        None

    """
    buffer = readline.get_line_buffer()
    available_models = get_available_models()

    # If completing a model name
    if "@" in buffer:
        at_index = buffer.rfind("@")
        model_text = buffer[at_index + 1 :]
        model_completions = [m for m in available_models if m.startswith(model_text)]

        try:
            # Return the full text including @ symbol
            return "@" + model_completions[state]
        except IndexError:
            return None

    # If completing a command
    elif text.startswith("/"):
        command_completions = [c for c in valid_commands if c.startswith(text)]
        try:
            return command_completions[state]
        except IndexError:
            return None

    return None


def global_completions(text: str, command_parts: list) -> list:
    """
    Function Description:
        Handles global autocompletions for the npcsh shell.
    Args:
        text : str : Text to autocomplete
        command_parts : list : List of command parts
    Keyword Args:
        None
    Returns:
        completions : list : List of completions

    """
    if not command_parts:
        return [c + " " for c in valid_commands if c.startswith(text)]
    elif command_parts[0] in ["/compile", "/com"]:
        # Autocomplete NPC files
        return [f for f in os.listdir(".") if f.endswith(".npc") and f.startswith(text)]
    elif command_parts[0] == "/read":
        # Autocomplete filenames
        return [f for f in os.listdir(".") if f.startswith(text)]
    else:
        # Default filename completion
        return [f for f in os.listdir(".") if f.startswith(text)]


def wrap_text(text: str, width: int = 80) -> str:
    """
    Function Description:
        Wraps text to a specified width.
    Args:
        text : str : Text to wrap
        width : int : Width of text
    Keyword Args:
        None
    Returns:
        lines : str : Wrapped text
    """
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width))
    return "\n".join(lines)


def get_file_color(filepath: str) -> tuple:
    """
    Function Description:
        Returns color and attributes for a given file path.
    Args:
        filepath : str : File path
    Keyword Args:
        None
    Returns:
        color : str : Color
        attrs : list : List of attributes

    """

    if os.path.isdir(filepath):
        return "blue", ["bold"]
    elif os.access(filepath, os.X_OK):
        return "green", []
    elif filepath.endswith((".zip", ".tar", ".gz", ".bz2", ".xz", ".7z")):
        return "red", []
    elif filepath.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")):
        return "magenta", []
    elif filepath.endswith((".py", ".pyw")):
        return "yellow", []
    elif filepath.endswith((".sh", ".bash", ".zsh")):
        return "green", []
    elif filepath.endswith((".c", ".cpp", ".h", ".hpp")):
        return "cyan", []
    elif filepath.endswith((".js", ".ts", ".jsx", ".tsx")):
        return "yellow", []
    elif filepath.endswith((".html", ".css", ".scss", ".sass")):
        return "magenta", []
    elif filepath.endswith((".md", ".txt", ".log")):
        return "white", []
    elif filepath.startswith("."):
        return "cyan", []
    else:
        return "white", []


def readline_safe_prompt(prompt: str) -> str:
    """
    Function Description:
        Escapes ANSI escape sequences in the prompt.
    Args:
        prompt : str : Prompt
    Keyword Args:
        None
    Returns:
        prompt : str : Prompt

    """
    # This regex matches ANSI escape sequences
    ansi_escape = re.compile(r"(\033\[[0-9;]*[a-zA-Z])")

    # Wrap them with \001 and \002
    def escape_sequence(m):
        return "\001" + m.group(1) + "\002"

    return ansi_escape.sub(escape_sequence, prompt)


def setup_readline() -> str:
    """
    Function Description:
        Sets up readline for the npcsh shell.
    Args:
        None
    Keyword Args:
        None
    Returns:
        history_file : str : History file
    """
    history_file = os.path.expanduser("~/.npcsh_history")
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass

    readline.set_history_length(1000)
    # Commented out because 'set' commands may not work as intended with parse_and_bind
    # readline.parse_and_bind("set editing-mode vi")
    readline.parse_and_bind('"\e[A": history-search-backward')
    readline.parse_and_bind('"\e[B": history-search-forward')
    readline.parse_and_bind('"\C-r": reverse-search-history')
    # Remove or adjust these lines if they cause issues
    # readline.parse_and_bind("set enable-bracketed-paste on")
    # readline.parse_and_bind("set mark-modified-lines on")
    readline.parse_and_bind("\C-e: end-of-line")
    readline.parse_and_bind("\C-a: beginning-of-line")

    return history_file


def save_readline_history():
    readline.write_history_file(os.path.expanduser("~/.npcsh_readline_history"))


def orange(text: str) -> str:
    """
    Function Description:
        Returns orange text.
    Args:
        text : str : Text
    Keyword Args:
        None
    Returns:
        text : str : Text

    """
    return f"\033[38;2;255;165;0m{text}{Style.RESET_ALL}"


def get_multiline_input(prompt: str) -> str:
    """
    Function Description:
        Gets multiline input from the user.
    Args:
        prompt : str : Prompt
    Keyword Args:
        None
    Returns:
        lines : str : Lines

    """
    lines = []
    current_prompt = prompt
    while True:
        try:
            line = input(current_prompt)
            if line.endswith("\\"):
                lines.append(line[:-1])  # Remove the backslash
                # Use a continuation prompt for the next line
                current_prompt = readline_safe_prompt("> ")
            else:
                lines.append(line)
                break
        except EOFError:
            break  # Handle EOF (Ctrl+D)
    return "\n".join(lines)


def start_interactive_session(command: list) -> int:
    """
    Function Description:
        Starts an interactive session.
    Args:
        command : list : Command to execute
    Keyword Args:
        None
    Returns:
        returncode : int : Return code

    """
    # Save the current terminal settings
    old_tty = termios.tcgetattr(sys.stdin)
    try:
        # Create a pseudo-terminal
        master_fd, slave_fd = pty.openpty()

        # Start the process
        p = subprocess.Popen(
            command,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            shell=True,
            preexec_fn=os.setsid,  # Create a new process group
        )

        # Set the terminal to raw mode
        tty.setraw(sys.stdin.fileno())

        def handle_timeout(signum, frame):
            raise TimeoutError("Process did not terminate in time")

        while p.poll() is None:
            r, w, e = select.select([sys.stdin, master_fd], [], [], 0.1)
            if sys.stdin in r:
                d = os.read(sys.stdin.fileno(), 10240)
                os.write(master_fd, d)
            elif master_fd in r:
                o = os.read(master_fd, 10240)
                if o:
                    os.write(sys.stdout.fileno(), o)
                else:
                    break

        # Wait for the process to terminate with a timeout
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(5)  # 5 second timeout
        try:
            p.wait()
        except TimeoutError:
            print("\nProcess did not terminate. Force killing...")
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            time.sleep(1)
            if p.poll() is None:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        finally:
            signal.alarm(0)

    finally:
        # Restore the terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, old_tty)

    return p.returncode


def validate_bash_command(command_parts: list) -> bool:
    """
    Function Description:
        Validate if the command sequence is a valid bash command with proper arguments/flags.
    Args:
        command_parts : list : Command parts
    Keyword Args:
        None
    Returns:
        bool : bool : Boolean
    """
    if not command_parts:
        return False

    COMMAND_PATTERNS = {
        "cat": {
            "flags": ["-n", "-b", "-E", "-T", "-s", "--number", "-A", "--show-all"],
            "requires_arg": True,
        },
        "find": {
            "flags": [
                "-name",
                "-type",
                "-size",
                "-mtime",
                "-exec",
                "-print",
                "-delete",
                "-maxdepth",
                "-mindepth",
                "-perm",
                "-user",
                "-group",
            ],
            "requires_arg": True,
        },
        "who": {
            "flags": [
                "-a",
                "-b",
                "-d",
                "-H",
                "-l",
                "-p",
                "-q",
                "-r",
                "-s",
                "-t",
                "-u",
                "--all",
                "--count",
                "--heading",
            ],
            "requires_arg": True,
        },
        "open": {
            "flags": ["-a", "-e", "-t", "-f", "-F", "-W", "-n", "-g", "-h"],
            "requires_arg": True,
        },
        "which": {"flags": ["-a", "-s", "-v"], "requires_arg": True},
    }

    base_command = command_parts[0]

    if base_command not in COMMAND_PATTERNS:
        return True  # Allow other commands to pass through

    pattern = COMMAND_PATTERNS[base_command]
    args = []
    flags = []

    for i in range(1, len(command_parts)):
        part = command_parts[i]
        if part.startswith("-"):
            flags.append(part)
            if part not in pattern["flags"]:
                return False  # Invalid flag
        else:
            args.append(part)

    # Check if 'who' has any arguments (it shouldn't)
    if base_command == "who" and args:
        return False

    # Handle 'which' with '-a' flag
    if base_command == "which" and "-a" in flags:
        return True  # Allow 'which -a' with or without arguments.

    # Check if any required arguments are missing
    if pattern.get("requires_arg", False) and not args:
        return False

    return True


def execute_slash_command(
    command: str,
    command_history: CommandHistory,
    db_path: str,
    db_conn: sqlite3.Connection,
    npc_compiler: NPCCompiler,
    valid_npcs: list,
    npc: NPC = None,
    retrieved_docs: list = None,
    embedding_model=None,
    text_data=None,
    text_data_embedded=None,
    messages=None,
    model: str = None,
    provider: str = None,
):
    """
    Function Description:
        Executes a slash command.
    Args:
        command : str : Command
        command_history : CommandHistory : Command history
        db_path : str : Database path
        npc_compiler : NPCCompiler : NPC compiler
    Keyword Args:
        embedding_model : None : Embedding model
        current_npc : None : Current NPC
        text_data : None : Text data
        text_data_embedded : None : Embedded text data
        messages : None : Messages
    Returns:
        dict : dict : Dictionary
    """

    command = command[1:]

    log_action("Command Executed", command)

    command_parts = command.split()
    command_name = command_parts[0]
    args = command_parts[1:]

    if command_name in valid_npcs:
        npc_path = get_npc_path(command_name, db_path)
        current_npc = load_npc_from_file(npc_path, db_conn)
        output = f"Switched to NPC: {current_npc.name}"
        print(output)
        return {"messages": messages, "output": output, "current_npc": current_npc}
    if command_name == "compile" or command_name == "com":
        try:
            if len(args) > 0:  # Specific NPC file(s) provided
                for npc_file in args:
                    # differentiate between .npc and .pipe
                    if npc_file.endswith(".pipe"):
                        compiled_script = npc_compiler.compile_pipe(npc_file)

                    elif npc_file.endswith(".npc"):
                        compiled_script = npc_compiler.compile(npc_file)

                        output += f"Compiled NPC profile: {compiled_script}\n"
            elif current_npc:  # Compile current NPC
                compiled_script = npc_compiler.compile(current_npc)
                output = f"Compiled NPC profile: {compiled_script}"
            else:  # Compile all NPCs in the directory
                for filename in os.listdir(npc_compiler.npc_directory):
                    if filename.endswith(".npc"):
                        try:
                            compiled_script = npc_compiler.compile(filename)
                            output += f"Compiled NPC profile: {compiled_script}\n"
                        except Exception as e:
                            output += f"Error compiling {filename}: {str(e)}\n"

        except Exception as e:
            import traceback

            output = f"Error compiling NPC profile: {str(e)}\n{traceback.format_exc()}"
            print(output)
    elif command_name == "pipe":
        if len(args) > 0:  # Specific NPC file(s) provided
            for npc_file in args:
                # differentiate between .npc and .pipe
                compiled_script = npc_compiler.compile_pipe(npc_file)
                # run through the steps in the pipe

    elif command_name == "list":
        output = list_directory()
    elif command_name == "read":
        if len(args) == 0:
            return {
                "messages": messages,
                "output": "Error: read command requires a filename argument",
            }
        filename = args[0]
        output = read_file(filename, npc=npc)
    elif command_name == "init":
        output = initialize_npc_project()
        return {"messages": messages, "output": output}
    elif (
        command.startswith("vixynt")
        or command.startswith("vix")
        or (command.startswith("v") and command[1] == " ")
    ):
        # check if "filename=..." is in the command
        filename = None
        if "filename=" in command:
            filename = command.split("filename=")[1].split()[0]
            command = command.replace(f"filename={filename}", "").strip()
        # Get user prompt about the image BY joining the rest of the arguments
        user_prompt = " ".join(command.split()[1:])

        output = generate_image(user_prompt, npc=npc, filename=filename)

    elif command.startswith("ots"):
        # check if there is a filename
        if len(command_parts) > 1:
            filename = command_parts[1]
            file_path = os.path.join(os.getcwd(), filename)
            # Get user prompt about the image
            user_prompt = input(
                "Enter a prompt for the LLM about this image (or press Enter to skip): "
            )

            output = analyze_image(
                command_history,
                user_prompt,
                file_path,
                filename,
                npc=npc,
            )

        else:
            output = capture_screenshot(npc=npc)
            user_prompt = input(
                "Enter a prompt for the LLM about this image (or press Enter to skip): "
            )
            output = analyze_image(
                command_history,
                user_prompt,
                output["file_path"],
                output["filename"],
                npc=npc,
                **output["model_kwargs"],
            )
        if output:
            if isinstance(output, dict) and "filename" in output:
                message = f"Screenshot captured: {output['filename']}\nFull path: {output['file_path']}\nLLM-ready data available."
            else:  # This handles both LLM responses and error messages (both strings)
                message = output
            return {"messages": messages, "output": message}  # Return the message
        else:  # Handle the case where capture_screenshot returns None
            print("Screenshot capture failed.")
            return {
                "messages": messages,
                "output": None,
            }  # Return None to indicate failure
    elif command_name == "help":  # New help command
        output = """# Available Commands

/compile [npc_file1.npc npc_file2.npc ...] #Compiles specified NPC profile(s). If no arguments are provided, compiles all NPCs in the npc_profiles directory.

/com [npc_file1.npc npc_file2.npc ...] #Alias for /compile.

/whisper   # Enter whisper mode.

/notes # Enter notes mode.

/data # Enter data mode.

/cmd <command> #Execute a command using the current NPC's LLM.

/command <command> #Alias for /cmd.

/set <model|provider|db_path> <value> #Sets the specified parameter. Enclose the value in quotes.

/sample <question> #Asks the current NPC a question.

/spool [inherit_last=<n>] #Enters spool mode. Optionally inherits the last <n> messages.

/sp [inherit_last=<n>] #Alias for /spool.

/<npc_name> #Enters the specified NPC's mode.

/help #Displays this help message.

/exit or /quit #Exits the current NPC mode or the npcsh shell.

#Note
Bash commands and other programs can be executed directly."""

        return {
            "messages": messages,
            "output": output,
        }

    elif command_name == "whisper":
        # try:
        output = enter_whisper_mode(command_history, npc=npc)
        # except Exception as e:
        #    print(f"Error entering whisper mode: {str(e)}")
        #    output = "Error entering whisper mode"

    elif command_name == "notes":
        output = enter_notes_mode(command_history, npc=npc)
    elif command_name == "data":
        # print("data")
        output = enter_data_mode(command_history, npc=npc)
        # output = enter_observation_mode(command_history, npc=npc)
    elif command_name == "cmd" or command_name == "command":
        output = execute_llm_command(
            command, command_history, npc=npc, retrieved_docs=retrieved_docs
        )
    elif command_name == "set":
        parts = command.split()
        if len(parts) == 3 and parts[1] in ["model", "provider", "db_path"]:
            output = execute_set_command(parts[1], parts[2])
        else:
            return {
                "messages": messages,
                "output": "Invalid set command. Usage: /set [model|provider|db_path] 'value_in_quotes' ",
            }
    elif command_name == "sample":
        output = execute_llm_question(
            command,
            command_history,
            npc=npc,
            retrieved_docs=retrieved_docs,
            messages=messages,
            model=model,
            provider=provider,
        )
    elif command_name == "spool" or command_name == "sp":
        inherit_last = 0
        for part in args:
            if part.startswith("inherit_last="):
                try:
                    inherit_last = int(part.split("=")[1])
                except ValueError:
                    return {
                        "messages": messages,
                        "output": "Error: inherit_last must be an integer",
                    }
                break

        output = enter_spool_mode(command_history, inherit_last, npc=npc)
        return {"messages": output["messages"], "output": output}

    else:
        output = f"Unknown command: {command_name}"

    subcommands = [f"/{command}"]
    return {"messages": messages, "output": output, "subcommands": subcommands}

    ## flush  the current shell context


def execute_set_command(command: str, value: str) -> str:
    """
    Function Description:
        This function sets a configuration value in the .npcshrc file.
    Args:
        command: The command to execute.
        value: The value to set.
    Keyword Args:
        None
    Returns:
        A message indicating the success or failure of the operation.
    """

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


def execute_hash_command(
    command: str,
    command_history: CommandHistory,
    npc: NPC = None,
    retrieved_docs: list = None,
    messages: list = None,
    model: str = None,
    provider: str = None,
) -> str:
    """
    Function Description:
        Executes a hash command.
    Args:
        command : str : Command
        command_history : CommandHistory : Command history
    Keyword Args:
        npc : NPC : NPC
        retrieved_docs : list : Retrieved documents
        messages : list : Messages
        model : str : Model
        provider : str : Provider
    Returns:
        output : str : Output

    """

    command_parts = command[1:].split()
    language = command_parts[0].lower()
    code = " ".join(command_parts[1:])

    if language == "python":
        output = execute_python(code)
    elif language == "r":
        output = execute_r(code)
    elif language == "sql":
        output = execute_sql(code)
    else:
        output = check_llm_command(
            command,
            command_history,
            npc=npc,
            retrieved_docs=retrieved_docs,
            messages=messages,
            model=model,
            provider=provider,
        )
    return output


def get_npc_from_command(command: str) -> Optional[str]:
    """
    Function Description:
        This function extracts the NPC name from a command string.
    Args:
        command: The command string.

    Keyword Args:
        None
    Returns:
        The NPC name if found, or None
    """

    parts = command.split()
    npc = None
    for part in parts:
        if part.startswith("npc="):
            npc = part.split("=")[1]
            break
    return npc


def open_terminal_editor(command: str) -> None:
    """
    Function Description:
        This function opens a terminal-based text editor.
    Args:
        command: The command to open the editor.
    Keyword Args:
        None
    Returns:
        None
    """

    try:
        os.system(command)
    except Exception as e:
        print(f"Error opening terminal editor: {e}")


def execute_command(
    command: str,
    command_history: CommandHistory,
    db_path: str,
    npc_compiler: NPCCompiler,
    embedding_model: Union[SentenceTransformer, Any] = None,
    current_npc: NPC = None,
    text_data: str = None,
    text_data_embedded: np.ndarray = None,
    messages: list = None,
):
    """
    Function Description:
        Executes a command.
    Args:
        command : str : Command
        command_history : CommandHistory : Command history
        db_path : str : Database path
        npc_compiler : NPCCompiler : NPC compiler
    Keyword Args:
        embedding_model : Union[SentenceTransformer, Any] : Embedding model
        current_npc : NPC : Current NPC
        text_data : str : Text data
        text_data_embedded : np.ndarray : Embedded text data
        messages : list : Messages
    Returns:
        dict : dict : Dictionary

    """
    subcommands = []
    output = ""
    location = os.getcwd()
    # Extract NPC from command
    db_conn = sqlite3.connect(db_path)
    # print(command)

    # Initialize retrieved_docs to None at the start
    retrieved_docs = None

    # Only try RAG search if text_data exists
    if text_data is not None:
        try:
            if embedding_model is None:
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            retrieved_docs = rag_search(
                command,
                text_data,
                embedding_model,
                text_data_embedded=text_data_embedded,
            )
        except Exception as e:
            print(f"Error searching text data: {str(e)}")
            retrieved_docs = None

    # print(retrieved_docs)
    if current_npc is None:
        valid_npcs = get_valid_npcs(db_path)

        npc_name = get_npc_from_command(command)
        if npc_name is None:
            npc_name = "sibiji"  # Default NPC
        # print(npc_name)
        npc_path = get_npc_path(npc_name, db_path)
        # print(npc_path)
        npc = load_npc_from_file(npc_path, db_conn)
    else:
        npc = current_npc
    # print(command, 'command', len(command), len(command.strip()))
    available_models = get_available_models()

    if len(command.strip()) == 0:
        return {"messages": messages, "output": output}
    if messages is None:
        messages = []
    messages.append({"role": "user", "content": command})  # Add user message
    model_override, provider_override, command = get_model_and_provider(
        command, available_models
    )
    # print(command, model_override, provider_override)
    if command.startswith("/"):

        result = execute_slash_command(
            command,
            command_history,
            db_path,
            db_conn,
            npc_compiler,
            valid_npcs,
            embedding_model=embedding_model,
            npc=current_npc,
            retrieved_docs=retrieved_docs,
            text_data=text_data,
            text_data_embedded=text_data_embedded,
            messages=messages,
            model=model_override,
            provider=provider_override,
        )
        output = result.get("output", "")
        new_messages = result.get("messages", None)
        subcommands = result.get("subcommands", [])

    elif command.startswith("#"):
        output = execute_hash_command(
            command,
            command_history,
            npc=npc,
            retrieved_docs=retrieved_docs,
            messages=messages,
        )

    else:
        # Check if it's a bash command
        # print(command)
        try:
            command_parts = shlex.split(command)
            # print('er')

        except ValueError as e:
            if "No closing quotation" in str(e):
                # Attempt to close unclosed quotes
                command += '"'
                try:
                    command_parts = shlex.split(command)
                except ValueError:
                    return {
                        "messages": messages,
                        "output": "Error: Unmatched quotation in command",
                    }
            else:
                return {"messages": messages, "output": f"Error: {str(e)}"}

        if command_parts[0] in interactive_commands:
            # print('ir')
            print(f"Starting interactive {command_parts[0]} session...")
            return_code = start_interactive_session(
                interactive_commands[command_parts[0]]
            )
            return {
                "messages": messages,
                "output": f"Interactive {command_parts[0]} session ended with return code {return_code}",
            }

        elif command_parts[0] == "cd":
            change_dir_result = change_directory(command_parts, messages)
            messages = change_dir_result["messages"]
            output = change_dir_result["output"]

        elif command_parts[0] in BASH_COMMANDS:
            if command_parts[0] in TERMINAL_EDITORS:
                return {"messages": messages, "output": open_terminal_editor(command)}

            elif command_parts[0] in ["cat", "find", "who", "open", "which"]:
                if not validate_bash_command(command_parts):
                    output = "Error: Invalid command syntax or arguments"  # Assign error message directly
                    output = check_llm_command(
                        command,
                        command_history,
                        npc=current_npc,
                        retrieved_docs=retrieved_docs,
                        messages=messages,
                        model=model_override,
                        provider=provider_override,
                    )
                else:  # ONLY execute if valid
                    try:
                        result = subprocess.run(
                            command_parts, capture_output=True, text=True
                        )
                        output = result.stdout + result.stderr
                    except Exception as e:
                        output = f"Error executing command: {e}"
            elif command.startswith("open "):
                try:
                    path_to_open = os.path.expanduser(
                        command.split(" ", 1)[1]
                    )  # Extract the path
                    absolute_path = os.path.abspath(path_to_open)  # Make it absolute
                    expanded_command = ["open", absolute_path]  # Use the absolute path
                    subprocess.run(expanded_command, check=True)
                    output = f"Launched: {command}"
                except subprocess.CalledProcessError as e:
                    output = colored(f"Error opening: {e}", "red")  # Show error message
                except Exception as e:
                    output = colored(
                        f"Error executing command: {str(e)}", "red"
                    )  # Show
            else:
                try:
                    result = subprocess.run(
                        command_parts, capture_output=True, text=True
                    )
                    output = result.stdout
                    if result.stderr:
                        output += colored(f"\nError: {result.stderr}", "red")

                    # Color code the output
                    colored_output = ""
                    for line in output.split("\n"):
                        parts = line.split()
                        if parts:
                            filepath = parts[-1]
                            color, attrs = get_file_color(filepath)
                            colored_filepath = colored(filepath, color, attrs=attrs)
                            colored_line = " ".join(parts[:-1] + [colored_filepath])
                            colored_output += colored_line + "\n"
                        else:
                            colored_output += line + "\n"

                    output = colored_output.rstrip()

                    if not output and result.returncode == 0:
                        output = colored(
                            f"Command '{command}' executed successfully (no output).",
                            "green",
                        )
                    print(output)
                except Exception as e:
                    output = colored(f"Error executing command: {e}", "red")

        else:
            # print(model_override, provider_override)
            output = check_llm_command(
                command,
                command_history,
                npc=current_npc,
                retrieved_docs=retrieved_docs,
                messages=messages,
                model=model_override,
                provider=provider_override,
            )

    # Add command to history
    command_history.add(command, subcommands, output, location)

    if isinstance(output, dict):
        response = output.get("output", "")
        new_messages = output.get("messages", None)
        if new_messages is not None:
            messages = new_messages
        output = response

    # Only render markdown once, at the end
    if output:
        render_markdown(output)

    return {"messages": messages, "output": output}


def enter_whisper_mode(command_history: Any, npc: Any = None) -> str:
    """
    Function Description:
        This function is used to enter the whisper mode.
    Args:
        command_history : Any : The command history object.
    Keyword Args:
        npc : Any : The NPC object.
    Returns:
        str : The output of the whisper mode.
    """

    if npc is not None:
        llm_name = npc.name
    else:
        llm_name = "LLM"
    try:
        model = whisper.load_model("base")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return "Error: Unable to load Whisper model"

    whisper_output = []
    npc_info = f" (NPC: {npc.name})" if npc else ""
    messages = []  # Initialize messages list for conversation history

    print(f"Entering whisper mode{npc_info}. Calibrating silence level...")
    try:
        silence_threshold = calibrate_silence()
    except Exception as e:
        print(f"Error calibrating silence: {e}")
        return "Error: Unable to calibrate silence"

    print("Ready. Speak after seeing 'Listening...'. Say 'exit' or type '/wq' to quit.")
    speak_text("Whisper mode activated. Ready for your input.")

    while True:
        # try:
        audio_data = record_audio(silence_threshold=silence_threshold)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            wf = wave.open(temp_audio.name, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data)
            wf.close()

            result = model.transcribe(temp_audio.name)
            text = result["text"].strip()

        os.unlink(temp_audio.name)

        print(f"You said: {text}")
        whisper_output.append(f"User: {text}")

        if text.lower() == "exit":
            print("Exiting whisper mode.")
            speak_text("Exiting whisper mode. Goodbye!")
            break

        messages.append({"role": "user", "content": text})  # Add user message

        llm_response = check_llm_command(
            text, command_history, npc=npc, messages=messages
        )  # Use check_llm_command
        # print(type(llm_response))
        if isinstance(llm_response, dict):
            print(f"{llm_name}: {llm_response['output']}")  # Print assistant's reply
            whisper_output.append(
                f"{llm_name}: {llm_response['output']}"
            )  # Add to output
            # speak_text(llm_response["output"])  # Speak assistant's reply
        elif isinstance(llm_response, list) and len(llm_response) > 0:
            assistant_reply = messages[-1]["content"]
            print(f"{llm_name}: {assistant_reply}")  # Print assistant's reply
            whisper_output.append(f"{llm_name}: {assistant_reply}")  # Add to output
            # speak_text(assistant_reply)  # Speak assistant's reply
        elif isinstance(
            llm_response, str
        ):  # Handle string responses (errors or direct replies)
            print(f"{llm_name}: {llm_response}")
            whisper_output.append(f"{llm_name}: {llm_response}")
            # speak_text(llm_response)

        # command_history.add(...)  This is now handled inside check_llm_command

        print("\nPress Enter to speak again, or type '/wq' to quit.")
        user_input = input()
        if user_input.lower() == "/wq":
            print("Exiting whisper mode.")
            speak_text("Exiting whisper mode. Goodbye!")
            break

    # except Exception as e:
    #    print(f"Error in whisper mode: {e}")
    #    whisper_output.append(f"Error: {e}")

    return "\n".join(whisper_output)


def enter_notes_mode(command_history: Any, npc: Any = None) -> None:
    """
    Function Description:

    Args:
        command_history : Any : The command history object.
    Keyword Args:
        npc : Any : The NPC object.
    Returns:
        None

    """

    npc_name = npc.name if npc else "base"
    print(f"Entering notes mode (NPC: {npc_name}). Type '/nq' to exit.")

    while True:
        note = input("Enter your note (or '/nq' to quit): ").strip()

        if note.lower() == "/nq":
            break

        save_note(note, command_history, npc)

    print("Exiting notes mode.")


def save_note(note: str, command_history: Any, npc: Any = None) -> None:
    """
    Function Description:
        This function is used to save a note.
    Args:
        note : str : The note to save.
        command_history : Any : The command history object.
    Keyword Args:
        npc : Any : The NPC object.
    Returns:
        None
    """
    current_dir = os.getcwd()
    timestamp = datetime.datetime.now().isoformat()
    npc_name = npc.name if npc else "base"

    # Assuming command_history has a method to access the database connection
    conn = command_history.conn
    cursor = conn.cursor()

    # Create notes table if it doesn't exist
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        note TEXT,
        npc TEXT,
        directory TEXT
    )
    """
    )

    # Insert the note into the database
    cursor.execute(
        """
    INSERT INTO notes (timestamp, note, npc, directory)
    VALUES (?, ?, ?, ?)
    """,
        (timestamp, note, npc_name, current_dir),
    )

    conn.commit()

    print("Note saved to database.")
    # save the note with the current datestamp to the current working directory
    with open(f"{current_dir}/note_{timestamp}.txt", "w") as f:
        f.write(note)


def enter_data_analysis_mode(command_history: Any, npc: Any = None) -> None:
    """
    Function Description:
        This function is used to enter the data analysis mode.
    Args:
        command_history : Any : The command history object.
    Keyword Args:
        npc : Any : The NPC object.
    Returns:
        None
    """

    npc_name = npc.name if npc else "data_analyst"
    print(f"Entering data analysis mode (NPC: {npc_name}). Type '/daq' to exit.")

    dataframes = {}  # Dict to store dataframes by name
    context = {"dataframes": dataframes}  # Context to store variables
    messages = []  # For conversation history if needed

    while True:
        user_input = input(f"{npc_name}> ").strip()

        if user_input.lower() == "/daq":
            break

        # Add user input to messages for context if interacting with LLM
        messages.append({"role": "user", "content": user_input})

        # Process commands
        if user_input.lower().startswith("load "):
            # Command format: load <file_path> as <df_name>
            try:
                parts = user_input.split()
                file_path = parts[1]
                if "as" in parts:
                    as_index = parts.index("as")
                    df_name = parts[as_index + 1]
                else:
                    df_name = "df"  # Default dataframe name
                # Load data into dataframe
                df = pd.read_csv(file_path)
                dataframes[df_name] = df
                print(f"Data loaded into dataframe '{df_name}'")
                # Record in command_history
                command_history.add(
                    user_input, ["load"], f"Loaded data into {df_name}", os.getcwd()
                )
            except Exception as e:
                print(f"Error loading data: {e}")

        elif user_input.lower().startswith("sql "):
            # Command format: sql <SQL query>
            try:
                query = user_input[4:]  # Remove 'sql ' prefix
                df = pd.read_sql_query(query, npc.db_conn)
                print(df)
                # Optionally store result in a dataframe
                dataframes["sql_result"] = df
                print("Result stored in dataframe 'sql_result'")
                command_history.add(
                    user_input, ["sql"], "Executed SQL query", os.getcwd()
                )
            except Exception as e:
                print(f"Error executing SQL query: {e}")

        elif user_input.lower().startswith("plot "):
            # Command format: plot <pandas plotting code>
            try:
                code = user_input[5:]  # Remove 'plot ' prefix
                # Prepare execution environment
                exec_globals = {"pd": pd, "plt": plt, **dataframes}
                exec(code, exec_globals)
                plt.show()
                command_history.add(user_input, ["plot"], "Generated plot", os.getcwd())
            except Exception as e:
                print(f"Error generating plot: {e}")

        elif user_input.lower().startswith("exec "):
            # Command format: exec <Python code>
            try:
                code = user_input[5:]  # Remove 'exec ' prefix
                # Prepare execution environment
                exec_globals = {"pd": pd, "plt": plt, **dataframes}
                exec(code, exec_globals)
                # Update dataframes with any new or modified dataframes
                dataframes.update(
                    {
                        k: v
                        for k, v in exec_globals.items()
                        if isinstance(v, pd.DataFrame)
                    }
                )
                command_history.add(user_input, ["exec"], "Executed code", os.getcwd())
            except Exception as e:
                print(f"Error executing code: {e}")

        elif user_input.lower().startswith("help"):
            # Provide help information
            print(
                """
Available commands:
- load <file_path> as <df_name>: Load CSV data into a dataframe.
- sql <SQL query>: Execute SQL query.
- plot <pandas plotting code>: Generate plots using matplotlib.
- exec <Python code>: Execute arbitrary Python code.
- help: Show this help message.
- /daq: Exit data analysis mode.
"""
            )

        else:
            # Unrecognized command
            print("Unrecognized command. Type 'help' for a list of available commands.")

    print("Exiting data analysis mode.")


def enter_data_mode(command_history: Any, npc: Any = None) -> None:
    """
    Function Description:
        This function is used to enter the data mode.
    Args:
        command_history : Any : The command history object.
    Keyword Args:
        npc : Any : The NPC object.
    Returns:
        None
    """
    npc_name = npc.name if npc else "data_analyst"
    print(f"Entering data mode (NPC: {npc_name}). Type '/dq' to exit.")

    exec_env = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "os": os,
        "npc": npc,
    }

    while True:
        try:
            user_input = input(f"{npc_name}> ").strip()
            if user_input.lower() == "/dq":
                break
            elif user_input == "":
                continue

            # First check if input exists in exec_env
            if user_input in exec_env:
                result = exec_env[user_input]
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        print(result.to_string())
                    else:
                        print(result)
                continue

            # Then check if it's a natural language query
            if not any(
                keyword in user_input
                for keyword in [
                    "=",
                    "+",
                    "-",
                    "*",
                    "/",
                    "(",
                    ")",
                    "[",
                    "]",
                    "{",
                    "}",
                    "import",
                ]
            ):
                if "df" in exec_env and isinstance(exec_env["df"], pd.DataFrame):
                    df_info = {
                        "shape": exec_env["df"].shape,
                        "columns": list(exec_env["df"].columns),
                        "dtypes": exec_env["df"].dtypes.to_dict(),
                        "head": exec_env["df"].head().to_dict(),
                        "summary": exec_env["df"].describe().to_dict(),
                    }

                    analysis_prompt = f"""Based on this DataFrame info: {df_info}
                    Generate Python analysis commands to answer: {user_input}
                    Return each command on a new line. Do not use markdown formatting or code blocks."""

                    analysis_response = npc.get_llm_response(analysis_prompt).get(
                        "response", ""
                    )
                    analysis_commands = [
                        cmd.strip()
                        for cmd in analysis_response.replace("```python", "")
                        .replace("```", "")
                        .split("\n")
                        if cmd.strip()
                    ]
                    results = []

                    print("\nAnalyzing data...")
                    for cmd in analysis_commands:
                        if cmd.strip():
                            try:
                                result = eval(cmd, exec_env)
                                if result is not None:
                                    render_markdown(f"\n{cmd} ")
                                    if isinstance(result, pd.DataFrame):
                                        render_markdown(result.to_string())
                                    else:
                                        render_markdown(result)
                                    results.append((cmd, result))
                            except SyntaxError:
                                try:
                                    exec(cmd, exec_env)
                                except Exception as e:
                                    print(f"Error in {cmd}: {str(e)}")
                            except Exception as e:
                                print(f"Error in {cmd}: {str(e)}")

                    if results:
                        interpretation_prompt = f"""Based on these analysis results:
                        {[(cmd, str(result)) for cmd, result in results]}

                        Provide a clear, concise interpretation of what we found in the data.
                        Focus on key insights and patterns. Do not use markdown formatting."""

                        print("\nInterpretation:")
                        interpretation = npc.get_llm_response(
                            interpretation_prompt
                        ).get("response", "")
                        interpretation = interpretation.replace("```", "").strip()
                        render_markdown(interpretation)
                    continue

            # If not in exec_env and not natural language, try as Python code
            try:
                result = eval(user_input, exec_env)
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        print(result.to_string())
                    else:
                        print(result)
            except SyntaxError:
                exec(user_input, exec_env)
            except Exception as e:
                print(f"Error: {str(e)}")

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Exiting data mode.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

    return


def enter_spool_mode(
    command_history: Any, inherit_last: int = 0, npc: Any = None
) -> Dict:
    """
    Function Description:
        This function is used to enter the spool mode.
    Args:
        command_history : Any : The command history object.
        inherit_last : int : The number of last commands to inherit.
    Keyword Args:
        npc : Any : The NPC object.
    Returns:
        Dict : The messages and output.

    """
    npc_info = f" (NPC: {npc.name})" if npc else ""
    print(f"Entering spool mode{npc_info}. Type '/sq' to exit spool mode.")
    spool_context = []
    system_message = get_system_message(npc) if npc else "You are a helpful assistant."
    # insert at the first position
    spool_context.insert(0, {"role": "assistant", "content": system_message})
    # Inherit last n messages if specified
    if inherit_last > 0:
        last_commands = command_history.get_all(limit=inherit_last)
        for cmd in reversed(last_commands):
            spool_context.append({"role": "user", "content": cmd[2]})  # command
            spool_context.append({"role": "assistant", "content": cmd[4]})  # output

    while True:
        try:
            user_input = input("spool> ").strip()
            if user_input.lower() == "/sq":
                print("Exiting spool mode.")
                break

            # Add user input to spool context
            spool_context.append({"role": "user", "content": user_input})

            # Prepare kwargs for get_conversation
            kwargs_to_pass = {}
            if npc:
                kwargs_to_pass["npc"] = npc
                if npc.model:
                    kwargs_to_pass["model"] = npc.model
                if npc.provider:
                    kwargs_to_pass["provider"] = npc.provider

            # Get the conversation
            conversation_result = get_conversation(spool_context, **kwargs_to_pass)

            # Handle potential errors in conversation_result
            if isinstance(conversation_result, str) and "Error" in conversation_result:
                print(conversation_result)  # Print the error message
                continue  # Skip to the next iteration of the loop
            elif (
                not isinstance(conversation_result, list)
                or len(conversation_result) == 0
            ):
                print("Error: Invalid response from get_conversation")
                continue

            spool_context = conversation_result  # update spool_context

            # Extract assistant's reply, handling potential KeyError
            try:
                # import pdb ; pdb.set_trace()

                assistant_reply = spool_context[-1]["content"]
            except (KeyError, IndexError) as e:
                print(f"Error extracting assistant's reply: {e}")
                print(
                    f"Conversation result: {conversation_result}"
                )  # Print for debugging
                continue

            command_history.add(
                user_input,
                ["spool", npc.name if npc else ""],
                assistant_reply,
                os.getcwd(),
            )
            render_markdown(assistant_reply)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting spool mode.")
            break

    return {
        "messages": spool_context,
        "output": "\n".join(
            [msg["content"] for msg in spool_context if msg["role"] == "assistant"]
        ),
    }
