# npcsh.py
import os

import readline
import atexit
from datetime import datetime
import pandas as pd

# Configure logging
import sqlite3
from termcolor import colored
from dotenv import load_dotenv
import subprocess
from .command_history import CommandHistory
from .llm_funcs import (
    render_markdown,
    execute_llm_command,
    execute_llm_question,
    generate_image,
    check_llm_command,
)
from .modes import (
    enter_whisper_mode,
    enter_notes_mode,
    enter_spool_mode,
    enter_data_mode,
)

import json
from .helpers import (
    log_action,
    load_all_files,
    rag_search,
    capture_screenshot,
    initialize_npc_project,
    list_directory,
    analyze_image,
    read_file,
    ensure_npcshrc_exists,
    setup_npcsh_config,
    execute_java,
    execute_scala,
    execute_r,
    execute_sql,
    execute_python,
    BASH_COMMANDS,
    TERMINAL_EDITORS,
    open_terminal_editor,
    get_npc_from_command,
    is_valid_npc,
    get_npc_path,
    initialize_base_npcs_if_needed,
    is_npcsh_initialized,
    set_npcsh_initialized,
    get_valid_npcs,
)
from sentence_transformers import SentenceTransformer, util

from .npc_compiler import NPCCompiler, NPC, load_npc_from_file

from colorama import Fore, Back, Style
import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers"
)  # transformers
# warnings.filterwarnings(
#    "ignore", message="CUDA initialization: The NVIDIA driver on your system is too old"
# )

import shlex
import subprocess
import os
from dotenv import load_dotenv
import json
import pandas as pd
import numpy as np


def get_file_color(filepath):
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


import pty
import select
import termios
import tty
import sys

import time
import signal


def start_interactive_session(command):
    """
    Start an interactive session for the given command.
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


interactive_commands = {
    "ipython": ["ipython"],
    "python": ["python", "-i"],
    "sqlite3": ["sqlite3"],
    "r": ["R", "--interactive"],
}


def validate_bash_command(command_parts):
    """Validate if the command sequence is a valid bash command with proper arguments/flags"""
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
            "requires_arg": False,
        },
        "open": {
            "flags": ["-a", "-e", "-t", "-f", "-F", "-W", "-n", "-g", "-h"],
            "requires_arg": True,
        },
        "which": {"flags": ["-a", "-s", "-v"], "requires_arg": False},
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


def execute_command(
    command,
    command_history,
    db_path,
    npc_compiler,
    embedding_model=None,
    current_npc=None,
    text_data=None,
    text_data_embedded=None,
    messages=None,
):
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

    if len(command.strip()) == 0:
        return {"messages": messages, "output": output}
    if messages is None:
        messages = []
    messages.append({"role": "user", "content": command})  # Add user message
    if command.startswith("/"):
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
                render_markdown(output)

            except Exception as e:
                import traceback

                output = (
                    f"Error compiling NPC profile: {str(e)}\n{traceback.format_exc()}"
                )
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
            output = """
            Available commands:

            /compile [npc_file1.npc npc_file2.npc ...]: Compiles specified NPC profile(s). If no arguments are provided, compiles all NPCs in the npc_profiles directory.
            /com [npc_file1.npc npc_file2.npc ...]: Alias for /compile.
            /whisper: Enter whisper mode.
            /notes: Enter notes mode.
            /data: Enter data mode.
            /cmd <command>: Execute a command using the current NPC's LLM.
            /command <command>: Alias for /cmd.
            /set <model|provider|db_path> <value>: Sets the specified parameter. Enclose the value in quotes.
            /sample <question>: Asks the current NPC a question.
            /spool [inherit_last=<n>]: Enters spool mode. Optionally inherits the last <n> messages.
            /sp [inherit_last=<n>]: Alias for /spool.
            /<npc_name>: Enters the specified NPC's mode.
            /help: Displays this help message.
            /exit or /quit: Exits the current NPC mode or the npcsh shell.

            Bash commands and other programs can be executed directly.
            """
            print(output)  # Print the help message
            return {"messages": messages, "output": None}
            # Don't add /help to command history

        elif command_name == "whisper":
            try:
                output = enter_whisper_mode(command_history, npc=npc)
            except Exception as e:
                print(f"Error entering whisper mode: {str(e)}")
                output = "Error entering whisper mode"

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
                command, command_history, npc=npc, retrieved_docs=retrieved_docs
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

        ## flush  the current shell context
    elif command.startswith("#"):
        command_parts = command[1:].split()
        language = command_parts[0].lower()
        code = " ".join(command_parts[1:])

        if language == "python":
            output = execute_python(code)
        elif language == "r":
            output = execute_r(code)
        elif language == "sql":
            output = execute_sql(code)
        elif language == "scala":
            output = execute_scala(code)
        elif language == "java":
            output = execute_java(code)
        else:
            output = check_llm_command(
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

        elif command_parts[0] in BASH_COMMANDS:
            if command_parts[0] in TERMINAL_EDITORS:
                return {"messages": messages, "output": open_terminal_editor(command)}
            elif command_parts[0] in ["cat", "find", "who", "open", "which"]:
                if not validate_bash_command(command_parts):
                    output = "Error: Invalid command syntax or arguments"  # Assign error message directly
                    output = check_llm_command(
                        command,
                        command_history,
                        npc=npc,
                        retrieved_docs=retrieved_docs,
                        messages=messages,
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
            # print('dsf')
            # print(npc)

            # print(messages)
            output = check_llm_command(
                command,
                command_history,
                npc=npc,
                retrieved_docs=retrieved_docs,
                messages=messages,
            )

    # Add command to history
    command_history.add(command, subcommands, output, location)

    # Print the output
    # if output:
    #    print(output)
    if isinstance(output, dict):
        response = output.get("output", "")
        new_messages = output.get("messages", None)
        if new_messages is not None:
            messages = new_messages
        output = response

    return {"messages": messages, "output": output}


def setup_readline():
    history_file = os.path.expanduser("~/.npcsh_history")
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass

    readline.set_history_length(1000)
    readline.parse_and_bind("set editing-mode vi")
    readline.parse_and_bind('"\e[A": history-search-backward')
    readline.parse_and_bind('"\e[B": history-search-forward')
    readline.parse_and_bind('"\C-r": reverse-search-history')
    # Add these lines to enable multiline input with readline:
    readline.parse_and_bind("set enable-bracketed-paste on")
    readline.parse_and_bind("set mark-modified-lines on")
    readline.parse_and_bind("\C-e: end-of-line")
    readline.parse_and_bind("\C-a: beginning-of-line")

    return history_file


def save_readline_history():
    readline.write_history_file(os.path.expanduser("~/.npcsh_readline_history"))


def orange(text):
    return f"\033[38;2;255;165;0m{text}{Style.RESET_ALL}"


# npcsh.py


def main():
    setup_npcsh_config()
    if "NPCSH_DB_PATH" in os.environ:
        db_path = os.path.expanduser(os.environ["NPCSH_DB_PATH"])
    else:
        db_path = os.path.expanduser("~/npcsh_history.db")

    command_history = CommandHistory(db_path)

    # Initialize base NPCs and tools
    initialize_base_npcs_if_needed(db_path)

    # Set npc_directory to the user's ~/.npcsh/npc_team directory
    user_npc_directory = os.path.expanduser("~/.npcsh/npc_team")

    # Check for project-specific npc_team directory
    project_npc_directory = os.path.abspath("./npc_team")
    npc_compiler = NPCCompiler(user_npc_directory, db_path)

    # Compile all NPCs in the user's npc_team directory
    for filename in os.listdir(user_npc_directory):
        if filename.endswith(".npc"):
            npc_compiler.compile(filename)

    # If project npc_team directory exists, compile NPCs from there as well
    if os.path.exists(project_npc_directory):
        for filename in os.listdir(project_npc_directory):
            if filename.endswith(".npc"):
                npc_file_path = os.path.join(project_npc_directory, filename)
                npc_compiler.compile(npc_file_path)
                
    # Load all files for RAG searches
    # Define the directory to load text files from
    text_data_directory = os.path.abspath("./")
    # Load all text files from the directory recursively
    text_data = load_all_files(text_data_directory)
    # embed all the text_data

    try:
        # Load the SentenceTransformer model
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Embed all the text data
        text_data_embedded = {
            filename: embedding_model.encode(
                text_data[filename], convert_to_tensor=True, show_progress_bar=False
            )
            for filename in text_data
        }
    except Exception as e:
        print(f"Error embedding text data: {str(e)}")
        text_data_embedded = None

    if not is_npcsh_initialized():
        print("Initializing NPCSH...")
        initialize_base_npcs_if_needed(db_path)
        print(
            "NPCSH initialization complete. Please restart your terminal or run 'source ~/.npcshrc' for the changes to take effect."
        )
    history_file = setup_readline()
    atexit.register(readline.write_history_file, history_file)
    atexit.register(command_history.close)

    print("Welcome to npcsh!")

    current_npc = None
    messages = None
    while True:
        try:
            if current_npc:
                prompt = f"{colored(os.getcwd(), 'blue')}:{orange(current_npc.name)}> "
            else:
                prompt = f"{colored(os.getcwd(), 'blue')}:{orange('npcsh')}> "

            user_input = input(prompt).strip()

            if user_input.lower() in ["exit", "quit"]:
                if current_npc:
                    print(f"Exiting {current_npc.name} mode.")
                    current_npc = None
                    continue
                else:
                    print("Goodbye!")
                    break

            # Execute the command and capture the result
            result = execute_command(
                user_input,
                command_history,
                db_path,
                npc_compiler,
                embedding_model,
                current_npc,
                text_data=text_data,
                text_data_embedded=text_data_embedded,
                messages=messages,
            )

            # Update messages with the new conversation history
            messages = result.get("messages", messages)

            if "current_npc" in result:
                current_npc = result["current_npc"]

            # Optionally, print the assistant's response
            output = result.get("output")
            # if output:
            #    print(output)

            # Optionally, print the result
            if (
                result["output"] is not None
                and not user_input.startswith("/")
                and not isinstance(result, dict)
            ):
                print("final", result)

        except (KeyboardInterrupt, EOFError):
            if current_npc:
                print(f"\nExiting {current_npc.name} mode.")
                current_npc = None
            else:
                print("\nGoodbye!")
                break
