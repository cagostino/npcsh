# npcsh.py
import os

import readline
import atexit
from datetime import datetime
import pandas as pd

# Configure logging
import sqlite3
from termcolor import colored

import subprocess
from .command_history import CommandHistory
from .llm_funcs import execute_llm_command, execute_llm_question, check_llm_command
from .modes import (
    enter_whisper_mode,
    enter_notes_mode,
    enter_spool_mode,
)
from .helpers import (
    log_action,
    capture_screenshot,
    load_npc_from_file,
    list_directory,
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

from .npc_compiler import NPCCompiler, NPC
from colorama import Fore, Back, Style

import shlex
import subprocess


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
            preexec_fn=os.setsid  # Create a new process group
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
    'ipython': ['ipython'],
    'python': ['python', '-i'],
    'sqlite3': ['sqlite3'],
    'r': ['R', '--interactive'],}

def execute_command(command, command_history, db_path, npc_compiler, current_npc=None):
    subcommands = []
    output = ""
    location = os.getcwd()
    # Extract NPC from command
    db_conn = sqlite3.connect(db_path)

    if current_npc is None:
        valid_npcs = get_valid_npcs(db_path)

        npc_name = get_npc_from_command(command)
        if npc_name is None:
            npc_name = "base"  # Default NPC
        # print(npc_name)
        npc_path = get_npc_path(npc_name, db_path)
        npc = load_npc_from_file(npc_path, db_conn)
    else:
        npc = current_npc
    if command.startswith("/"):
        command = command[1:]
        log_action("Command Executed", command)

        command_parts = command.split()
        command_name = command_parts[0]
        args = command_parts[1:]
        if current_npc is None and command_name in valid_npcs:
            npc_path = get_npc_path(command_name, db_path)
            npc = load_npc_from_file(npc_path, db_conn)

        if command_name == "compile" or command_name == "com":
            try:
                compiled_script = npc_compiler.compile(npc)
                output = f"Compiled NPC profile: {compiled_script}"
                print(output)
            except Exception as e:
                output = f"Error compiling NPC profile: {str(e)}"
                print(output)
        elif command_name == 'ots':
            output = capture_screenshot()
            #add llm part
            output =  f"Screenshot captured: {output['filename']}\nFull path: {output['file_path']}\nLLM-ready data available."
            return output
        elif command_name == "whisper":
            output = enter_whisper_mode(command_history, npc=npc)
        elif command_name == "notes":
            output = enter_notes_mode(command_history, npc=npc)
        elif command_name == "data":
            request = " ".join(args)
            output = get_data_response(request, npc=npc)
            # output = enter_observation_mode(command_history, npc=npc)
        elif command_name == "cmd" or command_name == "command":
            output = execute_llm_command(command, command_history, npc=npc)
        elif command_name == "set":
            parts = command.split()
            if len(parts) == 3 and parts[1] in ["model", "provider", "db_path"]:
                output = execute_set_command(parts[1], parts[2])
            else:
                return "Invalid set command. Usage: /set [model|provider|db_path] 'value_in_quotes' "
        elif command_name == "sample":
            output = execute_llm_question(command, command_history, npc=npc)
        elif command_name == "spool" or command_name == "sp":
            inherit_last = 0
            for part in args:
                if part.startswith("inherit_last="):
                    try:
                        inherit_last = int(part.split("=")[1])
                    except ValueError:
                        return "Error: inherit_last must be an integer"
                    break

            output = enter_spool_mode(command_history, inherit_last, npc=npc)
            return output

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
            output = check_llm_command(command, command_history, npc=npc)
    else:
        # Check if it's a bash command
        # print(command)
        command_parts = shlex.split(command)

        if command_parts[0] in interactive_commands:
            print(f"Starting interactive {command_parts[0]} session...")
            return_code = start_interactive_session(interactive_commands[command_parts[0]])
            return f"Interactive {command_parts[0]} session ended with return code {return_code}"

        elif command_parts[0] == "cd":
            try:
                if len(command_parts) > 1:
                    new_dir = os.path.expanduser(command_parts[1])
                else:
                    new_dir = os.path.expanduser("~")
                os.chdir(new_dir)
                return f"Changed directory to {os.getcwd()}"
            except FileNotFoundError:
                return f"Directory not found: {new_dir}"
            except PermissionError:
                return f"Permission denied: {new_dir}"

        elif command_parts[0] in BASH_COMMANDS:
            if command_parts[0] in TERMINAL_EDITORS:
                return open_terminal_editor(command)
            elif command.startswith("open "):
                try:
                    expanded_command = os.path.expanduser(command)
                    subprocess.Popen(shlex.split(expanded_command), 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL,
                                 start_new_session=True)
                    output = f"Launched: {command}"
                except Exception as e:
                    output = colored(f"Error executing command: {str(e)}", "red")
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
                except Exception as e:
                    output = colored(f"Error executing command: {e}", "red")

        else:
            output = check_llm_command(command, command_history, npc=npc)

    # Add command to history
    command_history.add(command, subcommands, output, location)

    # Print the output
    if output:
        print(output)

    return output


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

    return history_file


def save_readline_history():
    readline.write_history_file(os.path.expanduser("~/.npcsh_readline_history"))


def orange(text):
    return f"\033[38;2;255;165;0m{text}{Style.RESET_ALL}"


def main():
    setup_npcsh_config()
    if "NPCSH_DB_PATH" in os.environ:
        db_path = os.path.expanduser(os.environ["NPCSH_DB_PATH"])
    else:
        db_path = os.path.expanduser("~/npcsh_history.db")

    command_history = CommandHistory(db_path)

    os.makedirs("./npc_profiles", exist_ok=True)
    npc_directory = os.path.expanduser("./npc_profiles")
    npc_compiler = NPCCompiler(npc_directory)

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
            elif user_input.startswith("/") and not current_npc:
                command_parts = user_input[1:].split()
                command_name = command_parts[0]
                valid_npcs = get_valid_npcs(db_path)
                if command_name in valid_npcs:
                    npc_path = get_npc_path(command_name, db_path)
                    db_conn = sqlite3.connect(db_path)
                    current_npc = load_npc_from_file(npc_path, db_conn)
                    print(
                        f"Entered {current_npc.name} mode. Type 'exit' to return to main shell."
                    )
                    continue

            # Execute the command and capture the result
            result = execute_command(
                user_input, command_history, db_path, npc_compiler, current_npc
            )
            
            # If there's a result, print it
            # This is important for interactive sessions, which will return a message
            # when they end
            if result:
                print(result)

        except (KeyboardInterrupt, EOFError):
            if current_npc:
                print(f"\nExiting {current_npc.name} mode.")
                current_npc = None
            else:
                print("\nGoodbye!")
                break