# npcsh.py
import os

import readline
import atexit
from datetime import datetime
import pandas as pd
# Configure logging
import sqlite3

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
    load_npc_from_file,
    list_directory, 
    read_file, 
    ensure_npcshrc_exists,
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
    set_npcsh_initialized
)

from .npc_compiler import NPCCompiler, NPC

import shlex
import subprocess

def execute_command(command, command_history, db_path, npc_compiler):
    subcommands = []
    output = ""
    location = os.getcwd()
    # Extract NPC from command
    npc_name = get_npc_from_command(command)
    #print(npc_name)
    db_conn = sqlite3.connect(db_path)
    if npc_name is None:
        npc_name = "base"  # Default NPC
    #print(npc_name)
    npc_path = get_npc_path(npc_name, db_path)
    
    #print(npc_path)
    npc = load_npc_from_file(npc_path, db_conn)
    #print(npc)
    
    if command.startswith("/"):
        command = command[1:]
        log_action("Command Executed", command)

        command_parts = command.split()
        command_name = command_parts[0]
        args = command_parts[1:]
        # get the args that start with "npc="
        # if the args that start with npc= are more than 1 then raise an error
        # extract the npc
        
        if command_name == "compile" or command_name == "com":
            try:    
                compiled_script = npc_compiler.compile(npc)
                output = f"Compiled NPC profile: {compiled_script}"
                print(output)
            except Exception as e:
                output = f"Error compiling NPC profile: {str(e)}"
                print(output)

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
                output =  execute_set_command(parts[1], parts[2])
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

        if language == 'python':
            output = execute_python(code)
        elif language == 'r':
            output = execute_r(code)
        elif language == 'sql':
            output = execute_sql(code)
        elif language == 'scala':
            output = execute_scala(code)
        elif language == 'java':
            output = execute_java(code)
        else:
            output = check_llm_command(command, command_history, npc=npc)
    else:
        # Check if it's a bash command
        #print(command)
        command_parts = shlex.split(command)
        if command_parts[0] in BASH_COMMANDS:
            if command_parts[0] in TERMINAL_EDITORS:
                # Handle terminal editors
                return open_terminal_editor(command)
            else:
                try:
                    # Run the command and capture output
                    result = subprocess.run(command_parts, capture_output=True, text=True)
                    output = result.stdout
                    if result.stderr:
                        output += f"\nError: {result.stderr}"
                    
                    # If output is empty but the command ran successfully, mention that
                    if not output and result.returncode == 0:
                        output = f"Command '{command}' executed successfully (no output)."
                except Exception as e:
                    output = f"Error executing command: {e}"
        else:
            # If not a recognized bash command, fall back to LLM
            output = check_llm_command(command, command_history, npc=npc)

    # Add command to history
    command_history.add(command, subcommands, output, location)

    # Print the output
    if output:
        print(output)
    
    return output


def setup_readline():
    readline.set_history_length(1000)
    readline.parse_and_bind("set editing-mode vi")
    readline.parse_and_bind('"\e[A": history-search-backward')
    readline.parse_and_bind('"\e[B": history-search-forward')

    readline.parse_and_bind('"\C-r": reverse-search-history')


def save_readline_history():
    readline.write_history_file(os.path.expanduser("~/.npcsh_readline_history"))


def main():
    ensure_npcshrc_exists()
    # check if theres a set path to a local db in the os env
    if "NPCSH_DB_PATH" in os.environ:
        if '~' in os.environ["NPCSH_DB_PATH"]:

            db_path = os.path.expanduser(os.environ["NPCSH_DB_PATH"])
        else:
            db_path = os.environ["NPCSH_DB_PATH"]
        command_history = CommandHistory(db_path)

    else:
        db_path = os.path.expanduser('~/npcsh_history.db')
        command_history = CommandHistory()
    # Initialize NPCCompiler
    
    os.makedirs("./npc_profiles", exist_ok=True)
    npc_directory = os.path.expanduser(
        "./npc_profiles"
    )  # You can change this to your preferred directory
    npc_compiler = NPCCompiler(npc_directory)
    #print(os.environ)
    if not is_npcsh_initialized():
        print("Initializing NPCSH...")
        initialize_base_npcs_if_needed(db_path)
        print("NPCSH initialization complete. Please restart your terminal or run 'source ~/.npcshrc' for the changes to take effect.")
    
    
    setup_readline()
    atexit.register(save_readline_history)
    atexit.register(command_history.close)

    print("Welcome to npcsh!")
    while True:
        try:
            user_input = input("npcsh> ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            else:
                execute_command(user_input, command_history, db_path, npc_compiler)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
