# npcsh.py
import os

import readline
import atexit
from datetime import datetime
import pandas as pd

# Configure logging


from .command_history import CommandHistory
from .llm_funcs import (
    get_llm_response,
    execute_llm_command,
    check_llm_command,
    execute_llm_question,
    execute_llm_thought,
)
from .modes import (
    enter_bash_mode,
    enter_whisper_mode,
    enter_notes_mode,
    enter_observation_mode,
    enter_spool_mode,
)
from .helpers import log_action, list_directory, read_file
from .npc_compiler import NPCCompiler


def execute_command(command, command_history, db_path, npc_compiler):
    subcommands = []
    output = ""
    location = os.getcwd()

    if command.startswith("/"):
        command = command[1:]
        log_action("Command Executed", command)

        command_parts = command.split()
        command_name = command_parts[0]
        args = command_parts[1:]

        if (
            command_name == "b"
            or command_name == "bash"
            or command_name == "zsh"
            or command_name == "sh"
        ):
            output = enter_bash_mode()
        if command_name == "compile" or command_name == "com":
            try:
                compiled_script = npc_compiler.compile(args[0])
                output = f"Compiled NPC profile: {compiled_script}"
                print(output)
            except Exception as e:
                output = f"Error compiling NPC profile: {str(e)}"
                print(output)

        elif command_name == "whisper":
            output = enter_whisper_mode()
        elif command_name == "notes":
            output = enter_notes_mode(command_history)
        elif command_name == "obs":
            print(db_path)
            output = enter_observation_mode(command_history)
        elif command_name == "cmd" or command_name == "command":
            output = execute_llm_command(command, command_history)
        elif command_name == "?":
            output = execute_llm_question(command, command_history)
        elif command_name == "th" or command_name == "thought":
            output = execute_llm_thought(command, command_history)
        elif command_name == "spool" or command_name == "sp":
            inherit_last = int(args[0]) if args else 0

            output = enter_spool_mode(command_history, inherit_last)
        else:
            output = f"Unknown command: {command_name}"

        subcommands = [f"/{command}"]
    else:
        output = check_llm_command(command, command_history)

    command_history.add(command, subcommands, output, location)


def setup_readline():
    readline.set_history_length(1000)
    readline.parse_and_bind("set editing-mode vi")
    readline.parse_and_bind('"\C-r": reverse-search-history')


def save_readline_history():
    readline.write_history_file(os.path.expanduser("~/.npcsh_readline_history"))


def main():
    # check if theres a set path to a local db in the os env
    if "NPCSH_DB_PATH" in os.environ:
        db_path = os.environ["NPCSH_DB_PATH"]
        command_history = CommandHistory(db_path)

    else:
        db_path = "~/npcsh_history.db"
        command_history = CommandHistory()
    # Initialize NPCCompiler
    os.makedirs("./npc_profiles", exist_ok=True)
    npc_directory = os.path.expanduser(
        "./npc_profiles"
    )  # You can change this to your preferred directory
    npc_compiler = NPCCompiler(npc_directory)

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
