import os
import sys
import readline
import atexit
import re
import pty
import select
import termios
import tty
import shlex
import json
from datetime import datetime

# Third-party imports
import pandas as pd
import sqlite3
import numpy as np
from termcolor import colored
from dotenv import load_dotenv
import subprocess
from typing import Dict, Any, List, Optional

try:
    from sentence_transformers import SentenceTransformer
except:
    print("Could not load the sentence-transformers package.")
# Local imports
from .command_history import (
    CommandHistory,
    start_new_conversation,
    save_conversation_message,
)
from .llm_funcs import (
    execute_llm_command,
    execute_llm_question,
    generate_image,
    lookup_provider,
    check_llm_command,
    get_conversation,
    get_system_message,
)
from .search import rag_search, search_web
from .helpers import (
    load_all_files,
    initialize_npc_project,
    setup_npcsh_config,
    is_npcsh_initialized,
    initialize_base_npcs_if_needed,
)
from .shell_helpers import (
    complete,  # For command completion
    readline_safe_prompt,
    get_multiline_input,
    setup_readline,
    execute_command,
    orange,  # For colored prompt
)
from .npc_compiler import NPCCompiler, load_tools_from_directory

import argparse
from .serve import (
    start_flask_server,
)  # Assuming you move the Flask logic to a separate `serve.py`


def main() -> None:
    """
    Main function for the npcsh shell and server.
    Starts either the Flask server or the interactive shell based on the argument provided.
    """
    # Set up argument parsing to handle 'serve' and regular commands
    parser = argparse.ArgumentParser(description="npcsh CLI")
    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        help="The command to run ('serve' to start Flask server)",
    )
    args = parser.parse_args()

    # If 'serve' is not provided, proceed with the regular CLI
    setup_npcsh_config()
    if "NPCSH_DB_PATH" in os.environ:
        db_path = os.path.expanduser(os.environ["NPCSH_DB_PATH"])
    else:
        db_path = os.path.expanduser("~/npcsh_history.db")

    command_history = CommandHistory(db_path)
    valid_commands = [
        "/compile",
        "/com",
        "/whisper",
        "/notes",
        "/data",
        "/cmd",
        "/command",
        "/set",
        "/sample",
        "/spool",
        "/sp",
        "/help",
        "/exit",
        "/quit",
    ]

    readline.set_completer_delims(" \t\n")
    readline.set_completer(complete)
    if sys.platform == "darwin":
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")

    # Initialize base NPCs and tools
    initialize_base_npcs_if_needed(db_path)

    user_npc_directory = os.path.expanduser("~/.npcsh/npc_team")
    project_npc_directory = os.path.abspath("./npc_team/")

    npc_compiler = NPCCompiler(user_npc_directory, db_path)

    # Compile all NPCs in the user's npc_team directory
    for filename in os.listdir(user_npc_directory):
        if filename.endswith(".npc"):
            npc_file_path = os.path.join(user_npc_directory, filename)
            npc_compiler.compile(npc_file_path)

    # Compile NPCs from project-specific npc_team directory
    if os.path.exists(project_npc_directory):
        for filename in os.listdir(project_npc_directory):
            if filename.endswith(".npc"):
                npc_file_path = os.path.join(project_npc_directory, filename)
                npc_compiler.compile(npc_file_path)

    if not is_npcsh_initialized():
        print("Initializing NPCSH...")
        initialize_base_npcs_if_needed(db_path)
        print(
            "NPCSH initialization complete. Please restart your terminal or run 'source ~/.npcshrc' for the changes to take effect."
        )

    history_file = setup_readline()
    atexit.register(readline.write_history_file, history_file)
    atexit.register(command_history.close)
    # make npcsh into ascii art
    from colorama import init

    init()  # Initialize colorama for ANSI code support
    print(
        """
Welcome to \033[1;94mnpc\033[0m\033[1;38;5;202msh\033[0m!
  \033[1;94m                    \033[0m\033[1;38;5;202m               \\\\
  \033[1;94m _ __   _ __    ___ \033[0m\033[1;38;5;202m ___  | |___    \\\\
  \033[1;94m| '_ \ | '_ \  / __|\033[0m\033[1;38;5;202m/ __/ | |_ _|    \\\\
  \033[1;94m| | | || |_) |( |__ \033[0m\033[1;38;5;202m\_  \ | | | |    //
  \033[1;94m|_| |_|| .__/  \___|\033[0m\033[1;38;5;202m|___/ |_| |_|   //
         \033[1;94m| |          \033[0m\033[1;38;5;202m               //
         \033[1;94m| |
         \033[1;94m|_|

Begin by asking a question, issuing a bash command, or typing '/help' for more information.
"""
    )

    current_npc = None
    messages = None
    current_conversation_id = start_new_conversation()

    while True:
        try:
            if current_npc:
                prompt = f"{colored(os.getcwd(), 'blue')}:{orange(current_npc.name)}> "
            else:
                prompt = f"{colored(os.getcwd(), 'blue')}:\033[1;94mnpc\033[0m\033[1;38;5;202msh\033[0m!> "

            prompt = readline_safe_prompt(prompt)
            user_input = get_multiline_input(prompt).strip()

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
                current_npc,
                messages=messages,
                conversation_id=current_conversation_id,
            )

            messages = result.get("messages", messages)

            if "current_npc" in result:
                current_npc = result["current_npc"]

            output = result.get("output")
            save_conversation_message(
                command_history, current_conversation_id, "user", user_input
            )
            save_conversation_message(
                command_history,
                current_conversation_id,
                "assistant",
                result.get("output", ""),
            )

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
