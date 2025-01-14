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
from .cli_helpers import (
    complete,  # For command completion
    readline_safe_prompt,
    get_multiline_input,
    setup_readline,
    execute_command,
    orange,  # For colored prompt
)
from .npc_compiler import NPCCompiler


def main() -> None:
    """
    Function Description:
        Main function for the npcsh shell.
    Args:
        None
    Keyword Args:
        None
    Returns:
        None

    """

    setup_npcsh_config()
    if "NPCSH_DB_PATH" in os.environ:
        db_path = os.path.expanduser(os.environ["NPCSH_DB_PATH"])
    else:
        db_path = os.path.expanduser("~/npcsh_history.db")

    command_history = CommandHistory(db_path)
    global valid_commands  # Make sure it's global
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

    readline.set_completer_delims(" \t\n")  # Simplified delims
    readline.set_completer(complete)
    if sys.platform == "darwin":  # For macOS
        readline.parse_and_bind("bind ^I rl_complete")
    else:  # For Linux and others
        readline.parse_and_bind("tab: complete")

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
            npc_file_path = os.path.join(user_npc_directory, filename)
            npc_compiler.compile(npc_file_path)

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
    current_conversation_id = start_new_conversation()

    while True:
        try:
            if current_npc:
                prompt = f"{colored(os.getcwd(), 'blue')}:{orange(current_npc.name)}> "
            else:
                prompt = f"{colored(os.getcwd(), 'blue')}:{orange('npcsh')}> "

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
                embedding_model,
                current_npc,
                text_data=text_data,
                text_data_embedded=text_data_embedded,
                messages=messages,
                conversation_id=current_conversation_id,
            )

            # Update messages with the new conversation history
            messages = result.get("messages", messages)

            if "current_npc" in result:
                current_npc = result["current_npc"]

            # Optionally, print the assistant's response
            output = result.get("output")
            # if output:
            #    print(output)
            # Save messages to conversation history
            save_conversation_message(
                command_history, current_conversation_id, "user", user_input
            )
            save_conversation_message(
                command_history,
                current_conversation_id,
                "assistant",
                result.get("output", ""),
            )

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
