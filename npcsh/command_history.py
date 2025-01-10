import os
import sqlite3
import json
from datetime import datetime

from typing import Optional, List, Dict, Any


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Index, pd.Series, pd.DataFrame)):
            return obj.to_dict()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def show_history(command_history, args):
    if args:
        search_results = command_history.search(args[0])
        if search_results:
            return "\n".join(
                [f"{item[0]}. [{item[1]}] {item[2]}" for item in search_results]
            )
        else:
            return f"No commands found matching '{args[0]}'"
    else:
        all_history = command_history.get_all()
        return "\n".join([f"{item[0]}. [{item[1]}] {item[2]}" for item in all_history])


def query_history_for_llm(command_history, query):
    results = command_history.search(query)
    formatted_results = [
        f"Command: {r[2]}\nOutput: {r[4]}\nLocation: {r[5]}" for r in results
    ]
    return "\n\n".join(formatted_results)


class CommandHistory:
    def __init__(self, path="~/npcsh_history.db"):
        self.db_path = os.path.expanduser(path)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.create_command_table()
        self.create_conversation_table()

    def create_command_table(self):
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS command_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            command TEXT,
            subcommands TEXT,
            output TEXT,
            location TEXT
        )
        """
        )
        self.conn.commit()

    def create_conversation_table(self):
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            role TEXT,
            content TEXT,
            conversation_id TEXT,
            directory_path TEXT
        )
        """
        )
        self.conn.commit()

    def add_command(self, command, subcommands, output, location):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Convert everything to strings to ensure it can be stored
        safe_subcommands = str(subcommands)
        safe_output = str(output)

        self.cursor.execute(
            """
            INSERT INTO command_history (timestamp, command, subcommands, output, location)
            VALUES (?, ?, ?, ?, ?)
        """,
            (timestamp, command, safe_subcommands, safe_output, location),
        )
        self.conn.commit()

    def add_conversation(self, role, content, conversation_id, directory_path):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(content, dict):
            content = json.dumps(content, cls=CustomJSONEncoder)

        self.cursor.execute(
            """
            INSERT INTO conversation_history (timestamp, role, content, conversation_id, directory_path)
            VALUES (?, ?, ?, ?, ?)
        """,
            (timestamp, role, content, conversation_id, directory_path),
        )
        self.conn.commit()

    def get_last_command(self):
        self.cursor.execute(
            """
        SELECT * FROM command_history ORDER BY id DESC LIMIT 1
        """
        )
        return self.cursor.fetchone()

    def close(self):
        self.conn.close()

    def get_last_conversation(self, conversation_id):
        self.cursor.execute(
            """
        SELECT * FROM conversation_history WHERE conversation_id = ? ORDER BY id DESC LIMIT 1
        """,
            (conversation_id,),
        )

        return self.cursor.fetchone()

    def get_last_conversation_by_path(self, directory_path):
        most_recent_conversation_id = self.get_most_recent_conversation_id_by_path(
            directory_path
        )
        convo = self.get_conversations_by_id(most_recent_conversation_id[0])
        return convo

    def get_most_recent_conversation_id_by_path(self, path) -> Optional[str]:
        """Retrieve the most recent conversation ID for the current path."""
        current_path = os.getcwd()
        self.cursor.execute(
            """
            SELECT conversation_id FROM conversation_history
            WHERE directory_path = ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            (path,),
        )
        result = self.cursor.fetchone()
        return result

    def get_conversations_by_id(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve all messages for a specific conversation ID."""
        self.cursor.execute(
            """
            SELECT * FROM conversation_history
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
            """,
            (conversation_id,),
        )
        return self.cursor.fetchall()  #


def start_new_conversation() -> str:
    """
    Starts a new conversation and returns a unique conversation ID.
    """
    return f"conversation_{datetime.now().strftime('%Y%m%d%H%M%S')}"


def save_conversation_message(
    command_history: CommandHistory, conversation_id: str, role: str, content: str
):
    """
    Saves a conversation message linked to a conversation ID.
    """
    command_history.add_conversation(role, content, conversation_id, os.getcwd())


def retrieve_last_conversation(
    command_history: CommandHistory, conversation_id: str
) -> str:
    """
    Retrieves and formats all messages from the last conversation.
    """
    last_message = command_history.get_last_conversation(conversation_id)
    if last_message:
        return last_message[3]  # content
    return "No previous conversation messages found."
