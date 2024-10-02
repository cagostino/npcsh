import os
import sqlite3
import json
from datetime import datetime


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
        self.create_table()

    def create_table(self):
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

    def add(self, command, subcommands, output, location):
        timestamp = datetime.now().isoformat()
        self.cursor.execute(
            """
        INSERT INTO command_history (timestamp, command, subcommands, output, location)
        VALUES (?, ?, ?, ?, ?)
        """,
            (timestamp, command, json.dumps(subcommands), output, location),
        )
        self.conn.commit()

    def search(self, term):
        self.cursor.execute(
            """
        SELECT * FROM command_history WHERE command LIKE ?
        """,
            (f"%{term}%",),
        )
        return self.cursor.fetchall()

    def get_all(self, limit=100):
        self.cursor.execute(
            """
        SELECT * FROM command_history ORDER BY id DESC LIMIT ?
        """,
            (limit,),
        )
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()