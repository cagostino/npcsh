# modes.py
import os
import subprocess
from .llm_funcs import get_ollama_conversion
import sqlite3


def enter_bash_mode():
    print("Entering bash mode. Type '/bq' to exit bash mode.")
    current_dir = os.getcwd()
    bash_output = []
    while True:
        try:
            bash_input = input(f"bash {current_dir}> ").strip()
            if bash_input == "/bq":
                print("Exiting bash mode.")
                break
            else:
                try:
                    if bash_input.startswith("cd "):
                        new_dir = bash_input[3:].strip()
                        try:
                            os.chdir(os.path.expanduser(new_dir))
                            current_dir = os.getcwd()
                            bash_output.append(f"Changed directory to {current_dir}")
                            print(f"Changed directory to {current_dir}")
                        except FileNotFoundError:
                            bash_output.append(
                                f"bash: cd: {new_dir}: No such file or directory"
                            )
                    else:
                        result = subprocess.run(
                            bash_input,
                            shell=True,
                            text=True,
                            capture_output=True,
                            cwd=current_dir,
                        )
                        if result.stdout:
                            print(result.stdout.strip())
                            bash_output.append(result.stdout.strip())
                        if result.stderr:
                            bash_output.append(f"Error: {result.stderr.strip()}")
                except Exception as e:
                    bash_output.append(f"Error executing bash command: {e}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting bash mode.")
            break
    os.chdir(current_dir)
    return "\n".join(bash_output)


def enter_whisper_mode():
    pass


def enter_notes_mode(command_history):
    print("Entering notes mode. Type '/nq' to exit.")

    while True:
        note = input("Enter your note (or '/nq' to quit): ").strip()

        if note.lower() == "/nq":
            break

        save_note(note, command_history)

    print("Exiting notes mode.")


def save_note(note, command_history):
    current_dir = os.getcwd()
    readme_path = os.path.join(current_dir, "README.md")

    with open(readme_path, "a") as f:
        f.write(f"\n- {note}\n")

    print("Note added to README.md")
    command_history.add(f"/note {note}", ["note"], "", current_dir)


# Usage in your main script:
# enter_notes_mode()


def enter_observation_mode(command_history):
    conn = command_history.conn
    cursor = command_history.cursor

    print("Entering observation mode. Type '/obsq' or '/oq' to exit.")

    while True:
        # Show available tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name != 'command_history'"
        )
        tables = cursor.fetchall()

        print("\nAvailable tables:")
        for i, table in enumerate(tables, 1):
            print(f"{i}. {table[0]}")
        print("n. Create new table")
        print("d. Delete a table")
        print("q. Exit observation mode")

        choice = input("Choose an option: ").strip().lower()

        if choice == "q" or choice == "/obsq" or choice == "/oq":
            break
        elif choice == "n":
            create_new_table(cursor, conn)
        elif choice == "d":
            delete_table(cursor, conn)
        elif choice.isdigit() and 1 <= int(choice) <= len(tables):
            add_observation(cursor, conn, tables[int(choice) - 1][0])

    conn.close()
    print("Exiting observation mode.")


def enter_spool_mode(command_history, inherit_last=0, model="llama3.1"):
    print("Entering spool mode. Type '/sq' to exit spool mode.")
    spool_context = []

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

            # Process the spool context with LLM
            spool_context = get_ollama_conversion(spool_context, model=model)

            command_history.add(
                user_input, ["spool"], spool_context[-1]["content"], os.getcwd()
            )
            print(spool_context[-1]["content"])
        except (KeyboardInterrupt, EOFError):
            print("\nExiting spool mode.")
            break

    return "\n".join(
        [msg["content"] for msg in spool_context if msg["role"] == "assistant"]
    )


def create_new_table(cursor, conn):
    table_name = input("Enter new table name: ").strip()
    columns = input("Enter column names separated by commas: ").strip()

    create_query = (
        f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, {columns})"
    )
    cursor.execute(create_query)
    conn.commit()
    print(f"Table '{table_name}' created successfully.")


def delete_table(cursor, conn):
    table_name = input("Enter table name to delete: ").strip()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()
    print(f"Table '{table_name}' deleted successfully.")


def add_observation(cursor, conn, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall() if column[1] != "id"]

    values = []
    for column in columns:
        value = input(f"Enter value for {column}: ").strip()
        values.append(value)

    insert_query = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({','.join(['?' for _ in columns])})"
    cursor.execute(insert_query, values)
    conn.commit()
    print("Observation added successfully.")
