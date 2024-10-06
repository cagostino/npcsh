# modes.py
import os
import subprocess
from .llm_funcs import (
    get_ollama_conversation,
    get_llm_response,
    execute_data_operations,
)
from .helpers import (
    calibrate_silence,
    record_audio,
    speak_text, 
    is_silent
)
import sqlite3
import time
from gtts import gTTS
from playsound import playsound

import whisper
import pyaudio
import wave
import numpy as np
import tempfile
import os

def enter_whisper_mode(command_history, npc=None):
    try:
        model = whisper.load_model("base")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return "Error: Unable to load Whisper model"

    whisper_output = []
    npc_info = f" (NPC: {npc.name})" if npc else ""

    print(f"Entering whisper mode{npc_info}. Calibrating silence level...")
    try:
        silence_threshold = calibrate_silence()
    except Exception as e:
        print(f"Error calibrating silence: {e}")
        return "Error: Unable to calibrate silence"

    print("Ready. Speak after seeing 'Listening...'. Say 'exit' or type '/wq' to quit.")
    speak_text("Whisper mode activated. Ready for your input.")

    while True:
        try:
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

            llm_response = get_llm_response(text, npc=npc)
            print(f"LLM response: {llm_response}")
            whisper_output.append(f"LLM: {llm_response}")

            speak_text(llm_response)

            command_history.add(text, ["whisper", npc.name if npc else ""], llm_response, os.getcwd())

            print("\nPress Enter to speak again, or type '/wq' to quit.")
            user_input = input()
            if user_input.lower() == "/wq":
                print("Exiting whisper mode.")
                speak_text("Exiting whisper mode. Goodbye!")
                break

        except Exception as e:
            print(f"Error in whisper mode: {e}")
            whisper_output.append(f"Error: {e}")

    return "\n".join(whisper_output)

import datetime


def enter_notes_mode(command_history, npc=None):
    npc_name = npc.name if npc else 'base'
    print(f"Entering notes mode (NPC: {npc_name}). Type '/nq' to exit.")

    while True:
        note = input("Enter your note (or '/nq' to quit): ").strip()

        if note.lower() == '/nq':
            break

        save_note(note, command_history, npc)

    print("Exiting notes mode.")

def save_note(note, command_history, npc=None):
    current_dir = os.getcwd()
    timestamp = datetime.datetime.now().isoformat()
    npc_name = npc.name if npc else 'base'

    # Assuming command_history has a method to access the database connection
    conn = command_history.conn
    cursor = conn.cursor()

    # Create notes table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        note TEXT,
        npc TEXT,
        directory TEXT
    )
    ''')

    # Insert the note into the database
    cursor.execute('''
    INSERT INTO notes (timestamp, note, npc, directory)
    VALUES (?, ?, ?, ?)
    ''', (timestamp, note, npc_name, current_dir))

    conn.commit()

    print("Note saved to database.")



def enter_observation_mode(command_history, npc=None):
    conn = command_history.conn
    cursor = command_history.cursor

    npc_info = f" (NPC: {npc.name})" if npc else ""
    print(f"Entering observation mode{npc_info}. Type '/dq' to exit.")
    n_times = 0
    while True:
        # Show available tables
        if n_times == 0:
            initial_table_print(cursor)

        user_query = input(
            """
Enter a plain-text request or one using the dataframe manipulation framework of your choice. 
You can also have the data NPC ingest data into your database by pointing it to the right files.
data>"""
        )

        print(user_query)

        response = execute_data_operations(user_query, command_history, npc)

        answer_prompt = f"""
        Here is an input from the user:
        {user_query}
        Here is some useful data relevant to the query:
        {response}

        Now write a query to write a final response to be delivered to the user.

        Your answer must be in the format:
        {{"response": "Your response here."}}
        """
        final_response = get_llm_response(answer_prompt, format="json", npc=npc)
        print(final_response["response"])
        n_times += 1

        if user_query.lower() == "/dq":
            break

    conn.close()
    print("Exiting observation mode.")

def enter_spool_mode(command_history, inherit_last=0, model="llama3.1", npc=None):
    npc_info = f" (NPC: {npc.name})" if npc else ""
    print(f"Entering spool mode{npc_info}. Type '/sq' to exit spool mode.")
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
            spool_context = get_ollama_conversation(spool_context, model=model, npc=npc)

            command_history.add(
                user_input, ["spool", npc.name if npc else ""], spool_context[-1]["content"], os.getcwd()
            )
            print(spool_context[-1]["content"])
        except (KeyboardInterrupt, EOFError):
            print("\nExiting spool mode.")
            break

    return "\n".join(
        [msg["content"] for msg in spool_context if msg["role"] == "assistant"]
    )

def initial_table_print(cursor):
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name != 'command_history'"
    )
    tables = cursor.fetchall()

    print("\nAvailable tables:")
    for i, table in enumerate(tables, 1):
        print(f"{i}. {table[0]}")

def get_data_response(request, npc=None):
    data_output = npc.get_data_response(request) if npc else None
    return data_output

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