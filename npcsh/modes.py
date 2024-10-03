# modes.py
import os
import subprocess
from .llm_funcs import (
    get_ollama_conversation,
    get_llm_response,
    execute_data_operations,
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


def record_audio(duration=5, sample_rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024,
    )

    print("Recording...")
    frames = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b"".join(frames)


def get_audio_level(audio_data):
    return np.max(np.abs(np.frombuffer(audio_data, dtype=np.int16)))


def calibrate_silence(sample_rate=16000, duration=2):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024,
    )

    print("Calibrating silence level. Please remain quiet...")
    levels = []
    for _ in range(int(sample_rate * duration / 1024)):
        data = stream.read(1024)
        levels.append(get_audio_level(data))

    stream.stop_stream()
    stream.close()
    p.terminate()

    avg_level = np.mean(levels)
    silence_threshold = avg_level * 1.5  # Set threshold slightly above average
    print(f"Silence threshold set to: {silence_threshold}")
    return silence_threshold


def is_silent(audio_data, threshold=500):
    """Check if the audio chunk is silent."""
    return np.max(np.abs(np.frombuffer(audio_data, dtype=np.int16))) < threshold


def is_silent(audio_data, threshold):
    return get_audio_level(audio_data) < threshold


def record_audio(sample_rate=16000, max_duration=10, silence_threshold=None):
    if silence_threshold is None:
        silence_threshold = calibrate_silence()

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024,
    )

    print("Listening... (speak now)")
    frames = []
    silent_chunks = 0
    has_speech = False
    max_silent_chunks = int(sample_rate * 1.0 / 1024)  # 1.0 seconds of silence
    max_chunks = int(sample_rate * max_duration / 1024)  # Maximum duration in chunks

    start_time = time.time()
    for _ in range(max_chunks):
        data = stream.read(1024)
        frames.append(data)

        if is_silent(data, silence_threshold):
            silent_chunks += 1
            if has_speech and silent_chunks > max_silent_chunks:
                break
        else:
            silent_chunks = 0
            has_speech = True

        if len(frames) % 10 == 0:  # Print a dot every ~0.5 seconds
            print(".", end="", flush=True)

        if time.time() - start_time > max_duration:
            print("\nMax duration reached.")
            break

    print("\nProcessing...")

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b"".join(frames)


def speak_text(text):
    try:
        tts = gTTS(text=text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            playsound(fp.name)
        os.unlink(fp.name)
    except Exception as e:
        print(f"Text-to-speech error: {e}")


def enter_whisper_mode(command_history):
    try:
        model = whisper.load_model("base")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return "Error: Unable to load Whisper model"

    whisper_output = []

    print("Entering whisper mode. Calibrating silence level...")
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

            llm_response = get_llm_response(text)
            print(f"LLM response: {llm_response}")
            whisper_output.append(f"LLM: {llm_response}")

            speak_text(llm_response)

            command_history.add(text, ["whisper"], llm_response, os.getcwd())

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


def initial_table_print(cursor):
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name != 'command_history'"
    )
    tables = cursor.fetchall()

    print("\nAvailable tables:")
    for i, table in enumerate(tables, 1):
        print(f"{i}. {table[0]}")


def enter_observation_mode(command_history):
    conn = command_history.conn
    cursor = command_history.cursor

    print("Entering observation mode. Type '/dq' or to exit.")
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

        else:
            user_query = input(
                """
data>"""
            )
        print(user_query)

        response = execute_data_operations(user_query, command_history)

        answer_prompt = f"""

        Here is an input from the user:
        {user_query}
        Here is some useful data relevant to the query:
        {response}

        Now write a query to write a final response to be delivered to the user.

        Your answer must be in the format:
        {{"response": "Your response here."}}

        """
        final_response = get_llm_response(answer_prompt, format="json")
        print(final_response["response"])
        n_times += 1

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
            spool_context = get_ollama_conversation(spool_context, model=model)

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
