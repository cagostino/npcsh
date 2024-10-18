# modes.py
import os
import subprocess
from .llm_funcs import (
    get_llm_response,
    get_conversation,
    execute_data_operations,
    get_system_message,
    check_llm_command,
    get_data_response,
)
from .helpers import calibrate_silence, record_audio, speak_text, is_silent
import sqlite3
import time
from gtts import gTTS
from playsound import playsound

import whisper
import wave
import numpy as np
import tempfile
import os
import json


def enter_whisper_mode(command_history, npc=None):
    if npc is not None:
        llm_name = npc.name
    else:
        llm_name = "LLM"
    try:
        model = whisper.load_model("base")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return "Error: Unable to load Whisper model"

    whisper_output = []
    npc_info = f" (NPC: {npc.name})" if npc else ""
    messages = []  # Initialize messages list for conversation history

    print(f"Entering whisper mode{npc_info}. Calibrating silence level...")
    try:
        silence_threshold = calibrate_silence()
    except Exception as e:
        print(f"Error calibrating silence: {e}")
        return "Error: Unable to calibrate silence"

    print("Ready. Speak after seeing 'Listening...'. Say 'exit' or type '/wq' to quit.")
    speak_text("Whisper mode activated. Ready for your input.")

    while True:
        # try:
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

        messages.append({"role": "user", "content": text})  # Add user message

        llm_response = check_llm_command(
            text, command_history, npc=npc, messages=messages
        )  # Use check_llm_command
        # print(type(llm_response))
        if isinstance(llm_response, dict):
            print(f"{llm_name}: {llm_response['response']}")  # Print assistant's reply
            whisper_output.append(
                f"{llm_name}: {llm_response['response']}"
            )  # Add to output
            speak_text(llm_response["response"])  # Speak assistant's reply
        elif isinstance(llm_response, list) and len(llm_response) > 0:
            assistant_reply = messages[-1]["content"]
            print(f"{llm_name}: {assistant_reply}")  # Print assistant's reply
            whisper_output.append(f"{llm_name}: {assistant_reply}")  # Add to output
            speak_text(assistant_reply)  # Speak assistant's reply
        elif isinstance(
            llm_response, str
        ):  # Handle string responses (errors or direct replies)
            print(f"{llm_name}: {llm_response}")
            whisper_output.append(f"{llm_name}: {llm_response}")
            speak_text(llm_response)

        # command_history.add(...)  This is now handled inside check_llm_command

        print("\nPress Enter to speak again, or type '/wq' to quit.")
        user_input = input()
        if user_input.lower() == "/wq":
            print("Exiting whisper mode.")
            speak_text("Exiting whisper mode. Goodbye!")
            break

    # except Exception as e:
    #    print(f"Error in whisper mode: {e}")
    #    whisper_output.append(f"Error: {e}")

    return "\n".join(whisper_output)


import datetime


def enter_notes_mode(command_history, npc=None):
    npc_name = npc.name if npc else "base"
    print(f"Entering notes mode (NPC: {npc_name}). Type '/nq' to exit.")

    while True:
        note = input("Enter your note (or '/nq' to quit): ").strip()

        if note.lower() == "/nq":
            break

        save_note(note, command_history, npc)

    print("Exiting notes mode.")


def save_note(note, command_history, npc=None):
    current_dir = os.getcwd()
    timestamp = datetime.datetime.now().isoformat()
    npc_name = npc.name if npc else "base"

    # Assuming command_history has a method to access the database connection
    conn = command_history.conn
    cursor = conn.cursor()

    # Create notes table if it doesn't exist
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        note TEXT,
        npc TEXT,
        directory TEXT
    )
    """
    )

    # Insert the note into the database
    cursor.execute(
        """
    INSERT INTO notes (timestamp, note, npc, directory)
    VALUES (?, ?, ?, ?)
    """,
        (timestamp, note, npc_name, current_dir),
    )

    conn.commit()

    print("Note saved to database.")


def enter_data_mode(command_history, npc=None):
    conn = command_history.conn
    cursor = command_history.cursor
    dataframes = {}
    npc_info = f" (NPC: {npc.name})" if npc else ""
    print(f"Entering observation mode{npc_info}. Type '/dq' to exit.")
    n_times = 0
    while True:
        # Show available tables
        if n_times == 0:
            initial_table_print(cursor)

        user_query = input(
            """Enter a plain-text request, a data manipulation command, or a load command:
data> """
        )
        # print(user_query , 'user_query')
        if user_query.lower().startswith("load from "):
            try:
                parts = user_query.split(" as ")
                if len(parts) != 2:
                    raise ValueError("Invalid load command format.")

                file_path = parts[0].split("load from ")[1].strip()
                table_name = parts[1].strip()
                load_data_into_table(file_path, table_name, cursor, conn)
                dataframes[table_name] = pd.read_sql(
                    f"SELECT * FROM {table_name}", conn
                )
            except Exception as e:
                print(f"Error loading data: {e}")

        elif user_query.lower() == "/dq":
            break

        else:  # Process data operations or LLM requests
            response, engine = execute_data_operations(
                user_query, command_history, dataframes, npc=npc
            )
            if engine is not None and engine == "llm":
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
                dataframes["data_output"] = response
                dataframes["output"] = response

        n_times += 1

        if user_query.lower() == "/dq":
            break

    conn.close()
    print("Exiting observation mode.")



import cv2  # For video/image processing
import librosa  # For audio processing
import numpy as np


import base64


def process_video(file_path, table_name):
    embeddings = []
    texts = []
    try:
        video = cv2.VideoCapture(file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(frame_count):
            ret, frame = video.read()
            if not ret:
                break

            # Process every nth frame (adjust n as needed for performance)
            n = 10  # Process every 10th frame
            if i % n == 0:
                # Image Embeddings
                _, buffer = cv2.imencode(".jpg", frame)  # Encode frame as JPG
                base64_image = base64.b64encode(buffer).decode("utf-8")
                image_info = {
                    "filename": f"frame_{i}.jpg",
                    "file_path": f"data:image/jpeg;base64,{base64_image}",
                }  # Use data URL for OpenAI
                image_embedding_response = get_llm_response(
                    "Describe this image.",
                    image=image_info,
                    model="gpt-4",
                    provider="openai",
                )  # Replace with your image embedding model
                if (
                    isinstance(image_embedding_response, dict)
                    and "error" in image_embedding_response
                ):
                    print(
                        f"Error generating image embedding: {image_embedding_response['error']}"
                    )
                else:
                    # Assuming your image embedding model returns a textual description
                    embeddings.append(image_embedding_response)
                    texts.append(f"Frame {i}: {image_embedding_response}")

        video.release()
        return embeddings, texts

    except Exception as e:
        print(f"Error processing video: {e}")
        return [], []  # Return empty lists in case of error


def process_audio(file_path, table_name):
    embeddings = []
    texts = []
    try:
        audio, sr = librosa.load(file_path)
        # Transcribe audio using Whisper
        model = whisper.load_model("base")  # Or a larger model if available
        result = model.transcribe(file_path)
        transcribed_text = result["text"].strip()

        # Split transcribed text into chunks (adjust chunk_size as needed)
        chunk_size = 1000
        for i in range(0, len(transcribed_text), chunk_size):
            chunk = transcribed_text[i : i + chunk_size]
            text_embedding_response = get_llm_response(
                f"Generate an embedding for: {chunk}",
                model="text-embedding-ada-002",
                provider="openai",
            )  # Use a text embedding model
            if (
                isinstance(text_embedding_response, dict)
                and "error" in text_embedding_response
            ):
                print(
                    f"Error generating text embedding: {text_embedding_response['error']}"
                )
            else:
                embeddings.append(text_embedding_response)  # Store the embedding
                texts.append(chunk)  # Store the corresponding text chunk

        return embeddings, texts

    except Exception as e:
        print(f"Error processing audio: {e}")
        return [], []  # Return empty lists in case of error


import pandas as pd
import os
import sys
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader, UnstructuredExcelLoader, DirectoryLoader, UnstructuredFileLoader                                
from langchain_community.embeddings import OpenAIEmbeddings                                                                                                                          
from langchain_community.vectorstores import Chroma  
from langchain.text_splitter import RecursiveCharacterTextSplitter

import cv2
import librosa
import numpy as np
import tempfile
import json
from PIL import Image  # For image loading
import fitz  # PyMuPDF
import io


def load_data_into_table(file_path, table_name, cursor, conn):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type and load data
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".pdf"):
            # Extract text and images
            pdf_document = fitz.open(file_path)
            texts = []
            images = []

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]

                # Extract text
                text = page.get_text()
                texts.append({"page": page_num + 1, "content": text})

                # Extract images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Convert image to numpy array
                    image = Image.open(io.BytesIO(image_bytes))
                    img_array = np.array(image)

                    images.append(
                        {
                            "page": page_num + 1,
                            "index": img_index + 1,
                            "array": img_array.tobytes(),
                            "shape": img_array.shape,
                            "dtype": str(img_array.dtype),
                        }
                    )

            # Create DataFrame
            df = pd.DataFrame(
                {"texts": json.dumps(texts), "images": json.dumps(images)}, index=[0]
            )

            # Optionally create embeddings
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=0
                )
                split_texts = text_splitter.split_text(
                    "\n\n".join([t["content"] for t in texts])
                )
                embeddings = OpenAIEmbeddings()
                db = Chroma.from_texts(
                    split_texts, embeddings, collection_name=table_name
                )
                df["embeddings_collection"] = table_name
            except Exception as e:
                print(f"Warning: Could not create embeddings. Error: {e}")

        elif file_path.endswith((".txt", ".log", ".md")):
            with open(file_path, "r") as f:
                text = f.read()
            df = pd.DataFrame({"text": [text]})
        elif file_path.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        elif file_path.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")
        ):
            # Handle images as NumPy arrays
            img = Image.open(file_path)
            img_array = np.array(img)
            # Store image shape for reconstruction
            df = pd.DataFrame(
                {
                    "image_array": [img_array.tobytes()],
                    "shape": [img_array.shape],
                    "dtype": [img_array.dtype.str],
                }
            )

        elif file_path.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        ):  # Video files
            video_frames, audio_array = process_video(file_path)
            # Store video frames and audio
            df = pd.DataFrame(
                {
                    "video_frames": [video_frames.tobytes()],
                    "shape": [video_frames.shape],
                    "dtype": [video_frames.dtype.str],
                    "audio_array": [audio_array.tobytes()]
                    if audio_array is not None
                    else None,
                    "audio_rate": [sr] if audio_array is not None else None,
                }
            )

        elif file_path.lower().endswith((".mp3", ".wav", ".ogg")):  # Audio files
            audio_array, sr = process_audio(file_path)
            df = pd.DataFrame(
                {
                    "audio_array": [audio_array.tobytes()],
                    "audio_rate": [sr],
                }
            )
        else:
            # Attempt to load as text if no other type matches
            try:
                with open(file_path, "r") as file:
                    content = file.read()
                df = pd.DataFrame({"text": [content]})
            except Exception as e:
                print(f"Could not load file: {e}")
                return

        # Store DataFrame in the database
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Data from '{file_path}' loaded into table '{table_name}'")

    except Exception as e:
        raise e  # Re-raise the exception for handling in enter_observation_mode


def enter_spool_mode(command_history, inherit_last=0, npc=None):
    npc_info = f" (NPC: {npc.name})" if npc else ""
    print(f"Entering spool mode{npc_info}. Type '/sq' to exit spool mode.")
    spool_context = []
    system_message = get_system_message(npc) if npc else "You are a helpful assistant."
    # insert at the first position
    spool_context.insert(0, {"role": "assistant", "content": system_message})
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

            # Prepare kwargs for get_conversation
            kwargs_to_pass = {}
            if npc:
                kwargs_to_pass["npc"] = npc
                if npc.model:
                    kwargs_to_pass["model"] = npc.model
                if npc.provider:
                    kwargs_to_pass["provider"] = npc.provider

            # Get the conversation
            conversation_result = get_conversation(spool_context, **kwargs_to_pass)

            # Handle potential errors in conversation_result
            if isinstance(conversation_result, str) and "Error" in conversation_result:
                print(conversation_result)  # Print the error message
                continue  # Skip to the next iteration of the loop
            elif (
                not isinstance(conversation_result, list)
                or len(conversation_result) == 0
            ):
                print("Error: Invalid response from get_conversation")
                continue

            spool_context = conversation_result  # update spool_context

            # Extract assistant's reply, handling potential KeyError
            try:
                # import pdb ; pdb.set_trace()

                assistant_reply = spool_context[-1]["content"]
            except (KeyError, IndexError) as e:
                print(f"Error extracting assistant's reply: {e}")
                print(
                    f"Conversation result: {conversation_result}"
                )  # Print for debugging
                continue

            command_history.add(
                user_input,
                ["spool", npc.name if npc else ""],
                assistant_reply,
                os.getcwd(),
            )
            print(assistant_reply)

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
