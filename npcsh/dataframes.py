## functions for dataframes
import os
import sqlite3
import json
import pandas as pd
import numpy as np
import io
from PIL import Image
from typing import Optional

from .llm_funcs import get_llm_response
from .audio import process_audio
from .video import process_video

from .load_data import load_pdf, load_csv, load_json, load_excel, load_txt, load_image


def load_data_into_table(
    file_path: str, table_name: str, cursor: sqlite3.Cursor, conn: sqlite3.Connection
) -> None:
    """
    Function Description:
        This function is used to load data into a table.
    Args:
        file_path : str : The file path.
        table_name : str : The table name.
        cursor : sqlite3.Cursor : The SQLite cursor.
        conn : sqlite3.Connection : The SQLite connection.
    Keyword Args:
        None
    Returns:
        None
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type and load data
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".pdf"):
            df = load_pdf(file_path)
        elif file_path.endswith((".txt", ".log", ".md")):
            df = load_txt(file_path)
        elif file_path.endswith((".xls", ".xlsx")):
            df = load_excel(file_path)
        elif file_path.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")
        ):
            # Handle images as NumPy arrays
            df = load_image(file_path)
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
                    "audio_array": (
                        [audio_array.tobytes()] if audio_array is not None else None
                    ),
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
                df = load_txt(file_path)
            except Exception as e:
                print(f"Could not load file: {e}")
                return

        # Store DataFrame in the database
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Data from '{file_path}' loaded into table '{table_name}'")

    except Exception as e:
        raise e  # Re-raise the exception for handling in enter_observation_mode


def create_new_table(cursor: sqlite3.Cursor, conn: sqlite3.Connection) -> None:
    """
    Function Description:
        This function is used to create a new table.
    Args:
        cursor : sqlite3.Cursor : The SQLite cursor.
        conn : sqlite3.Connection : The SQLite connection.
    Keyword Args:
        None
    Returns:
        None
    """

    table_name = input("Enter new table name: ").strip()
    columns = input("Enter column names separated by commas: ").strip()

    create_query = (
        f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, {columns})"
    )
    cursor.execute(create_query)
    conn.commit()
    print(f"Table '{table_name}' created successfully.")


def delete_table(cursor: sqlite3.Cursor, conn: sqlite3.Connection) -> None:
    """
    Function Description:
        This function is used to delete a table.
    Args:
        cursor : sqlite3.Cursor : The SQLite cursor.
        conn : sqlite3.Connection : The SQLite connection.
    Keyword Args:
        None
    Returns:
        None
    """

    table_name = input("Enter table name to delete: ").strip()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()
    print(f"Table '{table_name}' deleted successfully.")


def add_observation(
    cursor: sqlite3.Cursor, conn: sqlite3.Connection, table_name: str
) -> None:
    """
    Function Description:
        This function is used to add an observation.
    Args:
        cursor : sqlite3.Cursor : The SQLite cursor.
        conn : sqlite3.Connection : The SQLite connection.
        table_name : str : The table name.
    Keyword Args:
        None
    Returns:
        None
    """

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
