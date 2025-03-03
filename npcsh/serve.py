from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import sqlite3
from datetime import datetime
import json
from pathlib import Path

from flask_sse import sse

import redis

# compress the image
from PIL import Image
from PIL import ImageFile
from io import BytesIO

from .command_history import (
    CommandHistory,
    save_conversation_message,
)
from .npc_compiler import NPCCompiler, NPC
from .npc_sysenv import get_model_and_provider, get_available_models, get_system_message

from .llm_funcs import (
    check_llm_command,
    get_stream,
)
from .image import capture_screenshot
from .helpers import get_db_npcs, get_directory_npcs, get_npc_path
from .npc_compiler import load_npc_from_file
from .shell_helpers import execute_command
import base64

# Configuration
db_path = os.path.expanduser("~/npcsh_history.db")
user_npc_directory = os.path.expanduser("~/.npcsh/npc_team")
project_npc_directory = os.path.abspath("./npc_team")

# Initialize components
npc_compiler = NPCCompiler(project_npc_directory, db_path)

app = Flask(__name__)
app.config["REDIS_URL"] = "redis://localhost:6379"
app.register_blueprint(sse, url_prefix="/stream")

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

CORS(
    app,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    supports_credentials=True,
)


@app.route("/api/capture_screenshot", methods=["GET"])
def capture():
    # Capture screenshot using NPC-based method
    screenshot = capture_screenshot(None, full=True)

    # Ensure screenshot was captured successfully
    if not screenshot:
        print("Screenshot capture failed")
        return None

    return jsonify({"screenshot": screenshot})


@app.route("/api/stream", methods=["POST"])
def stream():
    """SSE stream that takes messages, models, providers, and attachments from frontend."""
    data = request.json
    commandstr = data.get("commandstr")
    conversation_id = data.get("conversationId")
    model = data.get("model", None)
    provider = data.get("provider", None)
    save_to_sqlite3 = data.get("saveToSqlite3", False)
    npc = data.get("npc", None)
    attachments = data.get("attachments", [])
    current_path = data.get("currentPath")
    print(data)

    messages = data.get("messages", [])
    print("messages", messages)
    command_history = CommandHistory(db_path)

    images = []
    attachments_loaded = []

    if attachments:
        for attachment in attachments:
            extension = attachment["name"].split(".")[-1]
            extension_mapped = extension_map.get(extension.upper(), "others")
            file_path = os.path.expanduser(
                "~/.npcsh/" + extension_mapped + "/" + attachment["name"]
            )

            if extension_mapped == "images":
                # Open the image file and save it to the file path
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                img = Image.open(attachment["path"])

                # Save the image to a BytesIO buffer (to extract binary data)
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format="PNG")  # or the appropriate format
                img_byte_arr.seek(0)  # Rewind the buffer to the beginning

                # Save the image to a file
                img.save(file_path, optimize=True, quality=50)

                # Add to images list for LLM processing
                images.append({"filename": attachment["name"], "file_path": file_path})

                # Add the image data (in binary form) to attachments_loaded
                attachments_loaded.append(
                    {
                        "name": attachment["name"],
                        "type": extension_mapped,
                        "data": img_byte_arr.read(),  # Read binary data from the buffer
                        "size": os.path.getsize(file_path),
                    }
                )
    if save_to_sqlite3:
        if len(messages) == 0:
            # load the conversation messages
            messages = fetch_messages_for_conversation(conversation_id)
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        messages.append({"role": "user", "content": commandstr})
        message_id = command_history.generate_message_id()

        save_conversation_message(
            command_history,
            conversation_id,
            "user",
            commandstr,
            wd=current_path,
            model=model,
            provider=provider,
            npc=npc,
            attachments=attachments_loaded,
            message_id=message_id,
        )
        message_id = command_history.generate_message_id()

    stream_response = get_stream(
        messages,
        images=images,
        model=model,
        provider=provider,
        npc=npc if isinstance(npc, NPC) else None,
    )

    """else:

        stream_response = execute_command_stream(
            commandstr,
            command_history,
            db_path,
            npc_compiler,
            model=model,
            provider=provider,
            messages=messages,
            images=images,  # Pass the processed images
        )  # Get all conversation messages so far
    """
    final_response = ""  # To accumulate the assistant's response
    complete_response = []  # List to store all chunks

    def event_stream():
        for response_chunk in stream_response:
            chunk_content = ""

            # Extract content based on model type
            if model.startswith("gpt-4o"):
                chunk_content = "".join(
                    choice.delta.content
                    for choice in response_chunk.choices
                    if choice.delta.content is not None
                )
                if chunk_content:
                    complete_response.append(chunk_content)
                chunk_data = {
                    "id": response_chunk.id,
                    "object": response_chunk.object,
                    "created": response_chunk.created,
                    "model": response_chunk.model,
                    "choices": [
                        {
                            "index": choice.index,
                            "delta": {
                                "content": choice.delta.content,
                                "role": choice.delta.role,
                            },
                            "finish_reason": choice.finish_reason,
                        }
                        for choice in response_chunk.choices
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

            elif model.startswith("llama"):
                chunk_content = response_chunk["message"]["content"]
                if chunk_content:
                    complete_response.append(chunk_content)
                chunk_data = {
                    "id": None,
                    "object": None,
                    "created": response_chunk["created_at"],
                    "model": response_chunk["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk_content,
                                "role": response_chunk["message"]["role"],
                            },
                            "finish_reason": response_chunk.get("done_reason"),
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
            elif model.startswith("claude"):
                print(response_chunk)
                if response_chunk.type == "message_start":
                    chunk_data = {
                        "id": None,
                        "object": None,
                        "created": None,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": "",
                                    "role": "assistant",
                                },
                                "finish_reason": "",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                if response_chunk.type == "content_block_delta":
                    chunk_content = response_chunk.delta.text
                    if chunk_content:
                        complete_response.append(chunk_content)
                    chunk_data = {
                        "id": None,
                        "object": None,
                        "created": None,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": chunk_content,
                                    "role": "assistant",
                                },
                                "finish_reason": response_chunk.delta.type,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
            if save_to_sqlite3:
                save_conversation_message(
                    command_history,
                    conversation_id,
                    "assistant",
                    chunk_content,
                    wd=current_path,
                    model=model,
                    provider=provider,
                    npc=npc,
                    message_id=message_id,  # Save with the same message_id
                )

        # Send completion message
        yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
        if save_to_sqlite3:
            full_content = command_history.get_full_message_content(message_id)
            command_history.update_message_content(message_id, full_content)

    response = Response(event_stream(), mimetype="text/event-stream")

    return response


@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response


def get_db_connection():
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# Remove the global command_history instance
# command_history = CommandHistory(db_path)  # Remove this line

extension_map = {
    "PNG": "images",
    "JPG": "images",
    "JPEG": "images",
    "GIF": "images",
    "SVG": "images",
    "MP4": "videos",
    "AVI": "videos",
    "MOV": "videos",
    "WMV": "videos",
    "MPG": "videos",
    "MPEG": "videos",
    "DOC": "documents",
    "DOCX": "documents",
    "PDF": "documents",
    "PPT": "documents",
    "PPTX": "documents",
    "XLS": "documents",
    "XLSX": "documents",
    "TXT": "documents",
    "CSV": "documents",
    "ZIP": "archives",
    "RAR": "archives",
    "7Z": "archives",
    "TAR": "archives",
    "GZ": "archives",
    "BZ2": "archives",
    "ISO": "archives",
}


def fetch_messages_for_conversation(conversation_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT role, content, timestamp
        FROM conversation_history
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
    """
    cursor.execute(query, (conversation_id,))
    messages = cursor.fetchall()
    conn.close()

    return [
        {
            "role": message["role"],
            "content": message["content"],
            "timestamp": message["timestamp"],
        }
        for message in messages
    ]


@app.route("/api/get_attachment_response", methods=["POST"])
def get_attachment_response():
    data = request.json
    attachments = data.get("attachments", [])
    messages = data.get("messages")  # Get conversation ID

    # Process each attachment
    for attachment in attachments:
        extension = attachment["name"].split(".")[-1]
        extension_mapped = extension_map.get(extension.upper(), "others")
        file_path = os.path.expanduser(
            "~/.npcsh/" + extension_mapped + "/" + attachment["name"]
        )
        if extension_mapped == "images":
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(base64.b64decode(attachment["base64"]))
            img.save(file_path, optimize=True, quality=50)
    messages = (
        fetch_messages_for_conversation(conversation_id) if conversation_id else []
    )

    return jsonify(
        {
            "status": "success",
            "message": "Attachments processed.",
            "conversationId": conversation_id,
            "messages": messages,  # Optionally return fetched messages
        }
    )


@app.route("/api/execute", methods=["POST"])
def execute():
    try:
        data = request.json
        # print(data)
        # print(type(data))
        command = data.get("commandstr")  # .get("command")
        # strip it  to remove quotes/special marks
        command = command.strip()
        command = command.replace('"', "")
        command = command.replace("'", "")
        command = command.replace("`", "")

        current_path = data.get("currentPath")
        conversation_id = data.get("conversationId")
        # print(type(command))
        # print(command)
        if not command:
            return (
                jsonify(
                    {
                        "error": "No command provided",
                        "output": "Error: No command provided",
                    }
                ),
                400,
            )
        command_history = CommandHistory(db_path)

        # Use the existing command execution function from npcsh
        result = execute_command(
            command=command,
            command_history=command_history,
            db_path=db_path,
            npc_compiler=npc_compiler,
            conversation_id=conversation_id,
        )
        # print(current_path)
        save_conversation_message(
            command_history, conversation_id, "user", command, wd=current_path
        )
        save_conversation_message(
            command_history,
            conversation_id,
            "assistant",
            result.get("output", ""),
            wd=current_path,
        )

        return jsonify(
            {
                "output": result.get("output", ""),
                "currentPath": os.getcwd(),
                "error": None,
            }
        )

    except Exception as e:
        print(f"Error executing command: {str(e)}")
        import traceback

        print("Full traceback:")
        traceback.print_exc()
        return (
            jsonify(
                {
                    "error": str(e),
                    "output": f"Error: {str(e)}",
                    "currentPath": data.get("currentPath", None),
                }
            ),
            500,
        )


@app.route("/api/conversations", methods=["GET"])
def get_conversations():
    try:
        path = request.args.get("path")
        if not path:
            return jsonify({"error": "No path provided", "conversations": []}), 400

        conn = get_db_connection()
        try:
            cursor = conn.cursor()

            query = """
            SELECT DISTINCT conversation_id,
                   MIN(timestamp) as start_time,
                   GROUP_CONCAT(content) as preview
            FROM conversation_history
            WHERE directory_path = ?
            GROUP BY conversation_id
            ORDER BY start_time DESC
            """

            cursor.execute(query, [path])
            conversations = cursor.fetchall()

            return jsonify(
                {
                    "conversations": [
                        {
                            "id": conv["conversation_id"],
                            "timestamp": conv["start_time"],
                            "preview": (
                                conv["preview"][:100] + "..."
                                if conv["preview"] and len(conv["preview"]) > 100
                                else conv["preview"]
                            ),
                        }
                        for conv in conversations
                    ],
                    "error": None,
                }
            )

        finally:
            conn.close()

    except Exception as e:
        print(f"Error getting conversations: {str(e)}")
        return jsonify({"error": str(e), "conversations": []}), 500


@app.route("/api/conversation/<conversation_id>/messages", methods=["GET"])
def get_conversation_messages(conversation_id):
    try:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()

            query = """
                SELECT * FROM conversation_history
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
            """

            cursor.execute(query, [conversation_id])
            messages = cursor.fetchall()

            return jsonify(
                {
                    "messages": [
                        {
                            "role": msg["role"],
                            "content": msg["content"],
                            "timestamp": msg["timestamp"],
                        }
                        for msg in messages
                    ],
                    "error": None,
                }
            )

        finally:
            conn.close()

    except Exception as e:
        print(f"Error getting conversation messages: {str(e)}")
        return jsonify({"error": str(e), "messages": []}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "error": None})


def start_flask_server(port=5337):
    try:
        # Ensure the database tables exist
        conn = get_db_connection()
        try:
            cursor = conn.cursor()

            # Create tables if they don't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS command_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    command TEXT,
                    tags TEXT,
                    response TEXT,
                    directory TEXT,
                    conversation_id TEXT
                )
            """
            )

            cursor.execute(
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

            conn.commit()
        finally:
            conn.close()

        # Run the Flask app on all interfaces
        print("Starting Flask server on http://0.0.0.0:5337")
        app.run(host="0.0.0.0", port=5337, debug=True)
    except Exception as e:
        print(f"Error starting server: {str(e)}")


if __name__ == "__main__":
    start_flask_server()
