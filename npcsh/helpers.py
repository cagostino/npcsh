# helpers.py
import logging

logging.basicConfig(
    filename=".npcsh.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
import os
import sqlite3
import subprocess
from .npc_compiler import NPC

BASH_COMMANDS = [
    "alias",
    "bg",
    "bind",
    "break",
    "builtin",
    "case",
    "cd",
    "command",
    "compgen",
    "complete",
    "continue",
    "declare",
    "dirs",
    "disown",
    "echo",
    "enable",
    "eval",
    "exec",
    "exit",
    "export",
    "fc",
    "fg",
    "getopts",
    "hash",
    "help",
    "history",
    "if",
    "jobs",
    "kill",
    "let",
    "local",
    "logout",
    "popd",
    "printf",
    "pushd",
    "pwd",
    "read",
    "readonly",
    "return",
    "set",
    "shift",
    "shopt",
    "source",
    "suspend",
    "test",
    "times",
    "trap",
    "type",
    "typeset",
    "ulimit",
    "umask",
    "unalias",
    "unset",
    "until",
    "wait",
    "while",
    # Common Unix commands
    "ls",
    "cp",
    "mv",
    "rm",
    "mkdir",
    "rmdir",
    "touch",
    "cat",
    "less",
    "more",
    "head",
    "tail",
    "grep",
    "find",
    "sed",
    "awk",
    "sort",
    "uniq",
    "wc",
    "diff",
    "chmod",
    "chown",
    "chgrp",
    "ln",
    "tar",
    "gzip",
    "gunzip",
    "zip",
    "unzip",
    "ssh",
    "scp",
    "rsync",
    "wget",
    "curl",
    "ping",
    "netstat",
    "ifconfig",
    "route",
    "traceroute",
    "ps",
    "top",
    "htop",
    "kill",
    "killall",
    "su",
    "sudo",
    "whoami",
    "who",
    "w",
    "last",
    "finger",
    "uptime",
    "free",
    "df",
    "du",
    "mount",
    "umount",
    "fdisk",
    "mkfs",
    "fsck",
    "dd",
    "cron",
    "at",
    "systemctl",
    "service",
    "journalctl",
    "man",
    "info",
    "whatis",
    "whereis",
    "which",
    "date",
    "cal",
    "bc",
    "expr",
    "screen",
    "tmux",
    "git",
    "vim",
    "emacs",
    "nano",
]

TERMINAL_EDITORS = ["vim", "emacs", "nano"]
import yaml


def execute_set_command(command, value):
    config_path = os.path.expanduser("~/.npcshrc")

    # Map command to environment variable name
    var_map = {
        "model": "NPCSH_MODEL",
        "provider": "NPCSH_PROVIDER",
        "db_path": "NPCSH_DB_PATH",
    }

    if command not in var_map:
        return f"Unknown setting: {command}"

    env_var = var_map[command]

    # Read the current configuration
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            lines = f.readlines()
    else:
        lines = []

    # Check if the property exists and update it, or add it if it doesn't exist
    property_exists = False
    for i, line in enumerate(lines):
        if line.startswith(f"export {env_var}="):
            lines[i] = f"export {env_var}='{value}'\n"
            property_exists = True
            break

    if not property_exists:
        lines.append(f"export {env_var}='{value}'\n")

    # Save the updated configuration
    with open(config_path, "w") as f:
        f.writelines(lines)

    return f"{command.capitalize()} has been set to: {value}"


def execute_set_command(command, value):
    config_path = os.path.expanduser("~/.npcshrc")

    # Map command to environment variable name
    var_map = {
        "model": "NPCSH_MODEL",
        "provider": "NPCSH_PROVIDER",
        "db_path": "NPCSH_DB_PATH",
    }

    if command not in var_map:
        return f"Unknown setting: {command}"

    env_var = var_map[command]

    # Read the current configuration
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            lines = f.readlines()
    else:
        lines = []

    # Check if the property exists and update it, or add it if it doesn't exist
    property_exists = False
    for i, line in enumerate(lines):
        if line.startswith(f"export {env_var}="):
            lines[i] = f"export {env_var}='{value}'\n"
            property_exists = True
            break

    if not property_exists:
        lines.append(f"export {env_var}='{value}'\n")

    # Save the updated configuration
    with open(config_path, "w") as f:
        f.writelines(lines)

    return f"{command.capitalize()} has been set to: {value}"


def ensure_npcshrc_exists():
    npcshrc_path = os.path.expanduser("~/.npcshrc")
    if not os.path.exists(npcshrc_path):
        with open(npcshrc_path, "w") as npcshrc:
            npcshrc.write("# NPCSH Configuration File\n")
            npcshrc.write("export NPCSH_INITIALIZED=0\n")
            npcshrc.write("export NPCSH_PROVIDER='ollama'\n")
            npcshrc.write("export NPCSH_MODEL='phi3'\n")
            npcshrc.write("export NPCSH_DB_PATH='~/npcsh_history.db'\n")
    return npcshrc_path


def add_npcshrc_to_bashrc():
    bashrc_path = os.path.expanduser("~/.bashrc")
    npcshrc_line = "\n# Source NPCSH configuration\nif [ -f ~/.npcshrc ]; then\n    . ~/.npcshrc\nfi\n"

    with open(bashrc_path, "a+") as bashrc:
        bashrc.seek(0)
        content = bashrc.read()
        if "source ~/.npcshrc" not in content and ". ~/.npcshrc" not in content:
            bashrc.write(npcshrc_line)
            print("Added .npcshrc sourcing to .bashrc")
        else:
            print(".npcshrc already sourced in .bashrc")


def load_npc_from_file(npc_file: str, db_conn: sqlite3.Connection) -> NPC:
    # its a yaml filedef load_npc(npc_file: str, db_conn: sqlite3.Connection) -> NPC:
    """
    Load an NPC from a YAML file and initialize the NPC object.

    :param npc_file: Full path to the NPC file
    :param db_conn: SQLite database connection
    :return: Initialized NPC object
    """
    try:
        with open(npc_file, "r") as f:
            npc_data = yaml.safe_load(f)

        # Extract fields from YAML
        name = npc_data["name"]
        primary_directive = npc_data.get("primary_directive")
        suggested_tools_to_use = npc_data.get("suggested_tools_to_use")
        restrictions = npc_data.get("restrictions", [])
        model = npc_data.get("model", os.environ.get("NPCSH_MODEL", "phi3"))
        provider = npc_data.get("provider", os.environ.get("NPCSH_PROVIDER", "ollama"))

        # Initialize and return the NPC object
        return NPC(
            name,
            db_conn,
            primary_directive=primary_directive,
            suggested_tools_to_use=suggested_tools_to_use,
            restrictions=restrictions,
            model=model,
            provider=provider,
        )

    except FileNotFoundError:
        raise ValueError(f"NPC file not found: {npc_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML in NPC file {npc_file}: {str(e)}")
    except KeyError as e:
        raise ValueError(f"Missing required key in NPC file {npc_file}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading NPC from file {npc_file}: {str(e)}")


def is_npcsh_initialized():
    return os.environ.get("NPCSH_INITIALIZED", None) == "1"


def set_npcsh_initialized():
    npcshrc_path = ensure_npcshrc_exists()

    with open(npcshrc_path, "r+") as npcshrc:
        content = npcshrc.read()
        if "export NPCSH_INITIALIZED=0" in content:
            content = content.replace(
                "export NPCSH_INITIALIZED=0", "export NPCSH_INITIALIZED=1"
            )
            npcshrc.seek(0)
            npcshrc.write(content)
            npcshrc.truncate()

    # Also set it for the current session
    os.environ["NPCSH_INITIALIZED"] = "1"
    print("NPCSH initialization flag set in .npcshrc")


def get_npc_from_command(command):
    parts = command.split()
    npc = None
    for part in parts:
        if part.startswith("npc="):
            npc = part.split("=")[1]
            break
    return npc


def is_valid_npc(npc, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM compiled_npcs WHERE name = ?", (npc,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def get_npc_path(npc, db_path):
    initialize_base_npcs_if_needed(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT source_path FROM compiled_npcs WHERE name = ?", (npc,))

    result = cursor.fetchone()
    print(result)
    conn.close()
    return result[0] if result else None


def initialize_base_npcs_if_needed(db_path):
    if is_npcsh_initialized() == True:
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the compiled_npcs table if it doesn't exist
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS compiled_npcs (
        name TEXT PRIMARY KEY,
        source_path TEXT NOT NULL,
        compiled_path TEXT
    )
    """
    )

    # Check if base NPCs are already initialized
    cursor.execute("SELECT COUNT(*) FROM compiled_npcs")
    count = cursor.fetchone()[0]
    print(count)
    if count == 0:
        # Get the path to the npc_profiles directory
        current_file = os.path.abspath(__file__)
        folder_root = os.path.dirname(current_file)  # Go up two levels
        npc_profiles_dir = os.path.join(folder_root, "npc_profiles")

        # Insert base NPCs
        base_npcs = [
            ("base", os.path.join(npc_profiles_dir, "base.npc")),
            ("bash", os.path.join(npc_profiles_dir, "bash.npc")),
            ("data", os.path.join(npc_profiles_dir, "data.npc")),
        ]

        for npc_name, source_path in base_npcs:
            cursor.execute(
                """
            INSERT OR IGNORE INTO compiled_npcs (name, source_path, compiled_path)
            VALUES (?, ?, NULL)
            """,
                (npc_name, source_path),
            )

        conn.commit()
        print("Base NPCs initialized.")
    else:
        print("Base NPCs already initialized.")

    conn.close()
    set_npcsh_initialized()
    add_npcshrc_to_bashrc()


def is_valid_npc(npc, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM compiled_npcs WHERE name = ?", (npc,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


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


def open_terminal_editor(command):
    try:
        os.system(command)
    except Exception as e:
        print(f"Error opening terminal editor: {e}")


# Helper functions for language-specific execution remain the same


def execute_python(code):
    try:
        result = subprocess.run(
            ["python", "-c", code], capture_output=True, text=True, timeout=30
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out"


def execute_r(code):
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".R", delete=False
        ) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        result = subprocess.run(
            ["Rscript", temp_file_path], capture_output=True, text=True, timeout=30
        )
        os.unlink(temp_file_path)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        os.unlink(temp_file_path)
        return "Error: Execution timed out"


def execute_sql(code):
    # This is a placeholder. You'll need to implement SQL execution based on your database setup.
    return "SQL execution not implemented yet."


def execute_scala(code):
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".scala", delete=False
        ) as temp_file:
            temp_file.write(
                f"object TempObject {{ def main(args: Array[String]): Unit = {{ {code} }} }}"
            )
            temp_file_path = temp_file.name

        compile_result = subprocess.run(
            ["scalac", temp_file_path], capture_output=True, text=True, timeout=30
        )
        if compile_result.returncode != 0:
            os.unlink(temp_file_path)
            return f"Compilation Error: {compile_result.stderr}"

        run_result = subprocess.run(
            ["scala", "TempObject"], capture_output=True, text=True, timeout=30
        )
        os.unlink(temp_file_path)
        return (
            run_result.stdout
            if run_result.returncode == 0
            else f"Runtime Error: {run_result.stderr}"
        )
    except subprocess.TimeoutExpired:
        os.unlink(temp_file_path)
        return "Error: Execution timed out"


def execute_java(code):
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".java", delete=False
        ) as temp_file:
            temp_file.write(
                f"public class TempClass {{ public static void main(String[] args) {{ {code} }} }}"
            )
            temp_file_path = temp_file.name

        compile_result = subprocess.run(
            ["javac", temp_file_path], capture_output=True, text=True, timeout=30
        )
        if compile_result.returncode != 0:
            os.unlink(temp_file_path)
            return f"Compilation Error: {compile_result.stderr}"

        run_result = subprocess.run(
            ["java", "TempClass"], capture_output=True, text=True, timeout=30
        )
        os.unlink(temp_file_path)
        return (
            run_result.stdout
            if run_result.returncode == 0
            else f"Runtime Error: {run_result.stderr}"
        )
    except subprocess.TimeoutExpired:
        os.unlink(temp_file_path)
        return "Error: Execution timed out"


def list_directory(args):
    directory = args[0] if args else "."
    try:
        files = os.listdir(directory)
        for f in files:
            print(f)
    except Exception as e:
        print(f"Error listing directory: {e}")


def read_file(args):
    if not args:
        print("Usage: /read <filename>")
        return
    filename = args[0]
    try:
        with open(filename, "r") as file:
            content = file.read()
            print(content)
    except Exception as e:
        print(f"Error reading file: {e}")


def log_action(action, detail=""):
    logging.info(f"{action}: {detail}")
