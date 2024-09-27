# helpers.py
import logging

logging.basicConfig(
    filename=".npcsh.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
import os


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
