from setuptools import setup, find_packages
import os
import site

import platform
from pathlib import Path


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


def get_setup_message():
    if platform.system() == "Windows":
        user_scripts = Path(site.USER_BASE) / "Scripts"
        return f"""
==============================================
Important Setup Instructions for Windows Users
==============================================

The npcsh command line tool has been installed, but you need to add it to your PATH.

Please add this directory to your PATH:
{user_scripts}

You can do this in one of two ways:

1. Quick method (run in Command Prompt as Administrator):
   setx PATH "%PATH%;{user_scripts}"

2. Manual method:
   a. Press Win + X and select "System"
   b. Click "Advanced system settings"
   c. Click "Environment Variables"
   d. Under "User variables", find and select "Path"
   e. Click "Edit"
   f. Click "New"
   g. Add this path: {user_scripts}
   h. Click "OK" on all windows

After adding to PATH, restart your terminal/command prompt.

You can then run:
npcsh-setup

To configure your API keys and preferences.
==============================================
"""
    return ""  # Return empty string for non-Windows platforms


extra_files = package_files("npcsh/npc_team/")

setup(
    name="npcsh",
    version="0.3.23",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "redis",
        "flask_sse",
        "anthropic",
        "screeninfo",
        "sentence_transformers",
        "nltk",
        "thefuzz",
        "beautifulsoup4",
        "google-generativeai",
        "google-genai",
        "duckduckgo-search",
        "pypdf",
        "PyMuPDF",
        "opencv-python",
        "librosa",
        "openai",
        "jinja2",
        "pyautogui",
        "pandas",
        "matplotlib",
        "IPython",
        "ollama",
        "requests",
        "markdown",
        "PyYAML",
        "langchain",
        "langchain_community",
        "openai-whisper",
        "pyaudio",
        "pygments",
        "pyttsx3",
        "kuzu",
        "chromadb",
        "gtts",
        "playsound==1.2.2",
        "termcolor",
        "colorama",
        "python-dotenv",
        "pytest",
        "googlesearch-python",
        "flask",
        "flask_cors",
        "diffusers",
    ],
    entry_points={
        "console_scripts": [
            "npcsh=npcsh.shell:main",
            "npc=npcsh.cli:main",
        ],
    },
    author="Christopher Agostino",
    author_email="info@npcworldwi.de",
    description="npcsh is a command line tool for integrating LLMs into everyday workflows and for orchestrating teams of NPCs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cagostino/npcsh",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
    data_files=[("npcsh/npc_team", extra_files)],
    python_requires=">=3.10",
)
# Print setup message only on Windows
if platform.system() == "Windows":
    print(get_setup_message())
