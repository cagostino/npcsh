from setuptools import setup, find_packages
import site
import platform
from pathlib import Path
import os


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
    return ""


# Base requirements (no LLM packages)
base_requirements = [
    "jinja2",
    "scipy",
    "numpy",
    "requests",
    "markdown",
    "PyYAML",
    "pygments",
    "termcolor",
    "colorama",
    "Pillow",
    "python-dotenv",
    "pandas",
    "beautifulsoup4",
    "duckduckgo-search",
    "flask",
    "flask_cors",
    "redis",
    "flask_sse",
]

# API integration requirements
api_requirements = ["anthropic", "openai", "google-generativeai", "google-genai"]

# Local ML/AI requirements
local_requirements = [
    "sentence_transformers",
    "opencv-python",
    "ollama",
    "kuzu",
    "chromadb",
    "diffusers",
    "nltk",
]

# Voice/Audio requirements
voice_requirements = [
    "openai-whisper",
    "pyaudio",
    "gtts",
    "playsound==1.2.2",
    "pyttsx3",
]

extra_files = package_files("npcsh/npc_team/")

setup(
    name="npcsh",
    version="0.3.27.3",
    packages=find_packages(exclude=["tests*"]),
    install_requires=base_requirements,  # Only install base requirements by default
    extras_require={
        "lite": api_requirements,  # Just API integrations
        "local": local_requirements,  # Local AI/ML features
        "whisper": voice_requirements,  # Voice/Audio features
        "all": api_requirements + local_requirements + voice_requirements,  # Everything
    },
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

if platform.system() == "Windows":
    print(get_setup_message())
