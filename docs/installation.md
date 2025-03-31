
## Installation
`npcsh` is available on PyPI and can be installed using pip. Before installing, make sure you have the necessary dependencies installed on your system. Below are the instructions for installing such dependencies on Linux, Mac, and Windows. If you find any other dependencies that are needed, please let us know so we can update the installation instructions to be more accommodating.

### Linux install
```bash

# for audio primarily
sudo apt-get install espeak
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install alsa-base alsa-utils
sudo apt-get install libcairo2-dev
sudo apt-get install libgirepository1.0-dev
sudo apt-get install ffmpeg

# for triggers
sudo apt install inotify-tools


#And if you don't have ollama installed, use this:
curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcsh
# if you want to install with the API libraries
pip install npcsh[lite]
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install npcsh[local]
# if you want to use tts/stt
pip install npcsh[whisper]

# if you want everything:
pip install npcsh[all]




### Mac install
```bash
#mainly for audio
brew install portaudio
brew install ffmpeg
brew install pygobject3

# for triggers
brew install ...


brew install ollama
brew services start ollama
ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcsh
# if you want to install with the API libraries
pip install npcsh[lite]
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install npcsh[local]
# if you want to use tts/stt
pip install npcsh[whisper]

# if you want everything:
pip install npcsh[all]

```
### Windows Install

Download and install ollama exe.

Then, in a powershell. Download and install ffmpeg.

```
ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcsh
# if you want to install with the API libraries
pip install npcsh[lite]
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install npcsh[local]
# if you want to use tts/stt
pip install npcsh[whisper]

# if you want everything:
pip install npcsh[all]

```
As of now, npcsh appears to work well with some of the core functionalities like /ots and /whisper.


### Fedora Install (under construction)

python3-dev (fixes hnswlib issues with chroma db)
xhost +  (pyautogui)
python-tkinter (pyautogui)

## Startup Configuration and Project Structure
After it has been pip installed, `npcsh` can be used as a command line tool. Start it by typing:
```bash
npcsh
```
When initialized, `npcsh` will generate a .npcshrc file in your home directory that stores your npcsh settings.
Here is an example of what the .npcshrc file might look like after this has been run.
```bash
# NPCSH Configuration File
export NPCSH_INITIALIZED=1
export NPCSH_CHAT_PROVIDER='ollama'
export NPCSH_CHAT_MODEL='llama3.2'
export NPCSH_DB_PATH='~/npcsh_history.db'
```
`npcsh` also comes with a set of tools and NPCs that are used in processing. It will generate a folder at ~/.npcsh/ that contains the tools and NPCs that are used in the shell and these will be used in the absence of other project-specific ones. Additionally, `npcsh` records interactions and compiled information about npcs within a local SQLite database at the path specified in the .npcshrc file. This will default to ~/npcsh_history.db if not specified. When the data mode is used to load or analyze data in CSVs or PDFs, these data will be stored in the same database for future reference.

The installer will automatically add this file to your shell config, but if it does not do so successfully for whatever reason you can add the following to your .bashrc or .zshrc:

```bash
# Source NPCSH configuration
if [ -f ~/.npcshrc ]; then
    . ~/.npcshrc
fi
```

We support inference via `openai`, `anthropic`, `ollama`,`gemini`, `deepseek`,  and `openai-like` APIs. The default provider must be one of `['openai','anthropic','ollama', 'gemini', 'deepseek', 'openai-like']` and the model must be one available from those providers.

To use tools that require API keys, create an `.env` file up in the folder where you are working or place relevant API keys as env variables in your ~/.npcshrc. If you already have these API keys set in a ~/.bashrc or a ~/.zshrc or similar files, you need not additionally add them to ~/.npcshrc or to an `.env` file. Here is an example of what an `.env` file might look like:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export DEEPSEEK_API_KEY='your_deepseek_key'
export GEMINI_API_KEY='your_gemini_key'
export PERPLEXITY_API_KEY='your_perplexity_key'
```


 Individual npcs can also be set to use different models and providers by setting the `model` and `provider` keys in the npc files.
 Once initialized and set up, you will find the following in your ~/.npcsh directory:
```bash
~/.npcsh/
├── npc_team/           # Global NPCs
│   ├── tools/          # Global tools
│   └── assembly_lines/ # Workflow pipelines

```
For cases where you wish to set up a project specific set of NPCs, tools, and assembly lines, add a `npc_team` directory to your project and `npcsh` should be able to pick up on its presence, like so:
```bash
./npc_team/            # Project-specific NPCs
├── tools/             # Project tools #example tool next
│   └── example.tool
└── assembly_lines/    # Project workflows
    └── example.pipe
└── models/    # Project workflows
    └── example.model
└── example1.npc        # Example NPC
└── example2.npc        # Example NPC
└── example1.ctx        # Example NPC
└── example2.ctx        # Example NPC

```

## IMPORTANT: migrations and deprecations

### v0.3.4
-In v0.3.4, the structure for tools was adjusted. If you have made custom tools please refer to the structure within npc_compiler to ensure that they are in the correct format. Otherwise, do the following
```bash
rm ~/.npcsh/npc_team/tools/*.tool
```
and then
```bash
npcsh
```
and the updated tools will be copied over into the correct location.

### v0.3.5
-Version 0.3.5 included a complete overhaul and refactoring of the llm_funcs module. This was done to make it not as horribly long and to make it easier to add new models and providers


-in version 0.3.5, a change was introduced to the database schema for messages to add npcs, models, providers, and associated attachments to data. If you have used `npcsh` before this version, you will need to run this migration script to update your database schema:   [migrate_conversation_history_v0.3.5.py](https://github.com/cagostino/npcsh/blob/cfb9dc226e227b3e888f3abab53585693e77f43d/npcsh/migrations/migrate_conversation_history_%3Cv0.3.4-%3Ev0.3.5.py)

-additionally, NPCSH_MODEL and NPCSH_PROVIDER have been renamed to NPCSH_CHAT_MODEL and NPCSH_CHAT_PROVIDER
to provide a more consistent naming scheme now that we have additionally introduced `NPCSH_VISION_MODEL` and `NPCSH_VISION_PROVIDER`, `NPCSH_EMBEDDING_MODEL`, `NPCSH_EMBEDDING_PROVIDER`, `NPCSH_REASONING_MODEL`, `NPCSH_REASONING_PROVIDER`, `NPCSH_IMAGE_GEN_MODEL`, and `NPCSH_IMAGE_GEN_PROVIDER`.
- In addition, we have added NPCSH_API_URL to better accommodate openai-like apis that require a specific url to be set as well as `NPCSH_STREAM_OUTPUT` to indicate whether or not to use streaming in one's responses. It will be set to 0 (false) by default as it has only been tested  and verified for a small subset of the models and providers we have available (openai, anthropic, and ollama). If you try it and run into issues, please post them here so we can correct them as soon as possible !

