<p align="center">
  <img src="https://raw.githubusercontent.com/cagostino/npcsh/main/npcsh/npcsh.png" alt="npcsh logo with sibiji the spider">
</p>                                      


# npcsh


`npcsh` is a command-line tool designed to integrate Large Language Models (LLMs) into your daily workflow. It leverages the power of LLMs to understand your natural language commands and questions, executing tasks, answering queries, and providing relevant information from local files and the web.

`npcsh` allows users to coordinate agents to form assembly lines of NPCs that can reliably accomplish complicated multi-step procedures.

## Key Features

* **Natural Language Interface:** Interact with your system using plain English.  `npcsh` translates your requests into executable commands or provides direct answers.
* **NPC-Driven Interactions:** Define custom "NPCs" (Non-Player Characters) with specific personalities, directives, and tools. This allows for tailored interactions based on the task at hand.
* **Local File Integration:** Seamlessly access and query information from your local files using Retrieval Augmented Generation (RAG). `npcsh` understands the context of your project.
* **Web Search Capabilities:**  Don't have the answer locally? `npcsh` can search the web for you and integrate the findings into its responses.
* **Tool Use:** Define custom tools for your NPCs to use, expanding their capabilities beyond simple commands and questions. Tools can be anything from image generation to web searches.
* **Multiple LLM Providers:** Supports multiple LLM providers, including Ollama, OpenAI, and Anthropic, giving you flexibility in choosing the best model for your needs.
* **Interactive Modes:** Specialized modes for different interaction styles:
    * **Whisper Mode:**  Use your voice to interact with the LLM.
    * **Notes Mode:** Quickly jot down notes and store them within the `npcsh` database.
    * **Data Mode:**  Load, manipulate, and query data from various file formats.
    * **Spool Mode:** Engage in a continuous conversation with the LLM, maintaining context across multiple turns.
* **Extensible with Python:**  Write your own tools and extend `npcsh`'s functionality using Python.


## Dependencies

- ollama
- python >3.10





## Linux install
```bash

sudo apt-get install espeak
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install alsa-base alsa-utils
sudo apt-get install libcairo2-dev
sudo apt-get install libgirepository1.0-dev
sudo apt-get install ffmpeg
pip install npcsh
```
And if you don't have ollama installed, use this:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```



## Mac install
```bash
brew install portaudio
brew install ffmpeg
brew install ollama
brew services start ollama
brew install pygobject3
pip install npcsh
```

## Usage
After it has been pip installed, npcsh can be used as a command line tool. Start it by typing:
```bash
npcsh
```

Once in the npcsh, you can use bash commands or write natural language queries or commands. You can also switch between different modes defined below and you can compile a network of NPCs or use the macro tools we have developed.



## Basic Commands
```npcsh
/compile <npc_file>: Compiles the specified NPC profile.
/com <npc_file>: Alias for /compile. If no NPC file is specified, compiles all NPCs in the npc_team directory.
/<npc_name>: Switch to the specified NPC.
/whisper: Enter whisper mode.
/notes: Enter notes mode.
/data: Enter data mode to interact with data.
/spool: Enter spool mode for continuous conversation.
/ots: Take a screenshot and optionally analyze it with a prompt.
/vixynt <prompt>: Generate an image using the specified prompt.
/set <setting> <value>: Set configuration options (e.g., model, provider).
/help: Show the help message.
/exit or /quit: Exit npcsh.
```
## Executing Bash Commands
You can execute bash commands directly within npcsh. The LLM can also generate and execute bash commands based on your natural language requests.


## Configuration

The .npcshrc file in your home directory stores your npcsh settings. You can set your preferred LLM provider, model, and database path. The installer will automatically add this file to your shell config, but if it does not you can add the following to your .bashrc or .zshrc:

```bash
# Source NPCSH configuration
if [ -f ~/.npcshrc ]; then
    . ~/.npcshrc
fi
```

We support inference via openai and anthropic. To use them, set an ".env" file up in the folder where you are working and set the API keys there or set the environment variables in your ~/.npcshrc

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

The user can change the default model by setting the environment variable `NPCSH_MODEL` in their ~/.npcshrc to the desired model name and to change the provider by setting the environment variable `NPCSH_PROVIDER` to the desired provider name.

The provider must be one of ['ollama', 'openai', 'anthropic'] and the model must be one available from those providers. Individual npcs can also be set to use different models and providers by setting the `model` and `provider` keys in the npc files.


## compilation

Each NPC can be compiled to accomplish their primary directive and then any issues faced will be recorded and associated with the NPC so that it can reference it later through vector search. In any of the modes where a user requests input from an NPC, the NPC will include RAG search results before carrying out the request.


## Examples

Simple Bash Command: `ls -l`
LLM-Generated Command: `list all files in the current directory`
NPC Compilation and use: `/com foreman.npc \n /foreman \n what is the status of the project?`
Image Generation: `/vixynt a cat wearing a hat`
Screenshot Analysis: /ots What do you see in this screenshot?
Data Interaction: /data load from data.csv as my_data followed by /data pd.my_data.head()




## Creating NPCs
NPCs are defined in YAML files within the npc_team directory. Each NPC has a name, primary directive, and optionally, a list of tools. See the examples in the npc_profiles directory for guidance.

## Creating Tools
Tools are defined in YAML files within the npc_team/tools directory. Each tool has a name, inputs, a prompt, and pre/post-processing steps. Tools can be implemented in Python or other languages supported by npcsh.

## Contributing
Contributions are welcome! Please submit issues and pull requests on the GitHub repository.

## License
This project is licensed under the MIT License.
