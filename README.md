<p align="center">
  <img src="https://raw.githubusercontent.com/cagostino/npcsh/main/npcsh/npcsh.png" alt="npcsh logo with sibiji the spider">
</p>


# npcsh


- `npcsh` is a python-based command-line tool designed to integrate Large Language Models (LLMs) into one's daily workflow by making them available through the command line shell.

- `npcsh` leverages the power of LLMs to understand your natural language commands and questions, executing tasks, answering queries, and providing relevant information from local files and the web.

- `npcsh` provides macros to accomplish common tasks with LLMs like voice control (`/whisper`), image generation (`/vixynt`), screenshot capture and analysis (`/ots`), one-shot questions (`/sample`), and more.

- `npcsh` allows users to coordinate agents (i.e. NPCs) to form assembly lines that can reliably accomplish complicated multi-step procedures.

## Key Features

* **Natural Language Interface:** Interact with your system using natural language.  `npcsh` translates your requests into executable commands or provides direct answers.
* **NPC-Driven Interactions:** Define custom "NPCs" (Non-Player Characters) with specific personalities, directives, and tools. This allows for tailored interactions based on the task at hand.
* **Local File Integration:** Seamlessly access and query information from your local files using Retrieval Augmented Generation (RAG). `npcsh` understands the context of your project.
* **Web Search Capabilities:**  Don't have the answer locally? `npcsh` can search the web for you and integrate the findings into its responses.
* **Tool Use:** Define custom tools for your NPCs to use, expanding their capabilities beyond simple commands and questions. Tools can be anything from image generation to web searches.
* **Multiple LLM Providers:** Supports multiple LLM providers, including Ollama, OpenAI, OpenAI-like APIs, and Anthropic, giving you flexibility in choosing the best model for your needs.
* **Interactive Modes:** Specialized modes for different interaction styles:
    * **Whisper Mode:**  Use your voice to interact with the LLM.
    * **Notes Mode:** Quickly jot down notes and store them within the `npcsh` database.
    * **Data Mode:**  Load, manipulate, and query data from various file formats.
    * **Spool Mode:** Engage in a continuous conversation with the LLM, maintaining context across multiple turns.
* **Extensible with Python:**  Write your own tools and extend `npcsh`'s functionality using Python.


## Installation
`npcsh` is available on PyPI and can be installed using pip. Before installing, make sure you have the necessary dependencies installed on your system. Below are the instructions for installing such dependencies on Linux, Mac, and (soon-to-be) Windows. If you find any other dependencies that are needed, please let us know so we can update the installation instructions to be more accommodating.

### Linux install
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



### Mac install
```bash
brew install portaudio
brew install ffmpeg
brew install ollama
brew services start ollama
brew install pygobject3
pip install npcsh
```
### Widows Install

Coming soon!



## Startup Configuration and Project Structure
After it has been pip installed, npcsh can be used as a command line tool. Start it by typing:
```bash
npcsh
```
npcsh will generate a .npcshrc file in your home directory that stores your npcsh settings. You can set your preferred LLM provider, model, and database path. The installer will automatically add this file to your shell config, but if it does not do so successfully for whatever reason you can add the following to your .bashrc or .zshrc:

```bash
# Source NPCSH configuration
if [ -f ~/.npcshrc ]; then
    . ~/.npcshrc
fi
```

We support inference via openai, anthropic, ollama, and other openai-like APIs. To use tools that require API keys, create an ".env" file up in the folder where you are working or place relevant API keys as env variables in your ~/.npcshrc.

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

The user can change the default model by setting the environment variable `NPCSH_MODEL` in their ~/.npcshrc to the desired model name and to change the provider by setting the environment variable `NPCSH_PROVIDER` to the desired provider name.

The provider must be one of ['ollama', 'openai', 'anthropic', 'openai-like'] and the model must be one available from those providers. Individual npcs can also be set to use different models and providers by setting the `model` and `provider` keys in the npc files.
```bash
~/.npcsh/
├── npc_team/           # Global NPCs
│   ├── tools/          # Global tools
│   └── assembly_lines/ # Workflow pipelines
./npc_team/            # Project-specific NPCs
├── tools/             # Project tools
└── assembly_lines/    # Project workflows
```


## npcsh usage
npcsh is a command-line tool that allows you to interact with Large Language Models (LLMs) using natural language commands. It provides a variety of modes and features to help you accomplish tasks, answer questions, and integrate LLMs into your workflow. Through npcsh, users can have LLMs make use of tools that have become common in popular LLM applications, such as image generation, data analysis, web searches, and more.

Here are some examples of how you can use npcsh
```npcsh
Who was Simon Bolivar?
```

```npcsh
What is the capital of France?
```

```npcsh
What's the weather in Tokyo?
```

```npcsh
Why is the code in my VS code not working?
```

```npcsh
What is the biggest file in my current folder?
```

```npcsh
What is the best way to implement a linked list in Python?
```

```npcsh
In which of the files in the current directory is "Simon Bolivar" mentioned?
```


```npcsh
list all files in the current directory
```



## Macros

While npcsh can decide the best tool to use based on the user's input, the user can also specify certain tools to use by using a macro. Macros are commands that start with a forward slash (/) and are followed (in some cases) by the relevant arguments for those macros.
To learn about them from within the shell, type:
```npcsh
/help
```
To exit the shell:
```npcsh
/exit
```

Otherwise, here are some more detailed examples of macros that can be used in npcsh:

### Spool
Spool mode allows one to enter into a conversation with a specific LLM or a specific NPC. This is used for having distinct interactions from those in the base shell and these will be separately contained .
Start the spool mode:
```npcsh
/spool
```
Start the spool mode with a specific npc

Start the spool mode with a specific LLM model


Start the spool mode with specific files in context
```npcsh
/spool files=[*.py,*.md] # Load specific files for context
```




### Over-the-shoulder: Screenshots and image analysis

Use the /ots macro to take a screenshot and write a prompt for an LLM to answer about the screenshot.
```npcsh
/ots
```

Alternatively, pass an existing image in like :
```npcsh
/ots /path/to/image.png
```




### Image Generation
Image generation can be done with the /vixynt macro or through a general prompt that decides to make use of the relevant tool.
Use /vixynt like so
```npcsh
/vixynt A futuristic cityscape @dall-e-3
/vixynt A peaceful landscape @stable-diffusion
```
Alternatively:
```npcsh
Generate an image of a futuristic cityscape
```
should result in the llm deciding to make use of the image generation tool that it has available.


### Voice Control
Enter into a voice-controlled mode to interact with the LLM. This mode can executet commands and use tools just like the basic npcsh shell.
```npcsh
/whisper
```


### Executing Bash Commands
You can execute bash commands directly within npcsh. The LLM can also generate and execute bash commands based on your natural language requests.
For example:
```npcsh
ls -l
cp file1.txt file2.txt
mv file1.txt file2.txt
mkdir new_directory
```


### Compilation and NPC Interaction
Compile a specified NPC profile. This will make it available for use in npcsh interactions.
```npcsh
/compile <npc_file>
```
You can also use `/com` as an alias for `/compile`. If no NPC file is specified, all NPCs in the npc_team directory will be compiled.

Begin a conversations with a specified NPC by referencing their name
```npcsh
/<npc_name>:
```

### Data Interaction and analysis
Enter into data mode to load, manipulate, and query data from various file formats.
```npcsh
/data
load data.csv as df
df.describe()
```

### Notes
Jot down notes and store them within the npcsh database and in the current directory as a text file.
```npcsh
/notes
```

### Changing defaults from within npcsh
Users can change the default model and provider from within npcsh by using the following commands:
```npcsh
/set model ollama
/set provider llama3.2
```


## Creating NPCs
NPCs are defined in YAML files within the npc_team directory. Each NPC has a name, primary directive, and optionally, a list of tools. See the examples in the npc_profiles directory for guidance.





## Creating Tools
Tools are defined in YAML files within the npc_team/tools directory. Each tool has a name, inputs, a prompt, and pre/post-processing steps. Tools can be implemented in Python or other languages supported by npcsh.


## Python Examples
Integrate npcsh into your Python projects for additional flexibility. Below are a few examples of how to use the library programmatically.




### Example 1: Creating and Using an NPC
This example shows how to create and initialize an NPC and use it to answer a question.
```bash
import sqlite3
from npcsh import NPC, load_npc_from_file

# Set up database connection
db_path = '~/npcsh_history.db'
conn = sqlite3.connect(db_path)

# Load NPC from a file
npc = NPC(name='Simon Bolivar', db_conn=conn)

# Ask a question to the NPC
question = "What are the project updates?"
llm_response = npc.get_llm_response(question)

# Output the NPC's response
print(f"{npc.name}: {llm_response['response']}")
```
### Example 2: Using an NPC to Analyze Data
This example shows how to use an NPC to perform data analysis on a DataFrame using LLM commands.
```bash
import pandas as pd
from npcsh import NPC

# Create dummy data for analysis
data = {
    'feedback': ["Great product!", "Could be better", "Amazing service"],
    'customer_id': [1, 2, 3],
}
df = pd.DataFrame(data)

# Initialize the NPC
npc = NPC(name='Sibiji', db_conn=sqlite3.connect('~/npcsh_history.db'))

# Formulate a command for analysis
command = "Analyze customer feedback for sentiment."
```


### Example 3: Creating and Using a Tool
You can define a tool and execute it from within your Python script.
```bash

from npcsh import Tool, NPC
# Create a tool from a dictionary
tool_data = {
    "tool_name": "my_tool",
    "inputs": ["input_text"],
    "preprocess": [{"engine": "natural", "code": "Preprocessing: {{ inputs.input_text }}"}],
    "prompt": {"engine": "natural", "code": "Here is the output: {{ llm_response }}"},
    "postprocess": []
}

# Instantiate the tool
tool = Tool(tool_data)

# Create an NPC instance
npc = NPC(name='Sibiji', db_conn=sqlite3.connect('/path/to/npcsh_database.db'))

# Define input values dictionary
input_values = {
    "input_text": "User input goes here"
}

# Execute the tool
output = tool.execute(input_values, npc.tools_dict, None, 'Sample Command', npc)

print('Tool Output:', output)
```



## Contributing
Contributions are welcome! Please submit issues and pull requests on the GitHub repository.

## License
This project is licensed under the MIT License.
