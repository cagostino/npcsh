<p align="center">
  <img src="https://raw.githubusercontent.com/cagostino/npcsh/main/npcsh/npcsh.png" alt="npcsh logo with sibiji the spider">
</p>


# npcsh


- `npcsh` is a python-based command-line tool designed to integrate Large Language Models (LLMs) into one's daily workflow by making them available through the command line shell.

- **Smart Interpreter**: `npcsh` leverages the power of LLMs to understand your natural language commands and questions, executing tasks, answering queries, and providing relevant information from local files and the web.

- **Macros**: `npcsh` provides macros to accomplish common tasks with LLMs like voice control (`/whisper`), image generation (`/vixynt`), screenshot capture and analysis (`/ots`), one-shot questions (`/sample`), and more.

- **NPC-Driven Interactions**: `npcsh` allows users to coordinate agents (i.e. NPCs) to form assembly lines that can reliably accomplish complicated multi-step procedures. Define custom "NPCs" (Non-Player Characters) with specific personalities, directives, and tools. This allows for tailored interactions based on the task at hand.

* **Tool Use:** Define custom tools for your NPCs to use, expanding their capabilities beyond simple commands and questions. Some example tools include: image generation, local file search, data analysis, web search, local file search, bash command execution, and more.

* **Extensible with Python:**  Write your own tools and extend `npcsh`'s functionality using Python or use our functionis to simplify interactions with LLMs in your projects.

* **Bash Wrapper:** Execute bash commands directly without leaving the shell. Use your favorite command-line tools like VIM, Emacs, ipython, sqlite3, git, and more without leaving the shell!


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
After it has been pip installed, `npcsh` can be used as a command line tool. Start it by typing:
```bash
npcsh
```
When initialized, `npcsh` will generate a .npcshrc file in your home directory that stores your npcsh settings.
Here is an example of what the .npcshrc file might look like after this has been run.
```bash
# NPCSH Configuration File
export NPCSH_INITIALIZED=1
export NPCSH_PROVIDER='ollama'
export NPCSH_MODEL='llama3.2'
export NPCSH_DB_PATH='~/npcsh_history.db'
```
`npcsh` also comes with a set of tools and NPCs that are used in processing. It will generate a folder at ~/.npcsh/ that contains the tools and NPCs that are used in the shell and these will be used in the absence of other project-specific ones. Additionally, `npcsh` records interactions and compiled information about npcs within a local SQLite database at the path specified in the .npcshrc file. This will default to ~/npcsh_history.db if not specified. When the data mode is used to load or analayze data in CSVs or PDFs, these data will be stored in the same database for future reference.

The installer will automatically add this file to your shell config, but if it does not do so successfully for whatever reason you can add the following to your .bashrc or .zshrc:

```bash
# Source NPCSH configuration
if [ -f ~/.npcshrc ]; then
    . ~/.npcshrc
fi
```

We support inference via `openai`, `anthropic`, `ollama`, and `openai-like` APIs. The default provider must be one of `['ollama', 'openai', 'anthropic', 'openai-like']` and the model must be one available from those providers.

To use tools that require API keys, create an `.env` file up in the folder where you are working or place relevant API keys as env variables in your ~/.npcshrc. If you already have these API keys set in a ~/.bashrc or a ~/.zshrc or similar files, you need not additionally add them to ~/.npcshrc or to an `.env` file. Here is an example of what an `.env` file might look like:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
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
├── tools/             # Project tools
└── assembly_lines/    # Project workflows
```


## npcsh usage
`npcsh` is a command-line tool that allows you to interact with Large Language Models (LLMs) using natural language commands. It provides a variety of modes and features to help you accomplish tasks, answer questions, and integrate LLMs into your workflow. Through npcsh, users can have LLMs make use of tools that have become common in popular LLM applications, such as image generation, data analysis, web searches, and more.

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
