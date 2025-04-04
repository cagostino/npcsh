<p align="center">
  <img src="https://raw.githubusercontent.com/cagostino/npcsh/main/npcsh/npcsh.png" alt="npcsh logo with sibiji the spider">
</p>


# npcsh


- `npcsh` is a python-based AI Agent framework designed to integrate Large Language Models (LLMs) and Agents into one's daily workflow by making them available and easily configurable through a command line shell as well as an extensible python library.

- `npcsh` stores your executed commands, conversations, generated images, captured screenshots, and more in a central database

- The NPC shell understands natural language commands and provides a suite of built-in tools (macros) for tasks like voice control, image generation, and web searching, while allowing users to create custom NPCs (AI agents) with specific personalities and capabilities for complex workflows.

- `npcsh` is extensible through its python library implementation and offers a simple CLI interface with the `npc` cli.

- `npcsh` integrates with local and enterprise LLM providers like Ollama, LMStudio, OpenAI, Anthropic, Gemini, and Deepseek, making it a versatile tool for both simple commands and sophisticated AI-driven tasks.

Read the docs at [npcsh.readthedocs.io](https://npcsh.readthedocs.io/en/latest/)

`npcsh` can be used in a graphical user interface through the NPC Studio.
See the open source code for NPC Studio [here](https://github.com/). Download the executables at [our website](https://www.npcworldwi.de/npc-studio).



Interested to stay in the loop and to hear the latest and greatest about `npcsh` ? Be sure to sign up for the [npcsh newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A)!


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=cagostino/npcsh&type=Date)](https://star-history.com/#cagostino/npcsh&Date)

## TLDR Cheat Sheet for NPC shell and cli
Users can take advantage of `npcsh` through its custom shell or through a command-line interface (CLI) tool. Below is a cheat sheet that shows how to use `npcsh` commands in both the shell and the CLI. For the npcsh commands to work, one must activate `npcsh` by typing it in a shell.




| Task | npc CLI | npcsh |
|----------|----------|----------|
| Ask a generic question | npc 'prompt' | 'prompt' |
| Compile an NPC | npc compile /path/to/npc.npc | /compile /path/to/npc.npc |
| Computer use | npc plonk -n 'npc_name' -sp 'task for plonk to carry out '| /plonk -n 'npc_name' -sp 'task for plonk to carry out ' |
| Conjure an NPC team from context and templates | npc init -t 'template1, template2' -ctx 'context'   | /conjure  -t 'template1, 'template2' -ctx 'context'  |
| Enter a chat with an NPC (NPC needs to be compiled first) | npc chat -n npc_name | /spool npc=<npc_name> |
| Generate image    | npc vixynt 'prompt'  | /vixynt prompt   |
| Get a sample LLM response  | npc sample 'prompt'   | /sample prompt for llm  |
| Search for a term in the npcsh_db only in conversations with a specific npc | npc rag -n 'npc_name' -f 'filename' -q 'query' | /rag -n 'npc_name' -f 'filename' -q 'query' |
| Search the web | npc search -q "cal golden bears football schedule" -sp perplexity | /search -p perplexity 'cal bears football schedule' |
| Serve an NPC team | npc serve --port 5337 --cors='http://localhost:5137/' | /serve --port 5337 --cors='http://localhost:5137/' |
| Screenshot analysis  | npc ots |  /ots  |
| Voice Chat    | npc whisper -n 'npc_name'   | /whisper   |


## Python Examples
Integrate npcsh into your Python projects for additional flexibility. Below are a few examples of how to use the library programmatically.


### Example 1: usisng npcsh's get_llm_response and get_stream

```python
from npcsh.llm_funcs import get_llm_response

# ollama's llama3.2
response = get_llm_response("What is the capital of France? Respond with a json object containing 'capital' as the key and the capital as the value.",
                            model='llama3.2',
                            provider='ollama',
                            format='json')
#openai's gpt-4o-mini
response = get_llm_response("What is the capital of France? Respond with a json object containing 'capital' as the key and the capital as the value.",
                            model='gpt-4o-mini',
                            provider='openai',
                            format='json')
# anthropic's claude haikue 3.5 latest
response = get_llm_response("What is the capital of France? Respond with a json object containing 'capital' as the key and the capital as the value.",
                            model='claude-haiku-3-5-latest',
                            provider='anthropic',
                            format='json')

# alternatively, if you have NPCSH_CHAT_MODEL / NPCSH_CHAT_PROVIDER set in your ~/.npcshrc, it will use those values
response = get_llm_response("What is the capital of France? Respond with a json object containing 'capital' as the key and the capital as the value.",
                            format='json')




#stream responses

response = get_stream("What is the capital of France? Respond with a json object containing 'capital' as the key and the capital as the value.", )


```

### Example 3: Building a flow with check_llm_command

```python
#first let's demonstrate the capabilities of npcsh's check_llm_command
from npcsh.llm_funcs import check_llm_command

command = 'can you write a description of the idea of semantic degeneracy?'

response = check_llm_command(command,
                             model='gpt-4o-mini',
                             provider='openai')



# now to make the most of check_llm_command, let's add an NPC with a generic code execution tool


from npcsh.npc_compiler import NPC, Tool
from npcsh.llm_funcs import check_llm_command

code_execution_tool = Tool(
    {
        "tool_name": "execute_python",
        "description": """Executes a code block in python.
                Final output from script MUST be stored in a variable called `output`.
                          """,
        "inputs": ["script"],
        "steps": [
            {
                "engine": " python",
                "code": """{{ script }}""",
            }
        ],
    }
)


command = """can you write a description of the idea of semantic degeneracy and save it to a file?
             After, can you take that and make various versions of it from the points of
             views of different sub-disciplines of natural lanaguage processing?
             Finally produce a synthesis of the resultant various versions and save it."
            """
npc = NPC(
    name="NLP_Master",
    primary_directive="Provide astute anlayses on topics related to NLP. Carry out relevant tasks for users to aid them in their NLP-based analyses",
    model="gpt-4o-mini",
    provider="openai",
    tools=[code_execution_tool],
)
response = check_llm_command(
    command, model="gpt-4o-mini", provider="openai", npc=npc, stream=False
)


# or by attaching an NPC Team
from npcsh.npc_compiler import NPC

response = check_llm_command(command,
                             model='gpt-4o-mini',
                              provider='openai',)
```



### Example 2: Creating and Using an NPC
This example shows how to create and initialize an NPC and use it to answer a question.
```python
import sqlite3
from npcsh.npc_compiler import NPC

# Set up database connection
db_path = '~/npcsh_history.db'
conn = sqlite3.connect(db_path)

# Load NPC from a file
npc = NPC(
          name='Simon Bolivar',
          db_conn=conn,
          primary_directive='Liberate South America from the Spanish Royalists.',
          model='gpt-4o-mini',
          provider='openai',
          )

response = npc.get_llm_response("What is the most important territory to retain in the Andes mountains?")
print(response['response'])
```
```bash
'The most important territory to retain in the Andes mountains for the cause of liberation in South America would be the region of Quito in present-day Ecuador. This area is strategically significant due to its location and access to key trade routes. It also acts as a vital link between the northern and southern parts of the continent, influencing both military movements and the morale of the independence struggle. Retaining control over Quito would bolster efforts to unite various factions in the fight against Spanish colonial rule across the Andean states.'
```
### Example 3: Using an NPC to Analyze Data
This example shows how to use an NPC to perform data analysis on a DataFrame using LLM commands.
```python
from npcsh.npc_compiler import NPC
import sqlite3
import os
# Set up database connection
db_path = '~/npcsh_history.db'
conn = sqlite3.connect(os.path.expanduser(db_path))

# make a table to put into npcsh_history.db or change this example to use an existing table in a database you have
import pandas as pd
data = {
        'customer_feedback': ['The product is great!', 'The service was terrible.', 'I love the new feature.'],
        'customer_id': [1, 2, 3],
        'customer_rating': [5, 1, 3],
        'timestamp': ['2022-01-01', '2022-01-02', '2022-01-03']
        }


df = pd.DataFrame(data)
df.to_sql('customer_feedback', conn, if_exists='replace', index=False)


npc = NPC(
          name='Felix',
          db_conn=conn,
          primary_directive='Analyze customer feedback for sentiment.',
          model='gpt-4o-mini',
          provider='openai',
          )
response = npc.analyze_db_data('Provide a detailed report on the data contained in the `customer_feedback` table?')


```


### Example 4: Creating and Using a Tool
You can define a tool and execute it from within your Python script.
Here we'll create a tool that will take in a pdf file, extract the text, and then answer a user request about the text.

```python
from npcsh.npc_compiler import Tool, NPC
import sqlite3
import os

from jinja2 import Environment, FileSystemLoader

# Create a proper Jinja environment
jinja_env = Environment(loader=FileSystemLoader('.'))


tool_data = {
    "tool_name": "pdf_analyzer",
    "inputs": ["request", "file"],
    "steps": [{  # Make this a list with one dict inside
        "engine": "python",
        "code": """
try:
    import fitz  # PyMuPDF

    shared_context = {}
    shared_context['inputs'] = '{{request}}'

    pdf_path = '{{file}}'



    # Open the PDF
    doc = fitz.open(pdf_path)
    text = ""

    # Extract text from each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        text += page.get_text()

    # Close the document
    doc.close()

    print(f"Extracted text length: {len(text)}")
    if len(text) > 100:
        print(f"First 100 characters: {text[:100]}...")

    shared_context['extracted_text'] = text
    print("Text extraction completed successfully")

except Exception as e:
    error_msg = f"Error processing PDF: {str(e)}"
    print(error_msg)
    shared_context['extracted_text'] = f"Error: {error_msg}"
"""
    },
     {
        "engine": "natural",
        "code": """
{% if shared_context and shared_context.extracted_text %}
{% if shared_context.extracted_text.startswith('Error:') %}
{{ shared_context.extracted_text }}
{% else %}
Here is the text extracted from the PDF:

{{ shared_context.extracted_text }}

Please provide a response to user request: {{ request }} using the information extracted above.
{% endif %}
{% else %}
Error: No text was extracted from the PDF.
{% endif %}
"""
    },]
    }

# Instantiate the tool
tool = Tool(tool_data)

# Create an NPC instance
npc = NPC(
    name='starlana',
    primary_directive='Analyze text from Astrophysics papers with a keen attention to theoretical machinations and mechanisms.',
    model = 'llama3.2',
    provider='ollama',
    db_conn=sqlite3.connect(os.path.expanduser('~/npcsh_database.db'))
)

# Define input values dictionary
input_values = {
    "request": "what is the point of the yuan and narayanan work?",
    "file": os.path.abspath("test_data/yuan2004.pdf")
}

print(f"Attempting to read file: {input_values['file']}")
print(f"File exists: {os.path.exists(input_values['file'])}")

# Execute the tool
output = tool.execute(input_values, npc.tools_dict, jinja_env, 'Sample Command',model=npc.model, provider=npc.provider,  npc=npc)

print('Tool Output:', output)
```

### Example 5: Orchestrating a team



```python
import pandas as pd
import numpy as np
import os
from npcsh.npc_compiler import NPC, NPCTeam, Tool


# Create test data and save to CSV
def create_test_data(filepath="sales_data.csv"):
    sales_data = pd.DataFrame(
        {
            "date": pd.date_range(start="2024-01-01", periods=90),
            "revenue": np.random.normal(10000, 2000, 90),
            "customer_count": np.random.poisson(100, 90),
            "avg_ticket": np.random.normal(100, 20, 90),
            "region": np.random.choice(["North", "South", "East", "West"], 90),
            "channel": np.random.choice(["Online", "Store", "Mobile"], 90),
        }
    )

    # Add patterns to make data more realistic
    sales_data["revenue"] *= 1 + 0.3 * np.sin(
        np.pi * np.arange(90) / 30
    )  # Seasonal pattern
    sales_data.loc[sales_data["channel"] == "Mobile", "revenue"] *= 1.1  # Mobile growth
    sales_data.loc[
        sales_data["channel"] == "Online", "customer_count"
    ] *= 1.2  # Online customer growth

    sales_data.to_csv(filepath, index=False)
    return filepath, sales_data


code_execution_tool = Tool(
    {
        "tool_name": "execute_code",
        "description": """Executes a Python code block with access to pandas,
                          numpy, and matplotlib.
                          Results should be stored in the 'results' dict to be returned.
                          The only input should be a single code block with \n characters included.
                          The code block must use only the  libraries or methods contained withen the
                            pandas, numpy, and matplotlib libraries or using builtin methods.
                          do not include any json formatting or markdown formatting.

                          When generating your script, the final output must be encoded in a variable
                          named "output". e.g.

                          output  = some_analysis_function(inputs, derived_data_from_inputs)
                            Adapt accordingly based on the scope of the analysis

                          """,
        "inputs": ["script"],
        "steps": [
            {
                "engine": "python",
                "code": """{{script}}""",
            }
        ],
    }
)

# Analytics team definition
analytics_team = [
    {
        "name": "analyst",
        "primary_directive": "You analyze sales performance data, focusing on revenue trends, customer behavior metrics, and market indicators. Your expertise is in extracting actionable insights from complex datasets.",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "tools": [code_execution_tool],  # Only the code execution tool
    },
    {
        "name": "researcher",
        "primary_directive": "You specialize in causal analysis and experimental design. Given data insights, you determine what factors drive observed patterns and design tests to validate hypotheses.",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "tools": [code_execution_tool],  # Only the code execution tool
    },
    {
        "name": "engineer",
        "primary_directive": "You implement data pipelines and optimize data processing. When given analysis requirements, you create efficient workflows to automate insights generation.",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "tools": [code_execution_tool],  # Only the code execution tool
    },
]


def create_analytics_team():
    # Initialize NPCs with just the code execution tool
    npcs = []
    for npc_data in analytics_team:
        npc = NPC(
            name=npc_data["name"],
            primary_directive=npc_data["primary_directive"],
            model=npc_data["model"],
            provider=npc_data["provider"],
            tools=[code_execution_tool],  # Only code execution tool
        )
        npcs.append(npc)

    # Create coordinator with just code execution tool
    coordinator = NPC(
        name="coordinator",
        primary_directive="You coordinate the analytics team, ensuring each specialist contributes their expertise effectively. You synthesize insights and manage the workflow.",
        model="gpt-4o-mini",
        provider="openai",
        tools=[code_execution_tool],  # Only code execution tool
    )

    # Create team
    team = NPCTeam(npcs=npcs, foreman=coordinator)
    return team


def main():
    # Create and save test data
    data_path, sales_data = create_test_data()

    # Initialize team
    team = create_analytics_team()

    # Run analysis - updated prompt to reflect code execution approach
    results = team.orchestrate(
        f"""
    Analyze the sales data at {data_path} to:
    1. Identify key performance drivers
    2. Determine if mobile channel growth is significant
    3. Recommend tests to validate growth hypotheses

    Here is a header for the data file at {data_path}:
    {sales_data.head()}

    When working with dates, ensure that date columns are converted from raw strings. e.g. use the pd.to_datetime function.


    When working with potentially messy data, handle null values by using nan versions of numpy functions or
    by filtering them with a mask .

    Use Python code execution to perform the analysis - load the data and perform statistical analysis directly.
    """
    )

    print(results)

    # Cleanup
    os.remove(data_path)


if __name__ == "__main__":
    main()

```



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



## Contributing
Contributions are welcome! Please submit issues and pull requests on the GitHub repository.

## Support
If you appreciate the work here, [consider supporting NPC Worldwide](https://buymeacoffee.com/npcworldwide). If you'd like to explore how to use `npcsh` to help your business, please reach out to info@npcworldwi.de .


## NPC Studio
Coming soon! NPC Studio will be a desktop application for managing chats and agents on your own machine.
Be sure to sign up for the [npcsh newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A) to hear updates!

## License
This project is licensed under the MIT License.
