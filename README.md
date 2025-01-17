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


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=cagostino/npcsh&type=Date)](https://star-history.com/#cagostino/npcsh&Date)
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
`npcsh` also comes with a set of tools and NPCs that are used in processing. It will generate a folder at ~/.npcsh/ that contains the tools and NPCs that are used in the shell and these will be used in the absence of other project-specific ones. Additionally, `npcsh` records interactions and compiled information about npcs within a local SQLite database at the path specified in the .npcshrc file. This will default to ~/npcsh_history.db if not specified. When the data mode is used to load or analyze data in CSVs or PDFs, these data will be stored in the same database for future reference.

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
In the `npcsh` shell, users can ask LLMs questions, have LLMLs execute commands or use tools, or utilize macros that provide additional functionality. When a user does not invoke a specific macro, the shell will automatically decide which tool to use based on the user's input. Here are some examples of things one might ask the npcsh shell.

Here are some examples of how you can use npcsh
```npcsh
npcsh>Who was Simon Bolivar?

Simón Bolívar, often referred to as "El Libertador," was a Venezuelan military and political leader who played a key role in the Latin American wars of independence against Spanish
colonial rule in the early 19th century. He was born on July 24, 1783, in Caracas, Venezuela, into a wealthy Creole family.
Bolívar's vision extended beyond merely liberating territories; he aspired to unify the newly independent nations of South America into a single federation, which he called "Gran
Colombia," consisting of present-day Colombia, Venezuela, Ecuador, and Panama. He was known for his exceptional military strategies and leadership, which led to successful campaigns in
various regions, including the battles of Boyacá, Carabobo, and Ayacucho.
He faced numerous challenges, including political fragmentation, regional rivalries, and personal betrayals. Despite his initial successes, the unity he sought for Latin America proved
difficult to achieve, and Gran Colombia eventually disintegrated in the early 1830s.
Bolívar's influence and legacy extend far beyond his lifetime. He is celebrated in various countries across Latin America as a national hero and symbol of independence. His thoughts on
liberty, governance, and social issues continue to inspire movements for democracy and social justice in the region. Simón Bolívar died on December 17, 1830, but remains a pivotal figure
in the history of Latin America.
```


```npcsh
npcsh> What is the capital of France?
The capital of France is Paris. It is the largest city in the country and is known for its rich history, art, culture, and architecture, including famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum.
```


```npcsh
npcsh> What's the weather in Tokyo?

handle_tool_call invoked with tool_name: generic_search_tool

The weather in Tokyo, Japan, is expected to be mixed with sun and clouds. Here are some details from the recent forecasts:

 • Highs: Around 53°F to 58°F with a few variations depending on the day.

 • Lows: Approximately 35°F to 40°F.

 • Winds: Generally from the northwest at 5 to 10 mph.

 • Condition: Mainly sunny, but there may be periods of clouds and some overcast conditions throughout the week.
For more detailed information, you can refer to sources like The Weather Channel or AccuWeather.
/home/caug/npcww/npcsh:npcsh>

```
In the below example, the code that was open was the screen capture analysis tool itself.
```npcsh
npcsh>  Can you explain what the code does in the currently open VS code window?

handle_tool_call invoked with tool_name: screen_capture_analysis_tool

Screenshot saved as screenshot_20241223_225815.png

The code in the visible section of your VS Code window appears to be a script for capturing and analyzing screenshots. Here's a breakdown of what the code does:

 1 Import Necessary Libraries: It imports required libraries like system, datetime, and pyautogui, which are essential for capturing screenshots and handling date-time operations.

 2 Capture the Screen: The code captures the current screen using pyautogui.screenshot(), taking a screenshot of the entire screen.

 3 File Management: It generates a unique filename for the screenshot using the current date and time. This ensures that each screenshot file is distinct.

 4 Save the Screenshot: The captured screenshot is saved to a file with the generated filename.

 5 Image Analysis: It likely imports a functionality (analyze_image_base) to analyze the screenshot for specific content or information.

 6 Run the Analysis: While not explicitly shown, it appears to call this analysis function to process the saved screenshot.
In summary, this code automates the process of capturing a screenshot, saving it with a unique filename, and analyzing that image for specific content or patterns.


```

```npcsh
npcsh> What is the biggest file in my current folder?

LLM suggests the following bash command: ls -S | head -n 1

Running command: ls -S | head -n 1

Command executed with output: image_20241111_000033.png

I ran the command ls -S | head -n 1 in your current folder. This command sorts all files by size in descending order and then selects the first entry, which represents the largest file. The result of this operation shows that the biggest file in your current folder is image_20241111_000033.png.

```

```npcsh
npcsh>What is the best way to implement a linked list in Python?

The best way to implement a linked list in Python is to define a Node class for the individual elements and a LinkedList class to manage the nodes. Here's a basic implementation:
class Node:
    def __init__(self, data):
        self.data = data  # Store the data
        self.next = None  # Initialize the next node as None

class LinkedList:
    def __init__(self):
        self.head = None  # Initialize the head of the list

    def insert(self, data):
        """Inserts a new node at the end of the linked list."""
        new_node = Node(data)
        if not self.head:  # If list is empty, set new node as head
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:  # Traverse to the last node
            last_node = last_node.next
        last_node.next = new_node  # Link the new node

    def display(self):
        """Prints the linked list."""
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

# Example usage:
llist = LinkedList()
llist.insert(1)
llist.insert(2)
llist.insert(3)
llist.display()  # Output: 1 -> 2 -> 3 -> None
                                                                                            Explanation:

 1 Node Class: Represents each node in the linked list. It has two attributes: data (to store the value) and next (to point to the next node).

 2 LinkedList Class: Manages the linked list. It has a head attribute pointing to the first node.

 3 Insert Method: Adds a new node to the end of the list.

 4 Display Method: Prints the elements of the list in order.
This implementation is simple and can be extended with more methods (e.g., deletion, searching, etc.) as needed.

```

```npcsh
npcsh> In which of the files in the current directory is "Simon Bolivar" mentioned?

The user's query was about "Simon Bolivar." Based on the relevant context from the codebase, I found specific mentions and relevant information regarding Simon Bolívar in the documentation provided
in the README.md file.
From the relevant context, specifically in the section of README.md, it shared a code example that demonstrates how to load an NPC (Non-Player Character) named "Simon Bolivar" using SQLite and
answer a question about project updates. This code snippet indicates that "Simon Bolivar" is represented in the NPC system, allowing for interaction through questions to retrieve information.
Additionally, there is another passage that directly answers the query. It reads:

▌ "Simón Bolívar, often referred to as 'El Libertador,' was a Venezuelan military and political leader who played a key role in the Latin American wars of independence against Spanish colonial
▌ rule in the early 19th century."
This information highlights Bolívar's significant historical role and achievements, answering the query regarding who Simon Bolivar was, thus making it clear that he was a prominent leader in Latin
America's fight for independence.
To summarize, I arrived at this answer by examining the provided context in the README.md file that includes both the coding example that initializes an NPC named "Simon Bolivar" and a historical
overview of his contributions, which directly addresses the user's inquiry.


```


```npcsh
npcsh>what process is using the most ram on my computer right now?

LLM suggests the following bash command: ps aux --sort=-%mem | head -n 10
Running command: ps aux --sort=-%mem | head -n 10
Command executed with output: USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
caug     1040046  1.1  2.7 1226193464 1598980 ?  SNl  Dec22  23:30 /opt/vivaldi/vivaldi-bin --type=renderer --crashpad-handler-pid=4529 --enable-crash-reporter=,stable --change-stack-guard-on-fork=
caug      683148  0.0  1.9 2047464 1137232 ?     SNl  Dec13  10:11 io.elementary.appcenter
ollama      1884  0.0  1.8 3683920 1051020 ?     SNsl Dec02   0:30 /usr/local/bin/ollama serve
caug     1191836  5.3  1.6 7346052 964988 pts/17 Sl+  22:57   0:22 /home/caug/.pyenv/versions/npcsh/bin/python /home/caug/.pyenv/versions/3.11.0/envs/npcsh/bin/npcsh
caug     1145749  0.0  1.5 7068128 892500 pts/13 Sl+  Dec22   0:15 /home/caug/.pyenv/versions/3.11.0/bin/python3.11 /home/caug/.pyenv/versions/3.11.0/bin/npcsh
caug      493912  0.0  1.5 7050968 890012 pts/11 Sl+  Dec10   0:14 /home/caug/.pyenv/versions/npcsh/bin/python /home/caug/.pyenv/versions/npcsh/bin/npcsh
caug        4522  0.9  1.5 34391236 874756 ?     SNLl Dec02 295:06 /opt/vivaldi/vivaldi-bin
caug        3995  0.7  1.1 6051812 683916 ?      R<sl Dec02 221:19 /usr/bin/gnome-shell
caug      501721  0.3  1.0 1212584660 626248 ?   SNl  Dec10  60:22 /opt/vivaldi/vivaldi-bin --type=renderer --crashpad-handler-pid=4529 --enable-crash-reporter=,stable --change-stack-guard-on-fork=

To determine which processes are using the most RAM on your computer, I executed the command ps aux --sort=-%mem | head -n 10.
This command lists all running processes (ps aux), sorts them by memory usage in descending order (--sort=-%mem), and then retrieves the top 10 entries (head -n 10).
The output shows the processes currently consuming the most memory. In the results, the process using the most RAM is Vivaldi's renderer, followed by other applications such as the Elementary App Center, Ollama server, and several instances of Python related to your NPC shell.
```

```npcsh
npcsh> can you generate an image of a calico cat and an orange tabby cat in the style of a japanese woodblock print where the two cats are engaged in a samurai battle but instead of samurai swords they are holding cat toys that are like wands with felt birds on the ends of them on strings

handle_tool_call invoked with tool_name: image_generation_tool
```
![catfight](test_data/catfight.PNG)


In addition to its ability to execute one-off tasks or tool calls, npcsh also has
the ability to generate tool call sequences when asked:
```npcsh

npcsh> can you use a tool sequence to find the gdp of russia in 2024 and then to use that information to generate an image?

handle_tool_call invoked with tool_name: generic_search_tool
Tool found: generic_search_tool
handle_tool_call invoked with tool_name: image_generation_tool
Tool found: image_generation_tool

The Gross Domestic Product (GDP) of Russia is estimated to be $8.311 trillion by the end of 2024, according to World Economics.
This figure is significantly larger than the official estimate of $5.804 trillion published by the World Bank for the end of 2023.

It seems that you've generated an image. If you have any questions or need assistance related to this image, please let me know how I can help!
```
and then the associated image :
![gdp](test_data/r8ss9a.PNG)



## Macros

While npcsh can decide the best tool to use based on the user's input, the user can also specify certain tools to use with a macro. Macros are commands that start with a forward slash (/) and are followed (in some cases) by the relevant arguments for those macros.
To learn about them from within the shell, type:
```npcsh
npcsh> /help
```
To exit the shell:
```npcsh
npcsh> /exit
```

Otherwise, here are some more detailed examples of macros that can be used in npcsh:

### Spool
Spool mode allows one to enter into a conversation with a specific LLM or a specific NPC. This is used for having distinct interactions from those in the base shell and these will be separately contained .
Start the spool mode:
```npcsh
npcsh> /spool
```
Start the spool mode with a specific npc

```npcsh
npcsh> /spool npc=foreman
```

Start the spool mode with specific files in context that will be referenced through rag searches when relevant.

```npcsh
npcsh> /spool files=[*.py,*.md] # Load specific files for context
```


Start the spool with a specific llm model:
```npcsh
#note this is not yet implemented
npcsh> /spool model=llama3.3
```

### Over-the-shoulder: Screenshots and image analysis

Use the /ots macro to take a screenshot and write a prompt for an LLM to answer about the screenshot.
```npcsh
npcsh> /ots

Screenshot saved to: /home/caug/.npcsh/screenshots/screenshot_1735015011.png

Enter a prompt for the LLM about this image (or press Enter to skip): describe whats in this image

The image displays a source control graph, likely from a version control system like Git. It features a series of commits represented by colored dots connected by lines, illustrating the project's development history. Each commit message provides a brief description of the changes made, including tasks like fixing issues, merging pull requests, updating README files, and adjusting code or documentation. Notably, several commits mention specific users, particularly "Chris Agostino," indicating collaboration and contributions to the project. The graph visually represents the branching and merging of code changes.
```

Alternatively, pass an existing image in like :
```npcsh
npcsh> /ots test_data/catfight.PNG
Enter a prompt for the LLM about this image (or press Enter to skip): whats in this ?

The image features two cats, one calico and one orange tabby, playing with traditional Japanese-style toys. They are each holding sticks attached to colorful pom-pom balls, which resemble birds. The background includes stylized waves and a red sun, accentuating a vibrant, artistic style reminiscent of classic Japanese art. The playful interaction between the cats evokes a lively, whimsical scene.
```




### Vixynt: Image Generation
Image generation can be done with the /vixynt macro.
Use /vixynt like so where you can also specify the model to use with an @ reference. This @ reference will override the default model in ~/.npcshrc.

```npcsh
npcsh> /vixynt A futuristic cityscape @dall-e-3
```
![futuristic cityscape](test_data/futuristic_cityscape.PNG)

```npcsh
npcsh> /vixynt A peaceful landscape @runwayml/stable-diffusion-v1-5
```
![peaceful landscape](test_data/peaceful_landscape_stable_diff.png)



### Voice Control
Enter into a voice-controlled mode to interact with the LLM. This mode can executet commands and use tools just like the basic npcsh shell.
```npcsh
npcsh> /whisper
```


### Executing Bash Commands
You can execute bash commands directly within npcsh. The LLM can also generate and execute bash commands based on your natural language requests.
For example:
```npcsh
npcsh> ls -l

npcsh> cp file1.txt file2.txt
npcsh> mv file1.txt file2.txt
npcsh> mkdir new_directory
npcsh> git status
npcsh> vim file.txt

```


### Compilation and NPC Interaction
Compile a specified NPC profile. This will make it available for use in npcsh interactions.
```npcsh
npcsh> /compile <npc_file>
```
You can also use `/com` as an alias for `/compile`. If no NPC file is specified, all NPCs in the npc_team directory will be compiled.

Begin a conversations with a specified NPC by referencing their name
```npcsh
npcsh> /<npc_name>:
```

### Data Interaction and analysis
Enter into data mode to load, manipulate, and query data from various file formats.
```npcsh
npcsh> /data
data> load data.csv as df
data> df.describe()
```

### Notes
Jot down notes and store them within the npcsh database and in the current directory as a text file.
```npcsh
npcsh> /notes
```

### Changing defaults from within npcsh
Users can change the default model and provider from within npcsh by using the following commands:
```npcsh
npcsh> /set model ollama
npcsh> /set provider llama3.2
```


## Creating NPCs
NPCs are defined in YAML files within the npc_team directory. Each NPC has a name, primary directive, and optionally, a list of tools. See the examples in the npc_profiles directory for guidance.


Here is a typical NPC file:
```yaml
name: sibiji
primary_directive: You are a foundational AI assistant. Your role is to provide basic support and information. Respond to queries concisely and accurately.
suggested_tools_to_use:
  - simple data retrieval
model: llama3.2
provider: ollama
```


## Creating Tools
Tools are defined as YAMLs with `.tool` extension within the npc_team/tools directory. Each tool has a name, inputs, and consists of three distinct steps: preprocess, prompt, and postprocess. The idea here is that a tool consists of a stage where information is preprocessed and then passed to a prompt for some kind of analysis and then can be passed to another stage for postprocessing. In each of these three cases, the engine must be specified. The engine can be either "natural" for natural language processing or "python" for Python code. The code is the actual code that will be executed.

Here is an example of a tool file:
```yaml
tool_name: "screen_capture_analysis_tool"
inputs:
  - "prompt"
preprocess:
  - engine: "python"
    code: |
      # Capture the screen
      import pyautogui
      import datetime
      import os
      from PIL import Image
      from npcsh.image import analyze_image_base

      # Generate filename
      filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
      screenshot = pyautogui.screenshot()
      screenshot.save(filename)
      print(f"Screenshot saved as {filename}")

      # Load image
      image = Image.open(filename)

      # Full file path
      file_path = os.path.abspath('./'+filename)
      # Analyze the image

      llm_output = analyze_image_base(inputs['prompt']+ '\n\n attached is a screenshot of my screen currently.', file_path, filename, npc=npc)
prompt:
  engine: "natural"
  code: ""
postprocess:
  - engine: "natural"
    code: |
      Screenshot captured and saved as {{ filename }}.
      Analysis Result: {{ llm_output }}
```


## NPC Pipelines



Let's say you want to create a pipeline of steps where NPCs are used along the way. Let's initialize with a pipeline file we'll call `morning_routine.pipe`:
```yaml
steps:
  - step_name: "review_email"
    npc: "{{ ref('email_assistant') }}"
    task: "Get me up to speed on my recent emails: {{source('emails')}}."


  - step_name: "market_update"
    npc: "{{ ref('market_analyst') }}"
    task: "Give me an update on the latest events in the market: {{source('market_events')}}."

  - step_name: "summarize"
    npc: "{{ ref('sibiji') }}"
    model: llama3.2
    provider: ollama
    task: "Review the outputs from the {{review_email}} and {{market_update}} and provide me with a summary."

```
Now youll see that we reference NPCs in the pipeline file. We'll need to make sure we have each of those NPCs available.
Here is an example for the email assistant:
```yaml
name: email_assistant
primary_directive: You are an AI assistant specialized in managing and summarizing emails. You should present the information in a clear and concise manner.
model: gpt-4o-mini
provider: openai
```
Now for the marketing analyst:
```yaml
name: market_analyst
primary_directive: You are an AI assistant focused on monitoring and analyzing market trends. Provide de
model: llama3.2
provider: ollama
```
and then here is our trusty friend sibiji:
```yaml
name: sibiji
primary_directive: You are a foundational AI assistant. Your role is to provide basic support and information. Respond to queries concisely and accurately.
suggested_tools_to_use:
  - simple data retrieval
model: claude-3-5-sonnet-latest
provider: anthropic
```
Now that we have our pipeline and NPCs defined, we also need to ensure that the source data we are referencing will be there. When we use source('market_events') and source('emails') we are asking npcsh to pull those data directly from tables in our npcsh database. For simplicity we will just make these in python to insert them for this demo:
```python
import pandas as pd
from sqlalchemy import create_engine
import os

# Sample market events data
market_events_data = {
    "datetime": [
        "2023-10-15 09:00:00",
        "2023-10-16 10:30:00",
        "2023-10-17 11:45:00",
        "2023-10-18 13:15:00",
        "2023-10-19 14:30:00",
    ],
    "headline": [
        "Stock Market Rallies Amid Positive Economic Data",
        "Tech Giant Announces New Product Line",
        "Federal Reserve Hints at Interest Rate Pause",
        "Oil Prices Surge Following Supply Concerns",
        "Retail Sector Reports Record Q3 Earnings",
    ],
}

# Create a DataFrame
market_events_df = pd.DataFrame(market_events_data)

# Define database path relative to user's home directory
db_path = os.path.expanduser("~/npcsh_history.db")

# Create a connection to the SQLite database
engine = create_engine(f"sqlite:///{db_path}")
with engine.connect() as connection:
    # Write the data to a new table 'market_events', replacing existing data
    market_events_df.to_sql(
        "market_events", con=connection, if_exists="replace", index=False
    )

print("Market events have been added to the database.")

email_data = {
    "datetime": [
        "2023-10-10 10:00:00",
        "2023-10-11 11:00:00",
        "2023-10-12 12:00:00",
        "2023-10-13 13:00:00",
        "2023-10-14 14:00:00",
    ],
    "subject": [
        "Meeting Reminder",
        "Project Update",
        "Invoice Attached",
        "Weekly Report",
        "Holiday Notice",
    ],
    "sender": [
        "alice@example.com",
        "bob@example.com",
        "carol@example.com",
        "dave@example.com",
        "eve@example.com",
    ],
    "recipient": [
        "bob@example.com",
        "carol@example.com",
        "dave@example.com",
        "eve@example.com",
        "alice@example.com",
    ],
    "body": [
        "Don't forget the meeting tomorrow at 10 AM.",
        "The project is progressing well, see attached update.",
        "Please find your invoice attached.",
        "Here is the weekly report.",
        "The office will be closed on holidays, have a great time!",
    ],
}

# Create a DataFrame
emails_df = pd.DataFrame(email_data)

# Define database path relative to user's home directory
db_path = os.path.expanduser("~/npcsh_history.db")

# Create a connection to the SQLite database
engine = create_engine(f"sqlite:///{db_path}")
with engine.connect() as connection:
    # Write the data to a new table 'emails', replacing existing data
    emails_df.to_sql("emails", con=connection, if_exists="replace", index=False)

print("Sample emails have been added to the database.")

```


With these data now in place, we can proceed with running the pipeline. We can do this in npcsh by using the /compile command.




```npcsh
npcsh> /compile morning_routine.pipe
```



Alternatively we can run a pipeline like so in Python:

```bash
from npcsh.npc_compiler import PipelineRunner
import os

pipeline_runner = PipelineRunner(
    pipeline_file="morning_routine.pipe",
    npc_root_dir=os.path.abspath("./"),
    db_path="~/npcsh_history.db",
)
pipeline_runner.execute_pipeline(inputs)
```

What if you wanted to run operations on each row and some operations on all the data at once? We can do this with the pipelines as well. Here we will build a pipeline for news article analysis.
First we make the data for the pipeline that well use:
```python
import pandas as pd
from sqlalchemy import create_engine
import os

# Sample data generation for news articles
news_articles_data = {
    "news_article_id": list(range(1, 21)),
    "headline": [
        "Economy sees unexpected growth in Q4",
        "New tech gadget takes the world by storm",
        "Political debate heats up over new policy",
        "Health concerns rise amid new disease outbreak",
        "Sports team secures victory in last minute",
        "New economic policy introduced by government",
        "Breakthrough in AI technology announced",
        "Political leader delivers speech on reforms",
        "Healthcare systems pushed to limits",
        "Celebrated athlete breaks world record",
        "Controversial economic measures spark debate",
        "Innovative tech startup gains traction",
        "Political scandal shakes administration",
        "Healthcare workers protest for better pay",
        "Major sports event postponed due to weather",
        "Trade tensions impact global economy",
        "Tech company accused of data breach",
        "Election results lead to political upheaval",
        "Vaccine developments offer hope amid pandemic",
        "Sports league announces return to action",
    ],
    "content": ["Article content here..." for _ in range(20)],
    "publication_date": pd.date_range(start="1/1/2023", periods=20, freq="D"),
}
```

Then we will create the pipeline file:
```yaml
# news_analysis.pipe
steps:
  - step_name: "classify_news"
    npc: "{{ ref('news_assistant') }}"
    task: |
      Classify the following news articles into one of the categories:
      ["Politics", "Economy", "Technology", "Sports", "Health"].
      {{ source('news_articles') }}

  - step_name: "analyze_news"
    npc: "{{ ref('news_assistant') }}"
    batch_mode: true  # Process articles with knowledge of their tags
    task: |
      Based on the category assigned in {{classify_news}}, provide an in-depth
      analysis and perspectives on the article. Consider these aspects:
      ["Impacts", "Market Reaction", "Cultural Significance", "Predictions"].
      {{ source('news_articles') }}
```

Then we can run the pipeline like so:
```bash
/compile ./npc_team/news_analysis.pipe
```
or in python like:

```bash

from npcsh.npc_compiler import PipelineRunner
import os
runner = PipelineRunner(
    "./news_analysis.pipe",
    db_path=os.path.expanduser("~/npcsh_history.db"),
    npc_root_dir=os.path.abspath("."),
)
results = runner.execute_pipeline()
```

Alternatively, if youd like to use a mixture of agents in your pipeline, set one up like this:
```yaml
steps:
  - step_name: "classify_news"
    npc: "news_assistant"
    mixa: true
    mixa_agents:
      - "{{ ref('news_assistant') }}"
      - "{{ ref('journalist_npc') }}"
      - "{{ ref('data_scientist_npc') }}"
    mixa_voters:
      - "{{ ref('critic_npc') }}"
      - "{{ ref('editor_npc') }}"
      - "{{ ref('researcher_npc') }}"
    mixa_voter_count: 5
    mixa_turns: 3
    mixa_strategy: "vote"
    task: |
      Classify the following news articles...
      {{ source('news_articles') }}
```
You'll have to make npcs for these references to work, here are versions that should work with the above:
```yaml
name: news_assistant
```
Then, we can run the mixture of agents method like:

```bash
/compile ./npc_team/news_analysis_mixa.pipe
```
or in python like:

```bash

from npcsh.npc_compiler import PipelineRunner
import os

runner = PipelineRunner(
    "./news_analysis_mixa.pipe",
    db_path=os.path.expanduser("~/npcsh_history.db"),
    npc_root_dir=os.path.abspath("."),
)
results = runner.execute_pipeline()
```



Note, in the future we will aim to separate compilation and running so that we will have a compilation step that is more like a jinja rendering of the relevant information so that it can be more easily audited.


## Python Examples
Integrate npcsh into your Python projects for additional flexibility. Below are a few examples of how to use the library programmatically.



### Example 1: Creating and Using an NPC
This example shows how to create and initialize an NPC and use it to answer a question.
```bash
import sqlite3
from npcsh.npc_compiler import NPC

# Set up database connection
db_path = '~/npcsh_history.db'
conn = sqlite3.connect(db_path)

# Load NPC from a file
npc = NPC(db_conn=conn,
          name='Simon Bolivar',
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
### Example 2: Using an NPC to Analyze Data
This example shows how to use an NPC to perform data analysis on a DataFrame using LLM commands.
```bash
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


npc = NPC(db_conn=conn,
          name='Felix',
          primary_directive='Analyze customer feedback for sentiment.',
          model='llama3.2',
          provider='ollama',
          )
response = npc.analyze_db_data('Provide a detailed report on the data contained in the `customer_feedback` table?')


```


### Example 3: Creating and Using a Tool
You can define a tool and execute it from within your Python script.
Here we'll create a tool that will take in a pdf file, extract the text, and then answer a user request about the text.

```bash
from npcsh.npc_compiler import Tool, NPC
import sqlite3
import os

tool_data = {
    "tool_name": "pdf_analyzer",
    "inputs": ["request", "file"],
    "preprocess": [{  # Make this a list with one dict inside
        "engine": "python",
        "code": """
try:
    import fitz  # PyMuPDF

    shared_context = {}
    shared_context['inputs'] = inputs

    pdf_path = inputs['file']
    print(f"Processing PDF file: {pdf_path}")

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
    }],
    "prompt": {
        "engine": "natural",
        "code": """
{% if shared_context and shared_context.extracted_text %}
{% if shared_context.extracted_text.startswith('Error:') %}
{{ shared_context.extracted_text }}
{% else %}
Here is the text extracted from the PDF:

{{ shared_context.extracted_text }}

Please provide a response to user request: {{ inputs.request }} using the information extracted above.
{% endif %}
{% else %}
Error: No text was extracted from the PDF.
{% endif %}
"""
    },
    "postprocess": []  # Empty list instead of empty dict
}

# Instantiate the tool
tool = Tool(tool_data)

# Create an NPC instance
npc = NPC(
    name='starlana',
    primary_directive='Analyze text from Astrophysics papers with a keen attention to theoretical machinations and mechanisms.',
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
output = tool.execute(input_values, npc.tools_dict, None, 'Sample Command', npc)

print('Tool Output:', output)
```

## npcsql: SQL Integration and pipelines (UNDER CONSTRUCTION)

- to be filled in soon


## Contributing
Contributions are welcome! Please submit issues and pull requests on the GitHub repository.

## License
This project is licensed under the MIT License.
