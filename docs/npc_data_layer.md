

# NPC Data Layer

What principally powers the capabilities of npcsh is the NPC Data Layer. In the `~/.npcsh/` directory after installation, you will find
the npc teaam with its tools, models, contexts, assembly lines, and NPCs. By making tools, NPCs, contexts, and assembly lines simple data structures with
a fixed set of parameters, we can let users define them in easy-to-read YAML files, allowing for a modular and extensible system that can be easily modified and expanded upon. Furthermore, this data layer relies heavily on jinja templating to allow for dynamic content generation and the ability to reference other NPCs, tools, and assembly lines in the system.

## Creating NPCs
NPCs are defined in YAML files within the npc_team directory. Each NPC must have a name and a primary directive. Optionally, one can specify an LLM model/provider for the NPC as well as provide an explicit list of tools and whether or not to use the globally available tools. See the data models contained in `npcsh/data_models.py` for more explicit type details on the NPC data structure.



Here is a typical NPC file:
```yaml
name: sibiji
primary_directive: You are a foundational AI assistant. Your role is to provide basic support and information. Respond to queries concisely and accurately.
tools:
  - simple data retrieval
model: llama3.2
provider: ollama
```


## Creating Tools
Tools are defined as YAMLs with `.tool` extension within the npc_team/tools directory. Each tool has a name, inputs, and consists of three distinct steps: preprocess, prompt, and postprocess. The idea here is that a tool consists of a stage where information is preprocessed and then passed to a prompt for some kind of analysis and then can be passed to another stage for postprocessing. In each of these three cases, the engine must be specified. The engine can be either "natural" for natural language processing or "python" for Python code. The code is the actual code that will be executed.

Here is an example of a tool file:
```yaml
tool_name: "screen_capture_analysis_tool"
description: Captures the whole screen and sends the image for analysis
inputs:
  - "prompt"
steps:
  - engine: "python"
    code: |
      # Capture the screen
      import pyautogui
      import datetime
      import os
      from PIL import Image
      import time
      from npcsh.image import analyze_image_base, capture_screenshot

      out = capture_screenshot(npc = npc, full = True)

      llm_response = analyze_image_base( '{{prompt}}' + "\n\nAttached is a screenshot of my screen currently. Please use this to evaluate the situation. If the user asked for you to explain what's on their screen or something similar, they are referring to the details contained within the attached image. You do not need to actually view their screen. You do not need to mention that you cannot view or interpret images directly. You only need to answer the user's request based on the attached screenshot!",
                                        out['file_path'],
                                        out['filename'],
                                        npc=npc,
                                        **out['model_kwargs'])
      # To this:
      if isinstance(llm_response, dict):
          llm_response = llm_response.get('response', 'No response from image analysis')
      else:
          llm_response = 'No response from image analysis'

```


When you have created a tool, it will be surfaced as a potential option to be used when you ask a question in the base npcsh shell. The LLM will decide if it is the best tool to use based on the user's input. Alternatively, if you'd like, you can call the tools directly, without needing to let the AI decide if it's the right one to use.

  ```npcsh
  npcsh> /screen_cap_tool <prompt>
  ```
  or
  ```npcsh
  npcsh> /sql_executor select * from conversation_history limit 1

  ```
  or
  ```npcsh
  npcsh> /calculator 5+6
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


## npcsql: SQL Integration and pipelines (UNDER CONSTRUCTION)


In addition to NPCs being used in `npcsh` and through the python package, users may wish to take advantage of agentic interactions in SQL-like pipelines.
`npcsh` contains a pseudo-SQL interpreter that processes SQL models which lets users write queries containing LLM-function calls that reference specific NPCs. `npcsh` interprets these queries, renders any jinja template references through its python implementation, and then executes them accordingly.

Here is an example of a SQL-like query that uses NPCs to analyze data:
```sql
SELECT debate(['logician','magician'], 'Analyze the sentiment of the customer feedback.') AS sentiment_analysis
```

### squish
squish is an aggregate NPC LLM function that compresses information contained in whole columns or grouped chunks of data based on the SQL aggregation.

### splat
Splat is a row-wise NPC LLM function that allows for the application of an LLM function on each row of a dataset or a re-sampling




