

## TLDR Cheat Sheet
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



### Example 1: Creating and Using an NPC
This example shows how to create and initialize an NPC and use it to answer a question.
```bash
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


npc = NPC(
          name='Felix',
          db_conn=conn,
          primary_directive='Analyze customer feedback for sentiment.',
          model='gpt-4o-mini',
          provider='openai',
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

### Example 4: Orchestrating a team



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


