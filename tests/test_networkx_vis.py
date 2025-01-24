from npcsh.knowledge_graph import *

db_path = "./demo.db"  # Specify your database path here
path = "~/npcww/npcsh/tests/"

# First create some test data
conn = init_db(db_path, drop=True)  # Start fresh

# Create test groups
groups = ["Programming Tools", "AI Features", "Shell Integration", "Development"]

for group in groups:
    create_group(conn, group)

# Insert test facts
facts = [
    "npcsh is a Python-based command-line tool",
    "It integrates LLMs into daily workflow",
    "Users can execute bash commands directly",
    "Supports multiple AI models including GPT-4",
    "Provides voice control through /whisper command",
    "Can be extended with custom Python tools",
]

for fact in facts:
    insert_fact(conn, fact, path)

# Create relationships
relationships = [
    ("Programming Tools", "npcsh is a Python-based command-line tool"),
    ("Programming Tools", "Can be extended with custom Python tools"),
    ("AI Features", "It integrates LLMs into daily workflow"),
    ("AI Features", "Supports multiple AI models including GPT-4"),
    ("AI Features", "Provides voice control through /whisper command"),
    ("Shell Integration", "Users can execute bash commands directly"),
    ("Development", "Can be extended with custom Python tools"),
    ("Development", "npcsh is a Python-based command-line tool"),
]

for group, fact in relationships:
    assign_fact_to_group(conn, fact, group)

# Now visualize
visualize_graph(conn)
conn.close()
