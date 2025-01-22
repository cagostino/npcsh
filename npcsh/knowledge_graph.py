import kuzu
import json
from .llm_funcs import get_llm_response
from .npc_compiler import NPC
from typing import Optional, Dict, List
import os


def create_group(conn: kuzu.Connection, name: str, metadata: str = ""):
    """Create a new group in the database"""
    escaped_name = name.replace("'", "''")
    escaped_metadata = metadata.replace("'", "''")

    query = f"""
    CREATE (g:Groups {{
        name: '{escaped_name}',
        metadata: '{escaped_metadata}'
    }});
    """

    try:
        conn.execute(query)
        print(f"Created group: {escaped_name}")
    except Exception as e:
        print(f"Error creating group: {escaped_name}\n{e}")


def init_db(db_path: str, drop=False):
    """Initialize Kùzu database and create schema"""
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    if drop:
        conn.execute(""" drop table Contains""")
        conn.execute("""drop  table if exists Fact;""")
        conn.execute("""drop  table if exists Groups;""")

    # Create schema if the tables do not exist
    conn.execute(
        """CREATE NODE TABLE IF NOT EXISTS Fact(
           content STRING,
           path STRING,
           recorded_at STRING,
           PRIMARY KEY (content)
        );"""
    )
    print("Fact table created or already exists.")

    conn.execute(
        """CREATE NODE TABLE IF NOT EXISTS Groups(
           name STRING,
           metadata STRING,
           PRIMARY KEY (name)
        );"""
    )
    print("Groups table created or already exists.")

    conn.execute(
        """CREATE REL TABLE IF NOT EXISTS Contains(
           FROM Groups TO Fact
        );"""
    )
    print("Contains relationship table created or already exists.")

    return conn


def extract_facts(
    text: str, model: str = "llama3.2", provider: str = "ollama", npc: NPC = None
) -> List:
    """Extract facts from text using LLM"""
    prompt = """Extract facts from this text.
        A fact is a piece of information that makes a statement about the world.
        A fact is typically a sentence that is true or false.
        Facts may be simple or complex. They can also be conflicting with each other, usually
        because there is some hidden context that is not mentioned in the text.
        In any case, it is simply your job to extract a list of facts that could pertain to
        an individual's  personality.
        For example, if a user says :
            "since I am a doctor I am often trying to think up new ways to help people.
            Can you help me set up a new kind of software to help with that?"
        You might extract the following facts:
            - The user is a doctor
            - The user is helpful

        Another example:
            "I am a software engineer who loves to play video games. I am also a huge fan of the
            Star Wars franchise and I am a member of the 501st Legion."
        You might extract the following facts:
            - The user is a software engineer
            - The user loves to play video games
            - The user is a huge fan of the Star Wars franchise
            - The user is a member of the 501st Legion

        Thus, it is your mission to reliably extract litss of facts.


    Return a JSON object with the following structure:

        {{
            "fact_list": "a list containing the facts where each fact is a string",
        }}


    Return only the JSON object.
    Do not include any additional markdown formatting.

    """

    response = get_llm_response(
        prompt + f"\n\nText: {text}",
        model=model,
        provider=provider,
        format="json",
    )
    response = response["response"]
    print(response)
    return response["fact_list"]


def find_similar_groups(
    conn: kuzu.Connection,
    fact: str,  # Ensure fact is passed as a string
    model: str = "llama3.2",
    provider: str = "ollama",
    npc: NPC = None,
) -> List[str]:
    """Find existing groups that might contain this fact"""
    response = conn.execute(f"MATCH (g:Groups) RETURN g.name;")  # Execute query
    print(response)
    print(type(response))
    print(dir(response))
    groups = response.fetch_as_df()
    print(f"Groups: {groups}")
    if not groups:
        return []

    prompt = """Given a fact and a list of groups, determine which groups this fact belongs to.
        A fact should belong to a group if it is semantically related to the group's theme or purpose.
        For example, if a fact is "The user loves programming" and there's a group called "Technical_Interests",
        that would be a match.

    Return a JSON object with the following structure:
        {
            "group_list": "a list containing the names of matching groups"
        }

    Return only the JSON object.
    Do not include any additional markdown formatting.
    """

    response = get_llm_response(
        prompt + f"\n\nFact: {fact}\nGroups: {json.dumps(groups)}",
        model=model,
        provider=provider,
        format="json",
        npc=npc,
    )
    response = response["response"]
    return response["group_list"]


def identify_groups(
    facts: List[str],
    model: str = "llama3.2",
    provider: str = "ollama",
    npc: NPC = None,
) -> List[str]:
    """Identify natural groups from a list of facts"""
    prompt = """What are the main groups these facts could be organized into?
    Express these groups in plain, natural language.

    For example, given:
        - User enjoys programming in Python
        - User works on machine learning projects
        - User likes to play piano
        - User practices meditation daily

    You might identify groups like:
        - Programming
        - Machine Learning
        - Musical Interests
        - Daily Practices

    Return a JSON object with the following structure:
        `{
            "groups": ["list of group names"]
        }`


    Return only the JSON object. Do not include any additional markdown formatting or
    leading json characters.
    """

    response = get_llm_response(
        prompt + f"\n\nFacts: {json.dumps(facts)}",
        model=model,
        provider=provider,
        format="json",
        npc=npc,
    )
    return response["response"]["groups"]


def assign_to_groups(
    fact: str,
    groups: List[str],
    model: str = "llama3.2",
    provider: str = "ollama",
    npc: NPC = None,
) -> Dict[str, List[str]]:
    """Assign facts to the identified groups"""
    prompt = f"""Given this fact, assign it to any relevant groups.

    A fact can belong to multiple groups if it fits.

    Here is the facT: {fact}

    Here are the groups: {groups}

    Return a JSON object with the following structure:
        {{
            "groups": ["list of group names"]
        }}

    Do not include any additional markdown formatting or leading json characters.


    """

    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        format="json",
        npc=npc,
    )
    return response["response"]


def insert_fact(conn: kuzu.Connection, fact: str, path: str):
    """Insert a fact into the database"""
    # Escape special characters
    escaped_fact = fact.replace("'", "''")
    escaped_path = os.path.expanduser(path).replace("'", "''")  # Empty string if None

    import datetime

    recorded_at = datetime.datetime.now()
    escaped_recorded_at = recorded_at.strftime("%Y-%m-%d %H:%M:%S")

    # Construct the query
    query = f"""
    CREATE (f:Fact {{
        content: "{escaped_fact}",
        path: "`{escaped_path}`",
        recorded_at: '{escaped_recorded_at}'
    }});
    """

    # Debug output to ensure everything is correct
    print(f"Inserting fact: {escaped_fact}")
    print(f"With path: {escaped_path}")
    print(f"With recorded_at: {escaped_recorded_at}")

    try:
        conn.execute(query)
        print(f"Inserted fact: {escaped_fact}")
    except Exception as e:
        print(f"Error inserting fact: {escaped_fact}\n{e}")


def assign_fact_to_group(conn: kuzu.Connection, fact: str, group: str):
    """Create a relationship between a fact and a group"""
    escaped_fact = fact.replace("'", "''")
    escaped_group = group.replace("'", "''")

    query = f"""
    MATCH (f:Fact), (g:Groups)
    WHERE f.content = '{escaped_fact}' AND g.name = '{escaped_group}'
    CREATE (g)-[:Contains]->(f);
    """

    try:
        conn.execute(query)
        print(f"Assigned fact to group: {escaped_group}")
    except Exception as e:
        print(f"Error assigning fact to group: {e}")


def save_facts_to_db(conn: kuzu.Connection, facts: List[str], path: str):
    """Save a list of facts to the database"""
    for fact in facts:
        insert_fact(conn, fact, path)


def process_text(
    db_path: str,
    text: str,
    path: str,
    model: str = "llama3.2",
    provider: str = "ollama",
    npc: NPC = None,
):
    """Process text and add extracted facts to the Kùzu database"""
    conn = init_db(db_path, drop=False)
    facts = extract_facts(text, model=model, provider=provider, npc=npc)
    for fact in facts:
        print(fact)  # Confirm that 'fact' is correctly iterated

    save_facts_to_db(conn, facts, path)

    conn.close()

    return facts


import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(conn):
    """Visualize the knowledge graph using networkx"""
    # Create a networkx graph
    G = nx.DiGraph()

    # Get all facts and groups with their relationships
    facts_result = conn.execute("MATCH (f:Fact) RETURN f.content;").get_as_df()
    facts = [row["f.content"] for index, row in facts_result.iterrows()]

    groups_result = conn.execute("MATCH (g:Groups) RETURN g.name;").get_as_df()
    groups = [row["g.name"] for index, row in groups_result.iterrows()]

    relationships_result = conn.execute(
        """
        MATCH (g:Groups)-[r:Contains]->(f:Fact)
        RETURN g.name, f.content;
    """
    ).get_as_df()

    # Add nodes with different colors for facts and groups
    for fact in facts:
        G.add_node(fact, node_type="fact")
    for group in groups:
        G.add_node(group, node_type="group")

    # Add edges from relationships
    for index, row in relationships_result.iterrows():
        G.add_edge(row["g.name"], row["f.content"])  # group name -> fact content

    # Set up the visualization
    plt.figure(figsize=(20, 12))
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw groups (larger nodes, distinct color)
    group_nodes = [
        n for n, attr in G.nodes(data=True) if attr.get("node_type") == "group"
    ]
    nx.draw_networkx_nodes(
        G, pos, nodelist=group_nodes, node_color="lightgreen", node_size=3000, alpha=0.7
    )

    # Draw facts (smaller nodes, different color)
    fact_nodes = [
        n for n, attr in G.nodes(data=True) if attr.get("node_type") == "fact"
    ]
    nx.draw_networkx_nodes(
        G, pos, nodelist=fact_nodes, node_color="lightblue", node_size=2000, alpha=0.5
    )

    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20)

    # Add labels with different sizes for groups and facts
    group_labels = {node: node for node in group_nodes}
    fact_labels = {
        node: node[:50] + "..." if len(node) > 50 else node for node in fact_nodes
    }

    nx.draw_networkx_labels(G, pos, group_labels, font_size=10, font_weight="bold")
    nx.draw_networkx_labels(G, pos, fact_labels, font_size=8)

    plt.title("Knowledge Graph: Groups and Facts", pad=20, fontsize=16)
    plt.axis("off")
    plt.tight_layout()

    # Print statistics
    print("\nKnowledge Graph Statistics:")
    print(f"Number of facts: {len(facts)}")
    print(f"Number of groups: {len(groups)}")
    print(f"Number of relationships: {len(relationships_result)}")

    print("\nGroups:")
    for g in groups:
        related_facts = [
            row["f.content"]
            for index, row in relationships_result.iterrows()
            if row["g.name"] == g
        ]
        print(f"\n{g}:")
        for f in related_facts:
            print(f"  - {f}")

    plt.show()
