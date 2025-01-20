import kuzu
import json
from .llm_funcs import get_llm_response
from .npc_compiler import NPC
from typing import Optional, Dict, List
import os


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

    # Extracting group names properly from QueryResult
    groups = [row[0] for row in response.fetch_all()]  # Use fetch_all() to get rows

    print(f"Groups: {groups}")
    if not groups:
        return []

    prompt = f"""Given this fact: {json.dumps(fact)}
    And these groups: {json.dumps(groups)}

    Return an array of group names that this fact belongs to, if any.
    """

    response = get_llm_response(
        prompt, model=model, provider=provider, npc=npc, format="json"
    )
    return json.loads(response)


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
    conn = init_db(db_path, drop=True)
    facts = extract_facts(text, model=model, provider=provider, npc=npc)
    for fact in facts:
        print(fact)  # Confirm that 'fact' is correctly iterated

    save_facts_to_db(conn, facts, path)
    conn.close()
