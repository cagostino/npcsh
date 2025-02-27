import json
import os
import datetime

import numpy as np
import kuzu
from typing import Optional, Dict, List, Union, Tuple


from .llm_funcs import get_llm_response, get_embeddings
from .npc_compiler import NPC
import sqlite3


def safe_kuzu_execute(conn, query, error_message="Kuzu query failed"):
    """Execute a Kuzu query with proper error handling"""
    try:
        result = conn.execute(query)
        return result, None
    except Exception as e:
        error = f"{error_message}: {str(e)}"
        print(error)
        return None, error


def create_group(conn: kuzu.Connection, name: str, metadata: str = ""):
    """Create a new group in the database with robust error handling"""
    if conn is None:
        print("Cannot create group: database connection is None")
        return False

    try:
        # Properly escape quotes in strings
        escaped_name = name.replace('"', '\\"')
        escaped_metadata = metadata.replace('"', '\\"')

        query = f"""
        CREATE (g:Groups {{
            name: "{escaped_name}",
            metadata: "{escaped_metadata}"
        }});
        """

        result, error = safe_kuzu_execute(
            conn, query, f"Failed to create group: {name}"
        )
        if error:
            return False

        print(f"Created group: {name}")
        return True
    except Exception as e:
        print(f"Error creating group {name}: {str(e)}")
        traceback.print_exc()
        return False


import traceback


def init_db(db_path: str, drop=False):
    """Initialize Kùzu database and create schema with robust error handling"""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        try:
            db = kuzu.Database(db_path)
            conn = kuzu.Connection(db)
            print("Database connection established successfully")
        except Exception as e:
            print(f"Failed to connect to database: {str(e)}")
            traceback.print_exc()
            return None
        # Drop tables if requested
        if drop:
            safe_kuzu_execute(conn, "DROP REL TABLE IF EXISTS Contains")
            safe_kuzu_execute(conn, "DROP NODE TABLE IF EXISTS Fact")
            safe_kuzu_execute(conn, "DROP NODE TABLE IF EXISTS Groups")

        # Create tables with proper error handling
        safe_kuzu_execute(
            conn,
            """
            CREATE NODE TABLE IF NOT EXISTS Fact(
              content STRING,
              path STRING,
              recorded_at STRING,
              PRIMARY KEY (content)
            );
            """,
            "Failed to create Fact table",
        )
        print("Fact table created or already exists.")

        safe_kuzu_execute(
            conn,
            """
            CREATE NODE TABLE IF NOT EXISTS Groups(
              name STRING,
              metadata STRING,
              PRIMARY KEY (name)
            );
            """,
            "Failed to create Groups table",
        )
        print("Groups table created or already exists.")

        safe_kuzu_execute(
            conn,
            """
            CREATE REL TABLE IF NOT EXISTS Contains(
              FROM Groups TO Fact
            );
            """,
            "Failed to create Contains relationship table",
        )
        print("Contains relationship table created or already exists.")

        return conn
    except Exception as e:
        print(f"Fatal error initializing database: {str(e)}")
        traceback.print_exc()
        return None


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


def insert_fact(conn: kuzu.Connection, fact: str, path: str) -> bool:
    """Insert a fact into the database with robust error handling"""
    if conn is None:
        print("Cannot insert fact: database connection is None")
        return False

    try:
        # Properly escape quotes in strings
        escaped_fact = fact.replace('"', '\\"')
        escaped_path = os.path.expanduser(path).replace('"', '\\"')

        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Begin transaction
        safe_kuzu_execute(conn, "BEGIN TRANSACTION")

        # Check if fact already exists
        check_query = f"""
        MATCH (f:Fact {{content: "{escaped_fact}"}})
        RETURN f
        """

        result, error = safe_kuzu_execute(
            conn, check_query, "Failed to check if fact exists"
        )
        if error:
            safe_kuzu_execute(conn, "ROLLBACK")
            return False

        # Insert fact if it doesn't exist
        if not result.has_next():
            insert_query = f"""
            CREATE (f:Fact {{
                content: "{escaped_fact}",
                path: "{escaped_path}",
                recorded_at: "{timestamp}"
            }})
            """

            result, error = safe_kuzu_execute(
                conn, insert_query, "Failed to insert fact"
            )
            if error:
                safe_kuzu_execute(conn, "ROLLBACK")
                return False

        # Commit transaction
        safe_kuzu_execute(conn, "COMMIT")
        return True
    except Exception as e:
        print(f"Error inserting fact: {str(e)}")
        traceback.print_exc()
        safe_kuzu_execute(conn, "ROLLBACK")
        return False


def assign_fact_to_group(conn: kuzu.Connection, fact: str, group: str) -> bool:
    """Create a relationship between a fact and a group with robust error handling"""
    if conn is None:
        print("Cannot assign fact to group: database connection is None")
        return False

    try:
        # Properly escape quotes in strings
        escaped_fact = fact.replace('"', '\\"')
        escaped_group = group.replace('"', '\\"')

        # Check if both fact and group exist before creating relationship
        check_query = f"""
        MATCH (f:Fact {{content: "{escaped_fact}"}})
        RETURN f
        """

        result, error = safe_kuzu_execute(
            conn, check_query, "Failed to check if fact exists"
        )
        if error or not result.has_next():
            print(f"Fact not found: {fact}")
            return False

        check_query = f"""
        MATCH (g:Groups {{name: "{escaped_group}"}})
        RETURN g
        """

        result, error = safe_kuzu_execute(
            conn, check_query, "Failed to check if group exists"
        )
        if error or not result.has_next():
            print(f"Group not found: {group}")
            return False

        # Create relationship
        query = f"""
        MATCH (f:Fact), (g:Groups)
        WHERE f.content = "{escaped_fact}" AND g.name = "{escaped_group}"
        CREATE (g)-[:Contains]->(f)
        """

        result, error = safe_kuzu_execute(
            conn, query, f"Failed to assign fact to group: {group}"
        )
        if error:
            return False

        print(f"Assigned fact to group: {group}")
        return True
    except Exception as e:
        print(f"Error assigning fact to group: {str(e)}")
        traceback.print_exc()
        return False


def save_facts_to_db(
    conn: kuzu.Connection, facts: List[str], path: str, batch_size: int
):
    """Save a list of facts to the database in batches"""
    for i in range(0, len(facts), batch_size):
        batch = facts[i : i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} facts)")

        # Process each fact in the batch
        for fact in batch:
            try:
                print(f"Inserting fact: {fact}")
                print(f"With path: {path}")
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"With recorded_at: {timestamp}")

                insert_fact(conn, fact, path)
                print("Success!")
            except Exception as e:
                print(f"Failed to insert fact: {fact}")
                print(f"Error: {e}")
                continue

        print(f"Completed batch {i//batch_size + 1}")


def process_text(
    db_path: str,
    text: str,
    path: str,
    model: str = "llama3.2",
    provider: str = "ollama",
    npc: NPC = None,
    batch_size: int = 5,
    conn=None,
):
    """Process text and add extracted facts to the database with robust error handling"""

    try:
        # Initialize database
        if conn is None:
            conn = init_db(db_path, drop=False)

            return []

        # Extract facts
        facts = extract_facts(text, model=model, provider=provider, npc=npc)
        if not facts:
            print("No facts extracted")
            return []

        print(f"Extracted {len(facts)} facts")
        for fact in facts:
            print(f"- {fact}")

        # Process facts in batches
        for i in range(0, len(facts), batch_size):
            batch = facts[i : i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} facts)")

            for fact in batch:
                try:
                    print(f"Inserting fact: {fact}")
                    success = insert_fact(conn, fact, path)
                    if success:
                        print("Success!")
                    else:
                        print("Failed to insert fact")
                except Exception as e:
                    print(f"Error processing fact: {str(e)}")
                    traceback.print_exc()

            print(f"Completed batch {i//batch_size + 1}")

        return facts
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        traceback.print_exc()
        return []


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


import chromadb
import numpy as np
import os
import datetime
from typing import Optional, Dict, List, Union, Tuple


def setup_chroma_db(db_path: str):
    """Initialize Chroma vector database without a default embedding function"""
    try:
        # Create or connect to Chroma client with persistent storage
        client = chromadb.PersistentClient(path=db_path)

        # Check if collection exists, create if not
        try:
            collection = client.get_collection("facts_collection")
            print("Connected to existing facts collection")
        except ValueError:
            # Create new collection without an embedding function
            # We'll provide embeddings manually using get_embeddings
            collection = client.create_collection(
                name="facts_collection",
                metadata={"description": "Facts extracted from various sources"},
            )
            print("Created new facts collection")

        return client, collection
    except Exception as e:
        print(f"Error setting up Chroma DB: {e}")
        raise


def store_fact_with_embedding(
    collection, fact: str, metadata: dict, embedding: List[float]
) -> str:
    """Store a fact with its pre-generated embedding in Chroma DB

    Args:
        collection: Chroma collection
        fact: The fact text
        metadata: Dictionary with metadata (path, source, timestamp, etc.)
        embedding: Pre-generated embedding vector from get_embeddings

    Returns:
        ID of the stored fact
    """
    try:
        # Generate a deterministic ID from the fact content
        import hashlib

        fact_id = hashlib.md5(fact.encode()).hexdigest()

        # Store document with pre-generated embedding
        collection.add(
            documents=[fact],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[fact_id],
        )

        return fact_id
    except Exception as e:
        print(f"Error storing fact in Chroma: {e}")
        return None


def find_similar_facts_chroma(
    collection,
    query: str,
    query_embedding: List[float],
    n_results: int = 5,
    metadata_filter: Optional[Dict] = None,
) -> List[Dict]:
    """Find facts similar to the query using pre-generated embedding

    Args:
        collection: Chroma collection
        query: Query text (for reference only)
        query_embedding: Pre-generated embedding from get_embeddings
        n_results: Number of results to return
        metadata_filter: Optional filter for metadata fields

    Returns:
        List of dictionaries with results
    """
    try:
        # Perform query with optional metadata filtering
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=metadata_filter,
        )

        # Format results
        formatted_results = []
        for i, doc in enumerate(results["documents"][0]):
            formatted_results.append(
                {
                    "fact": doc,
                    "metadata": results["metadatas"][0][i],
                    "id": results["ids"][0][i],
                    "distance": (
                        results["distances"][0][i] if "distances" in results else None
                    ),
                }
            )

        return formatted_results
    except Exception as e:
        print(f"Error searching in Chroma: {e}")
        return []


def process_text_with_chroma(
    kuzu_db_path: str,
    chroma_db_path: str,
    text: str,
    path: str,
    npc: NPC = None,
    batch_size: int = 5,
):
    """Process text and store facts in both Kuzu and Chroma DB

    Args:
        kuzu_db_path: Path to Kuzu graph database
        chroma_db_path: Path to Chroma vector database
        text: Input text to process
        path: Source path or identifier
        model: LLM model to use
        provider: LLM provider
        embedding_model: Model to use for embeddings
        npc: Optional NPC instance
        batch_size: Batch size for processing

    Returns:
        List of extracted facts
    """
    # Initialize databases
    kuzu_conn = init_db(kuzu_db_path, drop=False)
    chroma_client, chroma_collection = setup_chroma_db(chroma_db_path)

    # Extract facts
    facts = extract_facts(text, model=model, provider=provider, npc=npc)

    # Process extracted facts
    for i in range(0, len(facts), batch_size):
        batch = facts[i : i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} facts)")

        # Generate embeddings for the batch using npcsh.llm_funcs.get_embeddings
        from .llm_funcs import get_embeddings

        batch_embeddings = get_embeddings(
            batch,
        )

        for j, fact in enumerate(batch):
            print(f"Processing fact: {fact}")
            embedding = batch_embeddings[j]

            # Check for similar facts in Chroma before inserting
            similar_facts = find_similar_facts_chroma(
                chroma_collection, fact, query_embedding=embedding, n_results=3
            )

            if similar_facts:
                print(f"Similar facts found:")
                for result in similar_facts:
                    print(f"  - {result['fact']} (distance: {result['distance']})")
                # Note: Could implement a similarity threshold here to skip highly similar facts

            # Prepare metadata
            metadata = {
                "path": path,
                "timestamp": datetime.datetime.now().isoformat(),
                "source_model": model,
                "source_provider": provider,
            }

            # Insert into Kuzu graph DB
            kuzu_success = insert_fact(kuzu_conn, fact, path)

            # Insert into Chroma vector DB if Kuzu insert was successful
            if kuzu_success:
                chroma_id = store_fact_with_embedding(
                    chroma_collection, fact, metadata, embedding
                )
                if chroma_id:
                    print(f"Successfully saved fact with ID: {chroma_id}")
                else:
                    print(f"Failed to save fact to Chroma")
            else:
                print(f"Failed to save fact to Kuzu graph")

    # Close Kuzu connection
    kuzu_conn.close()

    return facts


def hybrid_search_with_chroma(
    kuzu_conn: kuzu.Connection,
    chroma_collection,
    query: str,
    group_filter: Optional[List[str]] = None,
    top_k: int = 5,
    metadata_filter: Optional[Dict] = None,
) -> List[Dict]:
    """Perform hybrid search using both Chroma vector search and Kuzu graph relationships

    Args:
        kuzu_conn: Connection to Kuzu graph database
        chroma_collection: Chroma collection for vector search
        query: Search query text
        group_filter: Optional list of groups to filter by in graph
        top_k: Number of results to return
        metadata_filter: Optional metadata filter for Chroma search
        embedding_model: Model to use for embeddings
        provider: Provider for embeddings

    Returns:
        List of dictionaries with combined results
    """
    # Get embedding for query using npcsh.llm_funcs.get_embeddings
    from .llm_funcs import get_embeddings

    query_embedding = get_embeddings([query])[0]

    # Step 1: Find similar facts using Chroma vector search
    vector_results = find_similar_facts_chroma(
        chroma_collection,
        query,
        query_embedding=query_embedding,
        n_results=top_k,
        metadata_filter=metadata_filter,
    )

    # Extract just the fact texts from vector results
    vector_facts = [result["fact"] for result in vector_results]

    # Step 2: Expand context using graph relationships
    expanded_results = []

    # Add vector search results
    for result in vector_results:
        expanded_results.append(
            {
                "fact": result["fact"],
                "source": "vector_search",
                "relevance": "direct_match",
                "distance": result["distance"],
                "metadata": result["metadata"],
            }
        )

    # For each vector-matched fact, find related facts in the graph
    for fact in vector_facts:
        try:
            # Safely escape fact text for Kuzu query
            escaped_fact = fact.replace('"', '\\"')

            # Find groups containing this fact
            group_result = kuzu_conn.execute(
                f"""
                MATCH (g:Groups)-[:Contains]->(f:Fact)
                WHERE f.content = "{escaped_fact}"
                RETURN g.name
                """
            ).get_as_df()

            # Extract group names
            fact_groups = [row["g.name"] for _, row in group_result.iterrows()]

            # Apply group filter if provided
            if group_filter:
                fact_groups = [g for g in fact_groups if g in group_filter]

            # For each group, find other related facts
            for group in fact_groups:
                escaped_group = group.replace('"', '\\"')

                # Find facts in the same group
                related_facts_result = kuzu_conn.execute(
                    f"""
                    MATCH (g:Groups)-[:Contains]->(f:Fact)
                    WHERE g.name = "{escaped_group}" AND f.content <> "{escaped_fact}"
                    RETURN f.content, f.path, f.recorded_at
                    LIMIT 5
                    """
                ).get_as_df()

                # Add these related facts to results
                for _, row in related_facts_result.iterrows():
                    related_fact = {
                        "fact": row["f.content"],
                        "source": f"graph_relation_via_{group}",
                        "relevance": "group_related",
                        "path": row["f.path"],
                        "recorded_at": row["f.recorded_at"],
                    }

                    # Avoid duplicates
                    if not any(
                        r.get("fact") == related_fact["fact"] for r in expanded_results
                    ):
                        expanded_results.append(related_fact)

        except Exception as e:
            print(f"Error expanding results via graph: {e}")

    # Return results, limiting to top_k if needed
    return expanded_results[:top_k]


def get_facts_for_rag(
    kuzu_db_path: str,
    chroma_db_path: str,
    query: str,
    group_filters: Optional[List[str]] = None,
    top_k: int = 10,
) -> str:
    """Get facts for RAG by combining vector and graph search

    Args:
        kuzu_db_path: Path to Kuzu graph database
        chroma_db_path: Path to Chroma vector database
        query: Search query
        group_filters: Optional list of groups to filter by
        top_k: Number of results to return
        embedding_model: Model to use for embeddings
        provider: Provider for embeddings

    Returns:
        Formatted context string with retrieved facts
    """
    # Initialize connections
    kuzu_conn = init_db(kuzu_db_path)
    chroma_client, chroma_collection = setup_chroma_db(chroma_db_path)

    # Perform hybrid search
    results = hybrid_search_with_chroma(
        kuzu_conn=kuzu_conn,
        chroma_collection=chroma_collection,
        query=query,
        group_filter=group_filters,
        top_k=top_k,
    )

    # Format results as context for RAG
    context = "Related facts:\n\n"

    # First include direct vector matches
    context += "Most relevant facts:\n"
    vector_matches = [r for r in results if r["source"] == "vector_search"]
    for i, item in enumerate(vector_matches):
        context += f"{i+1}. {item['fact']}\n"

    # Then include graph-related facts
    context += "\nRelated concepts:\n"
    graph_matches = [r for r in results if r["source"] != "vector_search"]
    for i, item in enumerate(graph_matches):
        group = item["source"].replace("graph_relation_via_", "")
        context += f"{i+1}. {item['fact']} (related via {group})\n"

    # Close connections
    kuzu_conn.close()

    return context


# Example usage in a RAG context
def answer_with_rag(
    query: str,
    kuzu_db_path: str = os.path.expanduser("~/npcsh_graph.db"),
    chroma_db_path: str = os.path.expanduser("~/npcsh_chroma.db"),
    model: str = "ollama",
    provider: str = "llama3.2",
    embedding_model: str = "text-embedding-3-small",
) -> str:
    """Answer a query using RAG with facts from the knowledge base

    Args:
        query: User query
        kuzu_db_path: Path to Kuzu graph database
        chroma_db_path: Path to Chroma vector database
        model: LLM model to use
        provider: LLM provider
        embedding_model: Model to use for embeddings

    Returns:
        Answer from the model
    """
    # Get relevant facts using hybrid search
    context = get_facts_for_rag(
        kuzu_db_path,
        chroma_db_path,
        query,
    )

    # Craft prompt with retrieved context
    prompt = f"""
    Answer this question based on the retrieved information.

    Question: {query}

    {context}

    Please provide a comprehensive answer based on the facts above. If the information
    doesn't contain a direct answer, please indicate that clearly but try to synthesize
    from the available facts.
    """

    # Get response from LLM
    response = get_llm_response(prompt, model=model, provider=provider)

    return response["response"]
