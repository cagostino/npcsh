from npcsh.knowledge_graph import process_text_with_chroma
import os

# Paths to your databases
kuzu_db_path = os.path.expanduser("~/npcsh_graph.db")
chroma_db_path = os.path.expanduser("~/npcsh_chroma.db")

# Process text and store facts with embeddings from your function
text = """
npcsh is a python-based command-line tool designed to integrate Large Language Models (LLMs)
into one's daily workflow by making them available through the command line shell.
"""

facts = process_text_with_chroma(
    kuzu_db_path=kuzu_db_path,
    chroma_db_path=chroma_db_path,
    text=text,
    path="~/npcww/npcsh/docs/",
)

# Later, answer a question using RAG
answer = answer_with_rag(
    query="What can I do with npcsh?",
    kuzu_db_path=kuzu_db_path,
    chroma_db_path=chroma_db_path,
    model="gpt-4o-mini",
    provider="openai",
    embedding_model="text-embedding-3-small",
)

print(answer)
