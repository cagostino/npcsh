import chromadb
from sentence_transformers import SentenceTransformer

# Initialize Chroma PersistentClient (persistent on disk)
client = chromadb.PersistentClient(
    path="/home/caug/npcsh_chroma.db"
)  # Specify the path for saving the database

# Check if collection exists, create if not
collection_name = "state_union"
if collection_name not in client.list_collections():
    collection = client.create_collection(name=collection_name)
else:
    collection = client.get_collection(collection_name)

# Initialize SentenceTransformer model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample texts to be added
texts = ["Ketanji Brown Jackson is awesome", "foo", "bar"]

# Generate embeddings for the texts
embeddings = [model.encode(text) for text in texts]

# Generate unique IDs for each document (you can use any unique identifier)
ids = [str(i) for i in range(len(texts))]  # Simple IDs: "0", "1", "2", ...

# Add the texts and embeddings to the Chroma collection
for text, embedding, doc_id in zip(texts, embeddings, ids):
    collection.add(
        documents=[text],  # List of documents (texts)
        metadatas=[None],  # No metadata, pass None instead of empty dict
        embeddings=[embedding],  # Corresponding embeddings
        ids=[doc_id],  # Unique document IDs
    )

# Debugging: Check if texts were added
print(f"Added {len(texts)} texts to Chroma collection.")

# Querying: Example of querying the collection
query = "Ketanji Brown Jackson is awesome"
query_embedding = model.encode(query)

# Query the collection for similar results
results = collection.query(query_embeddings=[query_embedding], n_results=3)
print(f"Query results: {results}")


import chromadb

# Initialize Chroma PersistentClient (persistent on disk)
client = chromadb.PersistentClient(path="/home/caug/npcsh_chroma.db")


# List all collections
def list_collections():
    collections = client.list_collections()
    print("Collections available:")
    for collection in collections:
        print(collection)


# Inspect a specific collection
def inspect_collection(collection_name):
    collection = client.get_collection(collection_name)
    print(f"Inspecting collection: {collection_name}")

    # List the first 5 documents (for example)
    results = collection.query(
        query_embeddings=[[0] * ], n_results=5
    )  # Dummy query to fetch some data
    print("First 5 results in the collection:")
    for result in results["documents"]:
        print(result)


# CLI Loop
def cli():
    while True:
        print("\nCommands: [list] [inspect <collection_name>] [exit]")
        command = input("Enter command: ")

        if command == "list":
            list_collections()
        elif command.startswith("inspect"):
            collection_name = command.split(" ")[1]
            inspect_collection(collection_name)
        elif command == "exit":
            break
        else:
            print("Unknown command, try again.")


if __name__ == "__main__":
    cli()
