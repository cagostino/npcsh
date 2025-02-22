#######
#######
#######
#######
####### EMBEDDINGS
#######
from typing import List, Dict, Optional
import numpy as np
from npcsh.npc_sysenv import (
    NPCSH_VECTOR_DB_PATH,
    NPCSH_EMBEDDING_MODEL,
    NPCSH_EMBEDDING_PROVIDER,
    chroma_client,
)
import ollama
from openai import OpenAI
import anthropic


def get_ollama_embeddings(
    texts: List[str], model: str = "nomic-embed-text"
) -> List[List[float]]:
    """Generate embeddings using Ollama."""
    embeddings = []
    for text in texts:
        response = ollama.embeddings(model=model, prompt=text)
        embeddings.append(response["embedding"])
    return embeddings


def get_openai_embeddings(
    texts: List[str], model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """Generate embeddings using OpenAI."""
    client = OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(input=texts, model=model)
    return [embedding.embedding for embedding in response.data]


def get_openai_like_embeddings(
    texts: List[str], model, api_url=None, api_key=None
) -> List[List[float]]:
    """Generate embeddings using OpenAI."""
    client = OpenAI(api_key=openai_api_key, base_url=api_url)
    response = client.embeddings.create(input=texts, model=model)
    return [embedding.embedding for embedding in response.data]


def get_anthropic_embeddings(
    texts: List[str], model: str = "claude-3-haiku-20240307"
) -> List[List[float]]:
    """Generate embeddings using Anthropic."""
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    embeddings = []
    for text in texts:
        # response = client.messages.create(
        #    model=model, max_tokens=1024, messages=[{"role": "user", "content": text}]
        # )
        # Placeholder for actual embedding
        embeddings.append([0.0] * 1024)  # Replace with actual embedding when available
    return embeddings


def store_embeddings_for_model(
    texts,
    embeddings,
    metadata=None,
    model: str = NPCSH_EMBEDDING_MODEL,
    provider: str = NPCSH_EMBEDDING_PROVIDER,
):
    collection_name = f"{provider}_{model}_embeddings"
    collection = chroma_client.get_collection(collection_name)

    # Create meaningful metadata for each document (adjust as necessary)
    if metadata is None:
        metadata = [{"text_length": len(text)} for text in texts]  # Example metadata
        print(
            "metadata is none, creating metadata for each document as the length of the text"
        )
    # Add embeddings to the collection with metadata
    collection.add(
        ids=[str(i) for i in range(len(texts))],
        embeddings=embeddings,
        metadatas=metadata,  # Passing populated metadata
        documents=texts,
    )


def delete_embeddings_from_collection(collection, ids):
    """Delete embeddings by id from Chroma collection."""
    if ids:
        collection.delete(ids=ids)  # Only delete if ids are provided


def search_similar_texts(
    query: str,
    docs_to_embed: Optional[List[str]] = None,
    top_k: int = 5,
    db_path: str = NPCSH_VECTOR_DB_PATH,
    embedding_model: str = NPCSH_EMBEDDING_MODEL,
    embedding_provider: str = NPCSH_EMBEDDING_PROVIDER,
) -> List[Dict[str, any]]:
    """
    Search for similar texts using either a Chroma database or direct embedding comparison.
    """

    print(f"\nQuery to embed: {query}")
    embedded_search_term = get_ollama_embeddings([query], embedding_model)[0]
    # print(f"Query embedding: {embedded_search_term}")

    if docs_to_embed is None:
        # Fetch from the database if no documents to embed are provided
        collection_name = f"{embedding_provider}_{embedding_model}_embeddings"
        collection = chroma_client.get_collection(collection_name)
        results = collection.query(
            query_embeddings=[embedded_search_term], n_results=top_k
        )
        # Constructing and returning results
        return [
            {"id": id, "score": float(distance), "text": document}
            for id, distance, document in zip(
                results["ids"][0], results["distances"][0], results["documents"][0]
            )
        ]

    print(f"\nNumber of documents to embed: {len(docs_to_embed)}")

    # Get embeddings for provided documents
    raw_embeddings = get_ollama_embeddings(docs_to_embed, embedding_model)

    output_embeddings = []
    for idx, emb in enumerate(raw_embeddings):
        if emb:  # Exclude any empty embeddings
            output_embeddings.append(emb)

    # Convert to numpy arrays for calculations
    doc_embeddings = np.array(output_embeddings)
    query_embedding = np.array(embedded_search_term)

    # Check for zero-length embeddings
    if len(doc_embeddings) == 0:
        raise ValueError("No valid document embeddings found")

    # Normalize embeddings to avoid division by zeros
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query_embedding)

    # Ensure no zero vectors are being used in cosine similarity
    if query_norm == 0:
        raise ValueError("Query embedding is zero-length")

    # Calculate cosine similarities
    cosine_similarities = np.dot(doc_embeddings, query_embedding) / (
        doc_norms.flatten() * query_norm
    )

    # Get indices of top K documents
    top_indices = np.argsort(cosine_similarities)[::-1][:top_k]

    return [
        {
            "id": str(idx),
            "score": float(cosine_similarities[idx]),
            "text": docs_to_embed[idx],
        }
        for idx in top_indices
    ]
