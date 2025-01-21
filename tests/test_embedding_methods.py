import unittest
import random
from typing import List
from npcsh.llm_funcs import (
    get_embeddings,
    search_similar_texts_for_model,
)  # Ensure both functions are imported
import chromadb
from time import sleep


class TestEmbeddingSearch(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the environment for testing.
        You can adjust the setUp method if you need to prepare certain conditions.
        """
        self.texts = [
            "This is a test sentence.",
            "How are you doing today?",
            "Embedding text for search.",
            "Chroma vector search is fun.",
            "Ollama embedding test.",
        ]
        self.provider = "ollama"
        self.model = "nomic-embed-text"
        self.top_k = 3
        self.client = chromadb.PersistentClient(path="/home/caug/npcsh_chroma.db")

    def test_embeddings_creation(self):
        """Test if embeddings are created correctly for Ollama."""
        embeddings = get_embeddings(
            self.texts, provider=self.provider, model=self.model
        )
        self.assertEqual(len(embeddings), len(self.texts))
        self.assertTrue(all(isinstance(embedding, list) for embedding in embeddings))
        self.assertTrue(all(len(embedding) > 0 for embedding in embeddings))

    def test_embeddings_storage_in_chroma(self):
        """Test if embeddings are stored correctly in Chroma."""
        embeddings = get_embeddings(
            self.texts, provider=self.provider, model=self.model
        )
        collection_name = f"{self.provider}_{self.model}_embeddings"
        collection = self.client.get_collection(collection_name)

        # Ensure that the collection has the correct number of documents
        self.assertEqual(len(collection.get()["documents"]), len(self.texts))

    def test_search_similar_texts(self):
        query_embedding = get_embeddings(
            ["Embedding text for search."], provider=self.provider, model=self.model
        )[0]

        results = search_similar_texts_for_model(
            query_embedding,
            embedding_model=self.model,
            provider=self.provider,
            top_k=5,
        )
        print(results)
        self.assertTrue(len(results) > 0)  # Ensure there are some results

    def test_search_with_multiple_results(self):
        """Test searching and getting multiple results."""
        embeddings = get_embeddings(
            self.texts, provider=self.provider, model=self.model
        )

        search_text = "Embedding text for search."
        search_embedding = get_embeddings(
            [search_text], provider=self.provider, model=self.model
        )[0]

        # Perform the search
        results = search_similar_texts_for_model(
            search_embedding,
            embedding_model=self.model,
            provider=self.provider,
            top_k=5,
        )

        # Ensure multiple results are returned
        self.assertGreater(len(results), 1)

        # Check if the results are properly formatted
        print(results, type(results), len(results))
        self.assertTrue(
            all(
                "id" in result and "text" in result and "score" in result
                for result in results
            )
        )

    def test_search_empty_results(self):
        embedding_dim = 768
        query_embedding = [0.0] * embedding_dim  # Use a neutral embedding
        results = search_similar_texts_for_model(
            query_embedding,
            embedding_model=self.model,
            provider=self.provider,
            top_k=5,
        )

        print(results, len(results), type(results))

    def test_very_high_top_k(self):
        """Test search with a very high 'top_k' to ensure it doesn't break."""
        search_text = "go bears."
        search_embedding = get_embeddings(
            [search_text], provider=self.provider, model=self.model
        )[0]

        # Perform search with a high value for top_k
        results = search_similar_texts_for_model(
            search_embedding,
            embedding_model=self.model,
            provider=self.provider,
        )

        # Ensure no error occurs and results are returned
        self.assertGreater(len(results), 0)

    def tearDown(self) -> None:
        """Clean up resources after tests."""
        collection_name = f"{self.provider}_{self.model}_embeddings"
        collection = self.client.get_collection(collection_name)

        # Get current documents in the collection
        collection_data = collection.get()
        stored_ids = collection_data["ids"]  # Get the IDs of stored documents

        # Delete only if the IDs exist in the collection
        ids_to_delete = [str(i) for i in range(len(self.texts))]
        ids_to_delete = [
            id_ for id_ in ids_to_delete if id_ in stored_ids
        ]  # Filter out non-existing IDs

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)  # Delete the existing IDs


if __name__ == "__main__":
    unittest.main()
