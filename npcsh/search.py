# search.py

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from googlesearch import search
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sentence_transformers import util
except:
    pass


def search_web(
    query: str, num_results: int = 5, provider: str = "google"
) -> List[Dict[str, str]]:
    """
    Function Description:
        This function searches the web for information based on a query.
    Args:
        query: The search query.
    Keyword Args:
        num_results: The number of search results to retrieve.
        provider: The search engine provider to use ('google' or 'duckduckgo').
    Returns:
        A list of dictionaries with 'title', 'link', and 'content' keys.
    """
    results = []

    try:
        if provider == "duckduckgo":
            ddgs = DDGS()
            search_results = ddgs.text(query, max_results=num_results)
            urls = [r["link"] for r in search_results]
        else:  # google
            urls = list(search(query, num_results=num_results))

        for url in urls:
            try:
                # Fetch the webpage content
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status()

                # Parse with BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")

                # Get title and content
                title = soup.title.string if soup.title else url

                # Extract text content and clean it up
                content = " ".join([p.get_text() for p in soup.find_all("p")])
                content = " ".join(content.split())  # Clean up whitespace

                results.append(
                    {
                        "title": title,
                        "link": url,
                        "content": (
                            content[:500] + "..." if len(content) > 500 else content
                        ),
                    }
                )

            except Exception as e:
                print(f"Error fetching {url}: {str(e)}")
                continue

    except Exception as e:
        print(f"Search error: {str(e)}")

    return results


def rag_search(
    query: str,
    text_data: Dict[str, str],
    embedding_model: Any,
    text_data_embedded: Optional[Dict[str, np.ndarray]] = None,
    similarity_threshold: float = 0.2,
) -> List[str]:
    """
    Function Description:
        This function retrieves lines from documents that are relevant to the query.
    Args:
        query: The query string.
        text_data: A dictionary with file paths as keys and file contents as values.
        embedding_model: The sentence embedding model.
    Keyword Args:
        text_data_embedded: A dictionary with file paths as keys and embedded file contents as values.
        similarity_threshold: The similarity threshold for considering a line relevant.
    Returns:
        A list of relevant snippets.

    """

    results = []

    # Compute the embedding of the query
    query_embedding = embedding_model.encode(
        query, convert_to_tensor=True, show_progress_bar=False
    )

    for filename, content in text_data.items():
        # Split content into lines
        lines = content.split("\n")
        if not lines:
            continue
        # Compute embeddings for each line
        if text_data_embedded is None:
            line_embeddings = embedding_model.encode(lines, convert_to_tensor=True)
        else:
            line_embeddings = text_data_embedded[filename]
        # Compute cosine similarities
        cosine_scores = util.cos_sim(query_embedding, line_embeddings)[0].cpu().numpy()

        # Find indices of lines above the similarity threshold
        ##print("most similar", np.max(cosine_scores))
        ##print("most similar doc", lines[np.argmax(cosine_scores)])
        relevant_line_indices = np.where(cosine_scores >= similarity_threshold)[0]

        for idx in relevant_line_indices:
            idx = int(idx)  # Ensure idx is an integer
            # Get context lines (±10 lines)
            start_idx = max(0, idx - 10)
            end_idx = min(len(lines), idx + 11)  # +11 because end index is exclusive
            snippet = "\n".join(lines[start_idx:end_idx])
            results.append((filename, snippet))
    # print("results", results)
    return results
