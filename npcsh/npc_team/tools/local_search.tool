tool_name: local_search
description: |
    searches files in current and downstream directories to find items related to the user's query to help in answering.
inputs:
  - query
  - file_filter  # Optional - can be filename patterns or folder names
  - depth: 2  # Optional - search depth for nested directories
preprocess:
  - engine: python
    code: |
        import os
        import fnmatch
        from pathlib import Path
        import sklearn.feature_extraction.text
        import sklearn.metrics.pairwise
        import numpy as np
        TfidfVectorizer = sklearn.feature_extraction.text.TfidfVectorizer
        cosine_similary = sklearn.metrics.pairwise.cosine_similarity
        import numpy as np

        def find_files(file_filter=None):
            default_extensions = ['.py', '.txt', '.md', '.json', '.yml', '.yaml']
            matches = []

            # Walk through all directories
            for root, dirnames, filenames in os.walk('.'):
                # Skip common directories to ignore
                if any(ignore in root for ignore in ['.git', '__pycache__', 'node_modules', 'venv']):
                    continue

                for filename in filenames:
                    if not any(filename.endswith(ext) for ext in default_extensions):
                        continue

                    full_path = os.path.join(root, filename)

                    # If no filter specified, include all files
                    if not file_filter:
                        matches.append(full_path)
                        continue

                    # Check if file matches any of the filters
                    if isinstance(file_filter, str):
                        filters = [file_filter]
                    else:
                        filters = file_filter

                    for f in filters:
                        # Check if filter matches filename or any parent directory
                        if (fnmatch.fnmatch(filename, f) or
                            fnmatch.fnmatch(full_path, f'*{f}*')):
                            matches.append(full_path)
                            break

            return matches

        def load_document(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {
                'path': file_path,
                'content': content,
                'ext': Path(file_path).suffix.lower()
            }

        def chunk_document(content, chunk_size=1000, overlap=200):
            chunks = []
            start = 0
            while start < len(content):
                end = start + chunk_size
                chunk = content[start:end]
                # Try to break at newline if possible
                if end < len(content):
                    newline_pos = chunk.rfind('\n')
                    if newline_pos > chunk_size * 0.5:  # Only break if newline is past halfway
                        end = start + newline_pos + 1
                        chunk = content[start:end]
                chunks.append(chunk)
                start = end - overlap
            return chunks

        def find_relevant_chunks(query, chunks, top_k=3):
            vectorizer = TfidfVectorizer()
            try:
                chunk_vectors = vectorizer.fit_transform([c['content'] for c in chunks])
                query_vector = vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, chunk_vectors)[0]
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                return [(chunks[i], similarities[i]) for i in top_indices if similarities[i] > 0]
            except:
                return [(chunk, 0) for chunk in chunks[:top_k]]

        # Get files based on filter
        file_filter = inputs.get('file_filter', None)
        files = find_files(file_filter)

        print(f"Found {len(files)} files to search")

        # Load and chunk documents
        chunks = []
        for file_path in files:
            try:
                print(file_path)
                loaded_doc = load_document(file_path)
                print(file_path, 'loaded')
                if loaded_doc and isinstance(loaded_doc, dict):
                    content = loaded_doc.get('content', '')
                    if content:
                        doc_chunks = chunk_document(content)
                        for chunk in doc_chunks:
                            chunks.append({
                                'path': file_path,  # file_path is directly in scope
                                'content': chunk,
                                'ext': loaded_doc.get('ext', '')
                            })

            except Exception as e:
                print(files)
                print(f"Error loading {file_path}: {str(e)}")

        # Find relevant chunks
        relevant_chunks = find_relevant_chunks(inputs['query'], chunks)

        # Prepare context for LLM
        context_text = "\nRelevant code/documentation:\n\n"
        for chunk, similarity in relevant_chunks:
            context_text += f"File: {chunk['path']} (relevance: {similarity:.2f})\n```{chunk['ext'][1:]}\n{chunk['content']}\n```\n\n"

        context['relevant_context'] = context_text

prompt:
  engine: natural
  code: |
    You are a helpful coding assistant. Please help with this query:

    {{ inputs['query']  }}

    Here is the relevant context from the codebase:

    {{ relevant_context }}

    Please provide a clear and helpful response to the user's query.
    Explain exactly how it is that you arrived at your answer and how it answers the user's query.
    If you reference specific files or code sections, indicate which file they came from.
    In your response, you must explicitly mention what the users query was.

postprocess:
  - engine: natural
    code: ""