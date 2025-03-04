tool_name: local_search
description: |
    Searches files in current and downstream directories to find items related to the user's query using fuzzy matching.
    Returns only relevant snippets (10 lines around matches) to avoid including too much irrelevant content.
    Intended for fuzzy searches, not for understanding file sizes.
inputs:
  - query
  - summarize: false  # Optional - set to true to summarize the results
  - file_filter: 'none'  # Optional - can be filename patterns or folder names
  - depth: 2  # Optional - search depth for nested directories
  - fuzzy_threshold: 70  # Optional - minimum fuzzy match score (0-100)
steps:
  - engine: python
    code: |
        # Search parameters are directly available
        query = "{{ query }}"
        file_filter = "{{ file_filter | default('None') }}"
        if isinstance(file_filter, str) and file_filter.lower() == 'none':
            file_filter = None
        max_depth = {{ depth | default(2) }}
        fuzzy_threshold = {{ fuzzy_threshold | default(70) }}

        import os
        import fnmatch
        from pathlib import Path
        from thefuzz import fuzz  # Fuzzy string matching library

        def find_files(file_filter=None, max_depth=2):
            default_extensions = ['.py', '.txt', '.md',
                                '.json', '.yml', '.yaml',
                                '.log', '.csv', '.html',
                                '.js', '.css']
            matches = []
            root_path = Path('.').resolve()  # Resolve to absolute path

            # First, check files in the current directory
            for path in root_path.iterdir():
                if path.is_file():
                    # Skip hidden files
                    if path.name.startswith('.'):
                        continue

                    # If no filter specified, include files with default extensions
                    if file_filter is None:
                        if path.suffix in default_extensions:
                            matches.append(str(path))
                    else:
                        # If filter specified, check if file matches the filter
                        filters = [file_filter] if isinstance(file_filter, str) else file_filter
                        for f in filters:
                            if (fnmatch.fnmatch(path.name, f) or
                                fnmatch.fnmatch(str(path), f'*{f}*')):
                                matches.append(str(path))
                                break

            # Then, check subdirectories with depth control
            for path in root_path.rglob('*'):
                # Skip hidden folders and common directories to ignore
                if '/.' in str(path) or '__pycache__' in str(path) or '.git' in str(path) or 'node_modules' in str(path) or 'venv' in str(path):
                    continue

                # Skip if we've gone too deep
                relative_depth = len(path.relative_to(root_path).parts)
                if relative_depth > max_depth:
                    continue

                if path.is_file():
                    # If no filter specified, include files with default extensions
                    if file_filter is None:
                        if path.suffix in default_extensions:
                            matches.append(str(path))
                    else:
                        # If filter specified, check if file matches the filter
                        filters = [file_filter] if isinstance(file_filter, str) else file_filter
                        for f in filters:
                            if (fnmatch.fnmatch(path.name, f) or
                                fnmatch.fnmatch(str(path), f'*{f}*')):
                                matches.append(str(path))
                                break

            return matches

        # Find and load files
        files = find_files(file_filter, max_depth)

        # Process documents
        relevant_chunks = []
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()  # Read file as lines
            if lines:
                # Join lines into a single string for fuzzy matching
                content = ''.join(lines)
                match_score = fuzz.partial_ratio(query.lower(), content.lower())
                if match_score >= fuzzy_threshold:
                    # Find the best matching line
                    best_line_index = -1
                    best_line_score = 0
                    for i, line in enumerate(lines):
                        line_score = fuzz.partial_ratio(query.lower(), line.lower())
                        if line_score > best_line_score:
                            best_line_score = line_score
                            best_line_index = i

                    # Extract 10 lines around the best matching line
                    if best_line_index != -1:
                        start = max(0, best_line_index - 5)  # 5 lines before
                        end = min(len(lines), best_line_index + 6)  # 5 lines after
                        snippet = ''.join(lines[start:end])
                        relevant_chunks.append({
                            'path': file_path,
                            'snippet': snippet,
                            'ext': Path(file_path).suffix.lower(),
                            'score': match_score
                        })

        # Sort results by match score (highest first)
        relevant_chunks.sort(key=lambda x: x['score'], reverse=True)

        # Format results
        if relevant_chunks:
            context_text = "Here are the most relevant code sections:\n\n"
            for chunk in relevant_chunks:
                file_path = chunk['path'].replace('./', '')
                context_text += f"File: {file_path} (match score: {chunk['score']})\n"
                context_text += f"```{chunk['ext'][1:] if chunk['ext'] else ''}\n"
                context_text += f"{chunk['snippet'].strip()}\n"
                context_text += "```\n\n"
        else:
            context_text = "No relevant code sections found.\n"

        output = context_text

  - engine: natural
    code: |
        {% if summarize %}
        You are a helpful coding assistant.
        Please help with this query:

        `{{ query }}`

        The user is attempting to carry out a local search. This search returned the following results:

        `{{ results }}`

        Please analyze the code sections above and provide a clear, helpful response that directly addresses the query.
        If you reference specific files or code sections in your response, indicate which file they came from.
        Make sure to explain your reasoning and how the provided code relates to the query.
        {% endif %}
