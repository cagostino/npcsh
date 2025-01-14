tool_name: splonk
description: |
  Debugging tool to analyze and fix code issues based on user's description and test command.
  The tool uses a language model to provide insights and suggestions for code changes.
inputs:
  - issue_description   # User's description of the bug/issue
  - test_command       # Required - to verify fixes
  - file_patterns      # Optional - to narrow down search
  - max_attempts: 3    # Maximum debug iterations
preprocess:
  - engine: python
    code: |
        import os
        import subprocess
        from pathlib import Path
        import difflib
        import sklearn.feature_extraction.text
        import sklearn.metrics.pairwise
        import numpy as np
        from typing import List, Dict
        import fnmatch

        TfidfVectorizer = sklearn.feature_extraction.text.TfidfVectorizer
        cosine_similarity = sklearn.metrics.pairwise.cosine_similarity

        def find_files(file_filter=None):
            default_extensions = ['.py', '.txt', '.md', '.json', '.yml', '.yaml']
            matches = []

            for root, dirnames, filenames in os.walk('.'):
                if any(ignore in root for ignore in ['.git', '__pycache__', 'node_modules', 'venv']):
                    continue

                for filename in filenames:
                    if not any(filename.endswith(ext) for ext in default_extensions):
                        continue

                    full_path = os.path.join(root, filename)

                    if not file_filter:
                        matches.append(full_path)
                        continue

                    if isinstance(file_filter, str):
                        filters = [file_filter]
                    else:
                        filters = file_filter

                    for f in filters:
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

        def run_test(test_cmd):
            try:
                result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
                return {
                    'success': result.returncode == 0,
                    'output': result.stdout,
                    'error': result.stderr
                }
            except Exception as e:
                return {'success': False, 'error': str(e)}

        def backup_file(file_path):
            backup_path = str(file_path) + '.bak'
            with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            return backup_path

        def apply_changes(file_path, new_content):
            with open(file_path, 'r') as f:
                old_content = f.read()

            backup = backup_file(file_path)

            with open(file_path, 'w') as f:
                f.write(new_content)

            diff = list(difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=str(file_path),
                tofile=str(file_path)
            ))

            return {
                'backup': backup,
                'diff': ''.join(diff)
            }

        def analyze_and_fix(issue_description, debug_history=None):
            # Find and load relevant files
            files = find_files(inputs.get('file_patterns'))
            chunks = []
            for file_path in files:
                try:
                    doc = load_document(file_path)
                    if doc and isinstance(doc, dict):
                        chunks.append(doc)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")

            relevant_chunks = find_relevant_chunks(issue_description, chunks)

            # Build prompt
            prompt = f"""Debug this issue: {issue_description}"""

            if debug_history:
                prompt += "\n\nPrevious debug attempts:\n"
                for attempt in debug_history:
                    prompt += f"\nAttempt {attempt['attempt']}:\n"
                    prompt += f"Changes: {attempt['changes']}\n"
                    prompt += f"Error: {attempt['error']}\n"

            prompt += "\nRelevant code:\n"
            for chunk, similarity in relevant_chunks:
                prompt += f"\nFile: {chunk['path']} (relevance: {similarity:.2f})\n```{chunk['ext'][1:]}\n{chunk['content']}\n```\n"

            prompt += "\n\nProvide:\n1. Analysis of the issue\n2. Specific file changes needed\n3. Expected outcome"

            # Get and apply fixes
            llm_response = get_llm_response(prompt)
            changes = parse_llm_changes(llm_response)

            changes_made = []
            for change in changes:
                result = apply_changes(change['path'], change['content'])
                changes_made.append(result)

            return changes_made, llm_response

        # Main debugging loop
        max_attempts = inputs.get('max_attempts', 3)
        attempt = 0
        debug_history = []
        success = False

        while attempt < max_attempts and not success:
            attempt += 1
            print(f"\nDebug Attempt {attempt}/{max_attempts}")

            # Analyze and fix
            changes_made, analysis = analyze_and_fix(
                inputs['issue_description'],
                debug_history if debug_history else None
            )

            # Run tests
            test_result = run_test(inputs['test_command'])

            if test_result['success']:
                success = True
                print("✅ Fix successful!")
            else:
                debug_history.append({
                    'attempt': attempt,
                    'changes': changes_made,
                    'test_result': test_result,
                    'error': test_result['error'],
                    'analysis': analysis
                })
                print(f"❌ Fix attempt {attempt} failed. Error:\n{test_result['error']}")

        # Store final results
        context['debug_results'] = {
            'success': success,
            'attempts': attempt,
            'history': debug_history,
            'final_test': test_result,
            'final_analysis': analysis
        }

prompt:
  engine: natural
  code: |
    Debug Session Summary:
    Success: {{ debug_results['success'] }}
    Attempts: {{ debug_results['attempts'] }}

    {% if debug_results['success'] %}
    ✅ Issue resolved after {{ debug_results['attempts'] }} attempts.
    {% else %}
    ❌ Failed to resolve after {{ debug_results['attempts'] }} attempts.
    {% endif %}

    Debug History:
    {% for attempt in debug_results['history'] %}
    Attempt {{ attempt.attempt }}:
    Analysis: {{ attempt.analysis }}
    Changes: {{ attempt.changes }}
    Error: {{ attempt.error }}
    {% endfor %}

postprocess:
  - engine: natural
    code: ""