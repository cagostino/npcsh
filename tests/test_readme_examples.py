"""Run README.md Python examples as pytest tests."""

import re
from pathlib import Path

import pytest


README_PATH = Path(__file__).resolve().parents[1] / "README.md"


def _execute_block(code):
    namespace = {}
    exec(code, namespace)


def _load_blocks():
    if not README_PATH.exists():
        return []
    readme = README_PATH.read_text()
    blocks = re.findall(r"```python\n(.*?)```", readme, re.DOTALL)
    return [block.strip() for block in blocks if block.strip()]


def _normalize(code):
    code = re.sub(
        r"model\s*=\s*['\"][^'\"]+['\"]",
        "model='kimi-k2.7-code:cloud'",
        code,
    )
    code = re.sub(
        r"provider\s*=\s*['\"][^'\"]+['\"]",
        "provider='ollama'",
        code,
    )
    return code


def _build_tests():
    blocks = _load_blocks()
    count = 0
    for idx, block in enumerate(blocks, 1):
        code = _normalize(block)
        try:
            compile(code, f"<readme_example_{idx}>", "exec")
        except SyntaxError:
            continue

        def make_test(captured_code=code, captured_index=idx):
            def test_readme_example():
                _execute_block(captured_code)
            test_readme_example.__name__ = f"test_readme_example_{captured_index}"
            return test_readme_example

        count += 1
        globals()[f"test_readme_example_{idx}"] = make_test()
    return count


README_EXAMPLE_COUNT = _build_tests()
