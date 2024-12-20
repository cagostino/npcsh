import pytest
import os
import sqlite3
import tempfile
from npcsh.helpers import (
    ensure_npcshrc_exists,
    add_npcshrc_to_bashrc,
    is_npcsh_initialized,
    set_npcsh_initialized,
    get_npc_from_command,
    is_valid_npc,
    get_npc_path,
    initialize_base_npcs_if_needed,
    setup_npcsh_config,
    load_all_files,
)

@pytest.fixture
def temp_home_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        old_home = os.environ.get('HOME')
        os.environ['HOME'] = tmpdirname
        yield tmpdirname
        if old_home:
            os.environ['HOME'] = old_home
        else:
            del os.environ['HOME']

@pytest.fixture
def temp_db_path():
    _, path = tempfile.mkstemp()
    yield path
    os.unlink(path)

def test_ensure_npcshrc_exists(temp_home_dir):
    npcshrc_path = ensure_npcshrc_exists()
    assert os.path.exists(npcshrc_path)
    with open(npcshrc_path, 'r') as f:
        content = f.read()
        assert "export NPCSH_INITIALIZED=0" in content
        assert "export NPCSH_PROVIDER='ollama'" in content
        assert "export NPCSH_MODEL='phi3'" in content
        assert "export NPCSH_DB_PATH='~/npcsh_history.db'" in content

def test_add_npcshrc_to_bashrc(temp_home_dir):
    bashrc_path = os.path.join(temp_home_dir, '.bashrc')
    with open(bashrc_path, 'w') as f:
        f.write("# Existing .bashrc content")
    
    add_npcshrc_to_bashrc()
    
    with open(bashrc_path, 'r') as f:
        content = f.read()
        assert '. ~/.npcshrc' in content

def test_is_npcsh_initialized():
    os.environ['NPCSH_INITIALIZED'] = '0'
    assert not is_npcsh_initialized()
    
    os.environ['NPCSH_INITIALIZED'] = '1'
    assert is_npcsh_initialized()

def test_set_npcsh_initialized(temp_home_dir):
    npcshrc_path = ensure_npcshrc_exists()
    set_npcsh_initialized()
    
    with open(npcshrc_path, 'r') as f:
        content = f.read()
        assert "export NPCSH_INITIALIZED=1" in content
    
    assert os.environ['NPCSH_INITIALIZED'] == '1'

def test_get_npc_from_command():
    assert get_npc_from_command("hello world") is None
    assert get_npc_from_command("hello npc=test world") == "test"
    assert get_npc_from_command("npc=sample command") == "sample"

def test_is_valid_npc(temp_db_path):
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE compiled_npcs (
        name TEXT PRIMARY KEY,
        source_path TEXT NOT NULL,
        compiled_path TEXT
    )
    """)
    cursor.execute("INSERT INTO compiled_npcs (name, source_path) VALUES (?, ?)", ("test_npc", "/path/to/test.npc"))
    conn.commit()
    conn.close()

    assert is_valid_npc("test_npc", temp_db_path)
    assert not is_valid_npc("non_existent_npc", temp_db_path)

def test_get_npc_path(temp_db_path):
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE compiled_npcs (
        name TEXT PRIMARY KEY,
        source_path TEXT NOT NULL,
        compiled_path TEXT
    )
    """)
    cursor.execute("INSERT INTO compiled_npcs (name, source_path, compiled_path) VALUES (?, ?, ?)", 
                   ("test_npc", "/path/to/source.npc", "/path/to/compiled.npc"))
    conn.commit()
    conn.close()

    assert get_npc_path("test_npc", temp_db_path) == "/path/to/compiled.npc"
    assert get_npc_path("non_existent_npc", temp_db_path) is None

def test_initialize_base_npcs_if_needed(temp_db_path):
    os.environ['NPCSH_INITIALIZED'] = '0'
    initialize_base_npcs_if_needed(temp_db_path)

    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM compiled_npcs")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 3  # base, bash, and data NPCs
    assert os.environ['NPCSH_INITIALIZED'] == '1'

    # Test that it doesn't reinitialize if already initialized
    initialize_base_npcs_if_needed(temp_db_path)
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM compiled_npcs")
    new_count = cursor.fetchone()[0]
    conn.close()

    assert new_count == count  # Should not have added any new NPCs

def test_setup_npcsh_config(temp_home_dir):
    setup_npcsh_config()
    npcshrc_path = os.path.expanduser("~/.npcshrc")
    assert os.path.exists(npcshrc_path)
    with open(npcshrc_path, 'r') as f:
        content = f.read()
        assert "export NPCSH_INITIALIZED=0" in content
        assert "export NPCSH_PROVIDER='ollama'" in content
        assert "export NPCSH_MODEL='llama3.2'" in content
        assert "export NPCSH_DB_PATH='~/npcsh_history.db'" in content

def test_load_all_files(temp_home_dir):
    test_dir = os.path.join(temp_home_dir, "test_dir")
    os.makedirs(test_dir)
    test_file_path = os.path.join(test_dir, "test_file.txt")
    with open(test_file_path, 'w') as f:
        f.write("This is a test file.")

    result = load_all_files(test_dir)
    assert test_file_path in result
    assert result[test_file_path] == "This is a test file."
