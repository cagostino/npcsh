import pytest
from unittest.mock import patch, MagicMock
import os
import tempfile
import sqlite3
from modes import (
    enter_whisper_mode,
    enter_notes_mode,
    save_note,
    enter_observation_mode,
    enter_spool_mode,
    initial_table_print,
    get_data_response,
    create_new_table,
    delete_table,
    add_observation
)

@pytest.fixture
def mock_db():
    temp_db = tempfile.NamedTemporaryFile(delete=False)
    conn = sqlite3.connect(temp_db.name)
    cursor = conn.cursor()
    command_history = MagicMock()
    command_history.conn = conn
    command_history.cursor = cursor
    yield command_history
    conn.close()
    os.unlink(temp_db.name)

@pytest.mark.parametrize("user_input, expected_output", [
    (['', '/wq'], "Exiting whisper mode"),
    (['test speech', '/wq'], "User: test speech"),
])
@patch('modes.whisper.load_model')
@patch('modes.calibrate_silence')
@patch('modes.record_audio')
@patch('modes.get_llm_response')
@patch('modes.speak_text')
@patch('builtins.input')
def test_enter_whisper_mode(mock_input, mock_speak, mock_llm, mock_record, mock_calibrate, mock_load_model, user_input, expected_output, mock_db):
    mock_input.side_effect = user_input
    mock_calibrate.return_value = 100
    mock_record.return_value = b'audio_data'
    mock_llm.return_value = "LLM response"
    mock_load_model.return_value.transcribe.return_value = {"text": "test speech"}

    result = enter_whisper_mode(mock_db)
    assert expected_output in result

@patch('builtins.input')
def test_enter_notes_mode(mock_input, mock_db):
    mock_input.side_effect = ['Test note', '/nq']
    enter_notes_mode(mock_db)
    mock_db.conn.cursor().execute("SELECT * FROM notes")
    result = mock_db.conn.cursor().fetchone()
    assert result[2] == 'Test note'

def test_save_note(mock_db):
    save_note("Test note", mock_db)
    mock_db.conn.cursor().execute("SELECT * FROM notes")
    result = mock_db.conn.cursor().fetchone()
    assert result[2] == 'Test note'

@patch('modes.execute_data_operations')
@patch('builtins.input')
def test_enter_observation_mode(mock_input, mock_execute, mock_db):
    mock_input.side_effect = ['SELECT * FROM test_table', '/dq']
    mock_execute.return_value = "Data operation result"
    enter_observation_mode(mock_db)
    mock_execute.assert_called_once_with('SELECT * FROM test_table', mock_db, None)

@patch('modes.get_ollama_conversation')
@patch('builtins.input')
def test_enter_spool_mode(mock_input, mock_ollama, mock_db):
    mock_input.side_effect = ['Test input', '/sq']
    mock_ollama.return_value = [{"role": "assistant", "content": "AI response"}]
    result = enter_spool_mode(mock_db)
    assert "AI response" in result

def test_initial_table_print(capsys, mock_db):
    mock_db.cursor.fetchall.return_value = [('table1',), ('table2',)]
    initial_table_print(mock_db.cursor)
    captured = capsys.readouterr()
    assert "1. table1" in captured.out
    assert "2. table2" in captured.out

def test_get_data_response():
    npc = MagicMock()
    npc.get_data_response.return_value = "NPC data response"
    result = get_data_response("request", npc)
    assert result == "NPC data response"

@patch('builtins.input')
def test_create_new_table(mock_input, mock_db):
    mock_input.side_effect = ['new_table', 'column1, column2']
    create_new_table(mock_db.cursor, mock_db.conn)
    mock_db.cursor.execute.assert_called_with(
        "CREATE TABLE new_table (id INTEGER PRIMARY KEY AUTOINCREMENT, column1, column2)"
    )

@patch('builtins.input')
def test_delete_table(mock_input, mock_db):
    mock_input.return_value = 'table_to_delete'
    delete_table(mock_db.cursor, mock_db.conn)
    mock_db.cursor.execute.assert_called_with("DROP TABLE IF EXISTS table_to_delete")

@patch('builtins.input')
def test_add_observation(mock_input, mock_db):
    mock_db.cursor.fetchall.return_value = [(0, 'id'), (1, 'column1'), (2, 'column2')]
    mock_input.side_effect = ['value1', 'value2']
    add_observation(mock_db.cursor, mock_db.conn, 'test_table')
    mock_db.cursor.execute.assert_called_with(
        "INSERT INTO test_table (column1,column2) VALUES (?,?)",
        ['value1', 'value2']
    )

if __name__ == '__main__':
    pytest.main()