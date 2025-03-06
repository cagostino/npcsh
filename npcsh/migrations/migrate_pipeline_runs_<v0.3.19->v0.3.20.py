#!/usr/bin/env python3
"""
One-time migration script to add the pipeline_hash column to pipeline_runs table.
"""

import sqlite3
import os
import sys


def migrate_pipeline_runs_table(db_path):
    """Add pipeline_hash column to pipeline_runs table if it doesn't exist."""
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_runs'"
        )
        if not cursor.fetchone():
            print("Creating pipeline_runs table...")
            cursor.execute(
                "CREATE TABLE pipeline_runs ("
                "run_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "pipeline_hash TEXT, "
                "timestamp DATETIME)"
            )
            conn.commit()
            print("Table created successfully.")
            return True

        # Check if column exists
        cursor.execute("PRAGMA table_info(pipeline_runs)")
        columns = [info[1] for info in cursor.fetchall()]

        if "pipeline_hash" not in columns:
            print("Adding pipeline_hash column to pipeline_runs table...")
            cursor.execute("ALTER TABLE pipeline_runs ADD COLUMN pipeline_hash TEXT")
            conn.commit()
            print("Column added successfully.")
            return True
        else:
            print("Column pipeline_hash already exists in pipeline_runs table.")
            return True

    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    # Default path or take from command line
    default_db_path = os.path.expanduser("~/npcsh_history.db")

    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = default_db_path

    print(f"Migrating database at: {db_path}")
    success = migrate_pipeline_runs_table(db_path)

    if success:
        print("Migration completed successfully.")
        sys.exit(0)
    else:
        print("Migration failed.")
        sys.exit(1)
