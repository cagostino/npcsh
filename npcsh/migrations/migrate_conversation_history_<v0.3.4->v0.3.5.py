#!/usr/bin/env python3
import os
import sqlite3
import uuid
import shutil
from datetime import datetime


def migrate_command_history_db(db_path="~/npcsh_history.db"):
    """Standalone script to migrate the database schema to support attachments."""
    db_path = os.path.expanduser(db_path)

    # Create backup
    backup_path = f"{db_path}.backup.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    print(f"Creating backup at: {backup_path}")
    shutil.copy2(db_path, backup_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        print("Starting migration...")
        # Start transaction
        conn.execute("BEGIN TRANSACTION")

        # Check existing table structure
        cursor.execute("PRAGMA table_info(conversation_history)")
        columns = [info[1] for info in cursor.fetchall()]

        if "message_id" not in columns:
            print(
                "Adding message_id and additional columns to conversation_history table..."
            )

            # Create temporary table with new schema
            cursor.execute(
                """
                CREATE TABLE conversation_history_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT UNIQUE,
                    timestamp TEXT,
                    role TEXT,
                    content TEXT,
                    conversation_id TEXT,
                    directory_path TEXT,
                    model TEXT,
                    provider TEXT,
                    npc TEXT
                )
            """
            )

            # Generate UUIDs for each existing record
            print("Migrating existing conversation data...")
            cursor.execute("SELECT * FROM conversation_history")
            rows = cursor.fetchall()

            # Prepare INSERT statements for the new table
            for row in rows:
                # Generate a UUID for message_id
                message_id = str(uuid.uuid4())

                # Handle variable column counts in existing data
                if len(row) >= 6:
                    id, timestamp, role, content, conversation_id, directory_path = row[
                        :6
                    ]
                    cursor.execute(
                        """
                        INSERT INTO conversation_history_new
                        (id, message_id, timestamp, role, content, conversation_id, directory_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            id,
                            message_id,
                            timestamp,
                            role,
                            content,
                            conversation_id,
                            directory_path,
                        ),
                    )
                else:
                    print(f"Warning: Row has unexpected format: {row}")

            # Drop old table and rename new one
            cursor.execute("DROP TABLE conversation_history")
            cursor.execute(
                "ALTER TABLE conversation_history_new RENAME TO conversation_history"
            )
            print("Conversation table migration completed.")

        # Create attachments table if it doesn't exist
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS message_attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT,
                attachment_name TEXT,
                attachment_type TEXT,
                attachment_data BLOB,
                attachment_size INTEGER,
                upload_timestamp TEXT,
                FOREIGN KEY (message_id) REFERENCES conversation_history(message_id)
            )
        """
        )
        print("Attachment table created.")

        # Commit all changes
        conn.execute("COMMIT")
        print("Migration completed successfully!")

    except sqlite3.Error as e:
        conn.execute("ROLLBACK")
        print(f"Migration failed: {str(e)}")
        print("Database restored to original state.")

    finally:
        conn.close()


if __name__ == "__main__":
    migrate_command_history_db()
    print("Done. Run this script only once.")
