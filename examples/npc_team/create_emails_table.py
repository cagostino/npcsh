#!/usr/bin/env python3
"""
Script to create and populate the emails table needed by the morning_routine pipeline.
"""

import sqlite3
import os
import sys
import json
from datetime import datetime, timedelta


def create_emails_table(db_path):
    """Create and populate the emails table for the morning_routine pipeline."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='emails'"
        )
        if cursor.fetchone():
            print("Table 'emails' already exists.")
            return True

        # Create the emails table
        print("Creating 'emails' table...")
        cursor.execute(
            """
        CREATE TABLE emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT,
            recipient TEXT,
            subject TEXT,
            content TEXT,
            timestamp DATETIME,
            is_read BOOLEAN DEFAULT 0,
            priority INTEGER DEFAULT 0
        )
        """
        )

        # Add some sample data
        print("Adding sample email data...")
        now = datetime.now()
        sample_emails = [
            {
                "sender": "boss@company.com",
                "recipient": "user@company.com",
                "subject": "Weekly Progress Report",
                "content": "Please send me your weekly progress report by end of day.",
                "timestamp": (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                "is_read": 0,
                "priority": 2,
            },
            {
                "sender": "colleague@company.com",
                "recipient": "user@company.com",
                "subject": "Project Update",
                "content": "Here's the latest update on the project we're working on.",
                "timestamp": (now - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
                "is_read": 1,
                "priority": 1,
            },
            {
                "sender": "newsletter@tech.com",
                "recipient": "user@company.com",
                "subject": "Daily Tech News",
                "content": "Here are today's top tech stories...",
                "timestamp": (now - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
                "is_read": 0,
                "priority": 0,
            },
        ]

        for email in sample_emails:
            cursor.execute(
                """
            INSERT INTO emails (sender, recipient, subject, content, timestamp, is_read, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    email["sender"],
                    email["recipient"],
                    email["subject"],
                    email["content"],
                    email["timestamp"],
                    email["is_read"],
                    email["priority"],
                ),
            )

        conn.commit()
        print("Email table created and populated successfully.")
        return True

    except Exception as e:
        print(f"Error creating emails table: {e}")
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

    print(f"Setting up emails table in database at: {db_path}")
    success = create_emails_table(db_path)

    if success:
        print("Setup completed successfully.")
        sys.exit(0)
    else:
        print("Setup failed.")
        sys.exit(1)
