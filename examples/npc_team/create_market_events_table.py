#!/usr/bin/env python3
"""
Script to create and populate the market_events table needed by the NPC profile compiler.
"""

import sqlite3
import os
import sys
import json
from datetime import datetime, timedelta


def create_market_events_table(db_path):
    """Create and populate the market_events table for the NPC profile compiler."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='market_events'"
        )
        if cursor.fetchone():
            print("Table 'market_events' already exists.")
            return True

        # Create the market_events table
        print("Creating 'market_events' table...")
        cursor.execute(
            """
        CREATE TABLE market_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            description TEXT,
            impact_level INTEGER,
            timestamp DATETIME,
            market_sector TEXT,
            price_change REAL,
            volume INTEGER,
            is_processed BOOLEAN DEFAULT 0
        )
        """
        )

        # Add some sample data
        print("Adding sample market event data...")
        now = datetime.now()
        sample_events = [
            {
                "event_type": "earnings_report",
                "description": "XYZ Corp reports quarterly earnings above expectations",
                "impact_level": 3,
                "timestamp": (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "market_sector": "technology",
                "price_change": 2.45,
                "volume": 1250000,
                "is_processed": 1,
            },
            {
                "event_type": "market_crash",
                "description": "Stock market drops 5% on inflation concerns",
                "impact_level": 5,
                "timestamp": (now - timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S"),
                "market_sector": "global",
                "price_change": -5.12,
                "volume": 8500000,
                "is_processed": 1,
            },
            {
                "event_type": "merger_announcement",
                "description": "ABC Inc announces acquisition of DEF Corp",
                "impact_level": 4,
                "timestamp": (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
                "market_sector": "healthcare",
                "price_change": 3.75,
                "volume": 3200000,
                "is_processed": 0,
            },
            {
                "event_type": "policy_change",
                "description": "Central bank raises interest rates by 0.25%",
                "impact_level": 4,
                "timestamp": (now - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"),
                "market_sector": "finance",
                "price_change": -1.20,
                "volume": 4800000,
                "is_processed": 0,
            },
            {
                "event_type": "product_launch",
                "description": "New smartphone model released with innovative features",
                "impact_level": 2,
                "timestamp": (now - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S"),
                "market_sector": "consumer_electronics",
                "price_change": 1.85,
                "volume": 2100000,
                "is_processed": 1,
            },
        ]

        for event in sample_events:
            cursor.execute(
                """
            INSERT INTO market_events (event_type, description, impact_level, timestamp, market_sector, price_change, volume, is_processed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event["event_type"],
                    event["description"],
                    event["impact_level"],
                    event["timestamp"],
                    event["market_sector"],
                    event["price_change"],
                    event["volume"],
                    event["is_processed"],
                ),
            )

        conn.commit()
        print("Market events table created and populated successfully.")
        return True

    except Exception as e:
        print(f"Error creating market_events table: {e}")
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

    print(f"Setting up market_events table in database at: {db_path}")
    success = create_market_events_table(db_path)

    if success:
        print("Setup completed successfully.")
        sys.exit(0)
    else:
        print("Setup failed.")
        sys.exit(1)
