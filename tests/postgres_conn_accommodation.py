import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
import sqlite3
from npcsh.npc_compiler import NPC


def setup_postgres_db():
    """Set up PostgreSQL database and test data"""
    try:
        # First connect to default 'postgres' database to create/drop our test db
        conn = psycopg2.connect(
            dbname="postgres",  # Connect to default db first
            user="caug",
            password="gobears",
            host="localhost",
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        with conn.cursor() as cur:
            # Drop and create database
            cur.execute("DROP DATABASE IF EXISTS npc_test")
            cur.execute("CREATE DATABASE npc_test")

        conn.close()

        # Now connect to our new database
        conn = psycopg2.connect(
            dbname="npc_test", user="caug", password="gobears", host="localhost"
        )

        # Create tables
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE users (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    email VARCHAR(100),
                    created_at TIMESTAMP
                )
            """
            )

            cur.execute(
                """
                CREATE TABLE posts (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    title VARCHAR(200),
                    content TEXT,
                    created_at TIMESTAMP
                )
            """
            )

            # Insert test data
            cur.execute(
                """
                INSERT INTO users (name, email, created_at) VALUES
                ('Alice', 'alice@example.com', NOW()),
                ('Bob', 'bob@example.com', NOW()),
                ('Charlie', 'charlie@example.com', NOW())
            """
            )

            cur.execute(
                """
                INSERT INTO posts (user_id, title, content, created_at) VALUES
                (1, 'First Post', 'Hello World!', NOW()),
                (1, 'Second Post', 'More content...', NOW()),
                (2, 'Bob''s Post', 'This is interesting', NOW()),
                (3, 'Welcome', 'Hi everyone!', NOW())
            """
            )

        conn.commit()
        print("PostgreSQL test database setup complete!")
        return conn

    except Exception as e:
        print(f"Error setting up PostgreSQL: {e}")
        return None


def setup_sqlite():
    """Set up SQLite database with test data"""
    # Create SQLite database file
    db_path = "test_sqlite.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)

    # Create tables
    conn.execute(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            created_at TIMESTAMP
        )
    """
    )

    conn.execute(
        """
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT,
            content TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """
    )

    # Insert test data
    conn.execute(
        """
        INSERT INTO users (name, email, created_at) VALUES
        ('Alice', 'alice@example.com', CURRENT_TIMESTAMP),
        ('Bob', 'bob@example.com', CURRENT_TIMESTAMP),
        ('Charlie', 'charlie@example.com', CURRENT_TIMESTAMP)
    """
    )

    conn.execute(
        """
        INSERT INTO posts (user_id, title, content, created_at) VALUES
        (1, 'First Post', 'Hello World!', CURRENT_TIMESTAMP),
        (1, 'Second Post', 'More content...', CURRENT_TIMESTAMP),
        (2, 'Bob''s Post', 'This is interesting', CURRENT_TIMESTAMP),
        (3, 'Welcome', 'Hi everyone!', CURRENT_TIMESTAMP)
    """
    )

    conn.commit()
    print("SQLite test database setup complete!")
    return conn


def test_database_setup():
    """Test both database setups with the NPC class"""
    # Test SQLite
    sqlite_conn = setup_sqlite()
    sqlite_npc = NPC(
        name="SQLiteAnalyst",
        primary_directive="Analyze SQLite database",
        model="gpt-4o-mini",
        db_conn=sqlite_conn,
    )

    # Test PostgreSQL
    postgres_conn = setup_postgres_db()
    postgres_npc = NPC(
        name="PostgresAnalyst",
        primary_directive="Analyze PostgreSQL database",
        model="gpt-4o-mini",
        db_conn=postgres_conn,
    )

    # Test queries
    test_queries = [
        "How many users do we have?",
        "Show me all posts with their authors",
        "Count posts per user",
        "Find users with more than one post",
    ]

    print("\nTesting SQLite:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = sqlite_npc.analyze_db_data(query)
        print("Result:", result)

    print("\nTesting PostgreSQL:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = postgres_npc.analyze_db_data(query)
        print("Result:", result)

    # Cleanup
    sqlite_conn.close()
    postgres_conn.close()
    if os.path.exists("test_sqlite.db"):
        os.remove("test_sqlite.db")


if __name__ == "__main__":
    test_database_setup()
