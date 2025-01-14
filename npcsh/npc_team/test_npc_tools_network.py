import sqlite3
import os
import yaml
from jinja2 import Environment, FileSystemLoader, Template, Undefined
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Set
from collections import defaultdict, deque
from npcsh.npc_compiler import NPC, Tool, NPCCompiler, load_npc_from_file

import sqlite3
import unittest


db_path = os.path.expanduser("~/npcsh_history.db")
db_conn = sqlite3.connect(db_path)


class TestNPCFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up a test SQLite database in memory
        cls.db_path = ":memory:"
        cls.conn = sqlite3.connect(cls.db_path)
        cls.create_test_tables()

        # Create test NPCs
        cls.npcs = cls.create_test_npcs()

    @classmethod
    def create_test_tables(cls):
        cursor = cls.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback TEXT,
                user_preference TEXT
            )
            """
        )
        cls.conn.commit()

    @classmethod
    def create_test_npcs(cls):
        # Create three test NPCs with A/B testing focus

        npc1 = load_npc_from_file("./npc_team/datacollector.npc", db_conn)
        npc2 = load_npc_from_file("./npc_team/analyzer.npc", db_conn)
        npc3 = load_npc_from_file("./npc_team/presenter.npc", db_conn)
        return [npc1, npc2, npc3]

    def test_npc_tool_loading(self):
        # Assert that default tools are loaded correctly
        for npc in self.npcs:
            self.assertGreater(
                len(npc.tools), 0, f"{npc.name} should have tools loaded."
            )

    def test_npc_responses(self):
        # Simulate interaction to verify expected responses
        feedbacks = [
            ("Feedback 1: I like the new layout!", "Positive"),
            ("Feedback 2: It's too cluttered.", "Negative"),
            ("Feedback 3: The color scheme needs to change.", "Neutral"),
        ]

        cursor = self.conn.cursor()
        cursor.executemany(
            "INSERT INTO user_feedback (feedback, user_preference) VALUES (?, ?);",
            feedbacks,
        )
        self.conn.commit()

        # Test the DataCollector NPC
        responses = []
        for feedback in feedbacks:
            response = self.npcs[0].get_llm_response(
                f"Gather feedback for: {feedback[0]}"
            )
            responses.append(response)

        self.assertTrue(
            all(r is not None for r in responses), "All responses must be valid"
        )

    def test_npc_interactions(self):
        # Assert that NPCs work together as expected
        feedback = "User prefers a minimalistic design with easy navigation."

        # Collect feedback using DataCollector
        self.npcs[0].get_llm_response(f"User Feedback: {feedback}")

        # Analyze the feedback using Analyzer
        analysis = self.npcs[1].get_llm_response(f"Analyze feedback: {feedback}")
        self.assertIsNotNone(analysis, "Analysis should return a valid result")

        # Create a presentation using Presenter
        presentation = self.npcs[2].get_llm_response(
            f"Create presentation from analysis: {analysis}"
        )
        self.assertIsNotNone(presentation, "Presentation should return a valid result")


if __name__ == "__main__":
    unittest.main()
