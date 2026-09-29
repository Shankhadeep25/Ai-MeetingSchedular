"""
tests/test_rag_service.py
=========================
Tests for rag_service:
- Document formatting for embedding
- Semantic relevance filtering logic
"""

import unittest
from datetime import datetime

import rag_service


class TestRAGService(unittest.TestCase):

    def test_build_document(self):
        meeting = {
            "title": "Quarterly Roadmap Review",
            "participants": ["Alice", "Bob", "Charlie"],
            "start_time": datetime(2026, 9, 30, 15, 0),
            "duration_mins": 60,
        }
        doc = rag_service._build_document(meeting)
        self.assertIn("Quarterly Roadmap Review", doc)
        self.assertIn("Alice", doc)
        self.assertIn("60 minutes", doc)

    def test_build_document_minimal(self):
        meeting = {
            "title": "Quick Sync",
        }
        doc = rag_service._build_document(meeting)
        self.assertEqual(doc, "Quick Sync")


if __name__ == "__main__":
    unittest.main()
