"""
tests/test_memory_service.py
============================
Tests for memory_service:
- Database schema initialization
- Meeting logging and history retrieval
- Preference read/update
- Automatic preference inference
"""

import os
import unittest
import tempfile
from datetime import datetime, timedelta

# Create a temporary database for isolated testing
test_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["SQLITE_DB_PATH"] = test_db_file.name

import memory_service


class TestMemoryService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        memory_service.init_db()

    @classmethod
    def tearDownClass(cls):
        try:
            if os.path.exists(test_db_file.name):
                os.remove(test_db_file.name)
        except Exception:
            pass

    def test_user_creation_and_preferences(self):
        user_id = "test_user_alpha"
        memory_service.get_or_create_user(user_id, name="Alpha", timezone="UTC")

        # Test defaults
        prefs = memory_service.get_preferences(user_id)
        self.assertIn("preferred_duration", prefs)

        # Update preference
        memory_service.update_preference(user_id, "preferred_duration", 45)
        updated = memory_service.get_preferences(user_id)
        self.assertEqual(updated["preferred_duration"], "45")

    def test_log_and_retrieve_meetings(self):
        user_id = "test_user_beta"
        now = datetime.utcnow()
        meeting_id = memory_service.log_meeting(
            user_id=user_id,
            title="Design Review",
            participants=["Alice", "Bob"],
            start_time=now,
            end_time=now + timedelta(minutes=45),
            duration_mins=45,
            event_link="https://meet.google.com/xyz",
            raw_user_input="Let's have a 45 min design review",
        )

        self.assertIsInstance(meeting_id, int)
        history = memory_service.get_meeting_history(user_id, limit=5)
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[0]["title"], "Design Review")
        self.assertIn("Alice", history[0]["participants"])

    def test_preference_inference(self):
        user_id = "test_user_gamma"
        base_time = datetime(2026, 9, 20, 14, 0)  # 2pm

        # Log 3 meetings with 45 mins at 2pm
        for i in range(3):
            memory_service.log_meeting(
                user_id=user_id,
                title=f"Sync {i}",
                participants=["Dev"],
                start_time=base_time + timedelta(days=i),
                end_time=base_time + timedelta(days=i, minutes=45),
                duration_mins=45,
            )

        inferred = memory_service.infer_preferences(user_id)
        self.assertEqual(inferred.get("preferred_duration"), 45)
        self.assertEqual(inferred.get("preferred_start_hour"), 14)


if __name__ == "__main__":
    unittest.main()
