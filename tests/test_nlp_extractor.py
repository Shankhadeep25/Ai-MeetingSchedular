"""
tests/test_nlp_extractor.py
===========================
Tests for nlp_extractor module:
- Schema validation
- Date resolution logic
- Ambiguity detection
- Groq tool schema integrity
"""

import unittest
from datetime import datetime
from pydantic import ValidationError

from nlp_extractor import (
    MeetingRequest,
    resolve_date,
    MEETING_TOOL_SCHEMA,
)


class TestNLPExtractor(unittest.TestCase):

    def test_meeting_request_valid(self):
        """Test valid MeetingRequest creation."""
        req = MeetingRequest(
            intent="schedule",
            title="Team Sync",
            participants=["Alice", "Bob"],
            date_phrase="next Monday at 10am",
            duration_minutes=45,
            is_ambiguous=False,
        )
        self.assertEqual(req.intent, "schedule")
        self.assertEqual(req.duration_minutes, 45)
        self.assertFalse(req.is_ambiguous)

    def test_meeting_request_invalid_intent(self):
        """Test that invalid intent raises ValidationError."""
        with self.assertRaises(ValidationError):
            MeetingRequest(
                intent="invalid_intent",  # Should fail regex/enum
                is_ambiguous=False,
            )

    def test_resolve_date_valid_phrases(self):
        """Test offline dateparser resolution for relative phrases."""
        # Tomorrow
        res_tomorrow = resolve_date("tomorrow at 3pm", timezone="UTC")
        self.assertIsNotNone(res_tomorrow)
        self.assertEqual(res_tomorrow.hour, 15)

        # Empty/whitespace phrase
        self.assertIsNone(resolve_date(""))
        self.assertIsNone(resolve_date("   "))

    def test_tool_schema_structure(self):
        """Verify the Groq tool schema meets OpenAI/Groq function calling format."""
        self.assertEqual(MEETING_TOOL_SCHEMA["type"], "function")
        self.assertIn("name", MEETING_TOOL_SCHEMA["function"])
        self.assertEqual(MEETING_TOOL_SCHEMA["function"]["name"], "extract_meeting_request")
        params = MEETING_TOOL_SCHEMA["function"]["parameters"]
        self.assertIn("intent", params["properties"])
        self.assertIn("is_ambiguous", params["required"])


if __name__ == "__main__":
    unittest.main()
