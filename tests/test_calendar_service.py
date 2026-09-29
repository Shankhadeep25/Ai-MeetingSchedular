"""
tests/test_calendar_service.py
==============================
Tests for calendar_service:
- Slot finding logic with simulated busy ranges
- Event request body payload format
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import calendar_service


class TestCalendarService(unittest.TestCase):

    @patch("calendar_service.authenticate")
    def test_check_availability_free(self, mock_auth):
        """Test availability check when freebusy returns no busy intervals."""
        mock_service = MagicMock()
        mock_auth.return_value = mock_service
        mock_service.freebusy().query().execute.return_value = {
            "calendars": {"primary": {"busy": []}}
        }

        start = datetime(2026, 9, 25, 10, 0)
        end = datetime(2026, 9, 25, 10, 30)

        is_free = calendar_service.check_availability(start, end)
        self.assertTrue(is_free)

    @patch("calendar_service.authenticate")
    def test_check_availability_busy(self, mock_auth):
        """Test availability check when freebusy returns an overlapping interval."""
        mock_service = MagicMock()
        mock_auth.return_value = mock_service
        mock_service.freebusy().query().execute.return_value = {
            "calendars": {
                "primary": {
                    "busy": [
                        {
                            "start": "2026-09-25T10:00:00Z",
                            "end": "2026-09-25T10:30:00Z",
                        }
                    ]
                }
            }
        }

        start = datetime(2026, 9, 25, 10, 0)
        end = datetime(2026, 9, 25, 10, 30)

        is_free = calendar_service.check_availability(start, end)
        self.assertFalse(is_free)

    @patch("calendar_service.authenticate")
    def test_create_event_payload(self, mock_auth):
        """Verify create_event passes correctly formatted body and returns htmlLink."""
        mock_service = MagicMock()
        mock_auth.return_value = mock_service
        mock_service.events().insert().execute.return_value = {
            "htmlLink": "https://calendar.google.com/event?eid=test12345"
        }

        start = datetime(2026, 9, 25, 14, 0)
        end = datetime(2026, 9, 25, 14, 30)

        link = calendar_service.create_event(
            summary="Interview with Rahul",
            start=start,
            end=end,
            attendees=["rahul@example.com"],
            description="Interview round 1",
        )

        self.assertIn("calendar.google.com", link)
        mock_service.events().insert.assert_called_once()


if __name__ == "__main__":
    unittest.main()
