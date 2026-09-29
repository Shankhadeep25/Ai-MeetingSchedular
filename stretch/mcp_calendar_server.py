"""
stretch/mcp_calendar_server.py — Model Context Protocol (MCP) Calendar Server
=============================================================================
Exposes calendar services as standard Model Context Protocol (MCP) tools.
Allows external AI clients (Claude Desktop, Cursor, Antigravity) to call
calendar operations as standardized MCP functions.

Protocol: Model Context Protocol (JSON-RPC over stdio or SSE)
"""

import sys
import os
import json
from datetime import datetime, timedelta

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from mcp.server.fastmcp import FastMCP
    mcp_available = True
except ImportError:
    mcp_available = False

if mcp_available:
    # Initialize FastMCP Server
    mcp = FastMCP("calendar-scheduler")

    from calendar_service import (
        check_availability,
        create_event,
        find_alternative_slots,
        list_upcoming_events,
    )

    @mcp.tool()
    def check_calendar_availability(start_iso: str, end_iso: str) -> dict:
        """
        Check if the user's primary calendar is free between start and end times.
        Args:
            start_iso: ISO 8601 string (e.g. '2026-09-30T10:00:00+05:30')
            end_iso:   ISO 8601 string (e.g. '2026-09-30T10:30:00+05:30')
        """
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        is_free = check_availability(start, end)
        return {"is_free": is_free, "start": start_iso, "end": end_iso}

    @mcp.tool()
    def book_calendar_event(
        summary: str,
        start_iso: str,
        end_iso: str,
        attendees: list[str] = None,
        description: str = "",
    ) -> dict:
        """
        Create a meeting event on the user's Google Calendar.
        """
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        link = create_event(
            summary=summary,
            start=start,
            end=end,
            attendees=attendees or [],
            description=description,
        )
        return {"status": "success", "event_link": link, "summary": summary}

    @mcp.tool()
    def find_free_slots(start_iso: str, duration_minutes: int = 30) -> list[dict]:
        """
        Find alternative free time slots starting from a requested time.
        """
        start = datetime.fromisoformat(start_iso)
        slots = find_alternative_slots(
            preferred_start=start,
            duration_minutes=duration_minutes,
            days_to_search=3,
            max_slots=3,
        )
        return [
            {"start": s["start"].isoformat(), "end": s["end"].isoformat(), "label": s["label"]}
            for s in slots
        ]

    @mcp.tool()
    def get_upcoming_schedule(days: int = 7) -> list[dict]:
        """
        Fetch upcoming scheduled events from Google Calendar.
        """
        events = list_upcoming_events(days=days)
        return [
            {
                "summary": ev.get("summary", "Untitled"),
                "start": ev["start"].get("dateTime", ev["start"].get("date")),
                "link": ev.get("htmlLink", ""),
            }
            for ev in events
        ]

    if __name__ == "__main__":
        mcp.run()

else:
    def main():
        print("MCP package is not installed.")
        print("To run the MCP calendar server, install: pip install 'mcp[cli]'")
        print("Usage: python stretch/mcp_calendar_server.py")

    if __name__ == "__main__":
        main()
