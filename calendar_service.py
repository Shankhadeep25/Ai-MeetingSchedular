"""
calendar_service.py — Google Calendar Integration
=================================================
Handles all Google Calendar API operations using free personal OAuth2.

Cost: $0 — Google Calendar API is free for personal OAuth2 usage.
Quota: 1,000,000 queries/day (standard free tier).

First run: Opens browser for OAuth consent → saves token.json locally.
Subsequent runs: Uses cached token.json (auto-refreshes on expiry).
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Scopes required: read/write calendar events
SCOPES = ["https://www.googleapis.com/auth/calendar"]

CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "./credentials.json")
TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", "./token.json")
DEFAULT_TZ = os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata")


# ─────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────

def authenticate():
    """
    Authenticate with Google Calendar API using OAuth2.

    Flow:
    1. If token.json exists and is valid → use it directly
    2. If token.json is expired → auto-refresh using refresh_token
    3. If neither → open browser for user consent → save token.json

    Returns:
        Authenticated Google API service object
    """
    creds = None

    # Load existing token
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    # Refresh or re-authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired Google OAuth token...")
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDENTIALS_PATH}.\n"
                    "Download it from Google Cloud Console:\n"
                    "  APIs & Services → Credentials → OAuth 2.0 Client IDs → Download JSON\n"
                    "  Rename to credentials.json and place in the project root."
                )
            logger.info("Opening browser for Google OAuth2 consent...")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save token for future runs
        with open(TOKEN_PATH, "w") as token_file:
            token_file.write(creds.to_json())
        logger.info(f"Token saved to {TOKEN_PATH}")

    service = build("calendar", "v3", credentials=creds)
    logger.info("Google Calendar API authenticated successfully")
    return service


# ─────────────────────────────────────────────
# Availability Check
# ─────────────────────────────────────────────

def check_availability(start: datetime, end: datetime) -> bool:
    """
    Check if the primary calendar is free during the given time window.

    Args:
        start: Start datetime (timezone-aware)
        end:   End datetime (timezone-aware)

    Returns:
        True if the slot is free, False if busy
    """
    service = authenticate()

    body = {
        "timeMin": start.isoformat(),
        "timeMax": end.isoformat(),
        "items": [{"id": "primary"}],
    }

    try:
        result = service.freebusy().query(body=body).execute()
        busy_periods = result["calendars"]["primary"]["busy"]

        if busy_periods:
            logger.info(f"Slot {start} – {end} is BUSY: {busy_periods}")
            return False

        logger.info(f"Slot {start} – {end} is FREE")
        return True

    except HttpError as e:
        logger.error(f"Google Calendar freebusy query failed: {e}")
        raise


# ─────────────────────────────────────────────
# Create Event
# ─────────────────────────────────────────────

def create_event(
    summary: str,
    start: datetime,
    end: datetime,
    attendees: list[str] | None = None,
    description: str = "",
    timezone: str = DEFAULT_TZ,
) -> str:
    """
    Create a Google Calendar event on the primary calendar.

    Args:
        summary:    Event title
        start:      Start datetime (timezone-aware)
        end:        End datetime (timezone-aware)
        attendees:  List of attendee email addresses
        description: Optional event description
        timezone:   IANA timezone string for display

    Returns:
        HTML link to the created calendar event
    """
    service = authenticate()

    event_body: dict = {
        "summary": summary,
        "description": description,
        "start": {
            "dateTime": start.isoformat(),
            "timeZone": timezone,
        },
        "end": {
            "dateTime": end.isoformat(),
            "timeZone": timezone,
        },
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email", "minutes": 30},
                {"method": "popup", "minutes": 10},
            ],
        },
    }

    # Add attendees if provided (must be email addresses)
    if attendees:
        valid_attendees = [
            {"email": a} for a in attendees if "@" in a
        ]
        if valid_attendees:
            event_body["attendees"] = valid_attendees

    try:
        event = (
            service.events()
            .insert(calendarId="primary", body=event_body, sendUpdates="all")
            .execute()
        )
        event_link = event.get("htmlLink", "")
        logger.info(f"Event created: {summary!r} → {event_link}")
        return event_link

    except HttpError as e:
        logger.error(f"Failed to create calendar event: {e}")
        raise


# ─────────────────────────────────────────────
# Search & Delete Events
# ─────────────────────────────────────────────

def search_events(
    query: str = "",
    time_min: Optional[datetime] = None,
    time_max: Optional[datetime] = None,
    max_results: int = 10,
) -> list[dict]:
    """
    Search for events on the primary calendar by text query or time window.

    Args:
        query: Search term (searches summary, description, attendee names/emails)
        time_min: Optional start of search window
        time_max: Optional end of search window
        max_results: Max events to return

    Returns:
        List of Google Calendar event dicts
    """
    service = authenticate()
    try:
        kwargs = {
            "calendarId": "primary",
            "maxResults": max_results,
            "singleEvents": True,
            "orderBy": "startTime",
        }
        if query:
            kwargs["q"] = query
        if time_min:
            kwargs["timeMin"] = time_min.isoformat()
        if time_max:
            kwargs["timeMax"] = time_max.isoformat()

        result = service.events().list(**kwargs).execute()
        return result.get("items", [])
    except HttpError as e:
        logger.error(f"Failed to search events: {e}")
        return []


def delete_event(event_id: str) -> bool:
    """
    Delete an event from Google Calendar by its event ID.

    Args:
        event_id: Google Calendar event ID string

    Returns:
        True if deleted successfully, False otherwise
    """
    service = authenticate()
    try:
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        logger.info(f"Successfully deleted event {event_id} from Google Calendar")
        return True
    except HttpError as e:
        logger.error(f"Failed to delete event {event_id}: {e}")
        return False


# ─────────────────────────────────────────────
# Find Alternative Slots
# ─────────────────────────────────────────────

def find_alternative_slots(
    preferred_start: datetime,
    duration_minutes: int = 30,
    days_to_search: int = 3,
    max_slots: int = 3,
    working_hours: tuple[int, int] = (9, 18),  # 9am – 6pm
) -> list[dict]:
    """
    Find available time slots near the requested time.

    Strategy:
    1. Try same day in 1-hour increments after the busy slot
    2. If none found, check the next N days during working hours

    Args:
        preferred_start:   The originally requested start time
        duration_minutes:  Meeting length
        days_to_search:    How many days ahead to look
        max_slots:         Maximum alternatives to return
        working_hours:     (start_hour, end_hour) for working day bounds

    Returns:
        List of dicts: [{"start": datetime, "end": datetime, "label": str}]
    """
    service = authenticate()  # noqa — warm up auth
    slots = []
    duration = timedelta(minutes=duration_minutes)
    work_start, work_end = working_hours

    # Build search range: preferred_start to preferred_start + days_to_search
    search_start = preferred_start
    search_end_bound = preferred_start + timedelta(days=days_to_search)

    # Query all busy periods in the search window at once (more efficient)
    body = {
        "timeMin": search_start.isoformat(),
        "timeMax": search_end_bound.isoformat(),
        "items": [{"id": "primary"}],
    }

    try:
        result = authenticate().freebusy().query(body=body).execute()
        busy_periods = result["calendars"]["primary"]["busy"]
    except HttpError as e:
        logger.error(f"freebusy query failed during slot search: {e}")
        return []

    # Convert busy periods to datetime objects
    busy_ranges = [
        (
            datetime.fromisoformat(b["start"]),
            datetime.fromisoformat(b["end"]),
        )
        for b in busy_periods
    ]

    # Probe candidate slots in 30-minute increments
    probe = search_start
    while probe < search_end_bound and len(slots) < max_slots:
        # Skip outside working hours
        if not (work_start <= probe.hour < work_end):
            # Jump to next working hour start
            probe = probe.replace(
                hour=work_start, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            continue

        candidate_end = probe + duration

        # Check for overlap with any busy period
        is_free = not any(
            b_start < candidate_end and b_end > probe
            for b_start, b_end in busy_ranges
        )

        if is_free:
            slots.append({
                "start": probe,
                "end": candidate_end,
                "label": probe.strftime("%A %b %d at %-I:%M %p"),
            })

        probe += timedelta(minutes=30)

    logger.info(f"Found {len(slots)} alternative slots")
    return slots


# ─────────────────────────────────────────────
# List Upcoming Events (for sidebar)
# ─────────────────────────────────────────────

def list_upcoming_events(days: int = 7, max_results: int = 10) -> list[dict]:
    """
    Fetch upcoming calendar events for the next N days.

    Args:
        days:        How many days ahead to look
        max_results: Maximum events to return

    Returns:
        List of event dicts with summary, start, end, htmlLink
    """
    service = authenticate()

    now = datetime.now(tz=timezone.utc)
    time_max = now + timedelta(days=days)

    try:
        result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=time_max.isoformat(),
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = result.get("items", [])
        logger.info(f"Fetched {len(events)} upcoming events")
        return events

    except HttpError as e:
        logger.error(f"Failed to list upcoming events: {e}")
        return []


# ─────────────────────────────────────────────
# CLI Test (run: python calendar_service.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 60)
    print("AI Meeting Scheduler — Calendar Service Test")
    print("=" * 60 + "\n")

    print("Step 1: Authenticating (may open browser)...")
    try:
        svc = authenticate()
        print("✅ Authentication successful!\n")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    print("Step 2: Listing upcoming events...")
    events = list_upcoming_events(days=7)
    if events:
        for ev in events:
            start = ev["start"].get("dateTime", ev["start"].get("date"))
            print(f"  📅 {ev.get('summary', 'No title')} — {start}")
    else:
        print("  (No upcoming events found)")

    print("\n✅ Calendar service is working correctly!")
    print("You can now create events using create_event()")
