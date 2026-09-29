"""
agent.py — Orchestration Engine
================================
Central brain of the AI Meeting Scheduler.

Coordinates all modules through the full agentic loop:
  extract → clarify? → RAG context → preferences → availability → book → log

Maintains conversation state via Streamlit session_state.
Also provides a standalone terminal interface for testing (Phase 3).
"""

import os
import logging
from datetime import timedelta
from typing import Optional

from dotenv import load_dotenv

from nlp_extractor import MeetingRequest, extract_meeting_info
from calendar_service import (
    check_availability,
    create_event,
    find_alternative_slots,
    search_events,
    delete_event,
    is_calendar_configured,
)
from memory_service import (
    log_meeting,
    get_preferences,
    infer_preferences,
    get_or_create_user,
    get_upcoming_meetings,
    delete_meeting_record,
)
from rag_service import retrieve_context, index_meeting

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_TZ = os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata")


# ─────────────────────────────────────────────
# Agent State (used by Streamlit session_state)
# ─────────────────────────────────────────────

def get_initial_state() -> dict:
    """Return a fresh agent state dict (for new sessions)."""
    return {
        "messages": [],                 # Full conversation history [{role, content}]
        "pending_request": None,        # Partial MeetingRequest awaiting clarification
        "user_id": "user_default",      # Session user identifier
        "awaiting_slot_selection": False,  # True when conflict alternatives are presented
        "alternative_slots": [],         # List of alternative slot dicts
        "conversation_history": [],      # Groq-format history for multi-turn NLP
    }


# ─────────────────────────────────────────────
# Core Agent Logic
# ─────────────────────────────────────────────

def handle_message(user_text: str, state: dict) -> tuple[str, dict]:
    """
    Process a user message through the full agentic pipeline.

    Args:
        user_text: Raw user input
        state:     Agent state dict (mutated in-place)

    Returns:
        (response_text, updated_state)
    """
    user_id = state.get("user_id", "user_default")
    get_or_create_user(user_id)

    # ── Branch: User is selecting from alternative slots ──────────────────
    if state.get("awaiting_slot_selection") and state.get("alternative_slots"):
        return _handle_slot_selection(user_text, state)

    # ── Branch: User is clarifying an ambiguous request ───────────────────
    if state.get("pending_request") and state["pending_request"].is_ambiguous:
        return _handle_clarification(user_text, state)

    # ── Main Branch: Fresh meeting request ────────────────────────────────
    return _handle_new_request(user_text, state)


def _handle_new_request(user_text: str, state: dict) -> tuple[str, dict]:
    """Process a new (or initial) meeting request."""
    user_id = state["user_id"]

    # Step 1: NLP extraction via Groq
    prefs = get_preferences(user_id)
    tz = prefs.get("timezone", DEFAULT_TZ)

    meeting: MeetingRequest = extract_meeting_info(
        user_text,
        timezone=tz,
        conversation_history=state.get("conversation_history", []),
    )

    # Update Groq conversation history for multi-turn context
    state.setdefault("conversation_history", []).append(
        {"role": "user", "content": user_text}
    )

    # Handle non-scheduling intents
    if meeting.intent == "query":
        return _handle_query(user_text, state)

    if meeting.intent == "cancel":
        return _handle_cancel(meeting, user_text, state)

    # Step 2: Check if ambiguous
    if meeting.is_ambiguous:
        state["pending_request"] = meeting
        clarification = (
            meeting.clarification_needed
            or "Could you provide more details about when you'd like to meet?"
        )
        logger.info(f"Request is ambiguous — asking: {clarification!r}")
        return f"🤔 {clarification}", state

    # Step 3: Retrieve RAG context (past meeting patterns)
    rag_context = _get_rag_context(meeting, user_id)

    # Step 4: Apply user preferences to fill in gaps
    if meeting.duration_minutes == 30:  # default — override with preference
        pref_duration = int(prefs.get("preferred_duration", 30))
        if pref_duration != meeting.duration_minutes:
            meeting.duration_minutes = pref_duration
            logger.info(f"Applied preferred duration: {pref_duration} min")

    # Step 5: Check calendar availability
    start_dt = meeting.resolved_datetime
    if not start_dt:
        state["pending_request"] = meeting
        return (
            "⏰ I couldn't determine the exact date and time. "
            "Could you specify it more precisely? For example: 'Monday October 2nd at 3pm'",
            state,
        )

    end_dt = start_dt + timedelta(minutes=meeting.duration_minutes)

    logger.info(f"Checking availability: {start_dt} → {end_dt}")
    is_free = True
    try:
        is_free = check_availability(start_dt, end_dt)
    except Exception as e:
        logger.warning(f"Google Calendar availability check skipped: {e}")
        is_free = True

    # Step 6a: Slot is free → book it
    if is_free:
        return _book_meeting(meeting, start_dt, end_dt, rag_context, state)

    # Step 6b: Slot is busy → find alternatives
    return _handle_conflict(meeting, start_dt, end_dt, state)


def _handle_clarification(user_text: str, state: dict) -> tuple[str, dict]:
    """
    Re-extract after user provides clarification for an ambiguous request.
    Combines original request + clarification into a fresh extraction.
    """
    pending: MeetingRequest = state["pending_request"]

    # Build a combined context message
    combined = (
        f"Original request: I want to {pending.intent} a meeting titled '{pending.title}' "
        f"with {', '.join(pending.participants) or 'no specific participants'}. "
        f"Clarification: {user_text}"
    )

    logger.info(f"Re-extracting with clarification: {combined[:100]!r}")

    # Clear pending and try again
    state["pending_request"] = None
    return _handle_new_request(combined, state)


def _handle_slot_selection(user_text: str, state: dict) -> tuple[str, dict]:
    """Parse user's slot selection from alternative options."""
    slots = state["alternative_slots"]
    pending: MeetingRequest = state.get("pending_request")

    selected_slot = None
    text_lower = user_text.lower()

    # Simple slot matching by index or keyword
    for i, slot in enumerate(slots):
        label_lower = slot["label"].lower()
        if (
            str(i + 1) in text_lower
            or any(word in text_lower for word in label_lower.split())
        ):
            selected_slot = slot
            break

    if not selected_slot and slots:
        # Default to first slot if no clear selection
        selected_slot = slots[0]

    if not selected_slot or not pending:
        state["awaiting_slot_selection"] = False
        state["alternative_slots"] = []
        state["pending_request"] = None
        return "I couldn't determine your selection. Please start a new booking request.", state

    state["awaiting_slot_selection"] = False
    state["alternative_slots"] = []

    return _book_meeting(
        pending,
        selected_slot["start"],
        selected_slot["end"],
        [],
        state,
    )


def _handle_conflict(
    meeting: MeetingRequest,
    start_dt,
    end_dt,
    state: dict,
) -> tuple[str, dict]:
    """Find and present alternative time slots when the requested slot is busy."""
    user_id = state["user_id"]

    logger.info("Slot is busy — searching for alternatives...")
    alt_slots = find_alternative_slots(
        preferred_start=start_dt,
        duration_minutes=meeting.duration_minutes,
        days_to_search=3,
        max_slots=3,
    )

    if not alt_slots:
        return (
            "😔 That slot is busy and I couldn't find free alternatives in the next 3 days. "
            "Please suggest a different time.",
            state,
        )

    # Store alternatives in state for follow-up selection
    state["awaiting_slot_selection"] = True
    state["alternative_slots"] = alt_slots
    state["pending_request"] = meeting

    options = "\n".join(
        f"  {i + 1}. {slot['label']}" for i, slot in enumerate(alt_slots)
    )

    return (
        f"⚠️ That time slot is already busy!\n\n"
        f"Here are some available alternatives:\n{options}\n\n"
        f"Which one works for you? (Say '1', '2', or '3')",
        state,
    )


def _book_meeting(
    meeting: MeetingRequest,
    start_dt,
    end_dt,
    rag_context: list[str],
    state: dict,
) -> tuple[str, dict]:
    """Create the calendar event and log everything."""
    user_id = state["user_id"]

    # Build description from RAG context if available
    description = ""
    if rag_context:
        description = "Context from past meetings:\n" + "\n".join(f"• {c}" for c in rag_context)

    logger.info(f"Creating event: {meeting.title!r} at {start_dt}")
    event_link = ""
    try:
        event_link = create_event(
            summary=meeting.title,
            start=start_dt,
            end=end_dt,
            attendees=meeting.participants,
            description=description,
        )
    except Exception as e:
        logger.warning(f"Google Calendar create_event skipped: {e}")

    # Log to SQLite
    db_meeting_id = log_meeting(
        user_id=user_id,
        title=meeting.title,
        participants=meeting.participants,
        start_time=start_dt,
        end_time=end_dt,
        duration_mins=meeting.duration_minutes,
        event_link=event_link,
        raw_user_input=state.get("conversation_history", [{}])[-1].get("content", ""),
        extracted_json=meeting.model_dump(mode="json"),
    )

    # Index in ChromaDB for future RAG retrieval
    index_meeting(
        db_meeting_id,
        {
            "title": meeting.title,
            "participants": meeting.participants,
            "start_time": start_dt,
            "duration_mins": meeting.duration_minutes,
            "event_link": event_link,
        },
        user_id,
    )

    # Infer preferences from updated history
    infer_preferences(user_id)

    # Format confirmation
    time_str = start_dt.strftime("%A, %B %d at %I:%M %p")
    duration_str = (
        f"{meeting.duration_minutes} minutes"
        if meeting.duration_minutes != 60
        else "1 hour"
    )
    has_participant_in_title = any(
        p.lower() in meeting.title.lower() for p in meeting.participants
    ) if meeting.participants else False

    participants_str = (
        f" with {', '.join(meeting.participants)}"
        if (meeting.participants and not has_participant_in_title)
        else ""
    )

    response = (
        f"✅ **Meeting booked!**\n\n"
        f"📅 **{meeting.title}**{participants_str}\n"
        f"🕐 {time_str} ({duration_str})\n"
    )

    if event_link:
        response += f"🔗 [View on Google Calendar]({event_link})"
    elif not is_calendar_configured():
        response += (
            "\n\n*(ℹ️ Note: Saved to database, but Google Calendar invite was not generated because Google Calendar is not connected yet. "
            "Add `GOOGLE_TOKEN_JSON` to Render's Environment Variables to enable live Google Calendar syncing.)*"
        )

    # Optional Push Notification (Stretch Goal - Free ntfy.sh)
    ntfy_topic = os.getenv("NTFY_TOPIC")
    if ntfy_topic:
        try:
            from stretch.notify_service import send_notification
            send_notification(
                title=f"🗓️ Booked: {meeting.title}",
                message=f"{time_str} ({duration_str}){participants_str}",
                click_url=event_link or None,
                tags=["calendar", "white_check_mark"],
            )
        except Exception as e:
            logger.debug(f"Push notification skipped/failed: {e}")

    # Reset state
    state["pending_request"] = None
    state["awaiting_slot_selection"] = False
    state["alternative_slots"] = []
    state["conversation_history"] = []

    return response, state


def _handle_query(user_text: str, state: dict) -> tuple[str, dict]:
    """Handle 'what meetings do I have?' type queries."""
    user_id = state["user_id"]
    upcoming = get_upcoming_meetings(user_id, days=7)

    if not upcoming:
        return "📭 You have no upcoming meetings in the next 7 days.", state

    lines = ["📅 **Your upcoming meetings:**\n"]
    for m in upcoming:
        time_str = m["start_time"].strftime("%A %b %d at %I:%M %p")
        participants = ", ".join(m.get("participants", []))
        line = f"• **{m['title']}** — {time_str}"
        if participants:
            line += f" (with {participants})"
        lines.append(line)

    return "\n".join(lines), state


def _handle_cancel(meeting: MeetingRequest, user_text: str, state: dict) -> tuple[str, dict]:
    """Handle cancelling a meeting on Google Calendar & SQLite history."""
    user_id = state["user_id"]

    # 1. Determine search query
    search_term = ""
    if meeting.participants:
        search_term = meeting.participants[0]
    elif meeting.title and meeting.title.lower() != "meeting":
        search_term = meeting.title
    else:
        # Filter filler words from raw user text
        stopwords = {"cancel", "delete", "remove", "the", "my", "meeting", "call", "with", "at", "on", "please"}
        words = [w for w in user_text.split() if w.lower() not in stopwords]
        search_term = " ".join(words)

    # 2. Search Google Calendar
    events = []
    try:
        events = search_events(query=search_term, max_results=5)
    except Exception as e:
        logger.warning(f"Google Calendar search skipped: {e}")

    if not events:
        return (
            f"🔍 I couldn't find any meeting matching **'{search_term or user_text}'** on your Google Calendar to cancel.",
            state,
        )

    # 3. Pick the matched event
    target_event = events[0]
    event_id = target_event["id"]
    event_summary = target_event.get("summary", "Meeting")
    start_time_raw = target_event["start"].get("dateTime", target_event["start"].get("date", ""))

    time_display = start_time_raw
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(start_time_raw)
        time_display = dt.strftime("%A, %B %d at %I:%M %p")
    except Exception:
        pass

    # 4. Delete from Google Calendar
    success = delete_event(event_id)
    state["conversation_history"] = []
    if success:
        # Also clean up SQLite history
        delete_meeting_record(user_id, title=event_summary)
        return (
            f"🗑️ **Meeting cancelled!**\n\n"
            f"Successfully removed **{event_summary}** ({time_display}) from your Google Calendar.",
            state,
        )
    else:
        return (
            f"⚠️ Found **{event_summary}**, but encountered an error trying to delete it from Google Calendar.",
            state,
        )


def _get_rag_context(meeting: MeetingRequest, user_id: str) -> list[str]:
    """Retrieve relevant past meeting context for enriching new booking."""
    query_parts = [meeting.title]
    if meeting.participants:
        query_parts.extend(meeting.participants)
    query = " ".join(query_parts)

    try:
        return retrieve_context(query, user_id, n_results=2)
    except Exception as e:
        logger.warning(f"RAG retrieval failed (non-fatal): {e}")
        return []


# ─────────────────────────────────────────────
# Terminal Interface (Phase 3 test)
# ─────────────────────────────────────────────

def run_terminal():
    """Run the agent as a simple terminal chatbot for testing."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 60)
    print("AI Meeting Scheduler — Terminal Interface")
    print("=" * 60)
    print("Type your meeting request, or 'quit' to exit.\n")

    state = get_initial_state()
    state["user_id"] = "terminal_user"

    print("🤖 Hello! I'm your AI meeting scheduler. How can I help you today?\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if user_input.lower() in ("quit", "exit", "bye"):
            print("🤖 Goodbye! Have a productive day!")
            break

        if not user_input:
            continue

        response, state = handle_message(user_input, state)
        print(f"\n🤖 {response}\n")


if __name__ == "__main__":
    run_terminal()
