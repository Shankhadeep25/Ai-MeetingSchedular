"""
nlp_extractor.py — Intent & Entity Extraction
=============================================
Uses Groq API (Llama 3.3 70B, FREE tier) with tool calling to extract
structured meeting details from natural language user input.

Free tier: 14,400 req/day | 500K tokens/min | $0 cost
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

import dateparser
from groq import Groq
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Pydantic Schema — the contract between LLM and code
# ─────────────────────────────────────────────

class MeetingRequest(BaseModel):
    """Structured representation of a meeting booking request."""

    intent: str = Field(
        description="One of: schedule, reschedule, cancel, query",
        pattern="^(schedule|reschedule|cancel|query)$",
    )
    title: str = Field(
        default="Meeting",
        description="Short meeting title or topic",
    )
    participants: list[str] = Field(
        default_factory=list,
        description="List of participant names or emails",
    )
    date_phrase: str = Field(
        default="",
        description="Raw date/time phrase extracted from user text, e.g. 'next Tuesday at 3pm'",
    )
    resolved_datetime: Optional[datetime] = Field(
        default=None,
        description="Parsed absolute datetime resolved from date_phrase",
    )
    duration_minutes: int = Field(
        default=30,
        description="Meeting duration in minutes",
    )
    is_ambiguous: bool = Field(
        description="True if key details are missing and clarification is needed",
    )
    clarification_needed: str = Field(
        default="",
        description="Specific question to ask the user if is_ambiguous is True",
    )


# ─────────────────────────────────────────────
# Groq Tool Schema — JSON Schema sent to Llama 3.3
# ─────────────────────────────────────────────

MEETING_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_meeting_request",
        "description": (
            "Extract structured meeting details from natural language. "
            "Set is_ambiguous=True if the date/time is missing or unclear. "
            "Always extract whatever is available even if ambiguous."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["schedule", "reschedule", "cancel", "query"],
                    "description": "The user's intent regarding the meeting",
                },
                "title": {
                    "type": "string",
                    "description": "Meeting title or topic (infer from context if not stated)",
                },
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of participant names or emails mentioned",
                },
                "date_phrase": {
                    "type": "string",
                    "description": "Exact date/time phrase from the user's message",
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Duration in minutes. Default 30 if not mentioned.",
                },
                "is_ambiguous": {
                    "type": "boolean",
                    "description": "True if date/time is missing, unclear, or contradictory",
                },
                "clarification_needed": {
                    "type": "string",
                    "description": "The specific question to ask the user if is_ambiguous is True",
                },
            },
            "required": ["intent", "is_ambiguous"],
        },
    },
}

SYSTEM_PROMPT = """You are an expert meeting scheduling assistant. 
Your job is to extract structured meeting information from natural language requests.

Rules:
1. Always call the extract_meeting_request tool — never respond in plain text.
2. If the user mentions a date/time phrase (e.g., "next Tuesday", "tomorrow at 3pm", "Friday afternoon"), 
   extract it verbatim into date_phrase, even if the time is vague.
3. Set is_ambiguous=True ONLY if no date/time information is present at all, 
   or if it's truly contradictory.
4. For duration: if user says "quick call" → 15 min, "30 minutes" → 30, 
   "an hour" → 60. Default is 30 if unspecified.
5. Infer meeting title from context: "call with Rahul" → "Call with Rahul", 
   "team standup" → "Team Standup".
6. For reschedule/cancel intents, still extract whatever meeting identifiers you can.
"""


# ─────────────────────────────────────────────
# Date Resolution
# ─────────────────────────────────────────────

def resolve_date(phrase: str, timezone: str = "Asia/Kolkata") -> Optional[datetime]:
    """
    Convert a natural language date phrase into an absolute datetime.
    Uses dateparser — runs fully offline, no API needed.

    Args:
        phrase: e.g. "next Tuesday at 3pm", "tomorrow afternoon"
        timezone: IANA timezone string

    Returns:
        Resolved datetime object or None if parsing fails
    """
    if not phrase or not phrase.strip():
        return None

    settings = {
        "PREFER_DATES_FROM": "future",
        "TIMEZONE": timezone,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DAY_OF_MONTH": "first",
    }

    try:
        parsed = dateparser.parse(phrase, settings=settings)
        if parsed:
            logger.debug(f"Resolved '{phrase}' → {parsed.isoformat()}")
        return parsed
    except Exception as e:
        logger.warning(f"dateparser failed for phrase '{phrase}': {e}")
        return None


# ─────────────────────────────────────────────
# Main Extraction Function
# ─────────────────────────────────────────────

def extract_meeting_info(
    user_text: str,
    timezone: str = None,
    conversation_history: list[dict] | None = None,
) -> MeetingRequest:
    """
    Parse natural language meeting request into a structured MeetingRequest.

    Args:
        user_text: Raw user message
        timezone: IANA timezone override (falls back to env DEFAULT_TIMEZONE)
        conversation_history: Optional prior conversation turns for context

    Returns:
        MeetingRequest Pydantic object
    """
    tz = timezone or os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com"
        )

    client = Groq(api_key=api_key)

    # Build messages — include prior conversation for multi-turn context
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_text})

    logger.info(f"Calling Groq (model={model}) for: {user_text[:80]!r}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[MEETING_TOOL_SCHEMA],
            tool_choice="required",  # always use the tool, never plain text
            temperature=0.1,         # low temp for deterministic extraction
            max_tokens=512,
        )
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        # Return a safe ambiguous request so the agent can ask for clarification
        return MeetingRequest(
            intent="schedule",
            is_ambiguous=True,
            clarification_needed=(
                "I had trouble understanding that. Could you rephrase your meeting request?"
            ),
        )

    # Parse tool call response
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        logger.warning("Groq returned no tool calls — falling back to ambiguous")
        return MeetingRequest(
            intent="schedule",
            is_ambiguous=True,
            clarification_needed="Could you provide more details about the meeting?",
        )

    raw_args = tool_calls[0].function.arguments
    logger.debug(f"Groq tool args: {raw_args}")

    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Groq tool args as JSON: {e}")
        return MeetingRequest(
            intent="schedule",
            is_ambiguous=True,
            clarification_needed="Could you please repeat your meeting request?",
        )

    # Resolve date phrase → absolute datetime
    date_phrase = args.get("date_phrase", "")
    resolved_dt = resolve_date(date_phrase, tz) if date_phrase else None

    # Build and validate the Pydantic model
    meeting = MeetingRequest(
        intent=args.get("intent", "schedule"),
        title=args.get("title", "Meeting"),
        participants=args.get("participants", []),
        date_phrase=date_phrase,
        resolved_datetime=resolved_dt,
        duration_minutes=args.get("duration_minutes", int(os.getenv("DEFAULT_MEETING_DURATION", 30))),
        is_ambiguous=args.get("is_ambiguous", False),
        clarification_needed=args.get("clarification_needed", ""),
    )

    # If date phrase exists but couldn't be resolved, mark as ambiguous
    if date_phrase and resolved_dt is None and not meeting.is_ambiguous:
        meeting.is_ambiguous = True
        meeting.clarification_needed = (
            f"I understood '{date_phrase}' but couldn't determine the exact date. "
            "Could you specify the date more precisely (e.g., 'Monday September 28 at 3pm')?"
        )

    logger.info(
        f"Extracted: intent={meeting.intent}, title={meeting.title!r}, "
        f"date={resolved_dt}, ambiguous={meeting.is_ambiguous}"
    )

    return meeting


# ─────────────────────────────────────────────
# CLI Test (run: python nlp_extractor.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_sentences = [
        "Schedule a call with Rahul next Tuesday at 3pm",
        "Book a 1-hour team standup for tomorrow morning",
        "Set up a quick 15-minute sync with Priya and John on Friday at 2pm",
        "I need to meet with the design team next Monday",
        "Cancel my meeting with Sarah",
        "Reschedule the product review to Thursday afternoon",
        "Schedule something with the sales team",  # ambiguous — no date
        "Can we do a call this weekend at 11am?",
        "Book a 30-minute catchup with Ankit for September 30th at noon",
        "Meeting with Vikram tomorrow",
    ]

    print("\n" + "=" * 60)
    print("AI Meeting Scheduler — NLP Extractor Test")
    print("=" * 60 + "\n")

    for i, sentence in enumerate(test_sentences, 1):
        print(f"[{i}] Input: {sentence!r}")
        try:
            result = extract_meeting_info(sentence)
            print(f"     Intent:      {result.intent}")
            print(f"     Title:       {result.title}")
            print(f"     Participants:{result.participants}")
            print(f"     Date phrase: {result.date_phrase!r}")
            print(f"     Resolved:    {result.resolved_datetime}")
            print(f"     Duration:    {result.duration_minutes} min")
            print(f"     Ambiguous:   {result.is_ambiguous}")
            if result.is_ambiguous:
                print(f"     Clarify:     {result.clarification_needed!r}")
        except Exception as e:
            print(f"     ERROR: {e}")
        print()
