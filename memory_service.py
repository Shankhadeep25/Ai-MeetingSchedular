"""
memory_service.py — Persistent User Memory (SQLite)
====================================================
Stores meeting history and user preferences across sessions.
Uses SQLite — a local file, completely free, zero setup.

Auto-creates the data/ directory and meetings.db on first run.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from collections import Counter

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Database Setup
# ─────────────────────────────────────────────

DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/meetings.db")

# Ensure data directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


# ─────────────────────────────────────────────
# SQLAlchemy Models
# ─────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    timezone = Column(String, default="Asia/Kolkata")
    created_at = Column(DateTime, default=datetime.utcnow)


class MeetingHistory(Base):
    __tablename__ = "meeting_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    participants = Column(Text)          # JSON array stored as text
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    duration_mins = Column(Integer)
    event_link = Column(Text)
    raw_user_input = Column(Text)
    extracted_json = Column(Text)        # Audit: raw LLM output
    created_at = Column(DateTime, default=datetime.utcnow)


class Preference(Base):
    __tablename__ = "preferences"

    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    pref_key = Column(String, primary_key=True)
    pref_value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ─────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist. Safe to call multiple times."""
    Base.metadata.create_all(engine)
    logger.info(f"SQLite database initialized at {DB_PATH}")


def get_or_create_user(user_id: str, name: str = "User", email: str = "", timezone: str = None) -> None:
    """Ensure a user row exists. Creates one if not found."""
    init_db()
    tz = timezone or os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata")

    with SessionLocal() as session:
        user = session.get(User, user_id)
        if not user:
            user = User(
                id=user_id,
                name=name,
                email=email or f"{user_id}@local",
                timezone=tz,
            )
            session.add(user)
            session.commit()
            logger.info(f"Created user: {user_id}")


# ─────────────────────────────────────────────
# Meeting History
# ─────────────────────────────────────────────

def log_meeting(
    user_id: str,
    title: str,
    participants: list[str],
    start_time: datetime,
    end_time: datetime,
    duration_mins: int,
    event_link: str = "",
    raw_user_input: str = "",
    extracted_json: dict | None = None,
) -> int:
    """
    Persist a booked meeting to SQLite.

    Returns:
        ID of the inserted meeting record
    """
    init_db()
    get_or_create_user(user_id)

    with SessionLocal() as session:
        record = MeetingHistory(
            user_id=user_id,
            title=title,
            participants=json.dumps(participants),
            start_time=start_time,
            end_time=end_time,
            duration_mins=duration_mins,
            event_link=event_link,
            raw_user_input=raw_user_input,
            extracted_json=json.dumps(extracted_json) if extracted_json else None,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        logger.info(f"Meeting logged: id={record.id}, title={title!r}")
        return record.id


def get_meeting_history(user_id: str, limit: int = 10) -> list[dict]:
    """
    Retrieve the most recent N meetings for a user.

    Returns:
        List of meeting dicts ordered by most recent first
    """
    init_db()

    with SessionLocal() as session:
        records = (
            session.query(MeetingHistory)
            .filter(MeetingHistory.user_id == user_id)
            .order_by(MeetingHistory.start_time.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "id": r.id,
                "title": r.title,
                "participants": json.loads(r.participants) if r.participants else [],
                "start_time": r.start_time,
                "end_time": r.end_time,
                "duration_mins": r.duration_mins,
                "event_link": r.event_link,
            }
            for r in records
        ]


def get_upcoming_meetings(user_id: str, days: int = 7) -> list[dict]:
    """Fetch meetings scheduled within the next N days."""
    init_db()
    now = datetime.utcnow()
    future = now + timedelta(days=days)

    with SessionLocal() as session:
        records = (
            session.query(MeetingHistory)
            .filter(
                MeetingHistory.user_id == user_id,
                MeetingHistory.start_time >= now,
                MeetingHistory.start_time <= future,
            )
            .order_by(MeetingHistory.start_time.asc())
            .all()
        )

        return [
            {
                "id": r.id,
                "title": r.title,
                "participants": json.loads(r.participants) if r.participants else [],
                "start_time": r.start_time,
                "end_time": r.end_time,
                "event_link": r.event_link,
            }
            for r in records
        ]


# ─────────────────────────────────────────────
# User Preferences
# ─────────────────────────────────────────────

def update_preference(user_id: str, key: str, value: str | int | float) -> None:
    """Set or update a user preference key-value pair."""
    init_db()
    get_or_create_user(user_id)

    with SessionLocal() as session:
        pref = session.get(Preference, (user_id, key))
        if pref:
            pref.pref_value = str(value)
            pref.updated_at = datetime.utcnow()
        else:
            pref = Preference(user_id=user_id, pref_key=key, pref_value=str(value))
            session.add(pref)
        session.commit()
        logger.debug(f"Preference updated: {user_id} → {key}={value}")


def get_preferences(user_id: str) -> dict:
    """
    Return all stored preferences for a user.

    Returns:
        Dict of key → value. Includes defaults if user has none yet.
    """
    init_db()
    defaults = {
        "preferred_duration": str(os.getenv("DEFAULT_MEETING_DURATION", 30)),
        "timezone": os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata"),
        "preferred_start_hour": "9",
        "preferred_end_hour": "18",
    }

    with SessionLocal() as session:
        prefs = (
            session.query(Preference)
            .filter(Preference.user_id == user_id)
            .all()
        )
        result = dict(defaults)
        for p in prefs:
            result[p.pref_key] = p.pref_value
        return result


# ─────────────────────────────────────────────
# Preference Inference from History
# ─────────────────────────────────────────────

def infer_preferences(user_id: str) -> dict:
    """
    Analyze meeting history to infer user preferences automatically.

    Infers:
    - preferred_duration: most common meeting length
    - preferred_start_hour: most common hour for meetings

    Stores inferred preferences back to the DB.

    Returns:
        Dict of inferred preferences
    """
    history = get_meeting_history(user_id, limit=50)

    if len(history) < 3:
        logger.info(f"Not enough history to infer preferences for {user_id} (need 3+)")
        return {}

    # Most common duration
    durations = [m["duration_mins"] for m in history if m.get("duration_mins")]
    if durations:
        preferred_duration = Counter(durations).most_common(1)[0][0]
        update_preference(user_id, "preferred_duration", preferred_duration)
    else:
        preferred_duration = None

    # Most common booking hour
    hours = [m["start_time"].hour for m in history if m.get("start_time")]
    if hours:
        preferred_hour = Counter(hours).most_common(1)[0][0]
        update_preference(user_id, "preferred_start_hour", preferred_hour)
    else:
        preferred_hour = None

    inferred = {
        "preferred_duration": preferred_duration,
        "preferred_start_hour": preferred_hour,
    }

    logger.info(f"Inferred preferences for {user_id}: {inferred}")
    return inferred


# ─────────────────────────────────────────────
# CLI Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 60)
    print("AI Meeting Scheduler — Memory Service Test")
    print("=" * 60 + "\n")

    TEST_USER = "test_user_001"

    print("Step 1: Init DB + create user...")
    init_db()
    get_or_create_user(TEST_USER, name="Test User", timezone="Asia/Kolkata")
    print("✅ DB initialized\n")

    print("Step 2: Log a test meeting...")
    now = datetime.utcnow()
    meeting_id = log_meeting(
        user_id=TEST_USER,
        title="Test Standup",
        participants=["Rahul", "Priya"],
        start_time=now,
        end_time=now + timedelta(minutes=30),
        duration_mins=30,
        event_link="https://calendar.google.com/test",
        raw_user_input="Schedule a standup with Rahul and Priya",
    )
    print(f"✅ Logged meeting ID: {meeting_id}\n")

    print("Step 3: Retrieve history...")
    history = get_meeting_history(TEST_USER, limit=5)
    for m in history:
        print(f"  📋 [{m['id']}] {m['title']} — {m['start_time']}")
    print()

    print("Step 4: Set + get preferences...")
    update_preference(TEST_USER, "preferred_duration", 45)
    prefs = get_preferences(TEST_USER)
    print(f"  Preferences: {prefs}\n")

    print("✅ Memory service is working correctly!")
