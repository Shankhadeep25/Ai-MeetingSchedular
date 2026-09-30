"""
app.py — Streamlit Conversational UI
====================================
Interactive frontend for the AI Meeting Scheduler.
Features:
- Premium conversational chat interface with avatar styling
- Interactive quick prompt chips
- Action buttons for conflict resolution slot selection
- Sidebar: System health/status, upcoming meetings, timezone and preference controls
- Multi-turn session state management
"""

import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# Load local environment
load_dotenv()

# Import backend modules
from agent import handle_message, get_initial_state
from memory_service import (
    get_or_create_user,
    get_upcoming_meetings,
    get_preferences,
    update_preference,
)
from calendar_service import TOKEN_PATH, CREDENTIALS_PATH

# ─────────────────────────────────────────────
# Page Configuration & Aesthetics
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Meeting Scheduler",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark/Light mode adaptive, sleek cards, glowing chips)
st.markdown(
    """
    <style>
    /* Main container styling */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header hero styling */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-top: 6px;
    }
    
    /* Meeting card in sidebar */
    .meeting-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 12px 14px;
        border-radius: 8px;
        margin-bottom: 10px;
        transition: transform 0.15s ease-in-out;
    }
    .meeting-card:hover {
        transform: translateX(4px);
    }
    .meeting-title {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    .meeting-meta {
        font-size: 0.8rem;
        color: #94a3b8;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .status-connected {
        background-color: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .status-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
if "agent_state" not in st.session_state:
    st.session_state.agent_state = get_initial_state()

if "user_id" not in st.session_state:
    st.session_state.user_id = "user_default"
    st.session_state.agent_state["user_id"] = "user_default"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Hi! I'm your AI Meeting Assistant.**\n\n"
                "I can automatically parse your natural language requests, resolve conflicts, "
                "remember your preferences, and book real calendar invites.\n\n"
                "Try telling me something like:\n"
                "- *'Schedule a 30 min sync with Priya next Tuesday at 3pm'*\n"
                "- *'Set up a quick standup with Rahul tomorrow morning at 10am'*\n"
                "- *'What meetings do I have this week?'*"
            ),
        }
    ]

# ─────────────────────────────────────────────
# Sidebar: User Controls & Diagnostics
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System & Preferences")

    # User Profile
    user_id_input = st.text_input("Active User ID", value=st.session_state.user_id)
    if user_id_input != st.session_state.user_id:
        st.session_state.user_id = user_id_input
        st.session_state.agent_state["user_id"] = user_id_input
        get_or_create_user(user_id_input)
        st.rerun()

    # Timezone Selector
    user_prefs = get_preferences(st.session_state.user_id)
    tz_options = ["Asia/Kolkata", "UTC", "America/New_York", "Europe/London", "Asia/Tokyo"]
    current_tz = user_prefs.get("timezone", os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata"))
    selected_tz = st.selectbox(
        "Timezone",
        options=tz_options,
        index=tz_options.index(current_tz) if current_tz in tz_options else 0,
    )
    if selected_tz != current_tz:
        update_preference(st.session_state.user_id, "timezone", selected_tz)
        st.toast(f"Timezone updated to {selected_tz}")

    # Preferred Duration
    curr_duration = int(user_prefs.get("preferred_duration", 30))
    selected_duration = st.slider("Default Duration (mins)", 15, 120, curr_duration, step=15)
    if selected_duration != curr_duration:
        update_preference(st.session_state.user_id, "preferred_duration", selected_duration)

    st.markdown("---")
    st.markdown("### 🔌 Service Status")

    # Groq API Status
    has_groq = bool(os.getenv("GROQ_API_KEY")) and not os.getenv("GROQ_API_KEY", "").startswith("gsk_your")
    if has_groq:
        st.markdown('<span class="status-badge status-connected">● Groq LLM (Llama 3.3)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning">⚠ Groq Key Missing</span>', unsafe_allow_html=True)

    # Google Calendar Status
    has_token = (
        bool(os.getenv("GOOGLE_TOKEN_JSON"))
        or os.path.exists(TOKEN_PATH)
        or os.path.exists("/etc/secrets/token.json")
    )
    has_creds = (
        bool(os.getenv("GOOGLE_CREDENTIALS_JSON"))
        or os.path.exists(CREDENTIALS_PATH)
        or os.path.exists("/etc/secrets/credentials.json")
    )
    if has_token:
        st.markdown('<span class="status-badge status-connected">● Google Calendar Active</span>', unsafe_allow_html=True)
    elif has_creds:
        st.markdown('<span class="status-badge status-warning">○ OAuth Pending Auth</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning">⚠ Google Calendar Not Connected</span>', unsafe_allow_html=True)

    # Storage Status (Relational DB & Vector DB)
    if os.getenv("DATABASE_URL"):
        st.markdown('<span class="status-badge status-connected">● Cloud PostgreSQL Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-connected">● SQLite Local</span>', unsafe_allow_html=True)

    if os.getenv("PINECONE_API_KEY"):
        st.markdown('<span class="status-badge status-connected">● Pinecone Cloud Vector DB</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-connected">● ChromaDB Local</span>', unsafe_allow_html=True)

    st.markdown("---")
    # Public Calendar View Link
    public_cal_url = os.getenv(
        "PUBLIC_CALENDAR_URL",
        "https://calendar.google.com/calendar/embed?src=testuserkiit01%40gmail.com&ctz=Asia%2FKolkata",
    )
    st.markdown(
        f'<a href="{public_cal_url}" target="_blank" style="'
        'display:block; text-align:center; padding:8px 12px; '
        'background:linear-gradient(135deg,#1e3a8a,#3b82f6); '
        'color:white; border-radius:8px; text-decoration:none; '
        'font-weight:600; font-size:0.9rem; margin-bottom:12px;">'
        '📅 Open Live Calendar Grid</a>',
        unsafe_allow_html=True,
    )
    st.markdown("### 📋 Upcoming Schedule")
    try:
        upcoming = get_upcoming_meetings(st.session_state.user_id, days=7)
        if upcoming:
            for m in upcoming:
                start_str = m["start_time"].strftime("%b %d, %I:%M %p")
                participants = ", ".join(m.get("participants", []))
                link_html = f'<a href="{m["event_link"]}" target="_blank" style="color: #60a5fa; text-decoration: none;">🔗 Open</a>' if m.get("event_link") else ""
                st.markdown(
                    f"""
                    <div class="meeting-card">
                        <div class="meeting-title">{m['title']}</div>
                        <div class="meeting-meta">🕒 {start_str}</div>
                        {f'<div class="meeting-meta">👥 {participants}</div>' if participants else ''}
                        {link_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No meetings scheduled in the next 7 days.")
    except Exception as e:
        st.caption(f"Could not load upcoming meetings: {e}")

    # Action to clear conversation
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.agent_state = get_initial_state()
        st.session_state.agent_state["user_id"] = st.session_state.user_id
        st.rerun()

# ─────────────────────────────────────────────
# Main Hero Header
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1 class="hero-title">AI Meeting Scheduler</h1>
        <div class="hero-subtitle">Autonomous, Zero-Cost Intelligent Calendar Coordinator · Powered by Llama 3.3 70B</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Quick Action Suggestion Chips
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
quick_prompt = None

with col1:
    if st.button("⚡ Standup with Rahul tomorrow 10am", use_container_width=True):
        quick_prompt = "Schedule a team standup with Rahul tomorrow at 10am"
with col2:
    if st.button("☕ 15-min coffee chat next Friday 4pm", use_container_width=True):
        quick_prompt = "Book a quick 15-minute coffee chat with Priya next Friday at 4pm"
with col3:
    if st.button("📅 What meetings do I have?", use_container_width=True):
        quick_prompt = "What meetings do I have scheduled?"

# ─────────────────────────────────────────────
# Render Message Feed
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "🤖" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────
# Conflict Resolution Action Buttons
# ─────────────────────────────────────────────
state = st.session_state.agent_state
if state.get("awaiting_slot_selection") and state.get("alternative_slots"):
    st.info("👉 Select an alternative slot to confirm booking:")
    slot_cols = st.columns(len(state["alternative_slots"]))
    for idx, (slot, col) in enumerate(zip(state["alternative_slots"], slot_cols)):
        with col:
            if st.button(f"Option {idx + 1}\n{slot['label']}", key=f"slot_opt_{idx}", use_container_width=True):
                quick_prompt = str(idx + 1)

# ─────────────────────────────────────────────
# Chat Input & Agent Dispatch
# ─────────────────────────────────────────────
user_input = st.chat_input("Type your meeting request or reply here...")

prompt_to_process = quick_prompt or user_input

if prompt_to_process:
    # 1. Append user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_to_process)

    # 2. Dispatch to agent orchestrator
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing request and checking calendar availability..."):
            try:
                response_text, updated_state = handle_message(
                    prompt_to_process,
                    st.session_state.agent_state,
                )
                st.session_state.agent_state = updated_state
            except Exception as e:
                response_text = f"❌ **An error occurred:** {str(e)}\n\nPlease ensure your `.env` has a valid `GROQ_API_KEY` and Google Calendar is configured."

            st.markdown(response_text)

    # 3. Append assistant response to UI history
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Rerun to update sidebar upcoming meetings or slot buttons if changed
    st.rerun()
