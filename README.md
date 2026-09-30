# 🗓️ AI Meeting Scheduler
### An Autonomous, GenAI-Powered Conversational Calendar Coordinator

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-orange)](https://console.groq.com/)
[![Google Calendar](https://img.shields.io/badge/Google%20Calendar-API%20v3-4285F4?logo=googlecalendar)](https://developers.google.com/calendar)
[![Pinecone](https://img.shields.io/badge/Vector%20DB-Pinecone-brightgreen)](https://www.pinecone.io/)
[![Neon](https://img.shields.io/badge/Database-Neon%20PostgreSQL-9333ea)](https://neon.tech/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?logo=render)](https://ai-meetingschedular.onrender.com)

---

> **Autonomous, zero-cost intelligent meeting scheduling powered by Llama 3.3 70B.** Understands natural language, checks real calendar availability, resolves conflicts, remembers your preferences, and books live Google Calendar events — entirely through conversation.

**Live Demo:** [https://ai-meetingschedular.onrender.com](https://ai-meetingschedular.onrender.com)
**Public Calendar View:** [Open Live Calendar Grid (No login required)](https://calendar.google.com/calendar/embed?src=testuserkiit01%40gmail.com&ctz=Asia%2FKolkata)
**Source Code:** [github.com/Shankhadeep25/Ai-MeetingSchedular](https://github.com/Shankhadeep25/Ai-MeetingSchedular)

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Tech Stack](#-tech-stack--infrastructure)
3. [Features](#-features)
4. [System Architecture](#-system-architecture)
5. [Project Structure](#-project-directory-structure)
6. [Setup Guide](#-setup-guide-local)
7. [Cloud Deployment](#-cloud-deployment-render)
8. [How It Works](#-how-it-works--agent-flow)
9. [Running Tests](#-running-automated-tests)
10. [Stretch Goals](#-stretch-goals)
11. [Challenges & Solutions](#-challenges--solutions)

---

## 🎯 Project Overview

The **AI Meeting Scheduler** is a fully autonomous, conversational AI assistant that eliminates the friction of manual calendar management. Users interact through natural language — the system understands intent, extracts meeting details, checks real-time calendar availability, resolves scheduling conflicts, recalls past meeting context, and finally books a live Google Calendar event — all within a single conversation.

**What makes it unique:**
- 💬 **Natural Language Understanding** — No rigid command syntax. Talk to it like a human.
- 🤖 **Agentic Loop** — Multi-step reasoning: extract → clarify → check → resolve → book → log.
- 🧠 **Persistent Memory** — Learns your preferences (preferred duration, meeting hours) over time.
- 🔍 **RAG-Powered Context** — Retrieves relevant past meetings to auto-fill context.
- 🗓️ **Real Calendar Integration** — Creates actual Google Calendar events with reminders.
- ☁️ **100% Cloud-Native** — PostgreSQL + Pinecone + Render; zero local state required.
- 💸 **Zero Cost** — Entirely free-tier and open-source infrastructure.

---

## 🛠️ Tech Stack & Infrastructure

| Layer | Technology | Purpose | Cost |
|---|---|---|---|
| **LLM** | Groq API — Llama 3.3 70B | Intent extraction, tool calling | Free (14,400 req/day) |
| **UI** | Streamlit | Conversational web application | Free (open-source) |
| **Calendar** | Google Calendar API v3 | Real event booking & conflict checks | Free (personal OAuth2) |
| **Relational DB** | Neon PostgreSQL (Cloud) | Meeting history & user preferences | Free tier |
| **Vector DB** | Pinecone Serverless | Semantic search over past meetings | Free tier |
| **Embeddings** | Pinecone Inference (multilingual-e5-large) | Cloud-hosted embedding, no PyTorch | Free |
| **Date Parsing** | dateparser | Offline NL date resolution | Free (open-source) |
| **Deployment** | Render Web Service | 24/7 cloud hosting | Free tier |
| **Notifications** | ntfy.sh | Push notifications on booking | Free (open-source) |

---

## ✨ Features

### Core Features
- ✅ **Natural Language Scheduling** — "Schedule a 30-min sync with Rahul next Tuesday at 3pm"
- ✅ **Real Availability Checking** — Queries Google Calendar freebusy API before booking
- ✅ **Intelligent Conflict Resolution** — Proposes 3 alternative slots when requested time is busy
- ✅ **Meeting Cancellation** — "Cancel my meeting with Priya tomorrow"
- ✅ **Schedule Querying** — "What meetings do I have this week?"
- ✅ **Ambiguity Clarification** — Asks follow-up questions when details are missing
- ✅ **Multi-turn Conversation** — Remembers context across multiple conversation turns

### Memory & Intelligence
- ✅ **Persistent User Preferences** — Saves preferred duration, timezone, and working hours
- ✅ **Automatic Preference Inference** — Learns from booking history (peak hours, common durations)
- ✅ **RAG Context Retrieval** — Semantic search over past meetings to auto-populate context
- ✅ **User Profile System** — Supports multiple user IDs with separate histories

### Cloud & Deployment
- ✅ **Cloud PostgreSQL** — Neon PostgreSQL for persistent meeting history across sessions
- ✅ **Cloud Vector DB** — Pinecone Serverless for scalable semantic retrieval
- ✅ **Full Cloud Deployment** — Live on Render with all services connected
- ✅ **Environment-Aware Auth** — Reads GOOGLE_TOKEN_JSON env var (no file required on cloud)

### UI & Experience
- ✅ **Premium Dark Chat Interface** — Gradient hero, glassmorphism cards, micro-animations
- ✅ **Service Health Badges** — Live status for Groq, Google Calendar, PostgreSQL, Pinecone
- ✅ **Upcoming Schedule Sidebar** — Shows next 7 days of meetings with calendar links
- ✅ **Quick Action Chips** — One-click prompt suggestions
- ✅ **Interactive Slot Buttons** — Click to confirm alternative conflict resolution slots
- ✅ **Public Calendar Button** — Direct link to live Google Calendar grid view

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│              Streamlit Chat UI  (app.py)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ User message
┌──────────────────────────▼──────────────────────────────────┐
│                     Agent Core (agent.py)                    │
│          Orchestrator + Session State (Multi-turn)           │
└───┬──────────────┬──────────────┬──────────────┬────────────┘
    │              │              │              │
    ▼              ▼              ▼              ▼
┌───────────┐ ┌─────────┐ ┌──────────┐ ┌────────────┐
│  NLP      │ │ Google  │ │ Neon     │ │  Pinecone  │
│ Extractor │ │ Calendar│ │ Postgres │ │  Vector DB │
│ (Groq)   │ │ API v3  │ │ (Memory) │ │  (RAG)    │
└───────────┘ └─────────┘ └──────────┘ └────────────┘
```

### Agent Execution Flow

```
User Input
    │
    ▼
[1] Check multi-turn state (pending clarification? slot selection?)
    │
    ▼
[2] Groq Tool Calling → Extract: intent, title, participants, date_phrase
    │
    ▼
[3] dateparser → Resolve date_phrase → absolute datetime
    │
    ▼
[4] Ambiguous? → Ask clarification question → Wait for user reply
    │
    ▼
[5] RAG retrieval → Fetch semantically similar past meeting context
    │
    ▼
[6] Load user preferences → Apply default duration, timezone
    │
    ▼
[7] Google Calendar freebusy check → Is the slot free?
    ├── FREE  → Book event → Log to DB → Index to Pinecone → Notify
    └── BUSY  → Find 3 alternative slots → Present buttons to user
                    │
                    ▼
               User selects → Book confirmed slot
```

---

## 📁 Project Directory Structure

```
Ai-MeetingSchedular/
│
├── app.py                    # Streamlit conversational web interface
├── agent.py                  # Agentic loop orchestrator & state manager
├── nlp_extractor.py          # Groq tool calling & dateparser integration
├── calendar_service.py       # Google Calendar OAuth2, freebusy & event CRUD
├── memory_service.py         # SQLAlchemy ORM — PostgreSQL / SQLite adapter
├── rag_service.py            # Pinecone / ChromaDB semantic retrieval
│
├── requirements.txt          # Python dependencies (local dev)
├── requirements-cloud.txt    # Lean cloud deps (excludes PyTorch/ChromaDB)
├── .env.example              # Environment variables template
├── .gitignore                # Protects secrets, credentials, DBs
├── README.md                 # This file
├── PROCESS_LOG.md            # Development log — mistakes, decisions, fixes
│
├── tests/                    # Automated unit & integration tests
│   ├── test_nlp_extractor.py
│   ├── test_memory_service.py
│   ├── test_calendar_service.py
│   └── test_rag_service.py
│
├── stretch/                  # Stretch goal extensions
│   ├── notify_service.py     # ntfy.sh zero-cost push notifications
│   └── mcp_calendar_server.py # FastMCP calendar tool server
│
└── data/                     # Auto-created local data (gitignored)
    ├── meetings.db           # SQLite fallback database
    └── chroma/               # ChromaDB local vector index
```

---

## 🚀 Setup Guide (Local)

### Prerequisites
- Python 3.11+
- Git
- A Google account (for Calendar API)
- Free Groq API account

### Step 1: Clone & Create Virtual Environment
```bash
git clone https://github.com/Shankhadeep25/Ai-MeetingSchedular.git
cd Ai-MeetingSchedular

python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
source venv/bin/activate       # Linux/Mac

pip install -r requirements.txt
```

### Step 2: Get Your Free Groq API Key
1. Go to [console.groq.com](https://console.groq.com) and sign in for free.
2. Navigate to **API Keys** → **Create API Key**.
3. Copy the key (starts with `gsk_...`).

### Step 3: Configure Environment Variables
```bash
copy .env.example .env   # Windows
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
DEFAULT_TIMEZONE=Asia/Kolkata

# Optional cloud databases
DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require
PINECONE_API_KEY=pcsk_your_key_here
PINECONE_INDEX_NAME=ai-meetings
```

### Step 4: Google Calendar OAuth Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Create a project → Enable **Google Calendar API**.
3. Create an **OAuth 2.0 Desktop Client ID** → Download as `credentials.json`.
4. Place `credentials.json` in the project root.
5. Run once to authenticate:
```bash
python calendar_service.py
```
A browser window opens → Sign in → Grant permission → `token.json` is saved.

### Step 5: Run the Application
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## ☁️ Cloud Deployment (Render)

### Service Configuration
| Setting | Value |
|---|---|
| **Build Command** | `pip install -r requirements-cloud.txt` |
| **Start Command** | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| **Region** | Oregon (US West) |

### Required Environment Variables
| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` |
| `DATABASE_URL` | Neon PostgreSQL connection string |
| `PINECONE_API_KEY` | Your Pinecone API key |
| `PINECONE_INDEX_NAME` | `ai-meetings` |
| `PINECONE_INFERENCE_MODEL` | `multilingual-e5-large` |
| `GOOGLE_TOKEN_JSON` | Full contents of your local `token.json` |
| `DEFAULT_TIMEZONE` | `Asia/Kolkata` |
| `PUBLIC_CALENDAR_URL` | Your public Google Calendar embed URL |

---

## 🔄 How It Works — Agent Flow

### Example: Scheduling a Meeting
```
User:  "Schedule a 30-min call with Dr. Sharma next Monday at 2pm"

Agent: [1] Groq extracts intent=schedule, participants=["Dr. Sharma"],
              date_phrase="next Monday at 2pm", duration=30
       [2] dateparser resolves → 2026-10-05 14:00:00 IST
       [3] Pinecone RAG finds past context with Dr. Sharma
       [4] Google Calendar freebusy → slot is FREE ✅
       [5] Creates Google Calendar event with reminders
       [6] Logs to PostgreSQL, indexes to Pinecone
       [7] Sends ntfy.sh push notification

Response: "✅ Meeting booked!
           📅 Call with Dr. Sharma
           🕐 Monday, October 05 at 02:00 PM (30 minutes)
           🔗 Event Details · 📅 Open Live Calendar Grid"
```

### Example: Conflict Resolution
```
User:   "Book team sync with Priya tomorrow at 11am"

Agent:  [1] Extracts → tomorrow at 11am
        [2] Google Calendar → BUSY ⛔
        [3] Finds 3 alternative slots

Response: "⚠️ Slot is busy! Alternatives:
           [Option 1: 12:00 PM] [Option 2: 3:00 PM] [Option 3: Wed 11 AM]"

User:  clicks "Option 1" → Agent books immediately.
```

---

## 🧪 Running Automated Tests

```bash
# Run all tests
python -m unittest discover tests

# Individual tests
python -m unittest tests/test_nlp_extractor.py    # NLP & date parsing
python -m unittest tests/test_memory_service.py   # Database operations
python -m unittest tests/test_calendar_service.py # Calendar API (mocked)
python -m unittest tests/test_rag_service.py       # Vector search
```

---

## 📱 Stretch Goals

### 1. Free Push Notifications (ntfy.sh)
```env
NTFY_TOPIC=my-meeting-alerts-yourname
```
Visit `https://ntfy.sh/<your-topic>` or install the free ntfy app on Android/iOS.

### 2. Model Context Protocol (MCP) Server
Expose calendar tools to AI IDEs (Claude Desktop, Cursor, Antigravity):
```bash
pip install "mcp[cli]"
python stretch/mcp_calendar_server.py
```

---

## 🐛 Challenges & Solutions

| # | Challenge | Solution |
|---|---|---|
| 1 | LLM hallucinating wrong dates/years | Delegated all date math to `dateparser` with `PREFER_DATES_FROM=future` |
| 2 | LLM ignoring tool schema output | Enforced `tool_choice="required"` in Groq API |
| 3 | Naive datetime timezone errors | Set `RETURN_AS_TIMEZONE_AWARE=True` in dateparser |
| 4 | OAuth browser hang on cloud | Cached token in `GOOGLE_TOKEN_JSON` env var with auto-refresh |
| 5 | Multi-turn state loss between requests | Stored `pending_request` and `alternative_slots` in Streamlit session state |
| 6 | 512MB RAM crash on Render free tier | Replaced local PyTorch/ChromaDB with Pinecone hosted inference |
| 7 | `participants: null` Groq 400 validation error | Changed tool schema type from `"array"` to `["array", "null"]` |
| 8 | Unsafe model fallback (`models[0]` non-tool-call model) | Extended priority list, removed `models[0]` fallback entirely |

---

## 📄 License

MIT License — Free for portfolio and educational use.

---

## 👤 Author

**Shankhadeep Dey**
B.Tech CSCE | KIIT University
GitHub: [@Shankhadeep25](https://github.com/Shankhadeep25)
