# AI Meeting Scheduler — Development & Process Log

> **Repository:** [Shankhadeep25/Ai-MeetingSchedular](https://github.com/Shankhadeep25/Ai-MeetingSchedular)  
> **Architecture:** 100% Free-Tier GenAI Stack (Groq Llama 3.3 70B, Google Calendar OAuth2, SQLite, ChromaDB, Streamlit)

---

## 1. Project Directory Structure

```
Ai-MeetingSchedular/
├── .env.example              # Template for free environment keys & configuration
├── .gitignore                # Protects credentials, tokens, local DBs, and virtualenvs
├── requirements.txt          # Pinned dependencies for zero-cost stack
├── README.md                 # Project documentation & step-by-step setup guide
├── PROCESS_LOG.md            # Detailed audit of mistakes, design decisions & fixes
├── app.py                    # Streamlit conversational web interface (Phase 5)
├── agent.py                  # Agentic loop orchestrator & multi-turn state (Phase 3 & 4)
├── nlp_extractor.py          # Groq Llama 3.3 tool calling & dateparser (Phase 1)
├── calendar_service.py       # Google Calendar API OAuth2 & conflict detection (Phase 2)
├── memory_service.py         # SQLite persistent storage & preference inference (Phase 3)
├── rag_service.py            # ChromaDB + sentence-transformers semantic retrieval (Phase 3)
│
├── tests/                    # Automated unit & integration tests
│   ├── test_nlp_extractor.py
│   ├── test_memory_service.py
│   ├── test_calendar_service.py
│   └── test_rag_service.py
│
├── stretch/                  # Stretch goal extensions
│   ├── notify_service.py     # ntfy.sh zero-cost push notification dispatcher
│   └── mcp_calendar_server.py # FastMCP server exposing calendar tools
│
└── data/                     # Local auto-created directories (gitignored)
    ├── meetings.db           # SQLite database
    └── chroma/               # ChromaDB vector index
```

---

## 2. Mistake & Risk Analysis Log

| # | Component | Identified Mistake / Gotcha | Root Cause | Implemented Solution |
|---|---|---|---|---|
| **1** | **NLP Date Parsing** | LLM hallucinating absolute dates / years (e.g. inventing 2023 or wrong year). | LLM pretraining cutoff & non-deterministic date arithmetic. | Extracted only verbatim `date_phrase` via Groq tool calling, delegating all math to `dateparser` with `PREFER_DATES_FROM="future"`. |
| **2** | **NLP Tool Calling** | LLM outputting plain chat markdown instead of schema JSON. | Models occasionally default to conversational output when ambiguity occurs. | Enforced `tool_choice="required"` in Groq API, forcing function arguments even on ambiguous input. Added Pydantic schema validation. |
| **3** | **Timezone Safety** | Google Calendar API rejecting naive datetimes or assigning events to UTC. | Python datetime objects created without timezone offsets. | Set `RETURN_AS_TIMEZONE_AWARE: True` in `dateparser` and explicit IANA timezone (`Asia/Kolkata` / configurable) in Google Calendar API payload. |
| **4** | **OAuth Browser Hanging** | OAuth local server hangs in headless/background execution. | `InstalledAppFlow.run_local_server()` waiting indefinitely on browser interaction. | Cached credentials in `token.json` with automatic token refresh (`creds.refresh(Request())`) so OAuth consent only occurs once. |
| **5** | **State Loss in Multi-turn Flow** | User picking "Option 2" loses original meeting context (title, attendees). | Stateless HTTP / LLM invocations do not preserve previous context. | Maintained `pending_request` and `alternative_slots` inside session state, allowing one-click resolution. |
| **6** | **Cold Startup Latency** | ChromaDB and SentenceTransformer import taking 5-10 seconds on app load. | Eager importing of PyTorch / HuggingFace model weights. | Implemented lazy loading in `rag_service.py` (`_get_collection()` / `_get_embedding_function()`) initialized only on first search/index. |
| **7** | **Zero-Cost Violation** | Paid external APIs (OpenAI embeddings, Twilio SMS). | Standard industry templates assume paid cloud SaaS. | Replaced OpenAI with local `all-MiniLM-L6-v2` (22MB CPU) and Twilio with free open-source `ntfy.sh`. |
| **8** | **nlp_extractor.py SyntaxError** | `SyntaxError: invalid syntax` at line 284 — a duplicate `else` block from a previous partial edit left orphaned code after an already-closed `else`. | Incomplete prior edit merging two error-handler variants. | Removed the duplicate 24-line `elif/else` block; left only the canonical error handler. |
| **9** | **Wrong Groq Model Auto-selected** | `get_best_available_model()` fell back to `models[0]` which was `canopylabs/orpheus-arabic-saudi` — a model that requires terms acceptance and doesn't support tool calling. | The fallback to `models[0]` was unsafe when the account has non-standard models. | Removed `models[0]` fallback entirely. Extended priority list to include `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, and `qwen/qwen3.8-27b` which are available on this account. Updated `.env` `GROQ_MODEL=openai/gpt-oss-120b`. |

---

## 3. Step-by-Step Changes Completed

### Phase 1: Core NLP & Schema
- Created Pydantic `MeetingRequest` schema enforcing validation on `intent`, `duration_minutes`, and `is_ambiguous`.
- Created `nlp_extractor.py` using Groq's `llama-3.3-70b-versatile` tool calling.
- Integrated `dateparser` with future preference and timezone awareness.
- Added comprehensive unit tests in `tests/test_nlp_extractor.py`.

### Phase 2: Calendar Integration
- Implemented `calendar_service.py` handling Google Calendar OAuth2.
- Implemented `check_availability()` using `freebusy.query()`.
- Implemented `create_event()` with attendees and reminders.
- Implemented `search_events()` and `delete_event()` to enable real Google Calendar meeting cancellation.
- Implemented `find_alternative_slots()` probing working hours over 3 subsequent days.
- Added unit tests with mock Google API client in `tests/test_calendar_service.py`.

### Phase 3: Memory & RAG Services
- Implemented `memory_service.py` with SQLAlchemy models: `User`, `MeetingHistory`, and `Preference`.
- Implemented `infer_preferences()` to detect peak hours and common durations.
- Implemented `rag_service.py` with ChromaDB persistent storage and sentence-transformers.
- Added unit tests in `tests/test_memory_service.py` and `tests/test_rag_service.py`.

### Phase 4: Agent Orchestrator
- Implemented `agent.py` orchestrating the full loop:
  `extract` → `clarify` → `RAG retrieval` → `preferences` → `availability` → `book` → `log`.
- Connected multi-turn state handling for both ambiguous prompts and slot conflict picks.
- Built interactive CLI runner (`run_terminal()`) for terminal testing.

### Phase 5: Streamlit Chat Application
- Implemented `app.py` with custom dark/light adaptive styling.
- Added sidebar with User ID switcher, timezone selector, duration preference slider, and real-time service health checks.
- Added upcoming meetings feed in sidebar with direct Google Calendar links.
- Added dynamic interactive quick-action chips and conflict resolution buttons.

### Phase 6: Stretch Goals
- Implemented `stretch/notify_service.py` using `ntfy.sh` for push notifications.
- Integrated automatic push alert firing in `agent.py` when an event is booked.
- Implemented `stretch/mcp_calendar_server.py` exposing calendar tools over Model Context Protocol (FastMCP).

### Phase 7: Cloud Migration & Bug Fixes (Session 3)
- Migrated relational DB: SQLite → **Neon Cloud PostgreSQL** (`ap-southeast-1` Singapore region).
  - `memory_service.py` reads `DATABASE_URL` from env; falls back to SQLite when unset.
  - Added `psycopg2-binary` to `requirements.txt`.
- Migrated vector store: ChromaDB local → **Pinecone Cloud Serverless** (free tier).
  - `rag_service.py` reads `PINECONE_API_KEY`; falls back to local ChromaDB.
  - Added `pinecone>=5.0.0` to `requirements.txt`.
- **Fixed SyntaxError** in `nlp_extractor.py` (duplicate orphaned `else` block at line 284).
- **Fixed Groq model auto-selection**: removed unsafe `models[0]` fallback; extended priority list to include `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`; set `GROQ_MODEL=openai/gpt-oss-120b` in `.env`.
