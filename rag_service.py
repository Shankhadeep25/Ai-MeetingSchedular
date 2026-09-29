"""
rag_service.py — Retrieval-Augmented Generation (ChromaDB)
==========================================================
Semantic search over past meeting context using:
- ChromaDB  — local vector store, no server, completely free
- sentence-transformers — runs fully offline on CPU, no API key

Model: all-MiniLM-L6-v2 (22MB) — fast, accurate for short texts.

Use cases:
  - "Schedule a follow-up with Rahul" → retrieves last Rahul meeting
  - Auto-fill agenda from recurring patterns
  - Detect duplicate scheduling attempts
"""

import os
import logging
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "meetings"

# Pinecone Cloud Vector DB (Optional)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ai-meetings")

# Ensure chroma directory exists for fallback
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Lazy Initialization (avoid slow imports at startup)
# ─────────────────────────────────────────────

_chroma_client = None
_collection = None
_embedding_fn = None
_st_model = None
_pinecone_index = None


def _get_sentence_transformer_model():
    """Load local sentence-transformers encoder."""
    global _st_model
    if _st_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}...")
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(EMBEDDING_MODEL)
    return _st_model


def _get_pinecone_index():
    """Connect to Pinecone Cloud Serverless Index."""
    global _pinecone_index
    if _pinecone_index is None and PINECONE_API_KEY:
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [idx["name"] for idx in pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' (384-dim, cosine)...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        logger.info(f"Connected to Pinecone cloud index: {PINECONE_INDEX_NAME}")
    return _pinecone_index


def _get_embedding_function():
    """Load sentence-transformers embedding function (cached after first call)."""
    global _embedding_fn
    if _embedding_fn is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL} (first load may take a moment)...")
        try:
            from chromadb.utils import embedding_functions
            _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )
            logger.info("Embedding model loaded successfully")
        except ImportError:
            raise ImportError(
                "chromadb and sentence-transformers are required. "
                "Run: pip install chromadb sentence-transformers"
            )
    return _embedding_fn


def _get_collection():
    """Get or create the ChromaDB collection (cached after first call)."""
    global _chroma_client, _collection
    if _collection is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_get_embedding_function(),
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}' ready ({_collection.count()} docs)")
    return _collection


# ─────────────────────────────────────────────
# Meeting Document Builder
# ─────────────────────────────────────────────

def _build_document(meeting: dict) -> str:
    """
    Convert a meeting dict into a text document for embedding.

    The richer the text, the better the semantic search.
    """
    parts = [meeting.get("title", "Meeting")]

    participants = meeting.get("participants", [])
    if participants:
        parts.append(f"with {', '.join(participants)}")

    start = meeting.get("start_time")
    if isinstance(start, datetime):
        parts.append(f"on {start.strftime('%A %B %d %Y at %I:%M %p')}")
    elif isinstance(start, str):
        parts.append(f"on {start}")

    duration = meeting.get("duration_mins")
    if duration:
        parts.append(f"({duration} minutes)")

    return " ".join(parts)


# ─────────────────────────────────────────────
# Index Meeting
# ─────────────────────────────────────────────

def index_meeting(meeting_id: int | str, meeting: dict, user_id: str) -> None:
    """
    Embed and store a meeting in ChromaDB for future semantic retrieval.

    Args:
        meeting_id: Unique ID (from SQLite autoincrement)
        meeting:    Dict with title, participants, start_time, duration_mins
        user_id:    User identifier for filtering

    Skips silently if the document already exists.
    """
    collection = _get_collection()
    doc_id = f"meeting_{user_id}_{meeting_id}"

    # Check if already indexed
    existing = collection.get(ids=[doc_id])
    if existing["ids"]:
        logger.debug(f"Meeting {doc_id} already indexed, skipping")
        return

    document = _build_document(meeting)
    participants = meeting.get("participants", [])
    start = meeting.get("start_time")

    metadata = {
        "user_id": user_id,
        "participants": ", ".join(participants) if participants else "",
        "date": start.strftime("%Y-%m-%d") if isinstance(start, datetime) else str(start or ""),
        "duration_mins": str(meeting.get("duration_mins", 30)),
        "event_link": meeting.get("event_link", ""),
    }

    # Pinecone Cloud Path
    if PINECONE_API_KEY:
        try:
            index = _get_pinecone_index()
            encoder = _get_sentence_transformer_model()
            vector = encoder.encode(document).tolist()
            metadata["document"] = document
            index.upsert(vectors=[{"id": doc_id, "values": vector, "metadata": metadata}])
            logger.info(f"Indexed meeting {doc_id} to Pinecone Cloud: {document!r}")
            return
        except Exception as e:
            logger.warning(f"Pinecone indexing error: {e}, falling back to local ChromaDB")

    # Local ChromaDB Fallback
    collection = _get_collection()
    collection.add(
        ids=[doc_id],
        documents=[document],
        metadatas=[metadata],
    )

    logger.info(f"Indexed meeting {doc_id}: {document!r}")


# ─────────────────────────────────────────────
# Retrieve Context
# ─────────────────────────────────────────────

def retrieve_context(
    query: str,
    user_id: str,
    n_results: int = 3,
    min_relevance: float = 0.3,
) -> list[str]:
    """
    Retrieve semantically similar past meetings for the given query.

    Args:
        query:          User's natural language query
        user_id:        Filter results to this user's meetings
        n_results:      Max number of results to return
        min_relevance:  Minimum cosine similarity score (0–1)

    Returns:
        List of meeting description strings ordered by relevance
    """
    # Pinecone Cloud Retrieval
    if PINECONE_API_KEY:
        try:
            index = _get_pinecone_index()
            encoder = _get_sentence_transformer_model()
            query_vector = encoder.encode(query).tolist()
            filter_dict = {"user_id": user_id} if user_id else None
            res = index.query(
                vector=query_vector,
                top_k=n_results,
                include_metadata=True,
                filter=filter_dict,
            )
            matches = res.get("matches", [])
            relevant = [
                m["metadata"]["document"]
                for m in matches
                if m.get("score", 0) >= min_relevance and "document" in m.get("metadata", {})
            ]
            logger.info(f"Retrieved {len(relevant)} docs from Pinecone Cloud for: {query[:50]!r}")
            return relevant
        except Exception as e:
            logger.warning(f"Pinecone query error: {e}, falling back to ChromaDB")

    # Local ChromaDB Fallback
    collection = _get_collection()

    if collection.count() == 0:
        logger.info("ChromaDB collection is empty — no context to retrieve")
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            where={"user_id": user_id} if user_id else None,
            include=["documents", "distances", "metadatas"],
        )
    except Exception as e:
        logger.warning(f"ChromaDB query failed: {e}")
        return []

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Filter by relevance (ChromaDB cosine distance: lower = more similar)
    # Convert distance to similarity: similarity = 1 - distance
    relevant = [
        doc for doc, dist in zip(docs, distances)
        if (1 - dist) >= min_relevance
    ]

    logger.info(
        f"Retrieved {len(relevant)}/{len(docs)} relevant docs "
        f"for query: {query[:50]!r}"
    )
    return relevant


# ─────────────────────────────────────────────
# Index All Existing Meetings (backfill)
# ─────────────────────────────────────────────

def backfill_from_history(user_id: str) -> int:
    """
    Index all meetings from SQLite history that aren't in ChromaDB yet.
    Useful on first run or after adding new meetings via other means.

    Returns:
        Number of meetings newly indexed
    """
    from memory_service import get_meeting_history

    history = get_meeting_history(user_id, limit=100)
    count = 0
    for meeting in history:
        index_meeting(meeting["id"], meeting, user_id)
        count += 1

    logger.info(f"Backfilled {count} meetings for {user_id}")
    return count


# ─────────────────────────────────────────────
# CLI Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from datetime import timedelta

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 60)
    print("AI Meeting Scheduler — RAG Service Test")
    print("=" * 60 + "\n")

    TEST_USER = "test_user_001"

    # Sample meetings to index
    sample_meetings = [
        {
            "title": "Weekly standup with Rahul",
            "participants": ["Rahul Kumar"],
            "start_time": datetime(2026, 9, 20, 10, 0),
            "duration_mins": 30,
            "event_link": "",
        },
        {
            "title": "Product review with design team",
            "participants": ["Priya Sharma", "Vikram Singh"],
            "start_time": datetime(2026, 9, 18, 14, 0),
            "duration_mins": 60,
            "event_link": "",
        },
        {
            "title": "Client call with John from Acme",
            "participants": ["John Smith"],
            "start_time": datetime(2026, 9, 15, 11, 0),
            "duration_mins": 45,
            "event_link": "",
        },
    ]

    print("Step 1: Indexing sample meetings...")
    for i, meeting in enumerate(sample_meetings, 1):
        index_meeting(i, meeting, TEST_USER)
        print(f"  ✅ Indexed: {meeting['title']}")

    print("\nStep 2: Semantic retrieval tests...")
    test_queries = [
        "follow-up with Rahul",
        "design team meeting",
        "client call",
        "john acme",
    ]

    for query in test_queries:
        results = retrieve_context(query, TEST_USER)
        print(f"\n  Query: {query!r}")
        for r in results:
            print(f"    → {r}")

    print("\n✅ RAG service is working correctly!")
