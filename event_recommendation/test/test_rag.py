"""Tests for RAG retrieval functions."""

import os
import pytest
from langchain_core.documents import Document
from vector_db.chroma_store import RagStores


def test_format_event_card():
    """Test formatting event card."""
    from rag.retrieval import format_event_card

    doc = Document(
        page_content="Test event",
        metadata={
            "event_name": "Test Event",
            "event_type": "AQUATICS",
            "city": "Boston",
            "state": "Massachusetts",
            "center_name": "Test Center",
            "age_contains": "adults",
            "date_range": "Jan 1 - Dec 31",
            "time_slots": "10:00 AM",
            "duration": "1 hour",
            "instructor": "John Doe",
            "spots": "20",
            "source": "test.md",
        },
    )

    result = format_event_card(doc)
    assert "Test Event" in result
    assert "AQUATICS" in result
    assert "Boston" in result
    assert "Massachusetts" in result


def test_build_context_block():
    """Test building context block."""
    from rag.retrieval import build_context_block

    events = [
        Document(
            page_content="Event 1",
            metadata={"event_name": "Event 1", "source": "test1.md"},
        )
    ]
    activity_defs = [
        Document(
            page_content="Activity definition",
            metadata={"activity_heading": "AQUATICS", "source": "aquatics.md"},
        )
    ]

    result = build_context_block(events, activity_defs)
    assert "Retrieved Events" in result
    assert "Activity Definitions" in result
    assert "Event 1" in result
    assert "AQUATICS" in result


def test_rerank_no_reranker():
    """Test rerank function without reranker."""
    from rag.retrieval import rerank

    docs = [
        Document(page_content="Doc 1", metadata={}),
        Document(page_content="Doc 2", metadata={}),
    ]

    result = rerank("query", docs, top_n=1, reranker=None)
    assert len(result) == 1
    assert result[0].page_content == "Doc 1"


@pytest.mark.skipif(
    not os.path.exists("documents/Events") or not os.path.exists("documents/activityType"),
    reason="Documents directory not found",
)
def test_retrieve_activity_types():
    """Test retrieving activity types."""
    from vector_db.chroma_store import build_vectorstores
    from rag.input_documents.loader import load_documents
    from rag.retrieval import retrieve_activity_types
    import tempfile
    import shutil

    documents_path = "documents"
    events_files, activity_files = load_documents(documents_path)

    if events_files and activity_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            stores = build_vectorstores(
                events_files, activity_files, persist_dir=tmpdir
            )

            results = retrieve_activity_types(
                stores, "swimming", input_filter={}, k=3
            )

            assert isinstance(results, list)
            assert len(results) <= 3

