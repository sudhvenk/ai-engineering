"""Tests for ChromaDB vector store operations."""

import pytest
import os
import shutil
from vector_db.chroma_store import (
    RagStores,
    build_vectorstores,
    load_vectorstores,
    cleanup_vectorstores,
    build_chroma_where,
)


def test_build_chroma_where_single_filter():
    """Test building Chroma where clause with single filter."""
    filter_dict = {"city": "boston"}
    result = build_chroma_where(filter_dict)
    assert result == {"city": "boston"}


def test_build_chroma_where_multiple_filters():
    """Test building Chroma where clause with multiple filters."""
    filter_dict = {"city": "boston", "age_contains": "adults"}
    result = build_chroma_where(filter_dict)
    assert "$and" in result
    assert len(result["$and"]) == 2


def test_build_chroma_where_list_filter():
    """Test building Chroma where clause with list filter."""
    filter_dict = {"city": ["boston", "cambridge"]}
    result = build_chroma_where(filter_dict)
    assert "$or" in result["$and"][0] if "$and" in result else "$or" in result


def test_build_chroma_where_empty():
    """Test building Chroma where clause with empty filter."""
    filter_dict = {}
    result = build_chroma_where(filter_dict)
    assert result is None


def test_build_chroma_where_empty_string():
    """Test building Chroma where clause filters out empty strings."""
    filter_dict = {"city": "boston", "state": ""}
    result = build_chroma_where(filter_dict)
    # Should only include city, not empty state
    if "$and" in result:
        assert len(result["$and"]) == 1
    else:
        assert "city" in result


@pytest.mark.skipif(
    not os.path.exists("documents/Events") or not os.path.exists("documents/activityType"),
    reason="Documents directory not found",
)
def test_build_vectorstores(tmp_path):
    """Test building vector stores from documents."""
    from rag.input_documents.loader import load_documents

    documents_path = "documents"
    events_files, activity_files = load_documents(documents_path)

    if events_files and activity_files:
        persist_dir = str(tmp_path / "test_chroma")
        stores = build_vectorstores(
            events_files, activity_files, persist_dir=persist_dir
        )

        assert isinstance(stores, RagStores)
        assert stores.events is not None
        assert stores.activity_types is not None

        # Cleanup
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

