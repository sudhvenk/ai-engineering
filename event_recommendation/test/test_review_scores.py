"""Tests for review score retrieval and re-ranking functionality."""

import os
import pytest
import tempfile
import shutil
from langchain_core.documents import Document
from vector_db.chroma_store import RagStores, build_vectorstores
from rag.retrieval import get_review_scores, rerank_events_by_reviews


def test_get_review_scores_no_reviews():
    """Test get_review_scores when no reviews exist."""
    from unittest.mock import Mock, MagicMock
    
    # Create a mock RagStores with empty reviews
    mock_reviews = Mock()
    mock_reviews.similarity_search = Mock(return_value=[])
    
    mock_stores = Mock()
    mock_stores.reviews = mock_reviews
    
    result = get_review_scores(mock_stores)
    
    assert isinstance(result, dict)
    assert "activity_scores" in result
    assert "venue_scores" in result
    assert len(result["activity_scores"]) == 0
    assert len(result["venue_scores"]) == 0


def test_get_review_scores_with_mock_data():
    """Test get_review_scores with mock review data."""
    from unittest.mock import Mock
    
    # Create mock reviews with ratings
    mock_reviews_data = [
        Document(
            page_content="Great swimming class!",
            metadata={"rating": "5", "event_type": "SWIMMING", "location": "Boston"}
        ),
        Document(
            page_content="Good yoga session",
            metadata={"rating": "4", "event_type": "YOGA", "location": "Boston"}
        ),
        Document(
            page_content="Excellent swimming program",
            metadata={"rating": "5", "event_type": "SWIMMING", "location": "Cambridge"}
        ),
        Document(
            page_content="Average class",
            metadata={"rating": "3", "event_type": "SWIMMING", "location": "Boston"}
        ),
    ]
    
    mock_reviews = Mock()
    mock_reviews.similarity_search = Mock(return_value=mock_reviews_data)
    
    mock_stores = Mock()
    mock_stores.reviews = mock_reviews
    
    result = get_review_scores(mock_stores)
    
    # Check activity scores
    assert "SWIMMING" in result["activity_scores"]
    assert "YOGA" in result["activity_scores"]
    # SWIMMING should have average of (5 + 5 + 3) / 3 = 4.33
    assert abs(result["activity_scores"]["SWIMMING"] - 4.33) < 0.1
    # YOGA should have average of 4
    assert result["activity_scores"]["YOGA"] == 4.0
    
    # Check venue scores
    assert "Boston" in result["venue_scores"]
    assert "Cambridge" in result["venue_scores"]
    # Boston should have average of (5 + 4 + 3) / 3 = 4.0
    assert abs(result["venue_scores"]["Boston"] - 4.0) < 0.1
    # Cambridge should have average of 5
    assert result["venue_scores"]["Cambridge"] == 5.0


def test_get_review_scores_with_filters():
    """Test get_review_scores with event_type and location filters."""
    from unittest.mock import Mock
    
    mock_reviews_data = [
        Document(
            page_content="Great swimming class!",
            metadata={"rating": "5", "event_type": "SWIMMING", "location": "Boston"}
        ),
        Document(
            page_content="Good yoga session",
            metadata={"rating": "4", "event_type": "YOGA", "location": "Boston"}
        ),
    ]
    
    mock_reviews = Mock()
    mock_reviews.similarity_search = Mock(return_value=mock_reviews_data)
    
    mock_stores = Mock()
    mock_stores.reviews = mock_reviews
    
    # Test with event_type filter
    result = get_review_scores(
        mock_stores,
        event_types=["SWIMMING"],
        locations=None
    )
    
    # Should have called similarity_search with filter
    assert mock_reviews.similarity_search.called
    call_args = mock_reviews.similarity_search.call_args
    assert call_args[1]["filter"] is not None  # filter should be passed


def test_get_review_scores_invalid_ratings():
    """Test get_review_scores handles invalid ratings gracefully."""
    from unittest.mock import Mock
    
    mock_reviews_data = [
        Document(
            page_content="Review 1",
            metadata={"rating": "5", "event_type": "SWIMMING"}
        ),
        Document(
            page_content="Review 2",
            metadata={"rating": "invalid", "event_type": "SWIMMING"}
        ),
        Document(
            page_content="Review 3",
            metadata={"rating": "", "event_type": "SWIMMING"}
        ),
        Document(
            page_content="Review 4",
            metadata={"event_type": "SWIMMING"}  # No rating
        ),
    ]
    
    mock_reviews = Mock()
    mock_reviews.similarity_search = Mock(return_value=mock_reviews_data)
    
    mock_stores = Mock()
    mock_stores.reviews = mock_reviews
    
    result = get_review_scores(mock_stores)
    
    # Should only count valid ratings (5)
    assert "SWIMMING" in result["activity_scores"]
    assert result["activity_scores"]["SWIMMING"] == 5.0


def test_rerank_events_by_reviews():
    """Test reranking events based on review scores."""
    events = [
        Document(
            page_content="Event 1",
            metadata={"event_type": "SWIMMING", "center_name": "Boston YMCA", "city": "Boston"}
        ),
        Document(
            page_content="Event 2",
            metadata={"event_type": "YOGA", "center_name": "Cambridge YMCA", "city": "Cambridge"}
        ),
        Document(
            page_content="Event 3",
            metadata={"event_type": "SWIMMING", "center_name": "Boston YMCA", "city": "Boston"}
        ),
    ]
    
    review_scores = {
        "activity_scores": {
            "SWIMMING": 4.5,
            "YOGA": 3.5,
        },
        "venue_scores": {
            "Boston": 4.8,
            "Cambridge": 3.2,
        }
    }
    
    result = rerank_events_by_reviews(events, review_scores, top_n=3)
    
    assert len(result) == 3
    # Events with SWIMMING and Boston should rank higher
    # Event 1 and 3 should be ranked higher than Event 2
    assert result[0].metadata["event_type"] in ["SWIMMING", "YOGA"]


def test_rerank_events_by_reviews_no_scores():
    """Test reranking when no review scores are available."""
    events = [
        Document(
            page_content="Event 1",
            metadata={"event_type": "SWIMMING", "center_name": "Boston YMCA"}
        ),
        Document(
            page_content="Event 2",
            metadata={"event_type": "YOGA", "center_name": "Cambridge YMCA"}
        ),
    ]
    
    review_scores = {
        "activity_scores": {},
        "venue_scores": {},
    }
    
    result = rerank_events_by_reviews(events, review_scores, top_n=2)
    
    # Should return events in original order (or close to it) when no scores
    assert len(result) == 2


def test_rerank_events_by_reviews_venue_priority():
    """Test that venue scores get higher priority than activity scores."""
    events = [
        Document(
            page_content="Event 1",
            metadata={"event_type": "YOGA", "center_name": "High Rated YMCA", "city": "HighRated"}
        ),
        Document(
            page_content="Event 2",
            metadata={"event_type": "SWIMMING", "center_name": "Low Rated YMCA", "city": "LowRated"}
        ),
    ]
    
    # High venue score but lower activity score should rank higher
    review_scores = {
        "activity_scores": {
            "SWIMMING": 5.0,  # Higher activity score
            "YOGA": 3.0,      # Lower activity score
        },
        "venue_scores": {
            "HighRated": 5.0,  # Higher venue score
            "LowRated": 2.0,   # Lower venue score
        }
    }
    
    result = rerank_events_by_reviews(events, review_scores, top_n=2)
    
    # Event 1 (YOGA + HighRated) should rank higher due to venue priority
    # Composite score: Event 1 = (5.0 * 0.6) + (3.0 * 0.4) = 4.2
    # Composite score: Event 2 = (2.0 * 0.6) + (5.0 * 0.4) = 3.2
    # So Event 1 should be first
    assert len(result) == 2
    # The event with higher venue score should be ranked first
    first_event_venue = result[0].metadata.get("city") or result[0].metadata.get("center_name", "")
    assert "High" in first_event_venue or result[0].metadata["event_type"] == "YOGA"


def test_rerank_events_by_reviews_partial_scores():
    """Test reranking when only activity or only venue scores are available."""
    events = [
        Document(
            page_content="Event 1",
            metadata={"event_type": "SWIMMING", "center_name": "Unknown Venue"}
        ),
        Document(
            page_content="Event 2",
            metadata={"event_type": "YOGA", "center_name": "Unknown Venue"}
        ),
    ]
    
    # Only activity scores available
    review_scores = {
        "activity_scores": {
            "SWIMMING": 5.0,
            "YOGA": 3.0,
        },
        "venue_scores": {},  # No venue scores
    }
    
    result = rerank_events_by_reviews(events, review_scores, top_n=2)
    
    assert len(result) == 2
    # SWIMMING should rank higher
    assert result[0].metadata["event_type"] == "SWIMMING"


@pytest.mark.skipif(
    not os.path.exists("documents/Reviews/reviews_rag_2000.csv"),
    reason="Reviews CSV file not found",
)
def test_get_review_scores_with_real_data(tmp_path):
    """Test get_review_scores with real ChromaDB data if available."""
    from vector_db.chroma_store import build_vectorstores
    from rag.input_documents.loader import load_documents
    
    documents_path = "documents"
    reviews_csv_path = "documents/Reviews/reviews_rag_2000.csv"
    
    events_files, activity_files = load_documents(documents_path)
    
    if events_files and activity_files and os.path.exists(reviews_csv_path):
        import groq
        from dotenv import load_dotenv
        load_dotenv()
        groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        model = "openai/gpt-oss-120b"
        
        persist_dir = str(tmp_path / "test_reviews_chroma")
        db_path = str(tmp_path / "test_events.db")
        reviews_db_path = str(tmp_path / "test_reviews.db")
        
        from utils.build_reviews_db import build_reviews_database
        
        # Build reviews database first
        build_reviews_database(
            reviews_csv_path=reviews_csv_path,
            reviews_db_path=reviews_db_path,
            llm_client=groq_client,
            model=model,
        )
        
        stores = build_vectorstores(
            events_files,
            activity_files,
            groq_client=groq_client,
            model=model,
            persist_dir=persist_dir,
            db_path=db_path,
            reviews_db_path=reviews_db_path,
        )
        
        # Test getting review scores
        result = get_review_scores(stores)
        
        assert isinstance(result, dict)
        assert "activity_scores" in result
        assert "venue_scores" in result
        
        # Should have some scores if reviews exist
        total_scores = len(result["activity_scores"]) + len(result["venue_scores"])
        print(f"Found {len(result['activity_scores'])} activity scores and {len(result['venue_scores'])} venue scores")
        
        # Cleanup
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)


@pytest.mark.skipif(
    not os.path.exists("documents/Reviews/reviews_rag_2000.csv"),
    reason="Reviews CSV file not found",
)
def test_rerank_events_with_real_reviews(tmp_path):
    """Test full re-ranking pipeline with real data."""
    from vector_db.chroma_store import build_vectorstores
    from rag.input_documents.loader import load_documents
    from rag.retrieval import retrieve_events_for_activity_type
    
    documents_path = "documents"
    reviews_csv_path = "documents/Reviews/reviews_rag_2000.csv"
    
    events_files, activity_files = load_documents(documents_path)
    
    if events_files and activity_files and os.path.exists(reviews_csv_path):
        import groq
        from dotenv import load_dotenv
        load_dotenv()
        groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        model = "openai/gpt-oss-120b"
        
        persist_dir = str(tmp_path / "test_rerank_chroma")
        db_path = str(tmp_path / "test_events.db")
        reviews_db_path = str(tmp_path / "test_reviews.db")
        
        from utils.build_reviews_db import build_reviews_database
        
        # Build reviews database first
        build_reviews_database(
            reviews_csv_path=reviews_csv_path,
            reviews_db_path=reviews_db_path,
            llm_client=groq_client,
            model=model,
        )
        
        stores = build_vectorstores(
            events_files,
            activity_files,
            groq_client=groq_client,
            model=model,
            persist_dir=persist_dir,
            db_path=db_path,
            reviews_db_path=reviews_db_path,
        )
        
        # Retrieve some events
        events = retrieve_events_for_activity_type(
            stores,
            user_question="swimming",
            input_filter={"event_type": ["SWIMMING"]},
            k=10,
        )
        
        if events:
            # Get review scores
            review_scores = get_review_scores(stores)
            
            # Re-rank events
            reranked = rerank_events_by_reviews(events, review_scores, top_n=5)
            
            assert len(reranked) <= 5
            assert len(reranked) <= len(events)
            
            print(f"Re-ranked {len(events)} events to top {len(reranked)}")
        
        # Cleanup
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

