"""Tests for reviews vector store with where clause filtering."""

import os
import pytest
import shutil
import groq
import ollama
from vector_db.chroma_store import (
    RagStores,
    build_vectorstores,
    build_chroma_where,
    cleanup_vectorstores,
)
from utils.build_reviews_db import build_reviews_database
from rag.input_documents.loader import load_documents
from rag.retrieval import retrieve_reviews, get_review_scores

from dotenv import load_dotenv

load_dotenv()

ollama_client = ollama.Client(host="http://localhost:11434")
model = "llama3.2:latest"

@pytest.mark.skipif(
    not os.path.exists("documents/Events") 
    or not os.path.exists("documents/activityType")
    or not os.path.exists("documents/Reviews/reviews_rag_2000.csv"),
    reason="Required documents or reviews CSV not found",
)

def load_reviews(tmp_path) -> tuple[RagStores, str, str, str]:   # returns stores, persist_dir, db_path, reviews_db_path
    documents_path = "documents"
    reviews_csv_path = "documents/Reviews/reviews_rag_2000.csv"
    
    # Load documents
    events_files, activity_files = load_documents(documents_path)
    
    assert events_files, "No event files found"
    assert activity_files, "No activity type files found"
    assert os.path.exists(reviews_csv_path), f"Reviews CSV not found: {reviews_csv_path}"
    
    # Build vector stores with reviews
    persist_dir = str(tmp_path / "test_reviews_where")
    db_path = str(tmp_path / "test_events.db")
    reviews_db_path = str(tmp_path / "test_reviews.db")
    
    # Build reviews database upfront using utility
    # Build reviews database upfront using utility
    reviews_db = build_reviews_database(
        reviews_csv_path=reviews_csv_path,
        reviews_db_path=reviews_db_path,
        llm_client=ollama_client,
        model=model,
    )
    review_count = reviews_db.count_reviews()
    print(f"Loaded {review_count} reviews into database")
    assert review_count > 0, "No reviews were loaded"
    
    # Build vector stores (reviews DB must already exist)
    stores = build_vectorstores(
        events_files,
        activity_files,
        groq_client=ollama_client,
        model=model,
        persist_dir=persist_dir,
        db_path=db_path,
        reviews_db_path=reviews_db_path,
    )
    review_scores = stores.reviews.get_review_scores(event_types=["SENIOR CIRCUITS"], locations=["Summit Reach YMCA", "Pinecrest YMCA", "Seabrook Commons YMCA", "Harborlight YMCA", "Riverstone YMCA"])
    print(f"Review scores: {review_scores}")
    print(f"Review scores: {review_scores['activity_scores']}")
    print(f"Review scores: {review_scores['venue_scores']}")
    print(f"Review scores: {review_scores['activity_scores']['SENIOR CIRCUITS']}")
    print(f"Review scores: {review_scores['venue_scores']['Summit Reach YMCA']}")
    print(f"Review scores: {review_scores['venue_scores']['Pinecrest YMCA']}")
    print(f"Review scores: {review_scores['venue_scores']['Seabrook Commons YMCA']}")
    print(f"Review scores: {review_scores['venue_scores']['Harborlight YMCA']}")
    print(f"Review scores: {review_scores['venue_scores']['Riverstone YMCA']}")
    return stores, persist_dir, db_path, reviews_db_path

# test reviews for where clause filtering
# {'$and': [{'$or': [{'event_type': 'ARTHRITIS AQUA FITNESS'}, {'event_type': 'SENIOR CIRCUITS'}]}, {'$or': [{'location': 'Summit Reach YMCA'}, {'location': 'Summit Reach YMCA'}, {'location': 'Pinecrest YMCA'}, {'location': 'Pinecrest YMCA'}, {'location': 'Seabrook Commons YMCA'}, {'location': 'Seabrook Commons YMCA'}, {'location': 'Harborlight YMCA'}, {'location': 'Harborlight YMCA'}, {'location': 'Riverstone YMCA'}, {'location': 'Riverstone YMCA'}]}]}
def test_reviews_for_where_clause_filtering(tmp_path):
    # load reviews csv
    stores, persist_dir, db_path, reviews_db_path = load_reviews(tmp_path)

    event_types = ["CLASSIC MOVIES", "SENIOR CIRCUITS", "ARTHRITIS AQUA FITNESS"]
    #locations = ["Summit Reach YMCA", "Pinecrest YMCA", "Seabrook Commons YMCA", "Harborlight YMCA", "Riverstone YMCA"]
    locations = []
    if len(event_types) >= 2:
        test_event_types = list(event_types)[:2]
        # Query reviews from SQL database
        reviews = stores.reviews.query_reviews(event_types=test_event_types, limit=10)
        dict_scores = get_review_scores(stores, event_types=test_event_types, 
                            locations=locations)
        print(f"In test_reviews_for_where_clause_filtering **** PRINT dict_scores: {dict_scores}")
        print(f"In test_reviews_for_where_clause_filtering **** Found {len(reviews)} reviews")



    """
    #where_clause = build_chroma_where({"event_type": ["ARTHRITIS AQUA FITNESS", "SENIOR CIRCUITS"], "location": ["Summit Reach YMCA", "Summit Reach YMCA", "Pinecrest YMCA", "Pinecrest YMCA", "Seabrook Commons YMCA", "Seabrook Commons YMCA", "Harborlight YMCA", "Harborlight YMCA", "Riverstone YMCA", "Riverstone YMCA"]})
    where_clause = build_chroma_where({"event_type": ["SENIOR CIRCUITS"], "location": ["Summit Reach YMCA", "Summit Reach YMCA", "Pinecrest YMCA", "Pinecrest YMCA", "Seabrook Commons YMCA", "Seabrook Commons YMCA", "Harborlight YMCA", "Harborlight YMCA", "Riverstone YMCA", "Riverstone YMCA"]})
    print(f"In test_reviews_for_where_clause_filtering **** PRINT where_clause: {where_clause}")
    
    dict_scores = get_review_scores(stores, event_types=["SENIOR CIRCUITS"], 
                            locations=["Summit Reach YMCA", "Pinecrest YMCA", "Seabrook Commons YMCA", "Harborlight YMCA", "Riverstone YMCA"])
    print(f"In test_reviews_for_where_clause_filtering **** PRINT dict_scores: {dict_scores}")
    """
    

def test_reviews_store_with_where_clause_basic(tmp_path):
    """Test loading RAG store with reviews and basic where clause filtering."""
    stores, persist_dir, db_path, reviews_db_path = load_reviews(tmp_path)
    
    # Verify stores are loaded
    assert isinstance(stores, RagStores)
    assert stores.reviews is not None
    assert stores.activity_types is not None
    assert stores.events is not None
    # Verify reviews is a ReviewDB instance (not Chroma)
    from database.review_db import ReviewDB
    assert isinstance(stores.reviews, ReviewDB)
    
    # Check that reviews were loaded
    review_count = stores.reviews.count_reviews()
    print(f"Loaded {review_count} reviews into database")
    assert review_count > 0, "No reviews were loaded"
    
    # Test 1: Basic query without filters
    print("\n=== Test 1: Basic query without filters ===")
    results = stores.reviews.query_reviews(limit=5)
    assert len(results) > 0, "Should return some results"
    print(f"Found {len(results)} reviews")
    
    # Verify results have expected structure
    for doc in results:
        assert hasattr(doc, 'page_content'), "Document should have page_content"
        assert hasattr(doc, 'metadata'), "Document should have metadata"
        assert 'rating' in doc.metadata, "Review should have rating metadata"
        print(f"  - Rating: {doc.metadata.get('rating')}, "
              f"Event: {doc.metadata.get('event_type')}, "
              f"Location: {doc.metadata.get('location')}")
    
    # Test 2: Filter by rating
    print("\n=== Test 2: Filter by rating ===")
    results = stores.reviews.query_reviews(rating="5", limit=10)
    assert len(results) > 0, "Should return some 5-star reviews"
    print(f"Found {len(results)} reviews with rating 5")
    
    # Verify all results have rating 5
    for doc in results:
        assert doc.metadata.get("rating") == "5", f"Expected rating 5, got {doc.metadata.get('rating')}"
    
    # Test 3: Filter by event_type
    print("\n=== Test 3: Filter by event_type ===")
    # First, find what event types exist in the reviews
    all_reviews = stores.reviews.query_reviews(limit=100)
    event_types = set()
    for doc in all_reviews:
        event_type = doc.metadata.get("event_type")
        if event_type:
            event_types.add(event_type)
    
    print(f"Found event types in reviews: {list(event_types)[:5]}...")
    
    if event_types:
        # Test with first available event type
        test_event_type = list(event_types)[0]
        results = stores.reviews.query_reviews(
            event_types=[test_event_type],
            limit=10,
        )
        print(f"Found {len(results)} reviews for event type '{test_event_type}'")
        
        # Verify all results have the correct event type
        for doc in results:
            assert doc.metadata.get("event_type") == test_event_type, \
                f"Expected event_type '{test_event_type}', got '{doc.metadata.get('event_type')}'"
    
    # Test 4: Filter by location
    print("\n=== Test 4: Filter by location ===")
    # Find what locations exist
    locations = set()
    for doc in all_reviews:
        location = doc.metadata.get("location")
        if location:
            locations.add(location)
    
    print(f"Found locations in reviews: {list(locations)[:5]}...")
    
    if locations:
        # Test with first available location
        test_location = list(locations)[0]
        results = stores.reviews.query_reviews(
            locations=[test_location],
            limit=10,
        )
        print(f"Found {len(results)} reviews for location '{test_location}'")
        
        # Verify all results have the correct location
        for doc in results:
            assert doc.metadata.get("location") == test_location, \
                f"Expected location '{test_location}', got '{doc.metadata.get('location')}'"
    
    # Test 5: Multiple filters combined (rating AND event_type)
    print("\n=== Test 5: Multiple filters (rating AND event_type) ===")
    if event_types:
        test_event_type = list(event_types)[0]
        results = stores.reviews.query_reviews(
            rating="5",
            event_types=[test_event_type],
            limit=10,
        )
        print(f"Found {len(results)} reviews with rating 5 AND event_type '{test_event_type}'")
        
        # Verify all results match both criteria
        for doc in results:
            assert doc.metadata.get("rating") == "5", \
                f"Expected rating 5, got {doc.metadata.get('rating')}"
            assert doc.metadata.get("event_type") == test_event_type, \
                f"Expected event_type '{test_event_type}', got '{doc.metadata.get('event_type')}'"
    
    # Test 6: List filter (multiple event types with OR)
    print("\n=== Test 6: List filter (OR condition) ===")
    if len(event_types) >= 2:
        test_event_types = list(event_types)[:2]
        print(f"Querying for event types: {test_event_types}")
        
        results = stores.reviews.query_reviews(
            event_types=test_event_types,
            limit=20,
        )
        print(f"Found {len(results)} reviews for event types {test_event_types}")
        
        # Verify all results have one of the specified event types
        for doc in results:
            event_type = doc.metadata.get("event_type")
            assert event_type in test_event_types, \
                f"Expected one of {test_event_types}, got '{event_type}'"
    
    # Cleanup
    cleanup_vectorstores(stores)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(reviews_db_path):
        os.remove(reviews_db_path)
    
    print("\n✓ All where clause tests passed!")


@pytest.mark.skipif(
    not os.path.exists("documents/Reviews/reviews_rag_2000.csv"),
    reason="Reviews CSV file not found",
)
def test_retrieve_reviews_with_filters(tmp_path):
    """Test retrieve_reviews function with rating filter."""
    documents_path = "documents"
    reviews_csv_path = "documents/Reviews/reviews_rag_2000.csv"
    
    events_files, activity_files = load_documents(documents_path)
    
    if not events_files or not activity_files:
        pytest.skip("Required documents not found")
    
    persist_dir = str(tmp_path / "test_retrieve_reviews")
    db_path = str(tmp_path / "test_events_retrieve.db")
    reviews_db_path = str(tmp_path / "test_reviews_retrieve.db")
    
    # Build reviews database first
    build_reviews_database(
        reviews_csv_path=reviews_csv_path,
        reviews_db_path=reviews_db_path,
        llm_client=ollama_client,
        model=model,
    )
    
    stores = build_vectorstores(
        events_files,
        activity_files,
        groq_client=ollama_client,
        model=model,
        persist_dir=persist_dir,
        db_path=db_path,
        reviews_db_path=reviews_db_path,
    )
    
    # Test retrieve_reviews without filter
    results = retrieve_reviews(
        stores,
        user_question="swimming",
        k=5,
    )
    assert len(results) > 0, "Should return some reviews"
    print(f"Retrieved {len(results)} reviews without filter")
    
    # Test retrieve_reviews with rating filter
    results_filtered = retrieve_reviews(
        stores,
        user_question="swimming",
        k=5,
        rating_filter=5,
    )
    print(f"Retrieved {len(results_filtered)} reviews with rating filter=5")
    
    # Verify all filtered results have rating 5
    for doc in results_filtered:
        assert doc.metadata.get("rating") == "5", \
            f"Expected rating 5, got {doc.metadata.get('rating')}"
    
    # Cleanup
    cleanup_vectorstores(stores)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    if os.path.exists(db_path):
        os.remove(db_path)



if __name__ == "__main__":
    """Main function to run the chatbot."""
    pytest.main([__file__, "-v", "-s"])
