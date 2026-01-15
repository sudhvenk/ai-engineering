"""Utility to build reviews database from CSV using either Groq or Ollama client."""

import os
import json
from typing import Optional, Any
from database.review_db import ReviewDB, ReviewRecord, init_reviews_database


def build_reviews_database(
    reviews_csv_path: str,
    reviews_db_path: str,
    llm_client: Optional[Any] = None,
    model: str = "openai/gpt-oss-120b",
    use_llm: bool = True,
    batch_size: int = 10,
) -> ReviewDB:
    """
    Build reviews database from CSV file using either Groq or Ollama client.
    
    This is a unified utility that accepts either a Groq or Ollama client and
    processes reviews CSV to extract metadata (event_type, location, sentiment)
    using LLM, then stores them in a SQLite database.
    
    Args:
        reviews_csv_path: Path to reviews CSV file
        reviews_db_path: Path to SQLite database for reviews (will be created if doesn't exist)
        llm_client: Either groq.Groq or ollama.Client instance. If None, uses regex-based extraction
        model: Model name to use for LLM calls (default: "openai/gpt-oss-120b" for Groq, "llama3.2:latest" for Ollama)
        use_llm: If True, use LLM for metadata extraction; otherwise use regex
        batch_size: Number of reviews to process in a single LLM call (when use_llm=True)
        
    Returns:
        ReviewDB instance
        
    Raises:
        FileNotFoundError: If reviews CSV file doesn't exist
        ValueError: If llm_client is provided but is not a recognized type
    """
    if not os.path.exists(reviews_csv_path):
        raise FileNotFoundError(f"Reviews CSV file not found: {reviews_csv_path}")
    
    # Check if reviews database already exists and has data
    if os.path.exists(reviews_db_path):
        reviews_db = ReviewDB(reviews_db_path)
        review_count = reviews_db.count_reviews()
        if review_count > 0:
            print(f"Reviews database already exists with {review_count} reviews. Reusing existing database.")
            return reviews_db
    
    # Initialize reviews database
    init_reviews_database(reviews_db_path)
    reviews_db = ReviewDB(reviews_db_path)
    reviews_db.clear_reviews()  # Clear existing reviews
    
    # Determine client type and process reviews
    if use_llm and llm_client is not None:
        # Import here to avoid circular imports
        from rag.reviews_processing import build_review_documents, REVIEW_METADATA_SYSTEM_PROMPT
        from langchain_core.documents import Document
        
        # Check client type
        client_type = type(llm_client).__name__
        
        if client_type == "Groq":
            # Use Groq client - use existing function that processes with Groq
            from rag.reviews_processing import _extract_metadata_with_llm
            import csv
            
            reviews_data = []
            with open(reviews_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    review_text = row.get("review_text", "").strip()
                    if not review_text:
                        continue
                    reviews_data.append({
                        "review_text": review_text,
                        "created_at": row.get("created_at", "").strip(),
                        "rating": row.get("rating", "").strip(),
                    })
            
            print(f"Processing {len(reviews_data)} reviews with Groq LLM...")
            all_metadata = []
            for i, review_data in enumerate(reviews_data):
                if (i + 1) % 10 == 0:
                    print(f"Processing review {i + 1}/{len(reviews_data)}...")
                
                metadata = _extract_metadata_with_llm(
                    review_data["review_text"],
                    groq_client=llm_client,
                    model=model
                )
                all_metadata.append(metadata)
            
            # Create documents
            review_docs = []
            for review_data, metadata in zip(reviews_data, all_metadata):
                doc = Document(
                    page_content=review_data["review_text"],
                    metadata={
                        "source": os.path.basename(reviews_csv_path),
                        "created_at": review_data["created_at"],
                        "rating": review_data["rating"],
                        "event_type": metadata["event_type"],
                        "location": metadata["location"],
                        "sentiment": metadata["sentiment"],
                        "doc_type": "review",
                        "extraction_method": "llm",
                    }
                )
                review_docs.append(doc)
                
        elif client_type == "Client" and hasattr(llm_client, "chat"):
            # Use Ollama client
            from rag.reviews_processing import build_review_documents_using_llm
            review_docs = build_review_documents_using_llm(
                reviews_csv_path,
                ollama_client=llm_client,
                model=model,
                batch_size=batch_size,
            )
        else:
            raise ValueError(
                f"Unsupported LLM client type: {client_type}. "
                "Expected groq.Groq or ollama.Client"
            )
    else:
        # Import here to avoid circular imports
        from rag.reviews_processing import build_review_documents
        # Use regex-based extraction (no LLM)
        review_docs = build_review_documents(reviews_csv_path)
    
    # Convert documents to ReviewRecord objects and insert into database
    review_records = []
    for doc in review_docs:
        record = ReviewRecord(
            review_text=doc.page_content,
            rating=doc.metadata.get("rating"),
            created_at=doc.metadata.get("created_at"),
            event_type=doc.metadata.get("event_type"),
            location=doc.metadata.get("location"),
            sentiment=doc.metadata.get("sentiment"),
            source=doc.metadata.get("source", os.path.basename(reviews_csv_path)),
        )
        review_records.append(record)
    
    # Insert reviews into database
    if review_records:
        reviews_db.insert_reviews(review_records)
        print(f"✓ Stored {len(review_records)} reviews in SQL database at {reviews_db_path}")
    else:
        print(f"⚠️  No reviews were processed from {reviews_csv_path}")
    
    return reviews_db
