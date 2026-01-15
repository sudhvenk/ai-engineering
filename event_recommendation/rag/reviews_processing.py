"""Process reviews CSV for RAG vector store."""

import csv
import re
import json
import os
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

import groq
import ollama


def ollama_call(
    system_prompt: str,
    user_prompt: str,
    ollama_client: ollama.Client,
    model: str
) -> str:
    """
    Call Ollama LLM for text generation.
    
    Args:
        system_prompt: System prompt for the LLM
        user_prompt: User prompt for the LLM
        ollama_client: Ollama client instance
        model: Model name to use
        
    Returns:
        Raw text response from LLM
    """
    response = ollama_client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


def build_review_documents(csv_path: str) -> List[Document]:
    """
    Build Document objects from reviews CSV file.
    
    Args:
        csv_path: Path to the reviews CSV file
        
    Returns:
        List of Document objects with review text and metadata
    """
    documents: List[Document] = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            review_text = row.get("review_text", "").strip()
            if not review_text:
                continue
            
            # Extract metadata
            created_at = row.get("created_at", "").strip()
            rating = row.get("rating", "").strip()
            
            # Try to extract event type and location from review text
            # This is optional - can be enhanced with NLP if needed
            event_type = None
            location = None
            
            # Simple extraction: look for common patterns
            # e.g., "BEGINNER COOKING at Pinecrest YMCA"
            # Try to find event type (uppercase words before "at")
            event_match = re.search(r'([A-Z][A-Z\s/]+?)\s+at\s+', review_text)
            if event_match:
                event_type = event_match.group(1).strip()
            
            # Try to find location (city name or YMCA name)
            # dont make assumption that it is a YMCA, it could be a library, park, etc.
            # we need to extract the location from the review text
            location_match = re.search(r'at\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:\s+)?', review_text)
            if location_match:
                location = location_match.group(1).strip()
                # If location is YMCA, keep the YMCA in the name

            
            # Create document with review text as content
            doc = Document(
                page_content=review_text,
                metadata={
                    "source": "reviews_rag_2000.csv",
                    "created_at": created_at,
                    "rating": rating,
                    "event_type": event_type,
                    "location": location,
                    "doc_type": "review",
                }
            )
            documents.append(doc)
    
    print(f"Loaded {len(documents)} reviews from {csv_path}")
    return documents



REVIEW_METADATA_SYSTEM_PROMPT = """
You are a metadata extraction assistant for activity reviews.

Your task:
- Extract structured metadata from review text.
- Output ONLY valid JSON.
- Do NOT guess or invent information.
- If information is not present, return null.

Extract the following fields:
- event_type: string | null
  The specific activity, class, or event type mentioned (e.g., "BEGINNER COOKING", "SWIMMING", "YOGA", "SENIOR CIRCUITS")
  Use the exact name as mentioned in the review, or a normalized version if clear.
  If no specific event type is mentioned, return null.

- location: string | null
  The venue, facility, or location name (e.g., "Pinecrest YMCA", "Summit Reach YMCA", "Boston Library")
  Include the full name if available (e.g., "Pinecrest YMCA" not just "Pinecrest")
  If no location is mentioned, return null.

- sentiment: string | null
  The sentiment of the review, either "positive", "negative", or "neutral".
  Analyze the overall tone and content of the review to determine sentiment.
  If the review is clearly positive (praise, satisfaction, recommendation), return "positive".
  If the review is clearly negative (complaints, dissatisfaction, warnings), return "negative".
  If the review is neutral or mixed, return "neutral".
  If sentiment cannot be determined, return null.

Rules:
- Extract only information explicitly stated in the review text.
- Preserve capitalization and formatting of event types and locations.
- If the review mentions multiple events or locations, extract the primary one mentioned.
- Return JSON in this exact format: {"event_type": "...", "location": "...", "sentiment": "..."}
"""


def _extract_metadata_with_llm(review_text: str, groq_client: groq.Groq, model: str) -> Dict[str, Optional[str]]:
    """
    Extract metadata from a single review using LLM.
    
    Args:
        review_text: The review text to extract metadata from
        batch_mode: If True, expects review_text to contain multiple reviews
        
    Returns:
        Dictionary with 'event_type', 'location', and 'sentiment' keys
    """
    if not groq_client:
        return {"event_type": None, "location": None, "sentiment": None}
    
    # Import here to avoid circular imports
    from chat_ui.profile import llm_call_profile
    
    user_prompt = f"""
Extract metadata from this review text:

"{review_text}"

Return ONLY a JSON object with event_type, location, and sentiment fields.
Example: {{"event_type": "BEGINNER COOKING", "location": "Pinecrest YMCA", "sentiment": "positive"}}
If a field cannot be determined, use null.
"""
    
    if not groq_client:
        return {"event_type": None, "location": None, "sentiment": None}
    
    try:
        result_text = llm_call_profile(
            REVIEW_METADATA_SYSTEM_PROMPT,
            user_prompt,
            groq_client=groq_client,
            model=model
        ).strip()
        
        # Try to extract JSON from the response (in case LLM adds extra text)
        # Look for JSON object in the response
        json_start = result_text.find('{')
        json_end = result_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            result_text = result_text[json_start:json_end]
        
        metadata = json.loads(result_text)
        
        # Ensure we have the expected keys
        return {
            "event_type": metadata.get("event_type"),
            "location": metadata.get("location"),
            "sentiment": metadata.get("sentiment"),
        }
    except Exception as e:
        print(f"Warning: LLM metadata extraction failed for review: {e}")
        return {"event_type": None, "location": None, "sentiment": None}

# Import moved inside function to avoid circular imports

def _extract_metadata_batch_with_llm(reviews: List[str], ollama_client: ollama.Client, model: str) -> List[Dict[str, Optional[str]]]:
    """
    Extract metadata from multiple reviews in a single LLM call (more efficient).
    
    Args:
        reviews: List of review texts
        
    Returns:
        List of metadata dictionaries with 'event_type', 'location', and 'sentiment' keys
    """
    
    # Format reviews for batch processing
    reviews_text = "\n\n".join([
        f"Review {i+1}:\n{review}" for i, review in enumerate(reviews)
    ])
    
    user_prompt = f"""
Extract metadata from these reviews:

{reviews_text}

Return a JSON array with one object per review, each with event_type, location, and sentiment fields.
Example: [{{"event_type": "BEGINNER COOKING", "location": "Pinecrest YMCA", "sentiment": "positive"}}, {{"event_type": null, "location": "Boston", "sentiment": "negative"}}]
If a field cannot be determined, use null.
"""
    
    if not ollama_client or not reviews:
        return [{"event_type": None, "location": None, "sentiment": None} for _ in reviews]
    
    try:
        result_text = ollama_call(
            REVIEW_METADATA_SYSTEM_PROMPT, 
            user_prompt, 
            ollama_client=ollama_client,
            model=model
        ).strip()
        print(f"In _extract_metadata_batch_with_llm **** PRINT result_text: {result_text}")
        # Try to extract JSON array from the response
        json_start = result_text.find('[')
        json_end = result_text.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            result_text = result_text[json_start:json_end]
        
        metadata_list = json.loads(result_text)
        
        # Ensure we have the expected structure
        results = []
        for metadata in metadata_list:
            results.append({
                "event_type": metadata.get("event_type"),
                "location": metadata.get("location"),
                "sentiment": metadata.get("sentiment"),
            })
        
        # Pad if needed
        while len(results) < len(reviews):
            results.append({"event_type": None, "location": None, "sentiment": None})
        
        return results[:len(reviews)]
    except Exception as e:
        print(f"Warning: Batch LLM metadata extraction failed: {e}")
        return [{"event_type": None, "location": None, "sentiment": None} for _ in reviews]


def build_review_documents_using_llm(
    csv_path: str,
    ollama_client: ollama.Client,
    model: str,
    batch_size: int = 10,
    use_batch: bool = True,
) -> List[Document]:
    """
    Build Document objects from reviews CSV file using LLM to extract metadata.
    
    This function uses an LLM to extract event_type and location from review text,
    which is more accurate than regex-based extraction.
    
    Args:
        csv_path: Path to the reviews CSV file
        batch_size: Number of reviews to process in a single LLM call (when use_batch=True)
        use_batch: If True, process reviews in batches for efficiency. If False, process one at a time.
        
    Returns:
        List of Document objects with review text and metadata
    """
    if not ollama_client:
        print("Warning: GROQ_API_KEY not set. Falling back to regex-based extraction.")
        return build_review_documents(csv_path)
    
    documents: List[Document] = []
    reviews_data = []
    
    # First pass: read all reviews from CSV
    with open(csv_path, "r", encoding="utf-8") as f:
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
    
    print(f"Processing {len(reviews_data)} reviews with LLM...")
    
    # Extract metadata using LLM
    if use_batch and batch_size > 1:
        # Process in batches
        all_metadata = []
        for i in range(0, len(reviews_data), batch_size):
            batch = reviews_data[i:i + batch_size]
            review_texts = [r["review_text"] for r in batch]
            batch_metadata = _extract_metadata_batch_with_llm(review_texts, ollama_client, model)
            all_metadata.extend(batch_metadata)
    else:
        # Process one at a time
        all_metadata = []
        for i, review_data in enumerate(reviews_data):
            if (i + 1) % 10 == 0:
                print(f"Processing review {i + 1}/{len(reviews_data)}...")
            
            metadata = _extract_metadata_with_llm(review_data["review_text"], ollama_client, model)
            all_metadata.append(metadata)
    
    # Create documents with extracted metadata
    for review_data, metadata in zip(reviews_data, all_metadata):
        doc = Document(
            page_content=review_data["review_text"],
            metadata={
                "source": os.path.basename(csv_path),
                "created_at": review_data["created_at"],
                "rating": review_data["rating"],
                "event_type": metadata["event_type"],
                "location": metadata["location"],
                "sentiment": metadata["sentiment"],
                "doc_type": "review",
                "extraction_method": "llm",
            }
        )
        documents.append(doc)
    
    print(f"Loaded {len(documents)} reviews from {csv_path} (using LLM for metadata extraction)")
    return documents

from database.review_db import ReviewDB, ReviewRecord, init_reviews_database
def process_and_store_reviews_using_llm(
    reviews_csv_path: str,
    reviews_db_path: str,
    ollama_client: ollama.Client,
    model: str,
    use_llm: bool = True,
    batch_size: int = 10,
) -> ReviewDB:
    """
    Process reviews CSV and store in SQL database.
    
    Args:
        reviews_csv_path: Path to reviews CSV file
        reviews_db_path: Path to SQLite database for reviews
        groq_client: Groq client instance for LLM calls
        model: Model name to use for LLM calls
        use_llm: If True, use LLM for metadata extraction; otherwise use regex
        batch_size: Number of reviews to process in a single LLM call (when use_llm=True)
        
    Returns:
        ReviewDB instance
    """
    from rag.reviews_processing import build_review_documents, build_review_documents_using_llm
    
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
    
    # Process reviews
    if use_llm:
        review_docs = build_review_documents_using_llm(
            reviews_csv_path, 
            ollama_client=ollama_client, 
            model=model,
            batch_size=batch_size
        )
    else:
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
            source=doc.metadata.get("source", "reviews_rag_2000.csv"),
        )
        review_records.append(record)
    
    # Insert reviews into database
    if review_records:
        reviews_db.insert_reviews(review_records)
    
    print(f"Stored {len(review_records)} reviews in SQL database")
    return reviews_db