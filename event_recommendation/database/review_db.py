"""SQL database operations for reviews storage and retrieval."""

import sqlite3
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager

from langchain_core.documents import Document


@dataclass
class ReviewRecord:
    """Review record structure."""
    review_text: str
    rating: Optional[str]
    created_at: Optional[str]
    event_type: Optional[str]
    location: Optional[str]
    sentiment: Optional[str]
    source: str = "reviews_rag_2000.csv"


def init_reviews_database(db_path: str = "./reviews.db") -> None:
    """
    Initialize the reviews database with schema.
    
    Args:
        db_path: Path to SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT NOT NULL,
            rating TEXT,
            created_at TEXT,
            event_type TEXT,
            location TEXT,
            sentiment TEXT,
            source TEXT NOT NULL,
            created_at_db TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rating ON reviews(rating)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON reviews(event_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_location ON reviews(location)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON reviews(sentiment)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_type_location ON reviews(event_type, location)")
    
    conn.commit()
    conn.close()
    print(f"Reviews database initialized at {db_path}")


@contextmanager
def db_connection(db_path: str = "./reviews.db"):
    """Context manager for database connections."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class ReviewDB:
    """SQL database interface for reviews."""
    
    def __init__(self, db_path: str = "./reviews.db"):
        """
        Initialize ReviewDB.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        # Ensure database is initialized
        if not os.path.exists(db_path):
            init_reviews_database(db_path)
    
    def insert_review(self, review: ReviewRecord) -> int:
        """
        Insert a single review into the database.
        
        Args:
            review: ReviewRecord to insert
            
        Returns:
            ID of inserted review
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reviews (
                    review_text, rating, created_at, event_type, location, sentiment, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                review.review_text,
                review.rating,
                review.created_at,
                review.event_type,
                review.location,
                review.sentiment,
                review.source,
            ))
            return cursor.lastrowid
    
    def insert_reviews(self, reviews: List[ReviewRecord]) -> None:
        """
        Insert multiple reviews into the database.
        
        Args:
            reviews: List of ReviewRecord objects to insert
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO reviews (
                    review_text, rating, created_at, event_type, location, sentiment, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    review.review_text,
                    review.rating,
                    review.created_at,
                    review.event_type,
                    review.location,
                    review.sentiment,
                    review.source,
                )
                for review in reviews
            ])
        print(f"Inserted {len(reviews)} reviews into database")
    
    def clear_reviews(self) -> None:
        """Clear all reviews from the database."""
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM reviews")
        print("Cleared all reviews from database")
    
    def query_reviews(
        self,
        event_types: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        rating: Optional[str] = None,
        sentiment: Optional[str] = None,
        limit: int = 100,
    ) -> List[Document]:
        """
        Query reviews from the database with filters.
        
        Args:
            event_types: List of event types to match (OR clause)
            locations: List of locations to match (OR clause)
            rating: Rating filter (exact match)
            sentiment: Sentiment filter (exact match: "positive", "negative", "neutral")
            limit: Maximum number of results
            
        Returns:
            List of Document objects (compatible with existing code)
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build WHERE clause
            conditions = []
            params = []
            
            if event_types:
                # OR clause for event types
                placeholders = ",".join(["?"] * len(event_types))
                conditions.append(f"LOWER(event_type) IN ({placeholders})")
                params.extend([et.lower() for et in event_types])
            
            if locations:
                # OR clause for locations
                placeholders = ",".join(["?"] * len(locations))
                conditions.append(f"LOWER(location) IN ({placeholders})")
                params.extend([loc.lower() for loc in locations])
            
            if rating:
                conditions.append("rating = ?")
                params.append(rating)
            
            if sentiment:
                conditions.append("LOWER(sentiment) = ?")
                params.append(sentiment.lower())
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT 
                    review_text, rating, created_at, event_type, location, sentiment, source
                FROM reviews
                WHERE {where_clause}
                LIMIT ?
            """
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to Document objects
            documents = []
            for row in rows:
                doc = Document(
                    page_content=row[0],  # review_text
                    metadata={
                        "rating": row[1],
                        "created_at": row[2],
                        "event_type": row[3],
                        "location": row[4],
                        "sentiment": row[5],
                        "source": row[6],
                        "doc_type": "review",
                    }
                )
                documents.append(doc)
            
            return documents
    
    def count_reviews(self) -> int:
        """Get total number of reviews in database."""
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM reviews")
            return cursor.fetchone()[0]
    
    def get_review_scores(
        self,
        event_types: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate average review ratings for event types and locations/venues.
        
        Args:
            event_types: Optional list of event types to filter reviews
            locations: Optional list of locations/venues to filter reviews
            
        Returns:
            Dictionary with 'activity_scores' and 'venue_scores' containing
            average ratings for each activity type and venue
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build WHERE clause
            conditions = []
            params = []
            
            if event_types:
                placeholders = ",".join(["?"] * len(event_types))
                conditions.append(f"LOWER(event_type) IN ({placeholders})")
                params.extend([et.lower() for et in event_types])
            
            if locations:
                placeholders = ",".join(["?"] * len(locations))
                conditions.append(f"LOWER(location) IN ({placeholders})")
                params.extend([loc.lower() for loc in locations])
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # Query reviews with valid ratings
            query = f"""
                SELECT event_type, location, rating
                FROM reviews
                WHERE {where_clause} AND rating IS NOT NULL AND rating != ''
            """
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            activity_scores: Dict[str, List[float]] = {}
            venue_scores: Dict[str, List[float]] = {}
            
            # Calculate average ratings
            for row in rows:
                event_type, location, rating_str = row
                
                try:
                    rating = float(rating_str) if rating_str else None
                except (ValueError, TypeError):
                    rating = None
                
                if rating is None:
                    continue
                
                # Score by activity type
                if event_type:
                    event_type = event_type.strip()
                    if event_type:
                        if event_type not in activity_scores:
                            activity_scores[event_type] = []
                        activity_scores[event_type].append(rating)
                
                # Score by location/venue
                if location:
                    location = location.strip()
                    if location:
                        if location not in venue_scores:
                            venue_scores[location] = []
                        venue_scores[location].append(rating)
            
            # Calculate averages
            activity_avg = {
                event_type: sum(ratings) / len(ratings)
                for event_type, ratings in activity_scores.items()
                if len(ratings) > 0
            }
            
            venue_avg = {
                location: sum(ratings) / len(ratings)
                for location, ratings in venue_scores.items()
                if len(ratings) > 0
            }
            
            return {
                "activity_scores": activity_avg,
                "venue_scores": venue_avg,
            }
