"""SQL database operations for events storage and retrieval."""

import sqlite3
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

from langchain_core.documents import Document


@dataclass
class EventRecord:
    """Event record structure."""
    event_name: str
    event_type: Optional[str]
    event_type_raw: Optional[str]
    source: str
    city: Optional[str]
    state: Optional[str]
    age_min: Optional[int]
    age_max: Optional[int]
    age_contains: Optional[str]  # comma-separated string like "kids, teens"
    intensity: Optional[str]  # low/moderate/high
    instructor: Optional[str]
    date_range: Optional[str]
    time_slots: Optional[str]
    duration: Optional[str]
    spots: Optional[str]
    center_name: Optional[str]
    center_type: Optional[str]
    page_content: str  # Full event block text


def init_database(db_path: str = "./events.db") -> None:
    """
    Initialize the events database with schema.
    
    Args:
        db_path: Path to SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name TEXT NOT NULL,
            event_type TEXT,
            event_type_raw TEXT,
            source TEXT NOT NULL,
            city TEXT,
            state TEXT,
            age_min INTEGER,
            age_max INTEGER,
            age_contains TEXT,
            intensity TEXT,
            instructor TEXT,
            date_range TEXT,
            time_slots TEXT,
            duration TEXT,
            spots TEXT,
            center_name TEXT,
            center_type TEXT,
            page_content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_city ON events(city)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_state ON events(state)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_age_contains ON events(age_contains)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_intensity ON events(intensity)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_city_state ON events(city, state)")
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")


def get_database_connection(db_path: str = "./events.db"):
    """Get a database connection."""
    return sqlite3.connect(db_path)


@contextmanager
def db_connection(db_path: str = "./events.db"):
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


class EventDB:
    """SQL database interface for events."""
    
    def __init__(self, db_path: str = "./events.db"):
        """
        Initialize EventDB.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        # Ensure database is initialized
        if not os.path.exists(db_path):
            init_database(db_path)
    
    def insert_event(self, event: EventRecord) -> int:
        """
        Insert a single event into the database.
        
        Args:
            event: EventRecord to insert
            
        Returns:
            ID of inserted event
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (
                    event_name, event_type, event_type_raw, source,
                    city, state, age_min, age_max, age_contains,
                    intensity, instructor, date_range, time_slots,
                    duration, spots, center_name, center_type, page_content
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_name,
                event.event_type,
                event.event_type_raw,
                event.source,
                event.city,
                event.state,
                event.age_min,
                event.age_max,
                event.age_contains,
                event.intensity,
                event.instructor,
                event.date_range,
                event.time_slots,
                event.duration,
                event.spots,
                event.center_name,
                event.center_type,
                event.page_content,
            ))
            return cursor.lastrowid
    
    def insert_events(self, events: List[EventRecord]) -> None:
        """
        Insert multiple events into the database.
        
        Args:
            events: List of EventRecord objects to insert
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO events (
                    event_name, event_type, event_type_raw, source,
                    city, state, age_min, age_max, age_contains,
                    intensity, instructor, date_range, time_slots,
                    duration, spots, center_name, center_type, page_content
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    event.event_name,
                    event.event_type,
                    event.event_type_raw,
                    event.source,
                    event.city,
                    event.state,
                    event.age_min,
                    event.age_max,
                    event.age_contains,
                    event.intensity,
                    event.instructor,
                    event.date_range,
                    event.time_slots,
                    event.duration,
                    event.spots,
                    event.center_name,
                    event.center_type,
                    event.page_content,
                )
                for event in events
            ])
        print(f"Inserted {len(events)} events into database")
    
    def clear_events(self) -> None:
        """Clear all events from the database."""
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events")
        print("Cleared all events from database")
    
    def query_events(
        self,
        event_types: Optional[List[str]] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        age_contains: Optional[str] = None,
        intensity: Optional[str] = None,
        limit: int = 10,
    ) -> List[Document]:
        """
        Query events from the database with filters.
        
        Args:
            event_types: List of event types to match (OR clause)
            city: City filter (exact match, case-insensitive)
            state: State filter (exact match, case-insensitive)
            age_contains: Age group filter (checks if age_contains contains this value)
            intensity: Intensity filter (exact match)
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
            
            if city:
                conditions.append("LOWER(city) = ?")
                params.append(city.lower())
            
            if state:
                conditions.append("LOWER(state) = ?")
                params.append(state.lower())
            
            if age_contains:
                # Handle multiple age groups (comma-separated string or list)
                if isinstance(age_contains, str):
                    age_groups = [a.strip() for a in age_contains.split(",") if a.strip()]
                else:
                    age_groups = age_contains
                
                if age_groups:
                    # Create OR conditions for each age group
                    age_conditions = []
                    for age_group in age_groups:
                        age_conditions.append("(LOWER(age_contains) LIKE ? OR LOWER(age_contains) = ?)")
                        params.append(f"%{age_group.lower()}%")
                        params.append(age_group.lower())
                    if age_conditions:
                        conditions.append(f"({' OR '.join(age_conditions)})")
            
            if intensity:
                conditions.append("LOWER(intensity) = ?")
                params.append(intensity.lower())
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT 
                    event_name, event_type, event_type_raw, source,
                    city, state, age_min, age_max, age_contains,
                    intensity, instructor, date_range, time_slots,
                    duration, spots, center_name, center_type, page_content
                FROM events
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
                    page_content=row[17],  # page_content
                    metadata={
                        "event_name": row[0],
                        "event_type": row[1],
                        "event_type_raw": row[2],
                        "source": row[3],
                        "city": row[4],
                        "state": row[5],
                        "age_min": row[6],
                        "age_max": row[7],
                        "age_contains": row[8],
                        "intensity": row[9],
                        "instructor": row[10],
                        "date_range": row[11],
                        "time_slots": row[12],
                        "duration": row[13],
                        "spots": row[14],
                        "center_name": row[15],
                        "center_type": row[16],
                    }
                )
                documents.append(doc)
            
            return documents
    
    def count_events(self) -> int:
        """Get total number of events in database."""
        with db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM events")
            return cursor.fetchone()[0]

