"""RAG retrieval functions."""

from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from vector_db.chroma_store import RagStores, build_chroma_where
from utils.normalizers import (
    normalize_intensity,
    normalize_age_focus,
    normalize_city,
    normalize_state,
)


def retrieve_activity_types(
    stores: RagStores,
    user_question: str,
    input_filter: Dict[str, Any],
    k: int = 5,
    oversample: int = 8,
) -> List[Document]:
    """
    Retrieve activity types with deduplication.
    
    Args:
        stores: RagStores containing vector stores
        user_question: User query string
        input_filter: Filter dictionary for metadata
        k: Number of results to return
        oversample: Multiplier for initial retrieval (for deduplication)
        
    Returns:
        List of deduplicated activity type documents
    """
    raw_k = max(k * oversample, 20)
    print(f"In retrieve_activity_types **** input_filter: {input_filter}")

    # Build filter for chroma dict with and clause
    filter_dict = build_chroma_where(input_filter)

    raw = stores.activity_types.similarity_search(
        user_question, k=raw_k, filter=filter_dict
    )
    print(f"In retrieve_activity_types **** raw: {raw}")

    seen = set()
    out: List[Document] = []
    for d in raw:
        heading = (d.metadata.get("activity_heading_norm") or "").strip()
        key = heading.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(d)
        if len(out) >= k:
            break

    print(f"In retrieve_activity_types **** out: {out}")
    return out


def retrieve_events_for_activity_type(
    stores: RagStores,
    user_question: str,
    input_filter: Dict[str, Any],
    k: int = 50,  # Increased from 10 to 50 for better recall before re-ranking
) -> List[Document]:
    """
    Retrieve events matching activity type and filters using SQL database.
    
    Args:
        stores: RagStores containing events database and activity_types vector store
        user_question: User query string (not used for SQL queries, kept for compatibility)
        input_filter: Filter dictionary with keys: event_type, city, state, age_contains, intensity
        k: Number of results to return (increased for re-ranking)
        
    Returns:
        List of event documents
    """
    print(f"In retrieve_events_for_activity_type **** input_filter: {input_filter}")

    # Extract filter values
    event_types = input_filter.get("event_type")
    if isinstance(event_types, str):
        event_types = [event_types]
    elif not isinstance(event_types, list):
        event_types = None
    
    city = input_filter.get("city")
    state = input_filter.get("state")
    age_contains = input_filter.get("age_contains")
    intensity = input_filter.get("intensity")

    print(f"In retrieve_events_for_activity_type **** SQL query filters: event_types={event_types}, city={city}, state={state}, age_contains={age_contains}, intensity={intensity}")

    # Query SQL database with larger limit for re-ranking
    events = stores.events.query_events(
        event_types=event_types,
        city=city,
        state=state,
        age_contains=age_contains,
        intensity=intensity,
        limit=k,
    )
    print(f"In retrieve_events_for_activity_type **** events: {len(events)} found")
    return events


def retrieve_reviews(
    stores: RagStores,
    user_question: str,
    k: int = 5,
    rating_filter: Optional[int] = None,
) -> List[Document]:
    """
    Retrieve reviews from SQL database.
    
    Note: This function now queries SQL instead of ChromaDB for reviews.
    For semantic search, you may want to add a separate vector store for reviews
    or use a different retrieval strategy.
    
    Args:
        stores: RagStores containing reviews database
        user_question: User query string (currently not used for SQL queries)
        k: Number of results to return
        rating_filter: Optional rating filter (1-5)
        
    Returns:
        List of review documents
    """
    print(f"In retrieve_reviews **** query: {user_question}, k: {k}, rating_filter: {rating_filter}")
    
    # Query reviews from SQL database
    reviews = stores.reviews.query_reviews(
        rating=str(rating_filter) if rating_filter else None,
        limit=k,
    )
    
    print(f"In retrieve_reviews **** found {len(reviews)} reviews")
    return reviews


def get_review_scores(
    stores: RagStores,
    event_types: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate average review ratings for event types and locations/venues.
    
    Args:
        stores: RagStores containing reviews database
        event_types: Optional list of event types to filter reviews
        locations: Optional list of locations/venues to filter reviews
        
    Returns:
        Dictionary with 'activity_scores' and 'venue_scores' containing
        average ratings for each activity type and venue
    """
    # Use ReviewDB's built-in method to calculate scores
    scores = stores.reviews.get_review_scores(
        event_types=event_types,
        locations=locations,
    )
    
    print(f"Review scores calculated: {len(scores['activity_scores'])} activities, {len(scores['venue_scores'])} venues")
    return scores


def rerank_events_by_reviews(
    events: List[Document],
    review_scores: Dict[str, Dict[str, float]],
    top_n: int = 20,
) -> List[Document]:
    """
    Re-rank events based on review scores for activities and venues.
    
    Priority:
    1. Venue score (if available) - venues with higher ratings get priority
    2. Activity score (if available) - activities with higher ratings ranked higher
    3. Original order if no review data
    
    Args:
        events: List of event documents to re-rank
        review_scores: Dictionary with 'activity_scores' and 'venue_scores'
        top_n: Number of top events to return
        
    Returns:
        Re-ranked list of event documents
    """
    if not events:
        return []
    
    activity_scores = review_scores.get("activity_scores", {})
    venue_scores = review_scores.get("venue_scores", {})
    
    # Score each event
    scored_events = []
    for event in events:
        event_type = (event.metadata.get("event_type") or "").strip()
        center_name = (event.metadata.get("center_name") or "").strip()
        city = (event.metadata.get("city") or "").strip()
        
        # Try to match venue by center_name or city
        venue_score = None
        if center_name:
            # Try exact match first
            for venue, score in venue_scores.items():
                if venue.lower() in center_name.lower() or center_name.lower() in venue.lower():
                    venue_score = score
                    break
        
        # If no match by center_name, try city
        if venue_score is None and city:
            for venue, score in venue_scores.items():
                if venue.lower() == city.lower():
                    venue_score = score
                    break
        
        # Get activity score
        activity_score = None
        if event_type:
            # Try exact match
            activity_score = activity_scores.get(event_type)
            # Try case-insensitive match
            if activity_score is None:
                for act_type, score in activity_scores.items():
                    if act_type.lower() == event_type.lower():
                        activity_score = score
                        break
        
        # Calculate composite score
        # Venue score gets higher weight (0.6) than activity score (0.4)
        # If only one is available, use that
        if venue_score is not None and activity_score is not None:
            composite_score = (venue_score * 0.6) + (activity_score * 0.4)
        elif venue_score is not None:
            composite_score = venue_score
        elif activity_score is not None:
            composite_score = activity_score
        else:
            # No review data - use a default low score to keep original order
            composite_score = 0.0
        
        scored_events.append((composite_score, event))
    
    # Sort by score (descending) - higher scores first
    scored_events.sort(key=lambda x: x[0], reverse=True)
    
    # Return top N events
    reranked = [event for _score, event in scored_events[:top_n]]
    
    print(f"Re-ranked {len(events)} events to top {len(reranked)} based on reviews")
    return reranked


def format_event_card(d: Document) -> str:
    """Format a single event document as a card string."""
    m = d.metadata
    return (
        f"- **{m.get('event_name')}** ({m.get('event_type')}) — "
        f"{m.get('city')}, {m.get('state')} @ {m.get('center_name')}\n"
        f"  - Age: {m.get('age_contains')}\n"
        f"  - When: {m.get('date_range')} | {m.get('time_slots')} | {m.get('duration')}\n"
        f"  - Instructor: {m.get('instructor')} | Spots: {m.get('spots')}\n"
        f"  - Source: {m.get('source')}\n"
    )


def build_context_block(
    events: List[Document], activity_defs: List[Document]
) -> str:
    """Build context block from retrieved events and activity definitions."""
    parts = ["## Retrieved Events\n"]
    parts.extend([format_event_card(d) for d in events])

    if activity_defs:
        parts.append("\n## Activity Definitions\n")
        for d in activity_defs:
            parts.append(
                f"- Source: {d.metadata.get('source')} | "
                f"Section: {d.metadata.get('activity_heading')}\n"
                f"{d.page_content}\n"
            )

    return "\n".join(parts).strip()


def rerank(
    query: str, docs: List[Document], top_n: int = 5, reranker=None
) -> List[Document]:
    """
    Rerank documents using a cross-encoder reranker.
    
    Args:
        query: Query string
        docs: List of documents to rerank
        top_n: Number of top documents to return
        reranker: Optional reranker model (CrossEncoder)
        
    Returns:
        Top N reranked documents
    """
    if not docs:
        return []
    if reranker is None:
        return docs[:top_n]

    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _s, d in ranked[:top_n]]


def answer_user(
    stores: RagStores,
    user_question: str,
    user_profile: Dict[str, Any],
    prefer_reviews: bool = True,  # If True, prioritize review-based ranking; if False, prioritize city match
) -> str:
    """
    Two-stage retrieval with review-based re-ranking:
      1) user intent -> ACTIVITY TYPE definitions (activityType RAG)
      2) activity types + filters -> EVENTS (brochure RAG)
      3) Re-rank events based on review scores (activity and venue ratings)
    
    Uses: city/state, age, intensity, interests, and user_question.
    
    Args:
        stores: RagStores containing vector stores
        user_question: User query string
        user_profile: User profile dictionary
        prefer_reviews: If True, prioritize review-based ranking. If False and city is provided,
                       prioritize city-matched events. Default True.
        
    Returns:
        Formatted context block with events and activity definitions
    """
    profile = user_profile or {}
    RETRIEVAL_K = 50  # Increased from 5 to 50 for better recall before re-ranking

    def to_str_safe(v):
        if isinstance(v, list):
            return ", ".join(str(x) for x in v if x)
        return str(v).strip() if v else ""

    # Extract user constraints
    print(f"user_profile: {profile}")

    interests = to_str_safe(",".join(profile.get("interests", "")))  # free text
    intensity = normalize_intensity(profile.get("intensity", ""))
    activity_query_parts = dict({})

    if interests:
        activity_query_parts["activity_heading"] = interests
    if intensity:
        activity_query_parts["intensity"] = intensity

    print(f"activity_query_parts: {activity_query_parts}")

    # Retrieve more than 5 so dedupe returns multiple headings
    activity_type_docs = retrieve_activity_types(
        stores, user_question, activity_query_parts, k=5
    )

    print(f"**** activity_type_docs: {activity_type_docs}")

    events_query_parts = dict({})

    age_focus = normalize_age_focus(profile.get("age_focus"))
    city = normalize_city(profile.get("city"))
    state = normalize_state(profile.get("state"))

    # Choose top N headings (increase to 3 for better recall)
    chosen_headings: List[str] = []
    for d in activity_type_docs:
        h = (d.metadata.get("activity_heading_norm") or "").strip()
        if h and h not in chosen_headings:
            chosen_headings.append(h)
        if len(chosen_headings) >= 3:
            break

    # Filter out empty headings and add to query
    chosen_headings = [h for h in chosen_headings if h]
    print(f"****chosen_headings: {chosen_headings}")
    
    # Add event_type and age_contains to database query (no post-filtering needed)
    if chosen_headings:
        events_query_parts["event_type"] = chosen_headings
    
    if age_focus:
        # age_focus is already a comma-separated string from normalize_age_focus
        # Pass the full string - database query will handle multiple age groups
        events_query_parts["age_contains"] = age_focus

    # Handle city filtering based on prefer_reviews flag
    # If city is provided and prefer_reviews is False, filter by city
    # If prefer_reviews is True, don't filter by city initially (retrieve more, then re-rank)
    city_filtered = False
    if city:
        if not prefer_reviews:
            # User wants city-matched events prioritized
            events_query_parts["city"] = city
            city_filtered = True
            print(f"City filter applied: {city}")
        else:
            # Don't filter by city initially - retrieve more events for re-ranking
            print(f"City provided ({city}) but not filtering - will re-rank by reviews instead")
    
    if state:
        events_query_parts["state"] = state

    # Retrieve events with larger K for re-ranking
    events = retrieve_events_for_activity_type(
        stores=stores,
        user_question=user_question,
        input_filter=events_query_parts,
        k=RETRIEVAL_K,
    )

    print(f"Retrieved events: {len(events)}")

    # De-dupe events
    seen = set()
    deduped_events: List[Document] = []
    for e in events:
        key = (
            (e.metadata.get("source") or "").strip().lower(),
            (e.metadata.get("event_name") or e.metadata.get("event_title") or "").strip().lower(),
            (e.metadata.get("event_type") or "").strip().lower(),
            (e.metadata.get("city") or "").strip().lower(),
            (e.metadata.get("state") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped_events.append(e)

    # Optional: post-filter if filters weren't supported in vectorstore
    if intensity:
        deduped_events = [
            e
            for e in deduped_events
            if (e.metadata.get("intensity") == intensity)
        ]

    # Re-rank events based on reviews
    if prefer_reviews and deduped_events:
        # Get review scores for activities and venues
        event_types_list = [e.metadata.get("event_type") for e in deduped_events if e.metadata.get("event_type")]
        locations_list = [e.metadata.get("center_name") or e.metadata.get("city") for e in deduped_events]
        locations_list = [loc for loc in locations_list if loc]  # Remove None values
        
        review_scores = get_review_scores(
            stores=stores,
            event_types=event_types_list if event_types_list else None,
            locations=locations_list if locations_list else None,
        )
        
        # Re-rank events
        reranked_events = rerank_events_by_reviews(
            events=deduped_events,
            review_scores=review_scores,
            top_n=20,
        )
        
        # If city was provided but not filtered, prioritize city matches in final ranking
        if city and not city_filtered:
            # Split events into city matches and others
            city_matches = [e for e in reranked_events if (e.metadata.get("city") or "").strip().lower() == city.lower()]
            other_events = [e for e in reranked_events if (e.metadata.get("city") or "").strip().lower() != city.lower()]
            
            # Combine: city matches first (still ranked by reviews), then others
            top_events = city_matches + other_events
            top_events = top_events[:20]  # Limit to top 20
            print(f"Re-ranked with city priority: {len(city_matches)} city matches, {len(other_events)} others")
        else:
            top_events = reranked_events
    else:
        # No re-ranking - use original order
        # If city was filtered, events are already city-matched
        top_events = deduped_events[:20]
        print(f"Using original order (no re-ranking): {len(top_events)} events")

    # Include activity definitions for reasoning (intensity/benefits)
    activity_defs = [
        d
        for d in activity_type_docs
        if (d.metadata.get("activity_heading") or "").strip() in chosen_headings
    ]

    return build_context_block(top_events, activity_defs)

