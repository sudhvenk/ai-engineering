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
    k: int = 10,
) -> List[Document]:
    """
    Retrieve events matching activity type and filters.
    
    Args:
        stores: RagStores containing vector stores
        user_question: User query string
        input_filter: Filter dictionary for metadata
        k: Number of results to return
        
    Returns:
        List of event documents
    """
    print(f"In retrieve_events_for_activity_type **** input_filter: {input_filter}")

    # Build filter for chroma dict with and clause
    filter_dict = build_chroma_where(input_filter)

    print(f"In retrieve_events_for_activity_type **** output filter: {filter_dict}")

    events = stores.events.similarity_search(
        query=user_question,
        k=k,
        filter=filter_dict,
    )
    print(f"In retrieve_events_for_activity_type **** events: {events}")
    return events


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
) -> str:
    """
    Two-stage retrieval:
      1) user intent -> ACTIVITY TYPE definitions (activityType RAG)
      2) activity types + filters -> EVENTS (brochure RAG)
    Uses: city/state, age, intensity, interests, and user_question.
    
    Args:
        stores: RagStores containing vector stores
        user_question: User query string
        user_profile: User profile dictionary
        
    Returns:
        Formatted context block with events and activity definitions
    """
    profile = user_profile or {}
    RETRIEVAL_K = 5

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

    # Note: age_contains is NOT added to ChromaDB filter because it's stored as
    # comma-separated string (e.g., "kids, teens") and ChromaDB only does exact matching.
    # We'll post-filter after retrieval instead.
    if city:
        events_query_parts["city"] = city
    if state:
        events_query_parts["state"] = state

    # Choose top N headings (increase to 3 for better recall)
    chosen_headings: List[str] = []
    for d in activity_type_docs:
        h = (d.metadata.get("activity_heading_norm") or "").strip()
        if h and h not in chosen_headings:
            chosen_headings.append(h)
        if len(chosen_headings) >= 3:
            break

    # Filter out empty headings and assign as list for post-filtering
    chosen_headings = [h for h in chosen_headings if h]
    print(f"****chosen_headings: {chosen_headings}")
    
    # Note: event_type is NOT added to ChromaDB filter to allow more flexible matching.
    # We'll post-filter after retrieval instead.

    # Retrieve more results to account for post-filtering by age_contains and event_type
    needs_post_filtering = age_focus or chosen_headings
    retrieval_k = RETRIEVAL_K * 3 if needs_post_filtering else RETRIEVAL_K

    events = retrieve_events_for_activity_type(
        stores=stores,
        user_question=user_question,
        input_filter=events_query_parts,
        k=retrieval_k,
    )

    print(f"Retrieved events: {events}")

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

    # Post-filter by event_type (exact match with any of the chosen headings)
    if chosen_headings:
        chosen_headings_lower = [h.strip().lower() for h in chosen_headings]
        filtered_by_event_type = []
        for e in deduped_events:
            stored_event_type = (e.metadata.get("event_type") or "").strip().lower()
            if stored_event_type and stored_event_type in chosen_headings_lower:
                filtered_by_event_type.append(e)
        deduped_events = filtered_by_event_type
        print(f"After event_type post-filter: {len(deduped_events)} events")

    # Post-filter by age_contains (stored as "kids, teens" format)
    # Query format is "kids" or "kids,teens" (no space after comma)
    if age_focus:
        requested_ages = [a.strip().lower() for a in age_focus.split(",")]
        filtered_by_age = []
        for e in deduped_events:
            stored_age_contains = (e.metadata.get("age_contains") or "").strip()
            if stored_age_contains:
                # Stored format: "kids, teens" (with space after comma)
                stored_ages = [a.strip().lower() for a in stored_age_contains.split(", ")]
                # Check if any requested age group matches any stored age group
                if any(req_age in stored_ages for req_age in requested_ages):
                    filtered_by_age.append(e)
            else:
                # If no age_contains metadata, include it (fallback)
                filtered_by_age.append(e)
        deduped_events = filtered_by_age
        print(f"After age_contains post-filter: {len(deduped_events)} events")

    # Optional: post-filter if filters weren't supported in vectorstore
    if intensity:
        deduped_events = [
            e
            for e in deduped_events
            if (e.metadata.get("intensity") == intensity)
        ]

    top_events = deduped_events[:20]

    # Include activity definitions for reasoning (intensity/benefits)
    activity_defs = [
        d
        for d in activity_type_docs
        if (d.metadata.get("activity_heading") or "").strip() in chosen_headings
    ]

    return build_context_block(top_events, activity_defs)

