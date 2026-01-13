"""RAG (Retrieval Augmented Generation) modules."""

from .retrieval import (
    retrieve_activity_types,
    retrieve_events_for_activity_type,
    answer_user,
    build_context_block,
    format_event_card,
    rerank,
)

__all__ = [
    "retrieve_activity_types",
    "retrieve_events_for_activity_type",
    "answer_user",
    "build_context_block",
    "format_event_card",
    "rerank",
]

