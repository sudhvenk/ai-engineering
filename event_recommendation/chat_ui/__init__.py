"""Chat interface for the activity recommendation system."""

from .chat_interface import chat, launch_chat_interface
from .profile import (
    UserProfile,
    PROFILE_SYSTEM_PROMPT,
    PROFILE_USER_PROMPT_TEMPLATE,
    llm_call_profile,
    merge_profiles,
    get_recent_user_messages,
    build_retrieval_query,
)

__all__ = [
    "chat",
    "launch_chat_interface",
    "UserProfile",
    "PROFILE_SYSTEM_PROMPT",
    "PROFILE_USER_PROMPT_TEMPLATE",
    "llm_call_profile",
    "merge_profiles",
    "get_recent_user_messages",
    "build_retrieval_query",
]

