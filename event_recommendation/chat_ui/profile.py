"""User profile management and extraction."""

import json
from typing import Dict, Any, List, Tuple
from pydantic import BaseModel

import groq
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception,
    retry_if_not_exception_type,
)

groq_client = None
OPENSOURCE_OSS_MODEL = None

class UserProfile(BaseModel):
    """User profile schema."""
    location: str | None = None
    age_focus: str | None = None
    interests: list[str] | None = None
    time_prefs: list[str] | None = None
    city: str | None = None
    state: str | None = None
    budget_sensitivity: str | None = None


PROFILE_SYSTEM_PROMPT = """
You are a profile extraction assistant.

Your task:
- Extract structured user preferences from casual chat text.
- Output ONLY valid JSON.
- Do NOT guess.
- If unsure, return null or empty lists.

Allowed fields ONLY:
- location: string | null  (US city/state if present)
- age_focus: kids | teens | young_adults | adults | seniors | null
- interests: list of strings from: aquatics, athletics, dancing, cooking, drawing
- time_prefs: list of strings from: mornings, afternoons, evenings, weekends
- city: string | null (US city if present)
- state: string | null (US state if present)
- budget_sensitivity: low | medium | high | null

Rules:
- Never invent a location.
- Do not add fields not listed.
- If the user mentions multiple age groups, pick the dominant one; otherwise null.
"""

PROFILE_USER_PROMPT_TEMPLATE = """
Existing profile:
{existing_profile_json}

Recent user messages (for context):
{recent_user_messages}

New user message:
"{user_message}"

Return ONLY a JSON object with any fields you can confidently update.
If nothing can be updated, return:
{{"location": null, "age_focus": null, "interests": [], "time_prefs": [], "city": null, "state": null, "budget_sensitivity": null}}
""".strip()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((
        groq.RateLimitError,          # Rate limit exceeded
        groq.APIConnectionError,       # Connection issues
        groq.APITimeoutError,          # Request timeout
        groq.InternalServerError,      # Server errors (5xx)
        ConnectionError,               # Python connection errors
        TimeoutError,                  # Python timeout errors
    )),
    reraise=True,
)
def llm_call_profile(
    system_prompt: str, 
    user_prompt: str, 
    groq_client=None,
    model: str = "openai/gpt-oss-120b"
) -> str:
    """
    Call LLM for profile extraction.
    Must return raw text containing ONLY JSON.
    
    Args:
        system_prompt: System prompt for the LLM
        user_prompt: User prompt for the LLM
        groq_client: Groq client instance (if None, will be created)
        model: Model name to use (default: "openai/gpt-oss-120b")
        
    Returns:
        Raw text response from LLM containing JSON
        
    Raises:
        groq.RateLimitError: If rate limit is exceeded after retries
        groq.APIConnectionError: If connection fails after retries
        groq.APITimeoutError: If request times out after retries
        groq.InternalServerError: If server error occurs after retries
        groq.APIError: If API returns a non-retryable error
    """
    # Initialize Groq client if not provided
    if groq_client is None:
        groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    resp = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content


def merge_profiles(
    existing: Dict[str, Any], new: Dict[str, Any]
) -> Dict[str, Any]:
    """Non-destructive merge: keep existing unless new provides a confident update."""
    merged = dict(existing or {})

    for k, v in (new or {}).items():
        if v is None:
            continue
        if isinstance(v, list):
            cur = set(merged.get(k, []) or [])
            merged[k] = sorted(
                cur.union([x for x in v if isinstance(x, str) and x.strip()])
            )
        elif isinstance(v, str):
            if v.strip():
                merged[k] = v.strip()
        else:
            merged[k] = v

    # Normalize allowed enums (extra safety)
    allowed_age = {"kids", "teens", "young_adults", "adults", "seniors"}
    if merged.get("age_focus") not in allowed_age:
        merged["age_focus"] = None

    allowed_interests = {"aquatics", "athletics", "dancing", "cooking", "drawing"}
    merged["interests"] = [
        x for x in merged.get("interests", []) if x in allowed_interests
    ]

    allowed_time = {"mornings", "afternoons", "evenings", "weekends"}
    merged["time_prefs"] = [
        x for x in merged.get("time_prefs", []) if x in allowed_time
    ]

    if merged.get("city"):
        merged["city"] = merged["city"].strip()
    if merged.get("state"):
        merged["state"] = merged["state"].strip()

    allowed_budget = {"low", "medium", "high"}
    if merged.get("budget_sensitivity") not in allowed_budget:
        merged["budget_sensitivity"] = None

    return merged


def get_recent_user_messages(
    history: List[Tuple[str, str]], n: int = 4
) -> List[str]:
    """History is List[(user, assistant)]. Returns last n user messages."""
    if not history:
        return []
    users = []
    for u, _a in history:
        if isinstance(u, str) and u.strip():
            users.append(u.strip())
    return users[-n:]


def build_retrieval_query(
    message: str, profile: Dict[str, Any], history: List[Tuple[str, str]]
) -> str:
    """Build compact retrieval query from message, profile, and history."""
    recent = get_recent_user_messages(history, n=3)
    parts = [message.strip()]

    if profile.get("location"):
        parts.append(f"Location: {profile['location']}")
    if profile.get("age_focus"):
        parts.append(f"Age: {profile['age_focus']}")
    if profile.get("interests"):
        parts.append("Interests: " + ", ".join(profile["interests"]))
    if profile.get("time_prefs"):
        parts.append("Time prefs: " + ", ".join(profile["time_prefs"]))
    if profile.get("budget_sensitivity"):
        parts.append(f"Budget: {profile['budget_sensitivity']}")

    if recent:
        parts.append(
            "Recent user context: " + " | ".join([u[:120] for u in recent])
        )

    return "\n".join(parts)

