"""Gradio chat interface for activity recommendations."""

import json
import gradio as gr

from vector_db.chroma_store import RagStores
from rag.retrieval import answer_user
from .profile import (
    PROFILE_SYSTEM_PROMPT,
    PROFILE_USER_PROMPT_TEMPLATE,
    llm_call_profile,
    merge_profiles,
    get_recent_user_messages,
    build_retrieval_query,
)

import groq
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

ANSWER_SYSTEM_PROMPT = """You are an Activity Recommendation Assistant. 

Your job is to help users discover suitable activities, classes, or event types 
based on their age, interests, physical comfort level, time availability, and goals. 
You must follow these rules: 
1. Use ONLY the information provided in the context and user messages. 
2. Do NOT invent activities, classes, or benefits that are not explicitly stated. 
3. If information is missing, ask a clarifying question instead of guessing. 
4. Be respectful of physical limitations and accessibility needs. 
5. Do NOT provide medical advice. Phrase benefits in general wellness terms. 
6. When recommending activities, include: 
   - Activity name 
   - Typical intensity level 
   - Typical session length 
   - Recommended weekly frequency 
   - average user sentiment and rating of the activity
   - Why it fits the user's preferences 
7. If multiple activities fit, rank them from best to least suitable. 
8. If nothing fits well, explain why and suggest alternatives. 

Your tone should be friendly, practical, and encouraging. 
"""


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
def llm_call_answer(
    system_prompt: str, 
    user_prompt: str,
    groq_client: groq.Groq,
    model: str = "openai/gpt-oss-120b"
) -> str:
    """
    Call LLM to generate answer.
    
    Args:
        system_prompt: System prompt for the LLM
        user_prompt: User prompt for the LLM
        groq_client: Groq client instance
        model: Model name to use (default: "openai/gpt-oss-120b")
        
    Returns:
        Response text from LLM
        
    Raises:
        groq.RateLimitError: If rate limit is exceeded after retries
        groq.APIConnectionError: If connection fails after retries
        groq.APITimeoutError: If request times out after retries
        groq.InternalServerError: If server error occurs after retries
        groq.APIError: If API returns a non-retryable error
    """
    resp = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content


def convert_gradio_history(history):
    """Convert Gradio 6.x history format (list of dicts) to tuple format."""
    if not history:
        return []
    converted = []
    i = 0
    while i < len(history):
        msg = history[i]
        if isinstance(msg, dict):
            # Gradio 6.x format: {"role": "user"/"assistant", "content": "..."}
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                # Look for corresponding assistant message (next item)
                assistant_content = ""
                if i + 1 < len(history) and isinstance(history[i + 1], dict):
                    next_msg = history[i + 1]
                    if next_msg.get("role") == "assistant":
                        assistant_content = next_msg.get("content", "")
                        i += 1  # Skip the assistant message
                converted.append((content, assistant_content))
            elif role == "assistant":
                # Standalone assistant message (shouldn't happen, but handle it)
                converted.append(("", content))
        elif isinstance(msg, tuple):
            # Already in tuple format
            converted.append(msg)
        i += 1
    return converted


# Store profile state (simple approach - in production use Gradio's state management)
_user_profile_state = {}


def chat(
    message: str, 
    history, 
    stores: RagStores,
    groq_client: groq.Groq,
    model: str = "openai/gpt-oss-120b"
):
    """
    Chat function compatible with Gradio 6.x ChatInterface.
    
    Args:
        message: Current user message
        history: Gradio history format
        stores: RagStores containing vector stores
        groq_client: Groq client instance
        model: Model name to use
        
    Returns:
        Assistant response string
    """
    global _user_profile_state

    # Convert Gradio history format to tuple format for internal use
    history_tuples = convert_gradio_history(history)

    profile = dict(_user_profile_state or {})

    # Ask LLM to update profile using recent user turns + current message
    recent_msgs = get_recent_user_messages(history_tuples, n=4)
    profile_prompt = PROFILE_USER_PROMPT_TEMPLATE.format(
        existing_profile_json=json.dumps(profile, indent=2),
        recent_user_messages="\n".join([f"- {m}" for m in recent_msgs])
        if recent_msgs
        else "(none)",
        user_message=message,
    )

    try:
        extracted_raw = llm_call_profile(PROFILE_SYSTEM_PROMPT, profile_prompt, groq_client=groq_client, model=model)
        extracted = json.loads(extracted_raw)
        print(f"Extracted profile: {extracted}")
        profile = merge_profiles(profile, extracted)
        _user_profile_state.update(profile)  # Update global state
        print(f"Updated profile: {profile}")
    except Exception as e:
        # Fail-safe: keep existing profile if LLM extraction fails
        print(f"Profile extraction failed: {e}")
        pass

    # Build retrieval query
    retrieval_query = build_retrieval_query(message, profile, history_tuples)

    context = answer_user(stores, retrieval_query, profile)
    print(f"context: {context}")

    # Build answer prompt (history can be used here, not in embeddings)
    # Keep history minimal to avoid token bloat: last 6 turns
    last_turns = history_tuples[-6:] if history_tuples else []
    chat_history_text = "\n".join(
        [f"User: {u}\nAssistant: {a}" for (u, a) in last_turns if u and a]
    )

    answer_user_prompt = f"""
    You are continuing a conversation with a user about activities and events. 

    User question:
    {message}

    User profile (structured):
    {json.dumps(profile, indent=2)}

    Recent chat history:
    {chat_history_text if chat_history_text else "(none)"}

    Retrieved context:
    {context}
 
    Instructions: 
    - Recommend the top 2 suitable activity types from the knowledge provided. 
    - Explain clearly why each recommendation fits the user. 
    - Get the exact event name from the context provided and which location the event is located at. 
    - Include intensity, session length, and typical weekly frequency. 
    - Cite the activity source using [filename | activity name]. 
    - If the knowledge is insufficient, say what is missing. 
    """.strip()

    assistant_text = llm_call_answer(ANSWER_SYSTEM_PROMPT, answer_user_prompt, groq_client=groq_client, model=model)

    # Update global profile state
    _user_profile_state.update(profile)

    # Return only the message string (Gradio 6.x expects just the message)
    return assistant_text


def launch_chat_interface(stores: RagStores, groq_client: groq.Groq, model: str = "openai/gpt-oss-120b"):
    """
    Launch Gradio chat interface.
    
    Args:
        stores: RagStores containing vector stores
        groq_client: Groq client instance
        model: Model name to use
    """
    # Create a wrapper function that includes stores and client
    def chat_wrapper(message: str, history):
        return chat(message, history, stores, groq_client=groq_client, model=model)

    demo = gr.ChatInterface(chat_wrapper)
    demo.launch()

