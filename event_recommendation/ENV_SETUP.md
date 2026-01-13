# Environment Variables Setup

## Required Environment Variables

Your `.env` file should contain the following variables:

```bash
# Groq API Key (for LLM chat responses)
GROQ_API_KEY=your_groq_api_key_here

# OpenAI API Key (for embeddings)
OPENAI_API_KEY=your_openai_api_key_here
```

## Why Both Are Needed

1. **GROQ_API_KEY**: Used by the chat interface to generate responses using Groq's API
   - Used in: `chat_ui/profile.py`, `chat_ui/chat_interface.py`
   - Model: `openai/gpt-oss-120b`

2. **OPENAI_API_KEY**: Used by LangChain's OpenAIEmbeddings to create vector embeddings
   - Used in: `vector_db/chroma_store.py`
   - Model: `text-embedding-3-small`

## Current Issue

The error you're seeing:
```
"The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
```

This means `OPENAI_API_KEY` is missing from your `.env` file.

## Solution

Add `OPENAI_API_KEY` to your `.env` file:

1. Open the `.env` file
2. Add the line: `OPENAI_API_KEY=your_openai_api_key_here`
3. Replace `your_openai_api_key_here` with your actual OpenAI API key
4. Make sure `load_dotenv()` is called in `main.py` (it already is)

## Verification

After updating your `.env` file, you can verify it's loaded correctly:

```python
from dotenv import load_dotenv
import os

load_dotenv()
print("GROQ_API_KEY set:", bool(os.getenv("GROQ_API_KEY")))
print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
```

Both should print `True` if the keys are set correctly.

