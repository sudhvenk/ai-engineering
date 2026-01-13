# Code Refactoring Documentation

This document describes the refactored structure of the activity recommendation chatbot.

## Directory Structure

```
event_recommendation/
├── chat_ui/              # Chat interface components
│   ├── __init__.py
│   ├── chat_interface.py # Gradio chat interface
│   └── profile.py        # User profile management
├── vector_db/            # ChromaDB vector store operations
│   ├── __init__.py
│   └── chroma_store.py   # Vector store building and management
├── rag/                  # RAG (Retrieval Augmented Generation) modules
│   ├── __init__.py
│   ├── document_processing.py  # Document parsing and building
│   ├── retrieval.py           # RAG retrieval functions
│   └── input_documents/        # Document loading
│       ├── __init__.py
│       └── loader.py
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── normalizers.py    # Text normalization functions
│   ├── extractors.py     # Data extraction functions
│   └── helpers.py        # Helper utilities
├── test/                 # Test suite
│   ├── __init__.py
│   ├── test_chroma.py    # ChromaDB tests
│   ├── test_rag.py       # RAG tests
│   └── test_input_output.py  # Input/output tests
├── main.py               # Main entry point
└── documents/            # Input documents (unchanged)
    ├── Events/
    └── activityType/
```

## Module Descriptions

### chat_ui/
Contains the Gradio-based chat interface and user profile management.

- **chat_interface.py**: Main chat function, Gradio interface setup, LLM calls for answers
- **profile.py**: User profile extraction, merging, and management

### vector_db/
ChromaDB vector store operations.

- **chroma_store.py**: 
  - `RagStores`: Dataclass for storing event and activity type vector stores
  - `build_vectorstores()`: Build vector stores from markdown files
  - `load_vectorstores()`: Load existing vector stores from disk
  - `cleanup_vectorstores()`: Delete collections
  - `build_chroma_where()`: Build ChromaDB-compatible filter queries

### rag/
RAG processing and retrieval logic.

- **document_processing.py**:
  - `parse_center_metadata()`: Parse center information from markdown
  - `split_event_blocks()`: Split markdown into event blocks
  - `parse_event_metadata()`: Parse event metadata
  - `build_activitytype_documents()`: Build activity type documents
  - `build_event_documents()`: Build event documents

- **retrieval.py**:
  - `retrieve_activity_types()`: Retrieve activity types with deduplication
  - `retrieve_events_for_activity_type()`: Retrieve events matching filters
  - `answer_user()`: Main RAG function for two-stage retrieval
  - `format_event_card()`: Format event as card string
  - `build_context_block()`: Build context from retrieved documents
  - `rerank()`: Optional reranking of results

- **input_documents/loader.py**:
  - `load_documents()`: Load markdown files from documents directory

### utils/
Utility functions for text processing.

- **normalizers.py**: Text normalization (event types, activity headings, intensity, age, city, state)
- **extractors.py**: Data extraction (age ranges, age groups, intensity inference)
- **helpers.py**: General helper functions

### test/
Test suite for all components.

- **test_chroma.py**: Tests for ChromaDB operations
- **test_rag.py**: Tests for RAG retrieval functions
- **test_input_output.py**: Tests for document processing

## Usage

### Running the Chatbot

```bash
python main.py
```

This will:
1. Load documents from the `documents/` directory
2. Build or load vector stores
3. Launch the Gradio chat interface

### Running Tests

```bash
pytest test/
```

## Migration from Notebook

The code has been refactored from the Jupyter notebook (`helper/activity-chatbot v2.ipynb`) into this modular structure. Key changes:

1. **Separation of Concerns**: Each module has a clear responsibility
2. **Reusability**: Functions can be imported and reused
3. **Testability**: Each component can be tested independently
4. **Maintainability**: Easier to understand and modify

## Dependencies

All dependencies remain the same as in the original notebook:
- langchain
- langchain-openai
- langchain-chroma
- gradio
- groq
- python-dotenv

## Environment Variables

Required environment variables:
- `GROQ_API_KEY`: API key for Groq client

