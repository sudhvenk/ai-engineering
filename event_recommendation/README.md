# Event Recommendation System

An intelligent event recommendation chatbot powered by Retrieval-Augmented Generation (RAG) that helps users discover suitable activities, classes, and events based on their preferences, age, location, and interests.

## 🎯 Project Intent

This system is designed to provide personalized event recommendations through a conversational AI interface. It combines:

- **Vector Search**: Semantic similarity search using OpenAI embeddings and ChromaDB
- **SQL Database**: Structured storage for events and reviews with efficient querying
- **LLM Processing**: Groq or Ollama for natural language understanding and response generation
- **Review-Based Ranking**: User reviews stored in SQL database are used to score and rank event recommendations
- **Two-Stage RAG**: First retrieves activity types, then finds matching events with filtering

The system processes event brochures (markdown files), activity type definitions, and user reviews to create a comprehensive knowledge base for intelligent recommendations.

## 🎯 Project Overview

This system uses advanced AI engineering techniques to provide personalized event recommendations through a conversational interface. It combines vector search, semantic understanding, and large language models to deliver accurate, context-aware suggestions.

## 🤖 AI Engineering Technology Stack

### Core AI/ML Frameworks

#### 1. **LangChain** - RAG Framework & Orchestration
- **Purpose**: Provides the foundational framework for building RAG applications
- **Usage**:
  - `langchain_core`: Core abstractions for documents, embeddings, and chains
  - `langchain_openai`: OpenAI embeddings integration (`text-embedding-3-small`)
  - `langchain_chroma`: ChromaDB vector store integration
  - `langchain_community`: Document loaders for markdown files
  - `langchain_text_splitters`: Markdown header-based text splitting
- **Key Components**:
  - Document processing and chunking
  - Vector store management
  - Retrieval pipeline orchestration

#### 2. **OpenAI Embeddings** - Semantic Vector Representations
- **Model**: `text-embedding-3-small`
- **Purpose**: Converts text documents and queries into high-dimensional vector embeddings
- **Usage**: 
  - Embedding event descriptions and activity type definitions
  - Enabling semantic similarity search in the vector database
- **Integration**: Via `langchain_openai.OpenAIEmbeddings`

#### 3. **Groq API** - High-Performance LLM Inference
- **Model**: `openai/gpt-oss-120b` (OpenAI's open-source model)
- **Purpose**: 
  - Generates conversational responses
  - Extracts user profile information from conversations
  - Provides context-aware recommendations
- **Usage**:
  - User profile extraction and merging
  - Final answer generation with RAG context
- **Advantages**: Fast inference speeds with Groq's optimized infrastructure

#### 4. **ChromaDB** - Vector Database
- **Purpose**: Stores and retrieves vector embeddings for semantic search
- **Features Used**:
  - Persistent vector storage
  - Metadata filtering (city, state, age groups, event types)
  - Similarity search with filters
  - Collection: `activity_types` (events are stored in SQL, not ChromaDB)
- **Integration**: Via `langchain_chroma.Chroma`

#### 5. **SQLite** - Structured Data Storage
- **Purpose**: Stores events and reviews in structured format for efficient querying
- **Databases**:
  - `events.db`: Event records with metadata (city, state, age, intensity, etc.)
  - `reviews.db`: User reviews with extracted metadata (event_type, location, sentiment, rating)
- **Features**:
  - Fast SQL queries with indexes
  - Review scoring and aggregation
  - Metadata-based filtering

### UI & Interface

#### 6. **Gradio** - Chat Interface
- **Purpose**: Provides the web-based conversational UI
- **Features**:
  - Real-time chat interface
  - Conversation history management
  - User-friendly interaction design
- **Version**: Gradio 6.x compatible

### Development & Utilities

#### 7. **Python-dotenv** - Environment Management
- **Purpose**: Securely manages API keys and configuration
- **Usage**: Loads `GROQ_API_KEY` and `OPENAI_API_KEY` from `.env` file

#### 8. **Tenacity** - Retry Logic
- **Purpose**: Handles transient API failures with exponential backoff
- **Usage**: Automatic retries for Groq/Ollama API calls on rate limits and connection errors

#### 9. **Jupyter Notebooks** - Development & Experimentation
- **Purpose**: Used for iterative development and testing
- **Files**: `helper/activity-chatbot v2.ipynb`, `reviews_analysis.ipynb`

## 🏗️ Architecture

### Two-Stage RAG Pipeline with Review-Based Ranking

```
User Query
    ↓
1. Activity Type Retrieval (Stage 1)
   - Semantic search in activity_types ChromaDB collection
   - Filters: intensity, interests
   - Returns: Activity type definitions
    ↓
2. Event Retrieval (Stage 2)
   - SQL query in events.db database
   - Filters: city, state, age_contains, event_type
   - Post-filtering: age groups, event types
   - Returns: Matching events
    ↓
3. Review-Based Scoring
   - Query reviews.db for ratings by event_type and location
   - Calculate average scores for activities and venues
   - Re-rank events based on review scores
    ↓
4. Context Assembly
   - Combines top-ranked events + activity definitions
   - Formats for LLM context
    ↓
5. LLM Generation
   - Groq/Ollama API generates final response
   - Uses RAG context + conversation history + review scores
```

### Data Flow

1. **Document Ingestion**: 
   - Markdown files → Parsed documents → Vector embeddings (activity types) + SQL database (events)
   - Reviews CSV → LLM metadata extraction → SQL database (reviews)
2. **Query Processing**: User message → Profile extraction → Retrieval query
3. **Retrieval**: 
   - Vector similarity search (activity types) + SQL queries (events)
   - Review scoring from SQL database
4. **Ranking**: Events re-ranked by review scores (activity ratings + venue ratings)
5. **Generation**: Retrieved context + review scores → LLM → Personalized response

## 📁 Project Structure

```
event_recommendation/
├── chat_ui/                 # Chat interface components
│   ├── chat_interface.py    # Gradio UI & LLM integration
│   └── profile.py          # User profile extraction & management
├── vector_db/               # Vector database operations
│   └── chroma_store.py      # ChromaDB integration & filter building
├── database/                # SQL database operations
│   ├── event_db.py         # Events SQL database
│   └── review_db.py        # Reviews SQL database
├── rag/                     # RAG processing modules
│   ├── document_processing.py  # Document parsing & chunking
│   ├── reviews_processing.py   # Review processing & LLM extraction
│   ├── retrieval.py        # Two-stage retrieval pipeline with review ranking
│   └── input_documents/
│       └── loader.py        # Document loading utilities
├── utils/                   # Utility functions
│   ├── build_reviews_db.py  # Utility to build reviews database (Groq/Ollama)
│   ├── normalizers.py      # Text normalization
│   ├── extractors.py       # Data extraction (age, intensity)
│   └── helpers.py          # Helper utilities
├── test/                    # Test suite
│   ├── test_chroma.py      # Vector store tests
│   ├── test_rag.py         # RAG pipeline tests
│   └── test_input_output.py # I/O tests
├── documents/               # Input data
│   ├── Events/             # Event markdown files
│   └── activityType/      # Activity type definitions
├── main.py                 # Application entry point
└── requirements.txt        # Python dependencies
```

## 🚀 Setup & Installation

### Prerequisites

- Python 3.9+
- API Keys (choose one):
  - **Option 1**: Groq API key ([Get one here](https://console.groq.com/)) + OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
  - **Option 2**: Ollama installed locally ([Install here](https://ollama.ai/)) + OpenAI API key

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd event_recommendation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   # Groq API Key (for LLM chat responses and review processing)
   # Optional if using Ollama instead
   GROQ_API_KEY=your_groq_api_key_here
   
   # OpenAI API Key (for embeddings - REQUIRED)
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   
   **Note**: If using Ollama, you don't need `GROQ_API_KEY`, but you must have Ollama server running locally.

5. **Build reviews database (REQUIRED)**
   
   Before running the application, you must build the reviews database:
   
   **Option A: Using Groq (recommended)**
   ```bash
   python -m utils.build_reviews_db --client groq --model "openai/gpt-oss-120b"
   ```
   
   **Option B: Using Ollama (local)**
   ```bash
   # Make sure Ollama server is running: ollama serve
   python -m utils.build_reviews_db --client ollama --model "llama3.2:latest"
   ```
   
   **Option C: Using Python code**
   ```python
   from utils.build_reviews_db import build_reviews_database
   import groq
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
   
   build_reviews_database(
       reviews_csv_path="documents/Reviews/reviews_rag_2000.csv",
       reviews_db_path="reviews.db",
       llm_client=client,
       model="openai/gpt-oss-120b",
   )
   ```
   
   This will:
   - Process reviews CSV file
   - Extract metadata (event_type, location, sentiment) using LLM
   - Store reviews in SQLite database at `reviews.db`
   
   **Note**: The reviews database must exist and be populated before running the application.

6. **Verify setup**
   ```bash
   python main.py
   ```
   
   The application will:
   - Load documents from `documents/` directory
   - Build or load vector stores (first run builds, subsequent runs load)
   - Load reviews database (must be created in step 5)
   - Launch Gradio chat interface at `http://localhost:7860`

## 💻 Usage

### Running the Application

```bash
python main.py
```

### Using the Chat Interface

1. Open the Gradio interface (usually at `http://localhost:7860`)
2. Start chatting about activities you're interested in
3. The system will:
   - Extract your preferences (age, location, interests, intensity level)
   - Search for matching events
   - Provide personalized recommendations

### Example Queries

- "I'm looking for swimming classes for my 8-year-old in Framingham"
- "What aqua fitness programs are available for seniors in Pittsfield?"
- "I want low-intensity activities for adults in Boston"

## 🔧 Key Features

### 1. **Intelligent Profile Extraction**
- Automatically extracts user preferences from conversation
- Maintains profile state across conversation turns
- Merges new information with existing profile

### 2. **Two-Stage Retrieval**
- **Stage 1**: Finds relevant activity types based on user interests (ChromaDB vector search)
- **Stage 2**: Retrieves specific events matching activity types + filters (SQL database queries)

### 3. **Review-Based Ranking**
- Reviews stored in SQL database with extracted metadata (event_type, location, sentiment)
- Calculates average ratings for activities and venues
- Re-ranks events based on review scores (activity ratings + venue ratings)
- Provides personalized recommendations based on user feedback

### 4. **Advanced Filtering**
- **Metadata Filters**: City, state, age groups, event types
- **Post-Filtering**: Handles comma-separated values (e.g., "kids, teens")
- **Semantic Search**: Vector similarity for relevance
- **SQL Queries**: Fast structured queries for events and reviews

### 5. **Context-Aware Responses**
- Uses retrieved events and activity definitions
- Incorporates conversation history
- Includes review scores and sentiment in recommendations
- Provides detailed, personalized recommendations

## 🧪 Testing

Run the test suite:

```bash
pytest test/
```

Or run individual test files:

```bash
python test/test_chroma.py
python test/test_rag.py
python test/test_reviews_where_clause.py
```

**Note**: Tests that require reviews database will build it automatically using the test's temporary directory.

## 📊 Technology Stack Summary

| Category | Technology | Purpose |
|----------|-----------|---------|
| **RAG Framework** | LangChain | Document processing, retrieval orchestration |
| **Embeddings** | OpenAI `text-embedding-3-small` | Semantic vector representations |
| **LLM** | Groq API / Ollama | Response generation, profile extraction, review metadata extraction |
| **Vector DB** | ChromaDB | Persistent vector storage & search (activity types) |
| **SQL Database** | SQLite | Structured storage for events and reviews |
| **UI Framework** | Gradio | Web-based chat interface |
| **Language** | Python 3.9+ | Core development language |
| **Environment** | python-dotenv | API key management |
| **Retry Logic** | Tenacity | API failure handling with exponential backoff |

## 🔐 Environment Variables

Required environment variables (stored in `.env`):

- `OPENAI_API_KEY`: API key for OpenAI embeddings (REQUIRED)
- `GROQ_API_KEY`: API key for Groq LLM inference (optional if using Ollama)

**Note**: If using Ollama, you don't need `GROQ_API_KEY`, but you must have Ollama server running at `http://localhost:11434`.

See `ENV_SETUP.md` for detailed setup instructions.

## 📝 Documentation

- `ENV_SETUP.md`: Environment variable configuration
- `FILE_STRUCTURE.md`: Detailed project structure documentation
- `INSTALL.md`: Installation and setup instructions
- `utils/build_reviews_db.py`: Utility to build reviews database (see docstring for usage)

## 🔄 Building Reviews Database

The reviews database must be built **before** running the application. This is a one-time setup step that processes the reviews CSV and extracts metadata using LLM.

### Quick Start

```bash
# Using Groq (recommended)
python -m utils.build_reviews_db --client groq

# Using Ollama (local)
python -m utils.build_reviews_db --client ollama --model "llama3.2:latest"

# Without LLM (regex-based, faster but less accurate)
python -m utils.build_reviews_db --no-llm
```

### What It Does

1. Reads reviews from `documents/Reviews/reviews_rag_2000.csv`
2. Uses LLM to extract metadata:
   - `event_type`: Activity/class mentioned in review
   - `location`: Venue/facility name
   - `sentiment`: positive/negative/neutral
3. Stores reviews with metadata in `reviews.db` SQLite database
4. Creates indexes for fast querying

### Command-Line Options

```bash
python -m utils.build_reviews_db --help
```

Options:
- `--csv-path`: Path to reviews CSV (default: `documents/Reviews/reviews_rag_2000.csv`)
- `--db-path`: Path to reviews database (default: `reviews.db`)
- `--client`: LLM client to use: `groq` or `ollama` (default: `groq`)
- `--model`: Model name (default: auto-selected based on client)
- `--no-llm`: Use regex-based extraction instead of LLM
- `--batch-size`: Number of reviews per LLM call (default: 10)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

- LangChain team for the excellent RAG framework
- Groq for high-performance LLM inference
- OpenAI for embedding models
- ChromaDB for vector database capabilities
- Ed-donner's AI Engineering Course - https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/

---

**Built with ❤️ using modern AI engineering tools**
