# Event Recommendation System

An intelligent event recommendation chatbot powered by Retrieval-Augmented Generation (RAG) that helps users discover suitable activities, classes, and events based on their preferences, age, location, and interests.

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
  - Two separate collections: `events` and `activity_types`
- **Integration**: Via `langchain_chroma.Chroma`

### UI & Interface

#### 5. **Gradio** - Chat Interface
- **Purpose**: Provides the web-based conversational UI
- **Features**:
  - Real-time chat interface
  - Conversation history management
  - User-friendly interaction design
- **Version**: Gradio 6.x compatible

### Development & Utilities

#### 6. **Python-dotenv** - Environment Management
- **Purpose**: Securely manages API keys and configuration
- **Usage**: Loads `GROQ_API_KEY` and `OPENAI_API_KEY` from `.env` file

#### 7. **Jupyter Notebooks** - Development & Experimentation
- **Purpose**: Used for iterative development and testing
- **Files**: `helper/activity-chatbot v2.ipynb`

## 🏗️ Architecture

### Two-Stage RAG Pipeline

```
User Query
    ↓
1. Activity Type Retrieval (Stage 1)
   - Semantic search in activity_types collection
   - Filters: intensity, interests
   - Returns: Activity type definitions
    ↓
2. Event Retrieval (Stage 2)
   - Semantic search in events collection
   - Filters: city, state, age_contains, event_type
   - Post-filtering: age groups, event types
   - Returns: Matching events
    ↓
3. Context Assembly
   - Combines events + activity definitions
   - Formats for LLM context
    ↓
4. LLM Generation
   - Groq API generates final response
   - Uses RAG context + conversation history
```

### Data Flow

1. **Document Ingestion**: Markdown files → Parsed documents → Vector embeddings
2. **Query Processing**: User message → Profile extraction → Retrieval query
3. **Retrieval**: Vector similarity search + metadata filtering + post-filtering
4. **Generation**: Retrieved context → LLM → Personalized response

## 📁 Project Structure

```
event_recommendation/
├── chat_ui/                 # Chat interface components
│   ├── chat_interface.py    # Gradio UI & LLM integration
│   └── profile.py          # User profile extraction & management
├── vector_db/               # Vector database operations
│   └── chroma_store.py      # ChromaDB integration & filter building
├── rag/                     # RAG processing modules
│   ├── document_processing.py  # Document parsing & chunking
│   ├── retrieval.py        # Two-stage retrieval pipeline
│   └── input_documents/
│       └── loader.py        # Document loading utilities
├── utils/                   # Utility functions
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
- API Keys:
  - Groq API key ([Get one here](https://console.groq.com/))
  - OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

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
   # Groq API Key (for LLM chat responses)
   GROQ_API_KEY=your_groq_api_key_here
   
   # OpenAI API Key (for embeddings)
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Verify setup**
   ```bash
   python main.py
   ```
   
   The application will:
   - Load documents from `documents/` directory
   - Build or load vector stores (first run builds, subsequent runs load)
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
- **Stage 1**: Finds relevant activity types based on user interests
- **Stage 2**: Retrieves specific events matching activity types + filters

### 3. **Advanced Filtering**
- **Metadata Filters**: City, state, age groups, event types
- **Post-Filtering**: Handles comma-separated values (e.g., "kids, teens")
- **Semantic Search**: Vector similarity for relevance

### 4. **Context-Aware Responses**
- Uses retrieved events and activity definitions
- Incorporates conversation history
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
python test/test_input_output.py
```

## 📊 Technology Stack Summary

| Category | Technology | Purpose |
|----------|-----------|---------|
| **RAG Framework** | LangChain | Document processing, retrieval orchestration |
| **Embeddings** | OpenAI `text-embedding-3-small` | Semantic vector representations |
| **LLM** | Groq API (`openai/gpt-oss-120b`) | Response generation, profile extraction |
| **Vector DB** | ChromaDB | Persistent vector storage & search |
| **UI Framework** | Gradio | Web-based chat interface |
| **Language** | Python 3.9+ | Core development language |
| **Environment** | python-dotenv | API key management |

## 🔐 Environment Variables

Required environment variables (stored in `.env`):

- `GROQ_API_KEY`: API key for Groq LLM inference
- `OPENAI_API_KEY`: API key for OpenAI embeddings

See `ENV_SETUP.md` for detailed setup instructions.

## 📝 Documentation

- `ENV_SETUP.md`: Environment variable configuration
- `FILE_STRUCTURE.md`: Detailed project structure documentation
- `REFACTORING.md`: Code refactoring notes and architecture decisions

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

---

**Built with ❤️ using modern AI engineering tools**
