# AgentX - RAG-First Conversational AI Agent

A production-ready **Retrieval-Augmented Generation (RAG)** system that combines an intelligent agent with a knowledge base, web grounding, and multi-modal I/O (text, voice, and audio).

**Version:** 0.0.1 | **Author:** Mahir Baig

---

## Table of Contents

- [Overview](#overview)
- [Why AgentX?](#why-agentx)
- [Architecture](#architecture)
- [Execution Flows](#execution-flows)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Docker Deployment](#docker-deployment)
- [How It Works](#how-it-works)
- [Core Components](#core-components)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [RAG vs Grounding Decision Tree](#rag-vs-grounding-decision-tree)
- [Voice Features](#voice-features)
- [Logging & Monitoring](#logging--monitoring)
- [Security Best Practices](#security-best-practices)
- [Performance Optimization](#performance-optimization)
- [Testing](#testing)
- [Dependencies](#dependencies)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Configuration Reference](#configuration-reference)
- [Deployment](#deployment)
- [Additional Resources](#additional-resources)
- [License](#license)
- [Author](#author)
- [Contributing](#contributing)
- [Features Roadmap](#features-roadmap)

---

## Overview

AgentX is an advanced conversational agent that:

- **RAG-First Workflow** - Prioritizes internal knowledge base before web search
- **Intelligent Tool Selection** - Automatically routes between RAG and web grounding (sequentially)
- **Multi-Modal I/O** - Text input, speech-to-text (STT), and text-to-speech (TTS)
- **Conversation Memory** - Maintains context across multiple turns with thread IDs
- **FastAPI Backend** - RESTful API for programmatic access
- **Streamlit Frontend** - Interactive web UI with document upload
- **Azure Integration** - Cloud-based storage, embeddings, and LLM services
- **LangSmith Tracing** - Full observability and debugging

---

## Why AgentX?

| Feature | AgentX | Traditional RAG |
|---------|--------|-----------------|
| Knowledge Priority | RAG-first workflow (sequential) | Random tool selection |
| Multi-modal I/O | Text + Voice | Text only |
| Memory | Thread-based context | Stateless |
| Web Fallback | Automatic grounding (only when needed) | No fallback |
| Production Ready | FastAPI + Docker | Notebook only |
| Observability | LangSmith tracing | Limited logging |

---

## Architecture

### Standard RAG Architecture with 5-Layer Design

```
┌─────────────────────────────────────────────────────────────┐
│                   INTERFACE LAYER                            │
│                   (User Interaction)                         │
├──────────────────────┬──────────────────────────────────────┤
│   Streamlit App      │        FastAPI Backend               │
│   (Web UI)           │        (REST API)                    │
└──────────┬───────────┴──────────────┬───────────────────────┘
           │                          │
           └────────────┬─────────────┘
                        │
┌───────────────────────▼────────────────────────────────────┐
│               ORCHESTRATION LAYER                           │
│               (Agent & Memory)                              │
│                                                             │
│              ┌─────────────────────────┐                   │
│              │  LangGraph RAG Agent    │                   │
│              │  (Conversation Memory)  │                   │
│              └───────────┬─────────────┘                   │
└──────────────────────────┼─────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼───────┐  ┌───────▼───────┐  ┌──────▼─────────┐
│  RETRIEVAL    │  │  GENERATION   │  │   GROUNDING    │
│  COMPONENT    │  │  COMPONENT    │  │   COMPONENT    │
│  (RAG Tool)   │  │  (LLM Call)   │  │   (Fallback)   │
│               │  │               │  │                │
│  1. Embed     │  │  Context +    │  │  Web Search    │
│  2. Search    │  │  Retrieved    │  │  via           │
│  3. Retrieve  │  │  Documents    │  │  Perplexity    │
└───────┬───────┘  └───────▲───────┘  └──────┬─────────┘
        │                  │                  │
        │        Sequential Execution         │
        │        (RAG First → Grounding)      │
        │                  │                  │
┌───────▼──────────────────┴──────────────────▼─────────────┐
│                   DATA LAYER                               │
│                   (Storage & Retrieval)                    │
│                                                            │
│  ┌────────────────────────┐      ┌─────────────────────┐ │
│  │  ChromaDB Vector DB    │      │  Azure Blob Storage │ │
│  │  - Embeddings Storage  │      │  - Original Files   │ │
│  │  - Similarity Search   │      │  - Document Store   │ │
│  └────────────────────────┘      └─────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│                   SERVICE LAYER                             │
│                   (External APIs)                           │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────────────────┐   │
│  │  Azure OpenAI    │  │  Perplexity API              │   │
│  │  - Embeddings    │  │  - Web Search (Grounding)    │   │
│  │  - LLM (GPT-4)   │  │  - sonar-pro model           │   │
│  │  - Streaming     │  └──────────────────────────────┘   │
│  └──────────────────┘                                      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Azure Cognitive Services (STT/TTS)                  │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Architecture Principles

**1. Standard RAG Pattern:** Retrieve → Augment → Generate

**2. Five-Layer Design:**
- **Interface Layer**: User interaction (UI/API)
- **Orchestration Layer**: Agent logic and memory
- **Application Layer**: RAG retrieval, generation, grounding
- **Data Layer**: Vector and document storage
- **Service Layer**: External AI services

**3. Sequential Tool Execution:** RAG Tool (mandatory first) → Grounding Tool (fallback only)

**4. Separation of Concerns:** Ingestion pipeline separate from query pipeline

---

## Execution Flows

### 📤 Document Ingestion Flow (Separate Pipeline)

This flow runs **independently** when documents are uploaded. It happens **once per document**.

```
┌────────────────────────────────────────────────────────┐
│              DOCUMENT INGESTION FLOW                    │
└────────────────────────────────────────────────────────┘

User uploads PDF/TXT file
         ↓
app.py / api.py (File Upload Handler)
         ↓
pipelines/document_pipeline.py
         ↓
azure_blob_service.py
├─ Check if file already exists (duplicate detection)
└─ Upload original file to Azure Blob Storage
         ↓
components/extractor.py
└─ Extract text from PDF/TXT
         ↓
components/chunking.py
└─ Split text into chunks (1000 tokens, 200 overlap)
         ↓
components/embedding.py
└─ Generate embeddings via Azure OpenAI
         ↓
services/vector_database.py
└─ Store embeddings in ChromaDB with metadata
         ↓
✅ Document indexed and searchable
```

**Key Points:**
- Runs **asynchronously** from query processing
- Can be background/batch processed
- Happens **once per document**
- No user query involved

---

### 💬 Query/Response Flow (Sequential Tool Execution)

This flow runs **every time** a user asks a question. Tools execute **sequentially, not in parallel**.

```
┌────────────────────────────────────────────────────────┐
│              QUERY/RESPONSE FLOW                        │
└────────────────────────────────────────────────────────┘

User asks question
         ↓
app.py / api.py (Query Handler)
         ↓
services/agent_service.py (Query Wrapper)
         ↓
agents/agent.py (LangGraph RAG Agent)
         ↓
┌────────────────────────────────────────┐
│  STEP 1: RAG Tool (MANDATORY FIRST)    │
└────────────────────────────────────────┘
         ↓
tools/rag.py
├─ Embed user query (Azure OpenAI)
├─ Search ChromaDB (cosine similarity)
├─ Retrieve top-3 most relevant chunks
└─ Decision Point:
         ↓
    ┌────┴────┐
    │         │
  YES        NO
    │         │
    ↓         ↓
Found docs   No docs found
    │         │
    ↓         ↓
Return       ┌────────────────────────────────────┐
answer       │ STEP 2: Grounding Tool (FALLBACK)  │
with         └────────────────────────────────────┘
citations              ↓
    │            tools/grounding.py
    │            ├─ Call Perplexity API
    │            ├─ Search web with query
    │            └─ Return web answer with sources
    │                     ↓
    └─────────────────────┴──────────────────────
                          ↓
                    Agent formats response
                          ↓
                    ✅ Return to user
```

**Key Points:**
- **Sequential execution** (NOT parallel)
- RAG tool **ALWAYS runs first** (mandatory)
- Grounding tool **ONLY runs if RAG finds nothing**
- **Never skip RAG** - this is enforced by system prompt
- **Never use both tools** for same query

---

### 🔄 RAG-First Enforcement

The system prompt in `agents/agent.py` enforces this strict workflow:

```
MANDATORY WORKFLOW:
1. ALWAYS use RAG tool FIRST for every query
2. IF RAG finds ANY documents → Use them and STOP
3. IF RAG finds NO documents → Then use Grounding tool
4. NEVER skip the RAG tool
5. NEVER use both tools for the same query
```

**Why This Design?**
- **Cost optimization**: Internal knowledge is free, web search costs money
- **Accuracy**: Your documents are trusted sources
- **Privacy**: Keep queries internal when possible
- **Control**: You manage the knowledge base

---

## Project Structure

```
AgentX/
├── app.py                          # Streamlit web UI
├── setup.py                        # Package configuration
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── Dockerfile                      # Docker configuration
├── .dockerignore                   # Docker build exclusions
│
├── src/
│   ├── __init__.py
│   ├── api.py                      # FastAPI application
│   │
│   ├── agents/
│   │   ├── agent.py                # LangGraph RAG Agent (core)
│   │   └── __init__.py
│   │
│   ├── components/
│   │   ├── embedding.py            # Azure OpenAI embeddings
│   │   ├── chunking.py             # Text chunking (1000 tokens, 200 overlap)
│   │   ├── extractor.py            # PDF/TXT extraction
│   │   ├── ingest_files.py         # File ingestion pipeline
│   │   └── __init__.py
│   │
│   ├── services/
│   │   ├── agent_service.py        # Agent wrapper for FastAPI
│   │   ├── llm_service.py          # Multi-LLM support (Azure, Groq, Gemini)
│   │   ├── vector_database.py      # ChromaDB manager
│   │   ├── azure_blob_service.py   # Azure Storage integration
│   │   ├── stt.py                  # Speech-to-text (Azure Cognitive Services)
│   │   ├── tts.py                  # Text-to-speech (Azure Cognitive Services)
│   │   └── __init__.py
│   │
│   ├── pipelines/
│   │   ├── document_pipeline.py    # Complete document processing
│   │   └── __init__.py
│   │
│   ├── tools/
│   │   ├── rag.py                  # RAG retrieval tool (RUNS FIRST)
│   │   ├── grounding.py            # Web grounding via Perplexity (FALLBACK)
│   │   └── __init__.py
│   │
│   ├── config/
│   │   ├── constants.py            # Configuration constants
│   │   └── __init__.py
│   │
│   ├── exceptions/
│   │   └── __init__.py
│   │
│   ├── logger/
│   │   └── __init__.py             # Logging configuration
│   │
│   └── utils/
│       ├── common.py               # Utility functions
│       └── __init__.py
│
├── data/
│   └── chromadb/                   # Vector database (persistent)
│
├── logs/
│   └── *.log                       # Application logs
│
└── tests/
    └── test_complete_rag.py        # Integration tests
```

---

## System Requirements

- **OS:** Windows 10+, macOS 11+, or Linux
- **RAM:** 4GB minimum (8GB recommended)
- **Storage:** 2GB free space
- **Network:** Internet connection for Azure services
- **Python:** 3.10 or higher

---

## Quick Start

### One-Command Setup
```bash
git clone https://github.com/Mahir-Baig/Agentx.git
cd AgentX
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
# Edit .env with your API keys
```

### Run the App
```bash
# Option 1: Streamlit UI
streamlit run app.py

# Option 2: FastAPI Backend
uvicorn src.api:app --reload
```

**First time?** See [Detailed Setup](#detailed-setup) below.

---

## Detailed Setup

### 1. Prerequisites

- **Python 3.10+**
- **Docker** (optional)
- **Azure Account** (for OpenAI, Embeddings, Storage, STT/TTS)
- **API Keys:**
  - Azure OpenAI
  - Groq (optional)
  - Perplexity (for web grounding)
  - LangSmith (for tracing)

### 2. Installation
```bash
# Clone repository
git clone https://github.com/Mahir-Baig/Agentx.git
cd AgentX

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 3. Configuration
```bash
# Copy example env file
cp .env.example .env

# Edit .env with your credentials
# Required:
#   - AZURE_OPENAI_ENDPOINT
#   - AZURE_OPENAI_API_KEY
#   - AZURE_STORAGE_CONNECTION_STRING
#   - PERPLEXITY_API_KEY
#   - LANGCHAIN_API_KEY (for tracing)
```

### 4. Run Streamlit App
```bash
streamlit run app.py
```

Opens at: `http://localhost:8501`

### 5. Run FastAPI Backend
```bash
uvicorn src.api:app --reload --port 8000
```

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Docker Deployment

### Build Image
```bash
docker build -t agentx-api:latest .
```

### Run Container
```bash
docker run -p 8000:8000 \
  --env-file .env \
  -v ./data:/app/data \
  -v ./logs:/app/logs \
  agentx-api:latest
```

### Docker Compose (Optional)
```yaml
version: '3.8'
services:
  agentx:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
```

---

## How It Works

### Understanding RAG (Retrieval-Augmented Generation)

AgentX implements a **standard RAG architecture** with the following pattern:

**RAG = Retrieve → Augment → Generate**

1. **Retrieve**: Find relevant documents from knowledge base
2. **Augment**: Add retrieved context to the prompt
3. **Generate**: LLM creates answer based on context

---

### 1. Document Processing Pipeline (Indexing)

**Purpose:** Build a searchable knowledge base

```
┌──────────────────────────────────────────────────┐
│           INDEXING PHASE                          │
│        (Happens once per document)                │
└──────────────────────────────────────────────────┘

Step 1: Upload
└─ User uploads PDF/TXT via UI or API

Step 2: Storage
└─ Original file stored in Azure Blob Storage
└─ Purpose: Backup and retrieval

Step 3: Text Extraction
└─ PyMuPDF/PyPDF extracts all text
└─ Handles multi-page documents
└─ Preserves structure and formatting

Step 4: Chunking
└─ Text split into 1000-token chunks
└─ 200-token overlap between chunks
└─ Preserves semantic context
└─ Example: 50-page PDF → ~100 chunks

Step 5: Embedding
└─ Each chunk → vector embedding
└─ Azure OpenAI (text-embedding-3-small)
└─ Vector: 1536-dimensional array
└─ Captures semantic meaning

Step 6: Vector Storage
└─ Embeddings stored in ChromaDB
└─ Metadata: source, page, timestamp
└─ Indexed for fast retrieval
└─ Ready for semantic search

✅ Document now searchable by meaning, not just keywords
```

---

### 2. Query Processing Pipeline (Retrieval)

**Purpose:** Answer questions using RAG pattern

```
┌──────────────────────────────────────────────────┐
│           RETRIEVAL PHASE                         │
│        (Happens on every query)                   │
└──────────────────────────────────────────────────┘

Step 1: Query Embedding
└─ User question converted to vector
└─ Same embedding model as documents
└─ Enables semantic comparison

Step 2: Similarity Search (RETRIEVE)
└─ Compare query vector to document vectors
└─ Cosine similarity algorithm
└─ Find top-3 most relevant chunks
└─ Threshold: 0.7 minimum similarity

Step 3: Context Building (AUGMENT)
└─ Retrieved chunks combined
└─ Added to LLM prompt as context
└─ System instruction: "Answer based on context"
└─ User question included

Step 4: Answer Generation (GENERATE)
└─ Azure OpenAI (GPT-4) processes prompt
└─ Generates answer using provided context
└─ Streaming response (real-time tokens)
└─ Citations added automatically

Step 5: Fallback (if RAG fails)
└─ IF no documents found (similarity < 0.7)
└─ THEN call Grounding Tool
└─ Perplexity API searches web
└─ Returns answer with source URLs

✅ Answer based on your documents or latest web info
```

---

### 3. RAG vs Grounding Decision Logic

```
User Query: "What is AgentX?"
    │
    ▼
┌─────────────────────────────────┐
│ RAG Tool: Search Internal KB    │
└─────────────┬───────────────────┘
              │
    ┌─────────┴─────────┐
    │                   │
Documents Found    No Documents
    │                   │
    ▼                   ▼
┌─────────────┐   ┌──────────────────┐
│ RAG Answer  │   │ Grounding Tool   │
│             │   │ Search Web       │
│ "Based on   │   │                  │
│  your docs, │   │ "According to    │
│  AgentX     │   │  web sources,    │
│  is..."     │   │  AgentX is..."   │
│             │   │                  │
│ [Citation:  │   │ [Source:         │
│  doc.pdf,   │   │  https://...]    │
│  page 5]    │   │                  │
└─────────────┘   └──────────────────┘
    │                   │
    └─────────┬─────────┘
              │
              ▼
        User Answer
```

**Decision Criteria:**
- **Similarity score ≥ 0.7**: Use RAG (internal knowledge)
- **Similarity score < 0.7**: Use Grounding (web search)
- **Never use both**: One tool per query

---

### 4. Conversation Memory System

Each conversation has a unique `thread_id` that maintains context:

```
Thread ID: user-abc-123
┌──────────────────────────────────────┐
│ Turn 1:                              │
│ User: "What documents do you have?"  │
│ Bot: "I have 5 documents about..."  │
│                                      │
│ Turn 2: (remembers previous context) │
│ User: "Summarize the first one"     │
│ Bot: "The Python document covers..." │
│                                      │
│ Turn 3: (full conversation history)  │
│ User: "What about machine learning?" │
│ Bot: "Based on doc 2 about ML..."   │
└──────────────────────────────────────┘

Memory Storage:
└─ LangGraph MemorySaver
└─ Thread-based checkpointing
└─ Persists across sessions
└─ Enables natural conversations
```

---

### 5. Component Interactions

```
┌─────────────────────────────────────────────┐
│           COMPONENT FLOW                     │
└─────────────────────────────────────────────┘

User Interface (Streamlit/FastAPI)
         ↓
    [User Input]
         ↓
Agent Service (Wrapper)
         ↓
LangGraph Agent (Orchestrator)
         ↓
    ┌────┴────┐
    │         │
RAG Tool  Grounding Tool
    │         │
    ├─────────┴─────────┐
    │                   │
ChromaDB         Perplexity API
    │                   │
    └─────────┬─────────┘
              │
      Azure OpenAI
    (Embeddings + LLM)
              │
              ▼
       [Generated Answer]
              │
              ▼
        User Interface
```

---

### 6. Real-World Example

**Scenario:** User asks about "AgentX features"

**Step-by-Step:**

1. **User Input**: "What are the key features of AgentX?"

2. **Query Embedding**:
   - Text → Vector: `[0.123, -0.456, 0.789, ...]`
   - Takes ~100ms

3. **Vector Search**:
   - ChromaDB searches 1000 document chunks
   - Finds 3 most similar:
     * Chunk 1: "AgentX features include RAG-first..." (0.92 similarity)
     * Chunk 2: "Key capabilities: multi-modal I/O..." (0.87 similarity)
     * Chunk 3: "The system supports conversation..." (0.81 similarity)
   - Takes ~200ms

4. **Context Augmentation**:
   ```
   System: "Answer based on the provided context."
   Context: [Chunk 1 + Chunk 2 + Chunk 3]
   Question: "What are the key features of AgentX?"
   ```

5. **LLM Generation**:
   - Azure OpenAI processes augmented prompt
   - Generates: "Based on the documentation, AgentX's key features include..."
   - Streams response in real-time
   - Takes ~2-3 seconds

6. **Citation Addition**:
   - Automatically adds: `[Source: agentx_readme.pdf, page 1]`
   - Links answer back to source documents

7. **User Receives**: Complete answer with citations in ~3 seconds

**If RAG had failed:**
- Would call Perplexity API
- Search web for "AgentX features"
- Return web-based answer with URLs

---

### Key Advantages of This Approach

✅ **Accuracy**: Answers grounded in your documents, not hallucinations

✅ **Traceability**: Citations show exact source of information

✅ **Semantic Search**: Finds relevant info by meaning, not keywords

✅ **Scalability**: Handles thousands of documents efficiently

✅ **Cost Efficient**: Uses internal knowledge first, web search as fallback

✅ **Privacy**: Sensitive information stays in your knowledge base

✅ **Flexibility**: Easy to add/update documents without retraining

✅ **Context Aware**: Remembers conversation history for natural interactions

---

## Core Components

### LangGraph RAG Agent (`src/agents/agent.py`)

**The Brain of AgentX**

- Multi-tool agent using `create_react_agent`
- Enforces RAG-first workflow via system prompt
- Conversation memory via `MemorySaver` checkpoint
- Streaming and batch response modes

**System Prompt (Simplified):**
```
You are a RAG-first agent. MANDATORY workflow:
1. ALWAYS try RAG tool first
2. If RAG finds documents → use them (STOP)
3. If RAG finds nothing → use Grounding tool
4. Never skip RAG, never use both tools
```

**Available Tools:**
1. **RAG Tool** (`tools/rag.py`) - Searches internal knowledge base
2. **Grounding Tool** (`tools/grounding.py`) - Searches web via Perplexity

---

### Document Pipeline (`src/pipelines/document_pipeline.py`)

**The Ingestion Manager**

Orchestrates the complete document processing flow:
- Upload handling
- Duplicate detection
- Text extraction
- Chunking
- Embedding generation
- Storage (both ChromaDB and Azure Blob)

**Key Functions:**
- `process_document()` - Main pipeline orchestrator
- `check_duplicate()` - Prevents re-processing
- `extract_and_chunk()` - Text processing
- `embed_and_store()` - Vector storage

---

### LLM Service (`src/services/llm_service.py`)

**Multi-Provider LLM Support**

Supports multiple AI providers:
- **Azure OpenAI** (Primary) - GPT-4, GPT-3.5
- **Groq** (Fast alternative) - Meta Llama models
- **Google Gemini** (Optional) - Gemini Pro

**Features:**
- Configurable temperature, max_tokens
- Response streaming for real-time UX
- Request timeouts (10s default)
- LangSmith tracing integration
- Automatic retry logic

**Configuration:**
```python
llm = LLMService(
    provider="azure",
    model="gpt-4-mini",
    temperature=0.7,
    max_tokens=500,
    streaming=True
)
```

---

### Vector Database (`src/services/vector_database.py`)

**ChromaDB Manager**

- **Storage:** Persistent local storage in `data/chromadb/`
- **Search:** Cosine similarity search
- **Metadata:** Stores source, page numbers, timestamps
- **Collections:** Organized by document type

**Key Operations:**
```python
# Add documents
vector_db.add_documents(chunks, embeddings, metadata)

# Search
results = vector_db.similarity_search(query_embedding, top_k=3)

# Delete
vector_db.delete_collection(collection_name)
```

---

### RAG Tool (`src/tools/rag.py`)

**The Internal Knowledge Retriever**

**Process:**
1. Embed user query using Azure OpenAI
2. Search ChromaDB for top-3 similar chunks
3. Format results with citations
4. Return to agent

**Key Features:**
- Similarity threshold filtering (0.7 default)
- Source attribution (document name, page)
- Chunk context preservation
- Empty result handling

**Output Format:**
```
Based on internal documents:
[Content from chunk 1] [Source: doc.pdf, page 5]
[Content from chunk 2] [Source: doc.pdf, page 7]
[Content from chunk 3] [Source: guide.pdf, page 2]
```

---

### Grounding Tool (`src/tools/grounding.py`)

**The Web Search Fallback**

**When It Runs:**
- ONLY when RAG tool finds NO relevant documents
- Never runs if RAG succeeds
- Handles queries outside knowledge base

**Process:**
1. Receives query from agent
2. Calls Perplexity API (sonar-pro model)
3. Retrieves web search results
4. Formats with source URLs
5. Returns to agent

**Key Features:**
- Real-time web information
- Multiple source aggregation
- URL citation tracking
- Error handling with fallback

**Output Format:**
```
Based on web search:
[Web content summary]
Sources:
- https://example.com/article1
- https://example.com/article2
```

---

### Azure Integration

**Cloud Services Used:**

1. **Azure Blob Storage**
   - Original document storage
   - Duplicate detection
   - File versioning
   - Backup and recovery

2. **Azure OpenAI**
   - Text embeddings (text-embedding-3-small)
   - LLM inference (GPT-4, GPT-3.5)
   - Streaming responses
   - Token usage tracking

3. **Azure Cognitive Services**
   - Speech-to-Text (STT) - Real-time transcription
   - Text-to-Speech (TTS) - Natural voice synthesis
   - Multi-language support
   - Custom voice options

4. **LangSmith**
   - Request/response tracing
   - Performance monitoring
   - Debug insights
   - Token cost tracking

---

## API Reference

### FastAPI Endpoints

#### POST `/query`

Query the RAG agent with optional conversation context.

**Request:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is in the knowledge base?",
    "thread_id": "optional-uuid",
    "stream": false
  }'
```

**Response:**
```json
{
  "success": true,
  "query": "What is in the knowledge base?",
  "response": "Based on internal documents, AgentX contains information about...",
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "sources": [
    {
      "document": "agentx_guide.pdf",
      "page": 5,
      "content": "..."
    }
  ],
  "metadata": {
    "tool_used": "rag",
    "tokens": 245,
    "latency_ms": 1234
  }
}
```

**Parameters:**
- `query` (string, required): User question
- `thread_id` (string, optional): Conversation ID for context
- `stream` (boolean, optional): Enable streaming responses

---

#### GET `/health`

Health check endpoint for monitoring.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.0.1",
  "services": {
    "chromadb": "connected",
    "azure_openai": "connected",
    "azure_blob": "connected"
  }
}
```

---

#### POST `/upload`

Upload and process documents into the knowledge base.

**Request:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "metadata={\"category\":\"technical\"}"
```

**Response:**
```json
{
  "success": true,
  "filename": "document.pdf",
  "chunks_created": 15,
  "processing_time_seconds": 8.5,
  "document_id": "doc-uuid-123"
}
```

---

## Usage Examples

### Basic Query
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What are the key features?"}
)

print(response.json()["response"])
```

---

### With Conversation Memory
```python
import requests

thread_id = "user-123"
queries = [
    "What documents do you have?",
    "Summarize the main points",
    "What about machine learning?"
]

for query in queries:
    response = requests.post(
        "http://localhost:8000/query",
        json={"query": query, "thread_id": thread_id}
    )
    print(f"Q: {query}")
    print(f"A: {response.json()['response']}\n")
```

---

### Streaming Responses
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "Explain RAG systems", "stream": True},
    stream=True
)

for chunk in response.iter_content(chunk_size=None):
    if chunk:
        print(chunk.decode(), end="", flush=True)
```

---

### Document Upload
```python
import requests

with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload",
        files={"file": f}
    )

print(f"Uploaded: {response.json()['filename']}")
print(f"Chunks: {response.json()['chunks_created']}")
```

---

## RAG vs Grounding Decision Tree

```
User asks question
    ↓
Agent receives query
    ↓
STEP 1: Call RAG tool (MANDATORY - ALWAYS FIRST)
    ↓
RAG searches ChromaDB
    ↓
Did RAG find any documents?
    ├─ YES (1+ documents found)
    │  ├─ Format answer with citations
    │  ├─ Return response to user
    │  └─ DONE ✅ [Grounding NOT called]
    │
    └─ NO (0 documents found)
       ↓
       STEP 2: Call Grounding tool (FALLBACK ONLY)
       ↓
       Grounding searches web via Perplexity
       ↓
       Format answer with source URLs
       ↓
       Return response to user
       ↓
       DONE ✅
```

### Critical Rules

1. **RAG is ALWAYS attempted first** - No exceptions
2. **Grounding runs ONLY if RAG fails** - Never in parallel
3. **Once RAG succeeds, grounding is skipped** - Efficiency
4. **Agent never uses both tools** - One or the other

### Why This Matters

| Scenario | RAG Result | Grounding Called? | Reason |
|----------|-----------|------------------|---------|
| Query about uploaded docs | Found 3 chunks | ❌ No | RAG succeeded |
| Query about uploaded docs | Found 1 chunk | ❌ No | RAG succeeded (even with 1 result) |
| Query about recent news | Found 0 chunks | ✅ Yes | RAG failed, need web search |
| Query about unknown topic | Found 0 chunks | ✅ Yes | RAG failed, fallback to web |

---

## Voice Features

### Speech-to-Text (STT)

**Microphone Input in Streamlit UI**

- Azure Cognitive Services integration
- Real-time continuous recognition
- Multi-language support
- Automatic punctuation and formatting

**Usage:**
1. Click microphone button in UI
2. Speak your question
3. Text appears in input field
4. Submit to agent

**Supported Languages:** English, Spanish, French, German, Chinese, Japanese, and 60+ more

---

### Text-to-Speech (TTS)

**Read Aloud for Bot Responses**

- Natural-sounding voice synthesis
- Multiple voice options (male/female)
- Adjustable speech rate and pitch
- Background audio playback

**Usage:**
1. Agent responds to your query
2. Click "Read Aloud" button
3. Listen to response

**Voice Options:**
- en-US-JennyNeural (Female, friendly)
- en-US-GuyNeural (Male, professional)
- Custom voices available

---

## Logging & Monitoring

### Log Files

**Location:** `logs/` directory

**Structure:**
```
logs/
├── app_2024-12-29.log         # Application logs
├── agent_2024-12-29.log       # Agent decisions
├── api_2024-12-29.log         # API requests
└── errors_2024-12-29.log      # Error tracking
```

**Format:**
```
2024-12-29 14:23:45 [INFO] Query received: "What is RAG?"
2024-12-29 14:23:46 [DEBUG] RAG tool: Found 3 documents
2024-12-29 14:23:47 [INFO] Response sent (245 tokens, 1.2s)
```

**Log Levels:**
- `DEBUG` - Detailed diagnostic information
- `INFO` - General informational messages
- `WARNING` - Warning messages (non-critical)
- `ERROR` - Error messages (functionality impacted)
- `CRITICAL` - Critical errors (system failure)

**Rotation:** Daily rotation with 7-day retention

---

### LangSmith Integration

**Full Observability and Debugging**

**Features:**
- Request/response tracing
- Token usage tracking
- Performance metrics
- Cost analysis
- Error debugging
- Latency monitoring

**Access:** https://smith.langchain.com → Project: `rag-agent-project`

**What You Can Track:**
1. **Agent Decisions** - Which tool was used and why
2. **Token Consumption** - Input/output tokens per request
3. **Latency Breakdown** - Time spent in each component
4. **Error Traces** - Full stack traces for failures
5. **Tool Performance** - Success rate of RAG vs Grounding
6. **Cost Analysis** - API costs per query

---

## Security Best Practices

### IMPORTANT: Never Commit Secrets!

Your `.env` file contains sensitive API keys. Make sure:
- `.env` is in `.gitignore`
- Use `.env.example` as a template
- Never commit actual credentials
- Never share API keys publicly

### Additional Security Measures

1. **Environment Variables**
   - All credentials in `.env` (never commit)
   - `.env` in `.gitignore`
   - Use `.env.example` as template
   - Rotate keys regularly

2. **API Keys**
   - Rotate Azure keys every 90 days
   - Use Managed Identities when possible
   - Restrict API key scopes to minimum required
   - Monitor API usage for anomalies

3. **Data Privacy**
   - ChromaDB stored locally in `data/`
   - Logs in `logs/` directory
   - Exclude sensitive data from version control
   - Implement data encryption at rest

4. **Network Security**
   - Use HTTPS in production (TLS 1.2+)
   - Implement rate limiting (100 requests/minute)
   - Add authentication middleware (JWT/OAuth)
   - Use API keys for endpoint protection

5. **Input Validation**
   - Sanitize all user inputs
   - Limit file upload sizes (10MB default)
   - Validate file types (PDF, TXT only)
   - Prevent injection attacks

---

## Performance Optimization

### Implemented Features

- **Embedding caching** - Reuse embedding instances
- **Query result caching** - 1-hour TTL for identical queries
- **Reduced token limits** - 500 max tokens per response
- **Response streaming** - Real-time token generation
- **Request timeouts** - 10-second limit prevents hanging
- **Top-3 chunk retrieval** - Balance quality vs speed
- **Streamlit component caching** - Faster UI rendering
- **Batch processing** - Process multiple documents in parallel

### Expected Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| Cold start | <5s | 3-4s |
| Cached queries | <500ms | 300-400ms |
| Streaming response | 200-300ms | 200-250ms |
| Token cost reduction | 50% | 45-55% |
| RAG retrieval | <1s | 0.5-0.8s |
| Web grounding | <3s | 2-3s |

### Optimization Tips

1. **Adjust Chunk Size**
   - Edit `src/components/chunking.py`
   - Smaller chunks = faster retrieval, less context
   - Larger chunks = more context, slower retrieval
   - Default: 1000 tokens works well for most use cases

2. **Tune Retrieval Parameters**
   - Modify `top_k` in `src/tools/rag.py`
   - `top_k=3` is optimal for most queries
   - Increase to 5-7 for complex questions
   - Decrease to 1-2 for simple lookups

3. **Enable Redis Caching**
   - Implement Redis for distributed caching
   - Cache embedding vectors
   - Cache common query results
   - Set appropriate TTL values

4. **Use Lighter Models**
   - Switch to GPT-3.5 for faster responses
   - Use smaller embedding models
   - Trade-off: speed vs accuracy
   - Monitor quality degradation

5. **Optimize ChromaDB**
   - Regularly compact database
   - Use appropriate collection strategies
   - Index metadata fields
   - Monitor disk usage

6. **Parallel Processing**
   - Process multiple documents simultaneously
   - Use async/await for I/O operations
   - Batch embed multiple chunks
   - Queue long-running tasks

---

## Testing

### Run Integration Tests
```bash
python tests/test_complete_rag.py
```

**What It Tests:**
- Document upload and processing
- RAG retrieval accuracy
- Grounding fallback
- Conversation memory
- API endpoint functionality

### Test FastAPI Endpoint
```bash
# Health check
curl http://localhost:8000/health

# Query test
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}'

# Document upload test
curl -X POST "http://localhost:8000/upload" \
  -F "file=@test.pdf"
```

### Test Streamlit UI
```bash
streamlit run app.py
```

**Manual Test Checklist:**
- [ ] File upload works
- [ ] Query returns response
- [ ] RAG finds documents
- [ ] Grounding fallback works
- [ ] Conversation memory persists
- [ ] STT/TTS functional
- [ ] Streaming responses work

### Run Unit Tests (if available)
```bash
pytest tests/ -v
pytest tests/test_rag.py -v
pytest tests/test_grounding.py -v
```

---

## Dependencies

### Core Framework

- **LangChain 0.1.0+** - LLM orchestration framework
- **LangGraph 0.0.20+** - Agentic workflow management
- **FastAPI 0.104+** - RESTful API server
- **Streamlit 1.28+** - Interactive web UI

### AI/ML

- **Azure OpenAI** - LLM and embeddings
- **ChromaDB 0.4.18+** - Vector database
- **Groq SDK** - Alternative LLM provider
- **Perplexity API** - Web grounding/search

### Cloud Services

- **Azure Storage Blob** - Document storage
- **Azure Cognitive Services** - STT/TTS
- **LangSmith** - Tracing and observability

### Document Processing

- **PyMuPDF (fitz)** - PDF text extraction
- **PyPDF2** - Fallback PDF parser
- **python-docx** - Word document support (optional)
- **tiktoken** - Token counting

### Utilities

- **Python-dotenv** - Environment variable management
- **Requests** - HTTP client
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **Rich** - Terminal formatting

See `requirements.txt` for complete list with versions.

---

## FAQ

**Q: Does this work with OpenAI's API directly?**

A: Currently optimized for Azure OpenAI, but you can modify `llm_service.py` to use OpenAI's API directly. Change the client initialization and remove Azure-specific parameters like `azure_endpoint` and `api_version`.

**Q: Can I use a different vector database?**

A: Yes! Replace `vector_database.py` with implementations for:
- **Pinecone** - Cloud-hosted, serverless
- **Weaviate** - Open-source, GraphQL API
- **Qdrant** - Rust-based, high performance
- **Milvus** - Distributed, scalable

The interface remains similar - implement the same methods: `add_documents()`, `similarity_search()`, `delete_collection()`.

**Q: How much does it cost to run?**

A: Costs depend on Azure usage. Approximate costs per 1000 queries:
- **GPT-4-mini**: $15-25
- **GPT-3.5-turbo**: $2-5
- **Embeddings**: $0.10-0.50
- **Storage**: $0.01-0.05
- **Total**: ~$17-30 per 1000 queries

Cost optimization tips:
- Use GPT-3.5 instead of GPT-4
- Cache common queries
- Reduce max_tokens
- Batch embed operations

**Q: Can I run this completely offline?**

A: No, it requires Azure services for embeddings and LLM inference. However, you could modify it to use:
- **Ollama** - Local LLM inference
- **LM Studio** - Desktop LLM app
- **Sentence Transformers** - Local embeddings
- **Llama.cpp** - Optimized local models

This would make it fully offline but requires significant compute resources (16GB+ RAM, GPU recommended).

**Q: How many documents can it handle?**

A: ChromaDB can scale to millions of vectors. Practical limits:
- **Local deployment**: 10K-100K documents (depends on hardware)
- **Cloud deployment**: 100K-1M+ documents (with optimization)
- **Performance**: Retrieval time increases logarithmically

For large-scale deployments (1M+ documents), consider:
- Cloud-hosted vector databases (Pinecone, Weaviate Cloud)
- Hierarchical indexing
- Sharding across multiple collections

**Q: Does it support languages other than English?**

A: Yes! Azure OpenAI models support 50+ languages including:
- Spanish, French, German
- Chinese, Japanese, Korean
- Arabic, Hindi, Portuguese
- And many more

The RAG system works with any language the underlying LLM supports. For best results, use language-specific embedding models.

**Q: Can I customize the chunking strategy?**

A: Absolutely! Edit `src/components/chunking.py`:
```python
# Semantic chunking
# Sentence-based chunking
# Paragraph-based chunking
# Custom separators
```

Different strategies work better for different document types:
- **Technical docs**: Smaller chunks (500-750 tokens)
- **Narratives**: Larger chunks (1500-2000 tokens)
- **Mixed content**: Default (1000 tokens)

**Q: How do I add more tools to the agent?**

A: Create a new tool in `src/tools/` and register it in `agent.py`:
```python
# src/tools/calculator.py
from langchain.tools import Tool

def calculator(query: str) -> str:
    # Implementation
    pass

calculator_tool = Tool(
    name="calculator",
    description="Performs mathematical calculations",
    func=calculator
)

# src/agents/agent.py
tools = [rag_tool, grounding_tool, calculator_tool]
agent = create_react_agent(llm, tools, checkpointer)
```

**Q: Can I deploy this on AWS or GCP instead of Azure?**

A: Yes, with modifications:
- Replace Azure OpenAI with OpenAI API or Bedrock/Vertex AI
- Replace Azure Blob Storage with S3 or GCS
- Replace Azure Cognitive Services with alternative STT/TTS
- Update environment variables and service initialization

**Q: How do I handle PDF tables and images?**

A: Current version extracts text only. For advanced extraction:
- **Tables**: Use `tabula-py` or `camelot`
- **Images**: Use OCR (Tesseract, Azure Vision API)
- **Charts**: Use image embedding models
- **Multi-modal**: Upgrade to GPT-4 Vision

---

## Troubleshooting

### Common Issues

**1. "Azure OpenAI credentials not found"**
```bash
# Verify .env file exists and has AZURE_OPENAI_* variables
cat .env | grep AZURE_OPENAI

# Check if variables are loaded
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('AZURE_OPENAI_API_KEY'))"

# If still failing, check .env location
ls -la .env

# Ensure no trailing spaces in .env values
```

**2. ChromaDB connection error**
```bash
# Ensure data/chromadb/ exists
mkdir -p data/chromadb

# Check permissions
ls -la data/chromadb/

# Delete and recreate if corrupted
rm -rf data/chromadb/
mkdir -p data/chromadb

# Check if another process is using ChromaDB
lsof | grep chromadb
```

**3. Perplexity API key invalid**
```bash
# Verify PERPLEXITY_API_KEY in .env
grep PERPLEXITY_API_KEY .env

# Test API key directly
curl https://api.perplexity.ai/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "sonar-pro", "messages": [{"role": "user", "content": "test"}]}'

# Check for typos in key
echo $PERPLEXITY_API_KEY | wc -c  # Should be 64+ characters
```

**4. Streamlit port already in use**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process (Unix/Mac)
lsof -ti:8501 | xargs kill -9

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Or find and kill by name
pkill -f streamlit
```

**5. Import errors after installation**
```bash
# Reinstall in editable mode
pip uninstall agentx -y
pip install -e .

# Check if package is installed
pip show agentx

# Verify Python path
python -c "import sys; print('\n'.join(sys.path))"

# Clear pip cache if issues persist
pip cache purge
```

**6. Memory errors with large documents**
```bash
# Reduce chunk size in src/components/chunking.py
CHUNK_SIZE = 500  # Instead of 1000

# Decrease max_tokens in src/services/llm_service.py
max_tokens = 250  # Instead of 500

# Process documents in batches
# Split large PDFs into smaller files
# Use incremental processing
```

**7. Slow response times**
```bash
# Enable caching
# Reduce top_k in RAG retrieval (3 → 2)
# Use GPT-3.5 instead of GPT-4
# Implement query result caching
# Check network latency to Azure

# Monitor performance
import time
start = time.time()
# Your query here
print(f"Query took {time.time() - start:.2f}s")
```

**8. "Module not found" errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt

# Check for missing packages
pip list | grep langchain
pip list | grep chromadb
```

**9. RAG not finding documents**
```bash
# Check if documents are in ChromaDB
python -c "from src.services.vector_database import VectorDatabase; db = VectorDatabase(); print(db.get_collection_count())"

# Verify embeddings are generated
# Check logs for processing errors
cat logs/app_*.log | grep ERROR

# Re-index documents if needed
# Delete and re-upload documents
```

**10. Grounding tool always called (RAG bypassed)**
```bash
# Check agent system prompt in src/agents/agent.py
# Ensure RAG-first enforcement is present
# Verify RAG tool is returning results
# Check logs for agent decisions

# Test RAG tool directly
python -c "from src.tools.rag import rag_tool; print(rag_tool.run('test query'))"
```

---

## Configuration Reference

### Environment Variables

```bash
# ============================================
# Azure OpenAI Configuration
# ============================================
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# ============================================
# Groq Configuration (Optional)
# ============================================
GROQ_API_KEY=your-groq-key-here
GROQ_MODEL=llama3-70b-8192

# ============================================
# Perplexity Configuration (Web Grounding)
# ============================================
PERPLEXITY_API_KEY=your-perplexity-key-here
PERPLEXITY_MODEL=sonar-pro

# ============================================
# Azure Storage Configuration
# ============================================
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net
AZURE_STORAGE_CONTAINER_NAME=documents

# ============================================
# Azure Cognitive Services (STT/TTS)
# ============================================
AZURE_SPEECH_KEY=your-speech-key-here
AZURE_SPEECH_REGION=eastus

# ============================================
# LangSmith Configuration (Observability)
# ============================================
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your-langsmith-key-here
LANGCHAIN_PROJECT=rag-agent-project

# ============================================
# Application Settings
# ============================================
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=500
TEMPERATURE=0.7
TOP_K_RETRIEVAL=3

# ============================================
# Optional: Development Settings
# ============================================
DEBUG_MODE=false
LOG_LEVEL=INFO
ENABLE_STREAMING=true
CACHE_TTL=3600
```

### Chunking Configuration

Edit `src/components/chunking.py`:
```python
CHUNK_SIZE = 1000        # Tokens per chunk
CHUNK_OVERLAP = 200      # Overlap between chunks
SEPARATORS = ["\n\n", "\n", " ", ""]  # Split priority

# Advanced options
MIN_CHUNK_SIZE = 100     # Minimum viable chunk
MAX_CHUNK_SIZE = 1500    # Maximum chunk size
KEEP_SEPARATOR = True    # Preserve separators
```

### Retrieval Configuration

Edit `src/tools/rag.py`:
```python
TOP_K = 3                      # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7      # Minimum similarity score (0-1)
MAX_DISTANCE = 0.3              # Maximum cosine distance

# Reranking options
ENABLE_RERANKING = False        # Use cross-encoder reranking
RERANK_TOP_N = 10               # Candidates for reranking
```

### LLM Configuration

Edit `src/services/llm_service.py`:
```python
DEFAULT_TEMPERATURE = 0.7       # Randomness (0-1)
DEFAULT_MAX_TOKENS = 500        # Max response length
DEFAULT_TIMEOUT = 10            # Request timeout (seconds)

# Model selection
PRIMARY_MODEL = "gpt-4-mini"
FALLBACK_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"
```

---

## Deployment

### Development

```bash
# Streamlit UI (Development)
streamlit run app.py

# FastAPI with hot reload (Development)
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Both simultaneously (separate terminals)
Terminal 1: streamlit run app.py
Terminal 2: uvicorn src.api:app --reload
```

### Docker

```bash
# Build image
docker build -t agentx-api:latest .

# Run container
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  agentx-api:latest

# Docker Compose
docker-compose up -d

# View logs
docker logs -f agentx-api

# Stop container
docker-compose down
```

### Production (Azure Container Apps)

```bash
# Prerequisites
az login
az account set --subscription <your-subscription-id>

# Build and push to Azure Container Registry
az acr build \
  --registry <your-acr-name> \
  --image agentx:v1 \
  --file Dockerfile \
  .

# Create container app environment (first time only)
az containerapp env create \
  --name agentx-env \
  --resource-group <your-rg> \
  --location eastus

# Deploy to Azure Container Apps
az containerapp create \
  --name agentx-api \
  --resource-group <your-rg> \
  --environment agentx-env \
  --image <your-acr>.azurecr.io/agentx:v1 \
  --target-port 8000 \
  --ingress external \
  --cpu 1.0 \
  --memory 2.0Gi \
  --min-replicas 1 \
  --max-replicas 5 \
  --env-vars \
    AZURE_OPENAI_ENDPOINT=<value> \
    AZURE_OPENAI_API_KEY=secretref:openai-key \
  --secrets \
    openai-key=<your-secret>

# Update existing deployment
az containerapp update \
  --name agentx-api \
  --resource-group <your-rg> \
  --image <your-acr>.azurecr.io/agentx:v1.1
```

### Production (AWS ECS)

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker build -t agentx-api:latest .
docker tag agentx-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/agentx:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/agentx:latest

# Create ECS task definition and service
# See AWS ECS documentation for detailed steps
```

### Production Checklist

**Before Deployment:**
- [ ] `.env` file configured with production credentials
- [ ] LangSmith project created for monitoring
- [ ] Azure resources provisioned (OpenAI, Storage, Cognitive Services)
- [ ] Docker image built and tested locally
- [ ] Logs directory created with write permissions
- [ ] Backup strategy for `data/chromadb/` implemented
- [ ] Health checks configured (`/health` endpoint tested)
- [ ] Resource limits set (CPU: 1-2 cores, Memory: 2-4GB)
- [ ] Rate limiting implemented (100-1000 req/min)
- [ ] Authentication middleware added (JWT/API key)
- [ ] HTTPS/SSL certificates configured
- [ ] Monitoring and alerts setup (Azure Monitor, CloudWatch)
- [ ] Load balancer configured (if multi-instance)
- [ ] Database backup schedule (daily)
- [ ] Disaster recovery plan documented

**After Deployment:**
- [ ] Test all endpoints (query, upload, health)
- [ ] Monitor logs for errors
- [ ] Check LangSmith traces
- [ ] Verify response times (<3s)
- [ ] Test conversation memory
- [ ] Validate RAG-first workflow
- [ ] Test failover scenarios
- [ ] Load test with expected traffic
- [ ] Document API access details
- [ ] Setup monitoring dashboards

---

## Additional Resources

### Official Documentation
- **LangChain:** https://python.langchain.com/
- **LangGraph:** https://langchain-ai.github.io/langgraph/
- **FastAPI:** https://fastapi.tiangolo.com/
- **Streamlit:** https://docs.streamlit.io/
- **ChromaDB:** https://docs.trychroma.com/

### Cloud Services
- **Azure OpenAI:** https://azure.microsoft.com/en-us/products/ai-services/openai-service/
- **Azure Blob Storage:** https://azure.microsoft.com/en-us/products/storage/blobs/
- **Azure Cognitive Services:** https://azure.microsoft.com/en-us/products/cognitive-services/
- **Perplexity API:** https://docs.perplexity.ai/
- **LangSmith:** https://docs.smith.langchain.com/

### Learning Resources
- **RAG Fundamentals:** https://www.pinecone.io/learn/retrieval-augmented-generation/
- **Vector Databases:** https://www.pinecone.io/learn/vector-database/
- **LLM Best Practices:** https://platform.openai.com/docs/guides/gpt-best-practices
- **Prompt Engineering:** https://www.promptingguide.ai/

### Community
- **GitHub Issues:** https://github.com/Mahir-Baig/Agentx/issues
- **Discussions:** https://github.com/Mahir-Baig/Agentx/discussions
- **LangChain Discord:** https://discord.gg/langchain

---


## Author

**Mahir Baig**
- GitHub: [@Mahir-Baig](https://github.com/Mahir-Baig)
- Project: [AgentX](https://github.com/Mahir-Baig/Agentx)
- Email: [mahirbaig2@gmail.com]
---

## Contributing

We welcome contributions! Here's how to get started:

### How to Contribute

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
4. **Make** your changes with clear commit messages
5. **Test** your changes thoroughly
6. **Push** to your branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request with detailed description

### Development Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Agentx.git
cd AgentX

# Add upstream remote
git remote add upstream https://github.com/Mahir-Baig/Agentx.git

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Setup pre-commit hooks (if available)
pre-commit install

# Run tests before committing
pytest tests/ -v
```

**Commit Messages:**
- Use clear, descriptive messages
- Format: `type(scope): description`
- Examples:
  - `feat(rag): add semantic chunking strategy`
  - `fix(api): handle timeout errors gracefully`
  - `docs(readme): update deployment section`

**Pull Request Process:**
1. Update documentation
2. Add tests for new functionality
3. Request review from maintainers
4. Address review comments
5. Squash commits before merge


### Completed Features ✅

- [x] RAG-first workflow with sequential tool execution
- [x] Multi-modal I/O (text, voice)
- [x] Conversation memory with thread IDs
- [x] FastAPI backend with REST API
- [x] Streamlit frontend with document upload
- [x] Docker support with compose
- [x] Azure integration (OpenAI, Storage, Cognitive Services)
- [x] LangSmith tracing and observability
- [x] ChromaDB vector database
- [x] Web grounding via Perplexity
- [x] PDF and TXT document support
- [x] Token-based chunking
- [x] Response streaming
- [x] Error handling and logging

## Changelog

### Version 0.0.1 (2024-12-29)

**Initial Release**

**Features:**
- RAG-first conversational agent
- Sequential tool execution (RAG → Grounding)
- Multi-modal I/O (text, STT, TTS)
- Conversation memory with thread IDs
- FastAPI REST API
- Streamlit web UI
- Azure OpenAI integration
- ChromaDB vector database
- Document ingestion pipeline
- LangSmith tracing
- Docker deployment

**Known Issues:**
- Large PDF processing can be slow
- No support for images/tables in documents
- Limited to 3 chunks per retrieval

**Next Release:**
- Performance optimizations
- Multi-modal document support
- Advanced chunking strategies

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

---

**Last Updated:** December 30, 2025  
**Status:** Production Ready  
**Version:** 0.0.1

---

## Quick Links

- 📖 [Documentation](#table-of-contents)
- 🚀 [Quick Start](#quick-start)
- 🏗️ [Architecture](#architecture)
- 🔄 [Execution Flows](#execution-flows)
- 🐛 [Troubleshooting](#troubleshooting)
- 💬 [GitHub Issues](https://github.com/Mahir-Baig/Agentx/issues)
- 🤝 [Contributing](#contributing)

---

**Built with ❤️ by Mahir Baig**