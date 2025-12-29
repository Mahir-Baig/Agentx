# AgentX - RAG-First Conversational AI Agent
This project implements an agent with a RAG-first workflow that prioritizes an internal knowledge base and grounding via the Perplexity API. It supports both text-based and speech-based (STT) user prompts and can read generated responses aloud using text-to-speech (TTS).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

A production-ready **Retrieval-Augmented Generation (RAG)** system that combines an intelligent agent with a knowledge base, web grounding, and multi-modal I/O (text, voice, and audio).

**Version:** 0.0.1 | **Author:** Mahir Baig

---

## Table of Contents

- [Overview](#overview)
- [Why AgentX?](#why-agentx)
- [Architecture](#architecture)
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
- **Intelligent Tool Selection** - Automatically routes between RAG and web grounding
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
| Knowledge Priority | RAG-first workflow | Random tool selection |
| Multi-modal I/O | Text + Voice | Text only |
| Memory | Thread-based context | Stateless |
| Web Fallback | Automatic grounding | No fallback |
| Production Ready | FastAPI + Docker | Notebook only |
| Observability | LangSmith tracing | Limited logging |

---

## Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    User Interfaces                       │
├─────────────────┬─────────────────────────────────────┤
│ Streamlit App   │      FastAPI Backend                │
│ (Web UI)        │      (REST API)                     │
└────────┬────────┴──────────────┬──────────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
         ┌───────────▼──────────────┐
         │   LangGraph RAG Agent    │
         │  (Conversation Memory)   │
         └───────────┬──────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼──┐  ┌─────▼──┐  ┌─────▼──┐
   │  RAG  │  │Grounding│ │Document│
   │ Tool  │  │  Tool   │ │Pipeline│
   └────┬──┘  └─────┬──┘  └─────┬──┘
        │          │            │
   ┌────▼──────────▼──┐    ┌────▼──────────────┐
   │  ChromaDB Vector │    │ Azure Blob Storage│
   │  Database        │    │ (Documents)       │
   └──────────────────┘    └───────────────────┘
        │                  │
   ┌────▼────────────────────▼──┐
   │ Azure OpenAI Embeddings    │
   └────────────────────────────┘
```

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
│   │   ├── rag.py                  # RAG retrieval tool
│   │   ├── grounding.py            # Web grounding via Perplexity AI
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

### 1. Document Upload and Processing Pipeline
```
Upload PDF/TXT
     ↓
Duplicate Check (Azure Blob)
     ↓
Extract Text (PyMuPDF/PyPDF)
     ↓
Chunk Text (1000 tokens, 200 overlap)
     ↓
Generate Embeddings (Azure OpenAI)
     ↓
Store in ChromaDB (Vector DB)
```

### 2. User Query - RAG-First Workflow
```
User Question
     ↓
RAG Tool (MANDATORY FIRST)
├─ Embed query
├─ Search ChromaDB (top-3 chunks)
├─ Found relevant docs? 
│  ├─ YES → Return RAG answer with citations
│  └─ NO → Proceed to Grounding
└─ Grounding Tool (FALLBACK)
   ├─ Search web via Perplexity
   └─ Return web answer with sources
```

### 3. Conversation Memory

- Each user gets a unique `thread_id`
- All messages stored in LangGraph memory
- Enables multi-turn conversations with context

---

## Core Components

### LangGraph RAG Agent (`src/agents/agent.py`)

- Multi-tool agent using `create_react_agent`
- RAG-first system prompt enforces knowledge base priority
- Conversation memory via `MemorySaver` checkpoint
- Streaming and batch response modes

**Tools:**
1. **RAG Tool** - Retrieves from ChromaDB
2. **Grounding Tool** - Web search via Perplexity

### Document Pipeline (`src/pipelines/document_pipeline.py`)

- Handles: Upload → Duplicate Check → Processing → Embedding → Storage
- Integrates Azure Blob Storage for file management
- Automatic chunking and embedding

### LLM Service (`src/services/llm_service.py`)

**Supports Multiple Providers:**
- Azure OpenAI (Primary)
- Groq / Meta Llama (Fast alternative)
- Google Gemini (Optional)

**Features:**
- Configurable temperature, max_tokens, streaming
- Request timeouts for reliability
- LangSmith tracing integration

### Vector Database (`src/services/vector_database.py`)

- **ChromaDB** for vector embeddings
- Cosine similarity search
- Persistent storage in `data/chromadb/`

### Azure Integration

- **Blob Storage** - Document storage and retrieval
- **Cognitive Services** - STT/TTS for voice features
- **OpenAI** - Embeddings and LLM inference
- **LangSmith** - Tracing and observability

---

## API Reference

### FastAPI Endpoints

**POST `/query`**

Request:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is in the knowledge base?",
    "thread_id": "optional-uuid"
  }'
```

Response:
```json
{
  "success": true,
  "query": "Your question",
  "response": "Answer with citations [1][2]...",
  "thread_id": "uuid-for-conversation",
  "metadata": {}
}
```

**GET `/health`**
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.0.1"
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

## RAG vs Grounding Decision Tree
```
User asks question
    ↓
Call RAG tool (MANDATORY)
    ↓
Does RAG return information?
    ├─ YES (any documents found)
    │  └─ Return RAG answer [DONE]
    │
    └─ NO (no documents found)
       └─ Call Grounding tool
          └─ Return web answer [DONE]
```

**Key Rule:** Once RAG finds ANY information, grounding is NOT called.

---

## Voice Features

### Speech-to-Text (STT)

- Azure Cognitive Services
- Real-time continuous recognition
- Microphone input in Streamlit UI

### Text-to-Speech (TTS)

- Azure Cognitive Services
- Read-aloud for bot responses
- "Read Aloud" button in UI

---

## Logging & Monitoring

### Log Files

- **Location:** `logs/` directory
- **Format:** Timestamped with severity levels
- **Rotation:** Daily rotation with 7-day retention

### LangSmith Integration

- Full request/response tracing
- Token usage tracking
- Performance metrics
- Debug insights

**Access:** https://smith.langchain.com → Project: `rag-agent-project`

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

2. **API Keys**
   - Rotate Azure keys regularly
   - Use Managed Identities when possible
   - Restrict API key scopes

3. **Data Privacy**
   - ChromaDB stored locally in `data/`
   - Logs in `logs/` directory
   - Exclude from version control

4. **Network Security**
   - Use HTTPS in production
   - Implement rate limiting
   - Add authentication middleware

---

## Performance Optimization

### Implemented Features

- Embedding caching (reuse instances)
- Query result caching (1-hour TTL)
- Reduced token limits (500 max)
- Response streaming enabled
- Request timeouts (10 seconds)
- Top-3 chunk retrieval only
- Streamlit component caching

### Expected Metrics

- **Cold start:** 3-5 seconds
- **Cached queries:** <500ms
- **Streaming response:** 200-300ms chunks
- **Token cost:** 50% reduction vs 1000 limit

### Optimization Tips

1. **Adjust chunk size** in `chunking.py` based on your document types
2. **Tune retrieval** - Modify `top_k` parameter in `rag.py`
3. **Enable caching** - Implement Redis for distributed caching
4. **Use lighter models** - Switch to GPT-3.5 for faster responses

---

## Testing

### Run Integration Tests
```bash
python tests/test_complete_rag.py
```

### Test FastAPI Endpoint
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

### Test Streamlit UI
```bash
streamlit run app.py
```

### Run Unit Tests (if available)
```bash
pytest tests/ -v
```

---

## Dependencies

### Core Framework

- **LangChain** - LLM orchestration
- **LangGraph** - Agentic workflows
- **FastAPI** - REST API
- **Streamlit** - Web UI

### AI/ML

- **Azure OpenAI** - LLM and embeddings
- **ChromaDB** - Vector database
- **Groq API** - Alternative LLM
- **Perplexity API** - Web grounding

### Cloud Services

- **Azure Storage Blob** - Document storage
- **Azure Cognitive Services** - STT/TTS
- **LangSmith** - Tracing

### Utilities

- **PyMuPDF / PyPDF** - PDF extraction
- **Python-dotenv** - Environment management
- **Requests** - HTTP client
- **Uvicorn** - ASGI server

See `requirements.txt` for complete list with versions.

---

## FAQ

**Q: Does this work with OpenAI's API directly?**

A: Currently optimized for Azure OpenAI, but you can modify `llm_service.py` to use OpenAI's API directly. Change the client initialization and remove Azure-specific parameters.

**Q: Can I use a different vector database?**

A: Yes! Replace `vector_database.py` with implementations for Pinecone, Weaviate, Qdrant, or Milvus. The interface remains similar.

**Q: How much does it cost to run?**

A: Depends on Azure usage. Expect approximately $0.01-0.05 per query with GPT-4-mini. Costs scale with:
- Number of queries
- Document embeddings
- Storage usage
- Token consumption

**Q: Can I run this completely offline?**

A: No, it requires Azure services for embeddings and LLM inference. However, you could modify it to use local models with Ollama or LM Studio.

**Q: How many documents can it handle?**

A: ChromaDB can scale to millions of vectors. Performance depends on your hardware. For large-scale deployments, consider cloud-hosted vector databases.

**Q: Does it support languages other than English?**

A: Yes! Azure OpenAI models support multiple languages. The RAG system works with any language the underlying LLM supports.

---

## Troubleshooting

### Common Issues

**1. "Azure OpenAI credentials not found"**
```bash
# Verify .env file exists and has AZURE_OPENAI_* variables
cat .env | grep AZURE_OPENAI

# Check if variables are loaded
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('AZURE_OPENAI_API_KEY'))"
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
```

**3. Perplexity API key invalid**
```bash
# Verify PERPLEXITY_API_KEY in .env
grep PERPLEXITY_API_KEY .env

# Test API key
curl https://api.perplexity.ai/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "sonar-pro", "messages": [{"role": "user", "content": "test"}]}'
```

**4. Streamlit port already in use**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process (Unix)
lsof -ti:8501 | xargs kill -9

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

**5. Import errors after installation**
```bash
# Reinstall in editable mode
pip uninstall agentx
pip install -e .

# Check if package is installed
pip show agentx
```

**6. Memory errors with large documents**
```bash
# Reduce chunk size in chunking.py
# Decrease max_tokens in llm_service.py
# Process documents in batches
```

**7. Slow response times**
```bash
# Enable caching
# Reduce top_k in RAG retrieval
# Use GPT-3.5 instead of GPT-4
# Implement query result caching
```

---

## Configuration Reference

### Environment Variables
```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Groq (optional)
GROQ_API_KEY=your-groq-key
GROQ_MODEL=llama3-70b-8192

# Perplexity (web grounding)
PERPLEXITY_API_KEY=your-perplexity-key
PERPLEXITY_MODEL=sonar-pro

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
AZURE_STORAGE_CONTAINER_NAME=documents

# Azure Cognitive Services (STT/TTS)
AZURE_SPEECH_KEY=your-speech-key
AZURE_SPEECH_REGION=eastus

# LangSmith (observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=rag-agent-project

# Application Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=500
TEMPERATURE=0.7
TOP_K_RETRIEVAL=3
```

### Chunking Configuration

Edit `src/components/chunking.py`:
```python
CHUNK_SIZE = 1000        # Tokens per chunk
CHUNK_OVERLAP = 200      # Overlap between chunks
SEPARATORS = ["\n\n", "\n", " ", ""]
```

### Retrieval Configuration

Edit `src/tools/rag.py`:
```python
TOP_K = 3               # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score
```

---

## Deployment

### Development
```bash
# Streamlit
streamlit run app.py

# FastAPI with hot reload
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### Docker
```bash
# Build
docker build -t agentx-api:latest .

# Run
docker run -p 8000:8000 --env-file .env agentx-api:latest

# Docker Compose
docker-compose up -d
```

### Production (Azure)
```bash
# Build and push to Azure Container Registry
az acr build --registry <your-acr> --image agentx:v1 .

# Deploy to Azure Container Apps
az containerapp create \
  --name agentx-api \
  --resource-group <your-rg> \
  --environment <your-env> \
  --image <your-acr>.azurecr.io/agentx:v1 \
  --target-port 8000 \
  --ingress external \
  --env-vars-file .env
```

### Production Checklist

- [ ] `.env` file with all credentials
- [ ] LangSmith project created
- [ ] Azure resources provisioned
- [ ] Docker image built and tested
- [ ] Logs directory created with write permissions
- [ ] Backup strategy for `data/chromadb/`
- [ ] Health checks configured (`/health` endpoint)
- [ ] Resource limits set (CPU, memory)
- [ ] Rate limiting implemented
- [ ] Authentication middleware added
- [ ] HTTPS/SSL certificates configured
- [ ] Monitoring and alerts setup
- [ ] Load balancer configured (if needed)

---

## Additional Resources

- **LangChain Docs:** https://python.langchain.com/
- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Streamlit Docs:** https://docs.streamlit.io/
- **ChromaDB Docs:** https://docs.trychroma.com/
- **Azure OpenAI:** https://azure.microsoft.com/en-us/products/ai-services/openai-service/
- **Perplexity API:** https://docs.perplexity.ai/
- **LangSmith:** https://docs.smith.langchain.com/

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Mahir Baig**
- GitHub: [@Mahir-Baig](https://github.com/Mahir-Baig)
- Project: [AgentX](https://github.com/Mahir-Baig/Agentx)

---

## Contributing

We welcome contributions! Here's how to get started:

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Agentx.git
cd AgentX

# Add upstream remote
git remote add upstream https://github.com/Mahir-Baig/Agentx.git

# Install development dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks (if available)
pre-commit install

# Run tests
pytest tests/ -v
```

### Contribution Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Keep commits atomic and descriptive
- Ensure all tests pass before submitting PR

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines (if available).

---

## Features Roadmap

### Planned Features

- [ ] Advanced chunking strategies (semantic, hierarchical)
- [ ] Query rewriting and expansion
- [ ] Long-context support (100K+ tokens)
- [ ] Multi-modal RAG (images, tables, charts)
- [ ] Custom prompt templates
- [ ] Fine-tuned embeddings
- [ ] Hybrid search (keyword + vector)
- [ ] Admin dashboard
- [ ] Batch processing API
- [ ] Rate limiting and quotas
- [ ] Multi-tenancy support
- [ ] GraphRAG integration
- [ ] Document versioning
- [ ] Collaborative annotations
- [ ] Export to various formats

### Completed Features

- [x] RAG-first workflow
- [x] Multi-modal I/O (text, voice)
- [x] Conversation memory
- [x] FastAPI backend
- [x] Streamlit frontend
- [x] Docker support
- [x] Azure integration
- [x] LangSmith tracing

---

## Acknowledgments

Special thanks to:
- **LangChain** team for the excellent framework
- **Azure** for cloud infrastructure
- **ChromaDB** for vector database
- **Perplexity AI** for web grounding capabilities
- All contributors and users

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

**Last Updated:** December 29, 2025  
**Status:** Production Ready  
**Version:** 0.0.1