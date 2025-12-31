# AgentX - RAG-First Conversational AI Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready **Retrieval-Augmented Generation (RAG)** system with intelligent agent, knowledge base, and web grounding.

**Version:** 0.0.1 | **Author:** Mahir Baig

---

## Overview

AgentX is a conversational AI agent featuring:

- **RAG-First Workflow** - Internal knowledge base before web search
- **Multi-Modal I/O** - Text, speech-to-text, text-to-speech
- **Conversation Memory** - Thread-based context across turns
- **FastAPI + Streamlit** - REST API and interactive web UI
- **Azure Integration** - OpenAI, Blob Storage, Cognitive Services

---

## Architecture

### Standard RAG Pattern: Retrieve → Augment → Generate

```
┌─────────────────────────────────────────────────────────────┐
│                   INTERFACE LAYER                            │
├──────────────────────┬──────────────────────────────────────┤
│   Streamlit UI       │        FastAPI REST API              │
└──────────┬───────────┴──────────────┬───────────────────────┘
           └────────────┬─────────────┘
                        │
┌───────────────────────▼────────────────────────────────────┐
│            ORCHESTRATION LAYER                              │
│         LangGraph Agent + Conversation Memory               │
└───────────────────────┬────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
    RETRIEVE        GENERATE        GROUNDING
    (RAG Tool)      (LLM)          (Fallback)
        │               │               │
┌───────▼───────────────▼───────────────▼────────────────────┐
│                   DATA LAYER                                │
│  ChromaDB Vector Store  │  Azure Blob Storage               │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                  SERVICE LAYER                               │
│  Azure OpenAI (Embeddings + LLM)  │  Perplexity (Web Search)│
└──────────────────────────────────────────────────────────────┘
```

---

## Execution Flows

### 📤 Document Ingestion (Offline - Once per document)

```
Upload PDF/TXT
    ↓
Extract Text → Chunk (1000 tokens) → Embed (Azure OpenAI)
    ↓
Store in ChromaDB + Azure Blob
    ↓
✅ Ready for Search
```

### 💬 Query Processing (Online - Every query)

```
User Question
    ↓
Embed Query → Search ChromaDB (Top-3 similar chunks)
    ↓
    ├─ Found? → Generate answer with LLM + Citations → Done ✅
    └─ Not Found? → Web Search (Perplexity) → Done ✅
```

**Key Rule:** RAG always runs first. Grounding only if RAG finds nothing.

---

## Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/Mahir-Baig/Agentx.git
cd AgentX
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Configuration (.env)

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=your-connection-string
AZURE_STORAGE_CONTAINER_NAME=documents

# Perplexity (Web Grounding)
PERPLEXITY_API_KEY=your-perplexity-key

# LangSmith (Optional - for tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
```

### Run

```bash
# Option 1: Streamlit UI
streamlit run app.py

# Option 2: FastAPI Backend
uvicorn src.api:app --reload

# Docker
docker build -t agentx-api .
docker run -p 8000:8000 --env-file .env agentx-api
```

---

## Project Structure

```
AgentX/
├── app.py                      # Streamlit UI
├── src/
│   ├── api.py                  # FastAPI REST API
│   ├── agents/
│   │   └── agent.py            # LangGraph RAG Agent
│   ├── tools/
│   │   ├── rag.py              # RAG retrieval (runs first)
│   │   └── grounding.py        # Web search (fallback)
│   ├── services/
│   │   ├── llm_service.py      # LLM provider
│   │   ├── vector_database.py  # ChromaDB manager
│   │   └── azure_blob_service.py
│   ├── components/
│   │   ├── embedding.py        # Text embeddings
│   │   ├── chunking.py         # Text chunking
│   │   └── extractor.py        # PDF/TXT extraction
│   └── pipelines/
│       └── document_pipeline.py # Document processing
├── data/chromadb/              # Vector database
└── logs/                       # Application logs
```

---

## API Reference

### POST `/query`

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is AgentX?",
    "thread_id": "optional-uuid"
  }'
```

**Response:**
```json
{
  "success": true,
  "response": "Based on your documents, AgentX is...",
  "thread_id": "uuid",
  "sources": [{"document": "doc.pdf", "page": 5}]
}
```

### POST `/upload`

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

### GET `/health`

```bash
curl http://localhost:8000/health
```

---

## Usage Example

```python
import requests

# Query with conversation memory
thread_id = "user-123"
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What are the key features?",
        "thread_id": thread_id
    }
)

print(response.json()["response"])
```

---

## How RAG Works

1. **Upload Document** → Extract text → Chunk → Embed → Store in ChromaDB
2. **User Asks Question** → Embed query → Search similar chunks
3. **Found Docs?**
   - **Yes** → LLM generates answer using docs as context
   - **No** → Search web via Perplexity API
4. **Return Answer** with citations/sources

**Example:**
```
Query: "What is AgentX?"
→ Search ChromaDB (finds 3 chunks, similarity > 0.7)
→ LLM: "Based on your documents, AgentX is a RAG system..."
→ Citation: [doc.pdf, page 1]
```

---

## Key Features

- **RAG-First**: Always checks internal knowledge before web
- **Sequential Execution**: RAG → Grounding (never parallel)
- **Conversation Memory**: Maintains context via thread IDs
- **Semantic Search**: Vector embeddings (not just keywords)
- **Multi-Modal**: Text input + voice (STT/TTS)
- **Production Ready**: FastAPI, Docker, logging, tracing

---

## System Requirements

- Python 3.10+
- 4GB RAM (8GB recommended)
- Azure Account (OpenAI, Storage, Cognitive Services)
- Internet connection

---

## Configuration

### Chunking (`src/components/chunking.py`)
```python
CHUNK_SIZE = 1000          # Tokens per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks
```

### Retrieval (`src/tools/rag.py`)
```python
TOP_K = 3                  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7 # Minimum similarity (0-1)
```

### LLM (`src/services/llm_service.py`)
```python
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500
```

---

## Troubleshooting

**Azure credentials not found:**
```bash
cat .env | grep AZURE_OPENAI
```

**ChromaDB error:**
```bash
mkdir -p data/chromadb
rm -rf data/chromadb/  # If corrupted
```

**Port already in use:**
```bash
streamlit run app.py --server.port 8502
```

**Import errors:**
```bash
pip uninstall agentx -y
pip install -e .
```

---

## Testing

```bash
# Integration tests
python tests/test_complete_rag.py

# API test
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'

# UI test
streamlit run app.py
```

---

## Deployment

### Docker
```bash
docker build -t agentx-api .
docker run -p 8000:8000 --env-file .env agentx-api
```

### Azure Container Apps
```bash
az acr build --registry <your-acr> --image agentx:v1 .
az containerapp create \
  --name agentx-api \
  --resource-group <your-rg> \
  --image <your-acr>.azurecr.io/agentx:v1
```

---

## Dependencies

- **LangChain** - LLM orchestration
- **LangGraph** - Agent workflows
- **FastAPI** - REST API
- **Streamlit** - Web UI
- **ChromaDB** - Vector database
- **Azure OpenAI** - Embeddings + LLM
- **Perplexity API** - Web search

See `requirements.txt` for full list.

---

## FAQ

**Q: Can I use OpenAI API directly?**  
A: Yes, modify `llm_service.py` to use OpenAI instead of Azure OpenAI.

**Q: How much does it cost?**  
A: ~$17-30 per 1000 queries (GPT-4-mini + embeddings + storage).

**Q: Can I run offline?**  
A: No, requires Azure services. Use Ollama + local embeddings for offline.

**Q: How many documents?**  
A: ChromaDB scales to millions. Local: 10K-100K, Cloud: 100K-1M+.

---

## Resources

- **LangChain:** https://python.langchain.com/
- **ChromaDB:** https://docs.trychroma.com/
- **Azure OpenAI:** https://azure.microsoft.com/en-us/products/ai-services/openai-service/
- **GitHub Issues:** https://github.com/Mahir-Baig/Agentx/issues

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Author

**Mahir Baig**  
GitHub: [@Mahir-Baig](https://github.com/Mahir-Baig)  
Email: mahirbaig2@gmail.com  
Project: [AgentX](https://github.com/Mahir-Baig/Agentx)

---

**Last Updated:** December 30, 2025  
**Version:** 0.0.1

---

**Built with ❤️ by Mahir Baig**