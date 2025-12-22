"""
Simple FastAPI for RAG Agent
Single POST endpoint to query the agent with conversation memory
"""

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from typing import Optional
from src.services.agent_service import query_rag_agent
from src.logger import logger

app = FastAPI(title="RAG Agent API", version="1.0.0")


@app.post("/query")
async def query(
    query: str = Body(..., embed=True, description="Your question for the RAG agent"),
    thread_id: Optional[str] = Body(None, embed=True, description="Optional conversation thread ID (UUID auto-generated if not provided)")
):
    """
    Query the RAG agent with conversation memory.
    
    Send a query and get an answer with citations from your documents.
    Optionally provide a thread_id to continue a previous conversation.
    
    Example request:
    ```json
    {
        "query": "what is the job description?",
        "thread_id": "optional-uuid-here"
    }
    ```
    """
    try:
        # Validate query
        if not query or not query.strip():
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Query cannot be empty"}
            )
        
        # Call the agent service
        result = query_rag_agent(query=query, thread_id=thread_id, include_metadata=False)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG Agent API"}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG Agent API on http://localhost:8000")
    print("📝 POST /query - Query the agent")
    print("❤️  GET /health - Health check")
    uvicorn.run(app, host="127.0.0.1", port=8000)
