"""
Agent Service
Service layer for RAG Agent operations - provides a clean interface for FastAPI integration.
"""

import uuid
from typing import Dict, Optional, Any
from langsmith import traceable
from src.agents.agent import create_langgraph_agent
from src.logger import logger


_agent_instance = None


def _get_agent(model: Optional[str] = None):
    """
    Get or create the LangGraph RAG Agent instance (singleton pattern).
    
    Args:
        model: Optional Azure OpenAI deployment name
        
    Returns:
        LangGraphRAGAgent instance
    """
    global _agent_instance
    if _agent_instance is None:
        logger.info("Creating new LangGraph RAG Agent instance...")
        _agent_instance = create_langgraph_agent(model=model)
    return _agent_instance


@traceable(
    name="query_rag_agent", 
    tags=["agent", "rag", "main", "end_to_end"],
    metadata={
        "service": "agent_service",
        "operation": "query",
        "version": "1.0"
    }
)
def query_rag_agent(
    query: str,
    model: Optional[str] = None,
    include_metadata: bool = False,
    thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query the RAG agent and get the final output with conversation memory.
    This function is designed for FastAPI integration.
    
    Args:
        query: The user's question or request
        model: Optional Azure OpenAI deployment name to use
        include_metadata: Whether to include metadata in response (default: False)
        thread_id: Unique identifier for the conversation thread (auto-generated if not provided)
        
    Returns:
        Dictionary containing:
        - success: bool - Whether the query was successful
        - query: str - The original query
        - response: str - The agent's response
        - error: str - Error message (if any)
        - thread_id: str - The conversation thread ID
        - metadata: dict - Additional metadata (if include_metadata=True)
    
    Example:
        >>> result = query_rag_agent("What is the job description?")
        >>> print(result['response'])
        >>> print(result['thread_id'])  # Use this for follow-up queries
    """
    try:
        if thread_id is None:
            thread_id = str(uuid.uuid4())
            logger.info(f"Generated new thread_id: {thread_id}")
        
        logger.info(f"Received query: '{query}' [Thread: {thread_id}]")
        
        if not query or not query.strip():
            return {
                "success": False,
                "query": query,
                "response": "",
                "error": "Query cannot be empty",
                "thread_id": thread_id
            }
        
        agent = _get_agent(model=model)
        
        response = agent.invoke(query, thread_id=thread_id)
        
        result = {
            "success": True,
            "query": query,
            "response": response,
            "error": None,
            "thread_id": thread_id
        }
        
        if include_metadata:
            result["metadata"] = {
                "model": model or "default",
                "agent_type": "langgraph_react",
                "tools_available": ["rag"],
                "memory_enabled": True,
                "thread_id": thread_id
            }
        
        logger.info(f"Query processed successfully [Thread: {thread_id}]")
        return result
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "query": query,
            "response": "",
            "error": error_msg,
            "thread_id": thread_id
        }


async def query_rag_agent_async(
    query: str,
    model: Optional[str] = None,
    include_metadata: bool = False,
    thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Async version of query_rag_agent for FastAPI async endpoints with conversation memory.
    
    Args:
        query: The user's question or request
        model: Optional Azure OpenAI deployment name to use
        include_metadata: Whether to include metadata in response
        thread_id: Unique identifier for the conversation thread (auto-generated if not provided)
        
    Returns:
        Dictionary with success, query, response, error, thread_id, and optional metadata
    """
    return query_rag_agent(query=query, model=model, include_metadata=include_metadata, thread_id=thread_id)


def stream_rag_agent(query: str, model: Optional[str] = None, thread_id: Optional[str] = None):
    """
    Stream the RAG agent's response for real-time interaction with conversation memory.
    Useful for FastAPI streaming endpoints.
    
    Args:
        query: The user's question or request
        model: Optional Azure OpenAI deployment name to use
        thread_id: Unique identifier for the conversation thread (auto-generated if not provided)
        
    Yields:
        Chunks of the agent's response
        
    Example:
        >>> for chunk in stream_rag_agent("What is the job description?"):
        >>>     print(chunk, end='', flush=True)
    """
    try:
        if thread_id is None:
            thread_id = str(uuid.uuid4())
            logger.info(f"Generated new thread_id: {thread_id}")
        
        logger.info(f"Streaming response for query: '{query}' [Thread: {thread_id}]")
        
        if not query or not query.strip():
            yield {"error": "Query cannot be empty"}
            return
        
        agent = _get_agent(model=model)
        
        for chunk in agent.stream(query, thread_id=thread_id):
            yield chunk
            
    except Exception as e:
        error_msg = f"Error streaming response: {str(e)}"
        logger.error(error_msg)
        yield {"error": error_msg}


def get_agent_info() -> Dict[str, Any]:
    """
    Get information about the current agent configuration.
    Useful for health checks and debugging.
    
    Returns:
        Dictionary with agent configuration info
    """
    try:
        agent = _get_agent()
        return {
            "status": "active",
            "agent_type": "LangGraph ReAct Agent",
            "tools": [tool.name for tool in agent.tools],
            "model": "Azure OpenAI",
            "available": True
        }
    except Exception as e:
        logger.error(f"Error getting agent info: {str(e)}")
        return {
            "status": "error",
            "available": False,
            "error": str(e)
        }


if __name__ == "__main__":
    print("=" * 80)
    print("AGENT SERVICE DEMO WITH UUID-BASED CONVERSATION MEMORY")
    print("=" * 80)
    
    
    print(f"\n🔗 Starting NEW conversation (UUID will be auto-generated)")
    print("=" * 80)
    
    
    print("\n📝 Query 1:what is the job description")
    print("-" * 80)
    result1 = query_rag_agent("what is the job description", include_metadata=True)
    thread_id = result1["thread_id"] 
    
    if result1["success"]:
        print(f"✅ Success!")
        print(f"🆔 Thread ID: {thread_id}")
        print(f"🤖 Response:\n{result1['response']}")
        if result1.get("metadata"):
            print(f"📊 Metadata: {result1['metadata']}")
    else:
        print(f"❌ Error: {result1['error']}")
    
    print("\n" + "=" * 80)
    
   