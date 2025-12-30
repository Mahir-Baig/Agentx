"""
LangGraph RAG Agent
A conversational agent that uses LangGraph's prebuilt create_react_agent
to interact with the RAG tool for document retrieval and question answering.
"""

import uuid
from typing import Optional
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
from langsmith import traceable
from langgraph.checkpoint.memory import MemorySaver
from src.tools.rag import rag
from src.tools.grounding import grounding
from src.logger import logger

# Load environment variables
load_dotenv()

# LangSmith Configuration for tracing and monitoring
# Reads from .env if available, otherwise uses defaults below
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-agent-project")

class LangGraphRAGAgent:
    """
    LangGraph-based RAG Agent that uses create_react_agent for tool execution.
    
    This agent can:
    - Answer questions using the RAG tool to retrieve relevant documents
    - Maintain conversation context
    - Provide citations and sources for answers
    - Handle complex queries that require document retrieval
    """
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize the LangGraph RAG Agent.
        
        Args:
            model: Optional Azure OpenAI deployment name (defaults to config value)
        """
        logger.info("Initializing LangGraph RAG Agent...")
        
        # Initialize Azure OpenAI LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            deployment_name=model or os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # Fixed: Use AZURE_OPENAI_DEPLOYMENT
            temperature=0.1,
            # max_tokens=2000
        )
        
        # Define the system prompt - MUST encourage tool usage
        self.system_prompt = """You are an intelligent AI assistant with access to a RAG (Retrieval-Augmented Generation) system and web grounding capabilities.

**CRITICAL: MANDATORY RAG-FIRST WORKFLOW**

You MUST follow this exact sequence for EVERY user question. NO EXCEPTIONS.

═══════════════════════════════════════════════════════════════
STEP 1: ALWAYS USE RAG FIRST (MANDATORY)
═══════════════════════════════════════════════════════════════
For EVERY user question, you MUST:
1. Call the 'rag' tool first: rag(query="user's question")
2. Wait for the complete RAG response
3. Read and analyze what RAG returned

DO NOT proceed to Step 2 until you have received and read the RAG response.

═══════════════════════════════════════════════════════════════
STEP 2: EVALUATE RAG RESPONSE
═══════════════════════════════════════════════════════════════
After receiving RAG results, check:

✅ RAG SUCCESS - Use RAG answer if:
   - RAG returned any documents or information
   - RAG provided sources or references
   - RAG gave a partial answer
   - RAG found even somewhat relevant content
   
   → ACTION: STOP HERE. Return the RAG answer to user. DO NOT call grounding.

❌ RAG FAILURE - Only consider grounding if:
   - RAG explicitly says "No relevant documents found"
   - RAG returns completely empty results
   - RAG response indicates zero information available
   
   → ACTION: Only now may you proceed to Step 3.

═══════════════════════════════════════════════════════════════
STEP 3: USE GROUNDING ONLY AS FALLBACK
═══════════════════════════════════════════════════════════════
ONLY call grounding tool if RAG completely failed (Step 2 ❌):
1. Call: grounding(query="user's question")
2. Return web-based answer with citations
3. Label response as "🌐 Web Sources"

═══════════════════════════════════════════════════════════════

**TOOL PRIORITY HIERARCHY:**
1. 🥇 RAG Tool - Your PRIMARY and DEFAULT tool (use FIRST, ALWAYS)
2. 🥈 Grounding Tool - Your FALLBACK tool (use ONLY when RAG fails)

**FORBIDDEN ACTIONS:**
❌ NEVER call grounding before calling RAG
❌ NEVER skip RAG and go directly to grounding
❌ NEVER call grounding if RAG provided any answer at all
❌ NEVER use both tools when RAG succeeded

**RESPONSE FORMATTING:**

When RAG succeeds:
"📖 Knowledge Base Answer:
[RAG answer here with sources]"

When only grounding used:
"🌐 Web Search Answer:
(Note: No relevant information found in knowledge base)
[Grounding answer here with citations]"

**DECISION TREE:**
User asks question
    ↓
Call RAG (MANDATORY FIRST STEP)
    ↓
Does RAG have ANY information?
    ↓
YES → Return RAG answer, DONE ✓
NO → Call grounding, return web answer

Remember: Your default action for ANY question is to call RAG first. Grounding is only for when RAG has zero results.
"""
        # Create the tools list
        self.tools = [rag, grounding]
        
        # Initialize memory saver for conversation persistence
        self.memory = MemorySaver()
        logger.info("Memory saver initialized for conversation persistence")
        
        # Create the agent using langchain.agents.create_agent with memory
        # Latest API: create_agent(model, tools, system_prompt, checkpointer)
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer=self.memory  # Add memory for conversation history
        )
        
        logger.info("LangGraph RAG Agent initialized successfully with memory")
        logger.info(f"Tools available: {[tool.name for tool in self.tools]}")
    
    @traceable(name="langgraph_agent_invoke", tags=["agent", "invoke", "rag"])
    def invoke(self, query: str, thread_id: Optional[str] = None) -> str:
        """
        Process a user query using the LangGraph agent with conversation memory.
        
        Args:
            query: The user's question or request
            thread_id: Unique identifier for the conversation thread (auto-generated if not provided)
            
        Returns:
            The agent's response as a string
        """
        try:
            # Generate UUID if thread_id not provided
            if thread_id is None:
                thread_id = str(uuid.uuid4())
                logger.info(f"Generated new thread_id: {thread_id}")
            
            logger.info(f"Processing query: '{query}' [Thread: {thread_id}]")
            
            # Invoke the agent with the query using the new API format with thread_id
            inputs = {"messages": [{"role": "user", "content": query}]}
            config = {"configurable": {"thread_id": thread_id}}
            
            result = self.agent.invoke(inputs, config=config)
            
            # Extract the final response from messages
            messages = result.get("messages", [])
            if messages:
                # Get the last message (AI response)
                final_message = messages[-1]
                if isinstance(final_message, dict):
                    response = final_message.get("content", str(final_message))
                else:
                    response = final_message.content if hasattr(final_message, 'content') else str(final_message)
                logger.info(f"Query processed successfully [Thread: {thread_id}]")
                return response
            else:
                logger.warning("No response generated")
                return "I apologize, but I couldn't generate a response. Please try again."
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"
    
    @traceable(name="langgraph_agent_stream", tags=["agent", "stream", "rag"])
    def stream(self, query: str, thread_id: Optional[str] = None):
        """
        Stream the agent's response for real-time interaction with conversation memory.
        
        Args:
            query: The user's question or request
            thread_id: Unique identifier for the conversation thread (auto-generated if not provided)
            
        Yields:
            Chunks of the agent's response as they are generated
        """
        try:
            # Generate UUID if thread_id not provided
            if thread_id is None:
                thread_id = str(uuid.uuid4())
                logger.info(f"Generated new thread_id: {thread_id}")
            
            logger.info(f"Streaming response for query: '{query}' [Thread: {thread_id}]")
            
            # Stream the agent's response with the new API format and thread_id
            inputs = {"messages": [{"role": "user", "content": query}]}
            config = {"configurable": {"thread_id": thread_id}}
            
            for chunk in self.agent.stream(inputs, config=config, stream_mode="updates"):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}")
            yield {"error": str(e)}
    
    def get_conversation_history(self, thread_id: str = "default"):
        """
        Get the conversation history for a specific thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
            
        Returns:
            List of messages in the conversation
        """
        try:
            # Get the state from memory
            config = {"configurable": {"thread_id": thread_id}}
            state = self.agent.get_state(config)
            return state.values.get("messages", [])
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
    def clear_conversation_history(self, thread_id: str = "default"):
        """
        Clear the conversation history for a specific thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
        """
        try:
            logger.info(f"Clearing conversation history [Thread: {thread_id}]")
            # This will be implementation specific based on checkpointer
            # For MemorySaver, we can't directly clear, but new thread_id will start fresh
            logger.info(f"To start a new conversation, use a different thread_id")
        except Exception as e:
            logger.error(f"Error clearing conversation history: {str(e)}")


def create_langgraph_agent(model: Optional[str] = None) -> LangGraphRAGAgent:
    """
    Factory function to create a LangGraph RAG Agent.
    
    Args:
        model: Optional Azure OpenAI deployment name
        
    Returns:
        An initialized LangGraphRAGAgent instance
    """
    return LangGraphRAGAgent(model=model)


# Example usage
if __name__ == "__main__":
    # Create the agent
    agent = create_langgraph_agent()
    
    # Test queries
    test_queries = [
        # "what is job description",
        # "What are the main responsibilities?",
        "who won the 2025 ICC women's world cup?",
    ]
    
    print("=" * 80)
    print("LANGGRAPH RAG AGENT DEMO")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 80)
        response = agent.invoke(query)
        print(f"🤖 Response:\n{response}")
        print("=" * 80)
