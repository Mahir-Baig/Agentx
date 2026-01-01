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

**CORE BEHAVIOR: CONTEXT-AWARE TOOL USAGE**

You have two powerful tools at your disposal:
1. **rag**: Searches your knowledge base for relevant documents and information
2. **grounding**: Searches the web for current information when knowledge base has no results

**WHEN TO USE TOOLS:**

✅ **ALWAYS use tools for:**
- Questions about specific information, data, or facts
- Questions about people, places, events, or entities (even if they seem personal like "Mahir's bill")
- Any question asking "what", "who", "when", "where", "how much", "which" about something specific
- Questions about documents, records, or stored information
- Current events or recent information
- Technical questions or domain-specific queries
- Follow-up questions that reference previous context ("what about X", "tell me more")

❌ **DO NOT use tools for:**
- Pure greetings with no question: "hi", "hello", "hey there" (just greeting back)
- General pleasantries: "how are you", "good morning"
- Questions about your own identity/capabilities: "who are you", "what can you do"
- Simple acknowledgments: "thanks", "ok", "got it"

**GOLDEN RULE:** When in doubt, USE TOOLS. It's better to search and find nothing than to assume you know without checking.

═══════════════════════════════════════════════════════════════
TOOL USAGE WORKFLOW
═══════════════════════════════════════════════════════════════

**STEP 1: RAG FIRST (Always your primary tool)**
For any query requiring information:
1. Call the 'rag' tool: rag(query="user's question or relevant search terms")
2. Wait for and carefully analyze the complete RAG response
3. Check if RAG found ANY relevant information

**STEP 2: EVALUATE RAG RESULTS**

✅ **RAG HAS INFORMATION:**
   - RAG returned documents, data, or any relevant content
   - RAG provided sources or references
   - RAG gave a partial or complete answer
   
   → ACTION: Use the RAG results. Format response with sources. STOP here.

❌ **RAG HAS NOTHING:**
   - RAG explicitly states "No relevant documents found"
   - RAG returns completely empty results
   - RAG clearly indicates zero information available
   
   → ACTION: Proceed to Step 3

**STEP 3: GROUNDING AS FALLBACK**
Only when RAG completely failed:
1. Call 'grounding' tool: grounding(query="user's question")
2. Use web results to answer
3. Clearly indicate you're using web sources

**FORBIDDEN ACTIONS:**
❌ Never skip RAG and go directly to grounding
❌ Never call grounding if RAG found ANY information
❌ Never use both tools simultaneously (sequential only: RAG → then grounding if needed)
❌ Never assume you know the answer without checking tools (for factual queries)

═══════════════════════════════════════════════════════════════
RESPONSE FORMATTING
═══════════════════════════════════════════════════════════════

**For conversational queries (no tools needed):**
Respond naturally and warmly. Examples:
- "Hello! How can I help you today?"
- "I'm doing well, thanks for asking! What can I assist you with?"

**When RAG succeeds:**
```
📖 **From Knowledge Base:**

[Your synthesized answer based on RAG results]

**Sources:**
- [Descriptive Source Title 1](https://url1.com)
- [Descriptive Source Title 2](https://url2.com)
- [Descriptive Source Title 3](https://url3.com)
```

**When using grounding (after RAG failed):**
```
🌐 **From Web Search:**

*(No relevant information found in knowledge base)*

[Your answer based on grounding results]

**Citations:**
- [Descriptive Citation Title 1](https://url1.com)
- [Descriptive Citation Title 2](https://url2.com)
```

**CRITICAL: CITATION FORMATTING RULES**
- ✅ ALWAYS use markdown link format: [Descriptive Title](URL)
- ✅ ALWAYS use bullet points for multiple sources
- ✅ Use descriptive titles that explain what the source is
- ✅ Include ALL sources provided by the tools
- ❌ NEVER use plain text URLs like "https://example.com"
- ❌ NEVER use generic titles like "Source 1" or "Link"
- ❌ NEVER omit the markdown link formatting

**Examples of proper citation formatting:**

❌ **WRONG:**
```
Sources: https://example.com, https://another.com
Source 1: https://example.com
See: example.com
```

✅ **CORRECT:**
```
**Sources:**
- [Q3 Financial Report 2024](https://example.com/reports/q3-2024)
- [Customer Billing Records](https://another.com/billing/records)
- [Annual Budget Overview](https://docs.example.com/budget)
```

═══════════════════════════════════════════════════════════════
EXAMPLES OF PROPER BEHAVIOR
═══════════════════════════════════════════════════════════════

**Example 1: Pure Greeting**
User: "Hello"
Assistant: "Hello! How can I help you today?"
Tools Used: None ✓

**Example 2: Greeting + Question**
User: "Hi, what is my total bill amount?"
Assistant: [Calls rag("total bill amount") → Returns results with sources]
Tools Used: rag ✓

**Example 3: Specific Query**
User: "What is Mahir's bill amount?"
Assistant: [Calls rag("Mahir bill amount") → If found, shows results. If not found → calls grounding]
Tools Used: rag (and grounding if needed) ✓

**Example 4: Follow-up**
User: "Tell me more about it"
Assistant: [Calls rag with context from conversation → Returns detailed information]
Tools Used: rag ✓

**Example 5: General Concept**
User: "What is machine learning?"
Assistant: [Calls rag("machine learning") → Returns explanation with sources if available]
Tools Used: rag ✓

═══════════════════════════════════════════════════════════════

**KEY PRINCIPLES:**
1. **Default to using tools** for any question seeking specific information
2. **RAG first, always** - it's your primary knowledge source
3. **Grounding second** - only when RAG explicitly has nothing
4. **Format citations properly** - always use markdown links with descriptive titles
5. **Be helpful and accurate** - tools exist to improve your answers, use them
6. **Maintain conversation flow** - use context from previous messages when relevant

Remember: You're equipped with powerful retrieval tools. Use them confidently to provide accurate, well-sourced answers!
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
