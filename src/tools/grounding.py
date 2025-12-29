"""
Grounding Tool using Perplexity AI
This tool performs web-based grounding when knowledge base doesn't have sufficient information.
Uses Perplexity API for fact retrieval, verification, and augmentation.
"""

import os
from typing import Dict, List, Any, Optional
from langchain_core.tools import tool
from langsmith import traceable
from dotenv import load_dotenv
import requests
from src.logger import logger

load_dotenv()


def _call_perplexity_api(query: str) -> Optional[Dict[str, Any]]:
    """
    Call Perplexity AI API with online context.
    
    Args:
        query: The question to ask Perplexity
        
    Returns:
        Response with answer and citations, or None if failed
    """
    try:
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        perplexity_model = os.getenv("PERPLEXITY_MODEL")
        api_endpoint = "https://api.perplexity.ai/chat/completions"
        
        if not perplexity_api_key:
            logger.warning("⚠ Perplexity API key not configured")
            return None
        
        headers = {
            "Authorization": f"Bearer {perplexity_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": perplexity_model,
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant that provides accurate, well-researched answers using web sources.

**MANDATORY CITATION RULES:**

1. **Inline Citations**: Use numbered citations [1], [2], [3] immediately after each claim
   - Example: "Python was created in 1991 [1]. It emphasizes code readability [2]."

2. **Sources Section**: End response with clickable markdown links
   Format: [1] [Website Name](full_url)
   Examples: [1] [OpenAI](https://openai.com) [2] [Wikipedia](https://en.wikipedia.org/wiki/Topic)

3. **Rules**:
   - Cite every factual claim, statistic, and quote
   - Use authoritative sources
   - Keep display text concise (2-5 words)
   - Reuse citation numbers for the same source
   - NO inline citations without a Sources section

4. **Response Template**:
   [Your answer with inline citations [1][2][3]]
   
   Sources:
   [1] [Source Name](url)
   [2] [Source Name](url)
   [3] [Source Name](url)

Be factual, concise, and ground all answers in cited sources.
"""
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": 0.2,
            "top_k": 3,
            "stream": False
        }
        
        logger.info("Sending request to Perplexity API...")
        response = requests.post(
            api_endpoint,
            headers=headers,
            json=payload,
            # timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"Perplexity API error: {response.status_code} - {response.text}")
            return None
        
        data = response.json()
        
        # Extract content and citations
        message = data.get('choices', [{}])[0].get('message', {})
        content = message.get('content', '')
        citations = message.get('citations', [])
        
        logger.info(f"✓ Perplexity API response received")
        
        return {
            'content': content,
            'citations': citations,
            'model': data.get('model'),
            'usage': data.get('usage', {})
        }
        
    except requests.exceptions.Timeout:
        logger.error("Perplexity API request timed out")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to Perplexity API")
        return None
    except Exception as e:
        logger.error(f"Error calling Perplexity API: {str(e)}")
        return None


@tool
@traceable(name="grounding_tool", tags=["grounding", "perplexity"])
def grounding(query: str) -> str:
    """
    Perform web-based grounding using Perplexity AI.
    
    This tool retrieves current information from the web and provides
    answers with citations when knowledge base information is insufficient.
    
    Args:
        query: The question to ground with web sources
        
    Returns:
        A formatted string containing the answer and web sources (like Perplexity UI)
    """
    try:
        logger.info(f"Grounding Tool: Processing query: '{query}'")
        
        # Call Perplexity API
        response = _call_perplexity_api(query)
        
        if not response:
            return "⚠️ Unable to retrieve information from web sources at this time."
        
        answer = response.get('content', '')
        citations = response.get('citations', [])
        
        logger.info(f"✓ Grounded answer received ({len(answer)} characters)")
        logger.info(f"✓ Found {len(citations)} citations")
        
        # Format response similar to Perplexity UI with citations
        response_parts = []
        response_parts.append("🌐 Web Search Answer:")
        response_parts.append("")
        response_parts.append(answer)
        
        # Add numbered web sources/citations if available
        if citations:
            response_parts.append("")
            response_parts.append("📚 Sources:")
            for i, citation in enumerate(citations, 1):
                # Ensure proper markdown link format: [text](url)
                if citation.startswith('http'):
                    # URL only - extract domain as display text
                    from urllib.parse import urlparse
                    parsed = urlparse(citation)
                    domain = parsed.netloc.replace('www.', '')
                    response_parts.append(f"[{i}] [{domain}]({citation})")
                elif '|' in citation:
                    # Format: "Title|URL" - convert to markdown
                    title, url = citation.split('|', 1)
                    response_parts.append(f"[{i}] [{title.strip()}]({url.strip()})")
                elif ' - ' in citation and 'http' in citation:
                    # Format: "Title - URL"
                    title, url = citation.rsplit(' - ', 1)
                    response_parts.append(f"[{i}] [{title.strip()}]({url.strip()})")
                elif citation.startswith('[') and '](' in citation:
                    # Already in markdown format
                    response_parts.append(f"[{i}] {citation}")
                else:
                    # Plain text - try to make it a link if it contains URL
                    if 'http' in citation:
                        parts = citation.split()
                        url = [p for p in parts if p.startswith('http')]
                        if url:
                            text = citation.replace(url[0], '').strip()
                            response_parts.append(f"[{i}] [{text or 'Source'}]({url[0]})")
                        else:
                            response_parts.append(f"[{i}] {citation}")
                    else:
                        response_parts.append(f"[{i}] {citation}")
        
        final_response = "\n".join(response_parts)
        logger.info("Grounding completed successfully!")
        
        return final_response
        
    except Exception as e:
        logger.error(f"Grounding tool error: {str(e)}")
        return f"Error during grounding: {str(e)}"


def should_ground(
    similarity_scores: List[float],
    confidence_threshold: float = 0.6
) -> bool:
    """
    Determine if grounding is needed based on similarity scores from ChromaDB.
    
    Args:
        similarity_scores: List of similarity scores from vector DB (0-1)
        confidence_threshold: Minimum average similarity to skip grounding
        
    Returns:
        True if grounding should be used, False otherwise
    """
    if not similarity_scores:
        logger.info("No similarity scores provided, grounding recommended")
        return True
    
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    should_ground_result = avg_similarity < confidence_threshold
    
    logger.info(f"Average similarity: {avg_similarity:.3f} (threshold: {confidence_threshold})")
    logger.info(f"Grounding needed: {should_ground_result}")
    
    return should_ground_result
