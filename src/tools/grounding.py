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

**CITATION REQUIREMENTS (MANDATORY):**

1. **Inline Citations**: Use numbered citations [1], [2], [3] immediately after each claim or fact
   - Place citation numbers right after the relevant sentence or claim
   - Multiple sources for one claim: [1][2] or [1, 2]
   - Example: "The Eiffel Tower was completed in 1889 [1]. It stands 324 meters tall [2]."

2. **Source List**: Always end your response with a "Sources:" section containing:
   - Formatted as clickable links with source titles/domains
   - Format: [1] [Title or Domain Name](full_url)
   - Extract meaningful titles from the URL or use the domain name
   - Make links user-friendly and clean

3. **Citation Format Example**:
```
   The Python programming language was created by Guido van Rossum [1]. 
   It was first released in 1991 [2]. Python emphasizes code readability 
   and uses significant indentation [1][3].
   
   Sources:
   [1] [Python.org - About](https://www.python.org/about/)
   [2] [Wikipedia - Python Programming](https://en.wikipedia.org/wiki/Python_(programming_language))
   [3] [Python Documentation](https://docs.python.org/3/tutorial/)
```

4. **Source Link Formatting Rules**:
   - Use markdown link format: [Display Text](URL)
   - Display text should be the website name or article title
   - Keep display text concise (2-5 words max)
   - Examples:
     * [OpenAI Research](https://openai.com/research)
     * [Nature Journal](https://www.nature.com/articles/xyz)
     * [TechCrunch](https://techcrunch.com/article)
     * [GitHub Docs](https://docs.github.com)

5. **Best Practices**:
   - Cite every factual claim, statistic, or quote
   - Use the most authoritative and recent sources
   - Make source links clean and clickable
   - Number citations sequentially [1], [2], [3]...
   - If multiple facts come from the same source, reuse the citation number

6. **Response Structure**:
   [Your answer with inline citations [1][2]]
   
   Sources:
   [1] [Readable Source Name](https://full-url.com)
   [2] [Another Source Name](https://another-url.com)

**CRITICAL**: Every response MUST include:
- Inline citations [1][2] after each claim
- A Sources section with clickable markdown links [Display](URL)
- Clean, readable source names (not raw URLs)

Be factual, concise, and avoid speculation. Always ground your answers in the sources you cite.
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
            response_parts.append("� Sources:")
            for i, citation in enumerate(citations, 1):
                # Extract domain/title and make it a clickable link
                if citation.startswith('http'):
                    # If citation is a URL, extract domain
                    from urllib.parse import urlparse
                    parsed = urlparse(citation)
                    domain = parsed.netloc.replace('www.', '')
                    response_parts.append(f"• [{domain}]({citation})")
                elif ' - ' in citation and 'http' in citation:
                    # Format: "Title - URL"
                    parts = citation.rsplit(' - ', 1)
                    if len(parts) == 2:
                        title, url = parts
                        response_parts.append(f"• [{title}]({url.strip()})")
                    else:
                        response_parts.append(f"• {citation}")
                else:
                    # Already formatted or plain text
                    response_parts.append(f"• {citation}")
        
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
