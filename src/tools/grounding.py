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
import re
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
                    "content": """You are a helpful AI assistant. Provide accurate answers with proper citations.

**CRITICAL - YOU MUST FOLLOW THIS EXACTLY:**

1. **Answer Format**: Write your answer normally with citations [1][2][3] after facts

2. **Sources Section**: End EVERY response with this exact format:
   
   Sources:
   [1] [Full Source Title](https://exact-full-url.com/path)
   [2] [Another Source](https://another-url.com)
   [3] [Third Source](https://third-url.com)

**RULES:**
- ALWAYS include actual FULL URLs in sources, not just domain names
- Use markdown format: [DisplayName](fullurl)
- NEVER use plain text source names - they MUST be clickable links
- Include the complete URL including protocol (https://)
- If you don't have exact URL, provide the best matching source URL
- Separate numbered citations with newlines
- Do NOT include "Sources:" label with citations data - keep them as clean markdown links

**EXAMPLE:**
   The Earth orbits the Sun [1]. Python is a programming language [2].
   
   Sources:
   [1] [NASA Earth Information](https://www.nasa.gov/earth/)
   [2] [Python Official Documentation](https://www.python.org/doc/)

Remember: Clickable markdown links are MANDATORY. Use format [text](url)."""
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
        logger.info(f"Raw citations: {citations}")
        
        # Format response with proper markdown links
        response_parts = []
        response_parts.append("🌐 **Web Search Answer:**")
        response_parts.append("")
        response_parts.append(answer)
        
        # Extract markdown links from answer (Perplexity should format them)
        # Pattern: [Text](url)
        markdown_links = re.findall(r'\[([^\[\]]+)\]\(https?://[^\)]+\)', answer)
        logger.info(f"Found {len(markdown_links)} markdown links in answer")
        
        # Extract and format citations with fallback
        formatted_sources = []
        
        # PREFERRED: Extract markdown links directly from answer
        sources_match = re.search(
            r'(?:Sources?:|📚.*?Sources?:)\s*((?:\[[^\]]+\]\([^\)]+\)[\r\n]*)+)',
            answer,
            re.IGNORECASE | re.DOTALL
        )
        
        if sources_match:
            # Markdown sources already in answer - extract them
            sources_section = sources_match.group(1)
            source_links = re.findall(r'\[([^\[\]]+)\]\((https?://[^\)]+)\)', sources_section)
            
            for i, (title, url) in enumerate(source_links, 1):
                formatted_sources.append(f"[{i}] [{title}]({url})")
                logger.info(f"Extracted markdown source {i}: [{title}]({url})")
            
            # Remove the sources section from answer to avoid duplication
            answer = re.sub(
                r'\n*(?:Sources?:|📚.*?Sources?:)\s*(?:\[[^\]]+\]\([^\)]+\)[\r\n]*)+',
                '',
                answer,
                flags=re.IGNORECASE | re.DOTALL
            ).strip()
        
        # FALLBACK: Parse citations array if markdown extraction failed
        if not formatted_sources and citations:
            logger.info("No markdown links found in answer, parsing citations array as fallback")
            for i, citation in enumerate(citations, 1):
                source_link = None
                
                # Try multiple citation formats
                if isinstance(citation, dict):
                    # If citation is a dict with url/title
                    url = citation.get('url') or citation.get('href') or citation.get('link')
                    title = citation.get('title') or citation.get('name')
                    if url:
                        source_link = f"[{i}] [{title or 'Source'}]({url})"
                
                elif isinstance(citation, str):
                    if citation.startswith('http'):
                        # Pure URL
                        from urllib.parse import urlparse
                        parsed = urlparse(citation)
                        domain = parsed.netloc.replace('www.', '')
                        source_link = f"[{i}] [{domain}]({citation})"
                    
                    elif '|' in citation:
                        # Title|URL format
                        parts = citation.split('|', 1)
                        title = parts[0].strip()
                        url = parts[1].strip()
                        if url.startswith('http'):
                            source_link = f"[{i}] [{title}]({url})"
                    
                    elif ' - ' in citation:
                        # Title - URL format
                        parts = citation.rsplit(' - ', 1)
                        if len(parts) == 2 and parts[1].strip().startswith('http'):
                            title = parts[0].strip()
                            url = parts[1].strip()
                            source_link = f"[{i}] [{title}]({url})"
                    
                    # Fallback: check if citation contains URL anywhere
                    if not source_link and 'http' in citation:
                        urls = re.findall(r'https?://[^\s\]]+', citation)
                        if urls:
                            url = urls[0]
                            title = citation.replace(url, '').strip()
                            source_link = f"[{i}] [{title or 'Source'}]({url})"
                
                if source_link:
                    formatted_sources.append(source_link)
                    logger.info(f"Formatted citation {i}: {source_link}")
                else:
                    # Last resort: use plain text with warning
                    logger.warning(f"Citation {i} has no URL: {citation}")
                    formatted_sources.append(f"[{i}] {citation}")
        
        # Add sources section if we have them
        if formatted_sources:
            response_parts.append("")
            response_parts.append("---")
            response_parts.append("**📚 Sources:**")
            response_parts.extend(formatted_sources)
        
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
