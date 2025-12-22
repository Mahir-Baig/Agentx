"""
RAG Tool for LangGraph Integration
This tool performs Retrieval-Augmented Generation by:
1. Embedding the query
2. Searching ChromaDB for top 3 relevant chunks (cosine similarity)
3. Using Azure LLM to summarize and generate final answer with citations
"""

import os
from typing import Dict, List, Any
from langchain_core.tools import tool
from langsmith import traceable
from src.components.embedding import EmbeddingGenerator
from src.services.vector_database import ChromaDBManager
from src.services.llm_service import LLMService
from src.logger import logger


@tool
@traceable(name="rag_tool", tags=["rag", "retrieval"])
def rag(query: str) -> str:
    """
    Perform Retrieval-Augmented Generation (RAG) query.
    
    This tool retrieves relevant document chunks from the vector database,
    generates embeddings for the query, performs cosine similarity search,
    and uses Azure LLM to summarize the information with citations.
    
    Args:
        query: The question or query to search for
        
    Returns:
        A formatted string containing the answer and citations
    """
    try:
        logger.info(f"RAG Tool: Processing query: '{query}'")
        
        # Initialize components
        embedder = EmbeddingGenerator()
        
        chromadb_path = os.path.join(os.getcwd(), "data", "chromadb")
        db_manager = ChromaDBManager(
            persist_directory=chromadb_path,
            collection_name="documents"
        )
        
        llm_service = LLMService()
        
        # Step 1: Generate query embedding
        logger.info("Step 1: Generating query embedding...")
        query_embedding = embedder.embed_query(query)
        logger.info(f"✓ Query embedded (dimension: {len(query_embedding)})")
        
        # Step 2: Search ChromaDB for top 3 chunks using cosine similarity
        logger.info("Step 2: Searching ChromaDB for top 3 chunks (cosine similarity)...")
        results = db_manager.query(
            query_embedding=query_embedding,
            n_results=3
        )
        logger.info(f"✓ Retrieved {len(results.get('ids', []))} chunks")
        
        # Extract results properly from ChromaDB format
        if not results or not results.get('ids') or not results['ids'][0]:
            logger.warning("No relevant documents found")
            return "I couldn't find any relevant documents to answer your question. Please make sure documents are indexed in the database."
        
        # Step 3: Format context and extract metadata
        logger.info("Step 3: Formatting context...")
        context_parts = []
        citations = []
        seen_sources = set()
        
        # Azure Blob Storage configuration for generating URLs
        storage_account_name = "poc123"  # From connection string
        container_name = "accepted"  # Documents are in accepted container
        
        # ChromaDB returns results in lists
        ids = results['ids'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results.get('distances', [[]])[0]
        
        for i, (doc_id, text, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances), 1):
            # Extract document name from metadata
            source_filename = metadata.get('source_filename', 'Unknown Document')
            source_path = metadata.get('source', 'Unknown Path')
            
            # Format context with document reference
            context_parts.append(f"[Document {i}: {source_filename}]\n{text}\n")
            
            # Add to citations if not already seen
            if source_filename not in seen_sources:
                # Generate Azure Blob Storage URL
                # Format: https://{account}.blob.core.windows.net/{container}/{filename}
                blob_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}/{source_filename}"
                
                citations.append({
                    'filename': source_filename,
                    'path': source_path,
                    'blob_url': blob_url,
                    'similarity_score': round(1 - distance, 3) if distance is not None else 'N/A'
                })
                seen_sources.add(source_filename)
        
        context = "\n".join(context_parts)
        logger.info(f"✓ Context prepared with {len(citations)} unique sources")
        
        # Step 4: Generate answer using Azure LLM
        logger.info("Step 4: Generating answer with Azure LLM...")
        
        system_message = """You are a helpful AI assistant that answers questions based on the provided context.

Instructions:
1. Answer the question using ONLY the information from the provided context
2. Be concise and accurate
3. If the context doesn't contain enough information to answer the question, say so
4. DO NOT add inline citations or source references in your answer (e.g., don't write "(Source: file.pdf)")
5. Just provide the answer - the sources will be automatically added at the end
6. Provide a clear, well-structured answer without mentioning document names"""

        user_message = f"""Context from retrieved documents:

{context}

Question: {query}

Please provide a comprehensive answer based ONLY on the context above. Do not mention document names or add citations in your response."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        llm_response = llm_service.azure_chat_completion(
            messages=messages,
            temperature=0.1,  # Lower temperature for more factual responses
            # max_tokens=500
        )
        
        answer = llm_response.get('content', '')
        logger.info(f"✓ Answer generated ({len(answer)} characters)")
        
        # Format final response with PREFIX to identify it's from RAG
        response_parts = ["📖 **Knowledge Base Answer:**\n", answer]
        
        if citations:
            response_parts.append("\n\n📚 Sources:")
            for i, citation in enumerate(citations, 1):
                filename = citation['filename']
                blob_url = citation['blob_url']
                similarity = citation['similarity_score']
                # Format as markdown link: [Display Text](URL)
                response_parts.append(f"• [{filename}]({blob_url}) (Similarity: {similarity})")
        
        final_response = "\n".join(response_parts)
        logger.info("RAG query completed successfully!")
        logger.info(f"RAG response includes {len(citations)} citations with blob URLs")
        
        return final_response
        
    except Exception as e:
        logger.error(f"RAG tool error: {str(e)}")
        return f"Error: {str(e)}"
