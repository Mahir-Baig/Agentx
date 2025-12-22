from langchain_openai import AzureOpenAIEmbeddings
from langsmith import traceable
from src.logger import logger
from src.exceptions import RagException
import sys
from typing import List
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:
    """
    Generate embeddings using Azure OpenAI text-embedding-3-small model.
    """
    
    def __init__(
        self,
        azure_endpoint: str = None,
        api_key: str = None,
        api_version: str = None,
        deployment_name: str = None
    ):
        """
        Initialize the EmbeddingGenerator.
        
        :param azure_endpoint: Azure OpenAI endpoint URL.
        :param api_key: Azure OpenAI API key.
        :param api_version: Azure OpenAI API version.
        :param deployment_name: Name of the embedding model deployment.
        """
        logger.info("Initializing EmbeddingGenerator")
        
        # Get credentials from environment variables if not provided
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        
        if not self.azure_endpoint or not self.api_key:
            raise ValueError("Azure OpenAI endpoint and API key must be provided or set in environment variables")
        
        try:
            # Initialize Azure OpenAI Embeddings
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
                azure_deployment=self.deployment_name
            )
            logger.info(f"EmbeddingGenerator initialized with deployment: {self.deployment_name}")
        except Exception as e:
            logger.error(f"Failed to initialize AzureOpenAIEmbeddings: {e}")
            raise RagException(f"Failed to initialize AzureOpenAIEmbeddings: {e}", sys)
    
    @traceable(name="embed_documents", tags=["embedding", "documents"])
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for a list of LangChain Document objects.
        
        :param documents: List of Document objects to embed.
        :return: List of embedding vectors.
        """
        try:
            if not documents:
                logger.warning("No documents provided for embedding")
                return []
            
            logger.info(f"Generating embeddings for {len(documents)} documents")
            
            # Extract text content from documents
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            if embeddings:
                logger.info(f"Embedding dimension: {len(embeddings[0])}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings for documents: {e}")
            raise RagException(f"Error generating embeddings for documents: {e}", sys)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text strings.
        
        :param texts: List of text strings to embed.
        :return: List of embedding vectors.
        """
        try:
            if not texts:
                logger.warning("No texts provided for embedding")
                return []
            
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            embeddings = self.embeddings.embed_documents(texts)
            
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings for texts: {e}")
            raise RagException(f"Error generating embeddings for texts: {e}", sys)
    
    @traceable(name="embed_query", tags=["embedding", "query"])
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query string.
        
        :param query: Query string to embed.
        :return: Embedding vector.
        """
        try:
            logger.info(f"Generating embedding for query: {query[:100]}...")
            
            embedding = self.embeddings.embed_query(query)
            
            logger.info(f"Successfully generated embedding for query (dimension: {len(embedding)})")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for query: {e}")
            raise RagException(f"Error generating embedding for query: {e}", sys)
    
    @traceable(name="create_document_embeddings", tags=["embedding", "batch_processing"])
    def create_document_embeddings(self, documents: List[Document]) -> List[dict]:
        """
        Create a list of dictionaries containing documents and their embeddings.
        
        :param documents: List of Document objects.
        :return: List of dictionaries with 'document', 'text', and 'embedding' keys.
        """
        try:
            logger.info(f"Creating document embeddings for {len(documents)} documents")
            
            embeddings = self.embed_documents(documents)
            
            result = []
            for doc, embedding in zip(documents, embeddings):
                result.append({
                    'document': doc,
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'embedding': embedding
                })
            
            logger.info(f"Successfully created {len(result)} document-embedding pairs")
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating document embeddings: {e}")
            raise RagException(f"Error creating document embeddings: {e}", sys)
