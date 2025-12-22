from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from src.logger import logger
from src.exceptions import RagException
import sys
import os
from typing import List
from langchain_core.documents import Document


class TextChunker:
    """
    Chunk documents using LangChain RecursiveCharacterTextSplitter.
    Optimized for Azure OpenAI text-embedding-3-small model (1536 dimensions).
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: callable = len,
        separators: List[str] = None
    ):
        """
        Initialize the TextChunker.
        
        :param chunk_size: Maximum size of each chunk (optimized for Ada-3: ~250 tokens = 1000 chars).
        :param chunk_overlap: Number of characters to overlap between chunks.
        :param length_function: Function to measure text length.
        :param separators: List of separators to use for splitting.
        """
        logger.info("="*80)
        logger.info("Initializing TextChunker")
        logger.info("="*80)
        logger.info(f"Chunk size: {chunk_size} characters (~{chunk_size//4} tokens)")
        logger.info(f"Chunk overlap: {chunk_overlap} characters")
        logger.info("Optimized for Azure OpenAI text-embedding-3-small (1536 dimensions)")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators for better document structure preservation
        if separators is None:
            separators = [
                "\n\n",  # Double newline (paragraphs)
                "\n",    # Single newline
                ". ",    # Sentences
                ", ",    # Clauses
                " ",     # Words
                ""       # Characters
            ]
        
        try:
            # Initialize RecursiveCharacterTextSplitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=length_function,
                separators=separators,
                keep_separator=True  # Keep separators to maintain context
            )
            
            logger.info("✓ RecursiveCharacterTextSplitter initialized successfully")
            logger.info(f"✓ Separators: {separators}")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Failed to initialize TextChunker: {e}")
            raise RagException(f"Failed to initialize TextChunker: {e}", sys)
    
    @traceable(name="chunk_documents", tags=["chunking", "text_splitting"])
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of LangChain Document objects into chunks.
        
        :param documents: List of LangChain Document objects.
        :return: List of chunked Document objects.
        """
        try:
            if not documents:
                logger.warning("No documents provided for chunking")
                return []
            
            logger.info("="*80)
            logger.info("Chunking Documents")
            logger.info("="*80)
            logger.info(f"Input: {len(documents)} documents")
            
            # Track statistics
            total_input_chars = 0
            total_chunks = 0
            source_files = set()
            
            # Process all documents
            chunked_docs = []
            
            for i, doc in enumerate(documents):
                content_length = len(doc.page_content)
                total_input_chars += content_length
                
                # Extract source info for logging
                source = doc.metadata.get('source', f'document_{i}')
                source_files.add(source)
                
                logger.info(f"Processing document {i+1}/{len(documents)}: {os.path.basename(source)}")
                logger.info(f"  Content length: {content_length:,} characters")
                
                # Split the document
                doc_chunks = self.text_splitter.split_documents([doc])
                chunked_docs.extend(doc_chunks)
                
                logger.info(f"  Generated chunks: {len(doc_chunks)}")
                total_chunks += len(doc_chunks)
                
                # Log chunk size distribution for this document
                if doc_chunks:
                    chunk_sizes = [len(chunk.page_content) for chunk in doc_chunks]
                    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
                    min_chunk_size = min(chunk_sizes)
                    max_chunk_size = max(chunk_sizes)
                    
                    logger.info(f"  Chunk sizes - Avg: {avg_chunk_size:.0f}, Min: {min_chunk_size}, Max: {max_chunk_size}")
            
            # Overall statistics
            avg_chars_per_chunk = total_input_chars / total_chunks if total_chunks > 0 else 0
            compression_ratio = total_chunks / len(documents) if documents else 0
            
            logger.info("="*80)
            logger.info("CHUNKING SUMMARY")
            logger.info("="*80)
            logger.info(f"Input documents: {len(documents)}")
            logger.info(f"Source files: {len(source_files)}")
            logger.info(f"Total input characters: {total_input_chars:,}")
            logger.info(f"Output chunks: {total_chunks}")
            logger.info(f"Average chars per chunk: {avg_chars_per_chunk:.0f}")
            logger.info(f"Compression ratio: {compression_ratio:.1f}x (chunks per document)")
            logger.info(f"Estimated tokens per chunk: ~{avg_chars_per_chunk/4:.0f}")
            logger.info("✓ Chunking completed successfully")
            logger.info("="*80)
            
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error during document chunking: {e}")
            raise RagException(f"Error during document chunking: {e}", sys)
    
    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Split a single text string into chunks.
        
        :param text: Text string to split.
        :param metadata: Metadata to attach to each chunk.
        :return: List of chunked Document objects.
        """
        try:
            if not text:
                logger.warning("No text provided for chunking")
                return []
            
            logger.info(f"Chunking text ({len(text):,} characters)")
            
            # Create a temporary document
            if metadata is None:
                metadata = {}
            
            temp_doc = Document(page_content=text, metadata=metadata)
            
            # Use the document chunking method
            chunked_docs = self.chunk_documents([temp_doc])
            
            logger.info(f"✓ Text split into {len(chunked_docs)} chunks")
            
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error during text chunking: {e}")
            raise RagException(f"Error during text chunking: {e}", sys)
    
    def get_chunk_stats(self, documents: List[Document]) -> dict:
        """
        Get statistics about document chunks.
        
        :param documents: List of chunked Document objects.
        :return: Dictionary with chunk statistics.
        """
        try:
            if not documents:
                return {"error": "No documents provided"}
            
            chunk_sizes = [len(doc.page_content) for doc in documents]
            total_chunks = len(chunk_sizes)
            total_chars = sum(chunk_sizes)
            
            stats = {
                "total_chunks": total_chunks,
                "total_characters": total_chars,
                "average_chunk_size": total_chars / total_chunks if total_chunks > 0 else 0,
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
                "estimated_total_tokens": total_chars // 4,  # Rough estimate: 1 token ≈ 4 chars
                "estimated_avg_tokens_per_chunk": (total_chars // 4) // total_chunks if total_chunks > 0 else 0,
                "chunk_size_distribution": {
                    "under_500": len([s for s in chunk_sizes if s < 500]),
                    "500_to_1000": len([s for s in chunk_sizes if 500 <= s < 1000]),
                    "1000_to_1500": len([s for s in chunk_sizes if 1000 <= s < 1500]),
                    "over_1500": len([s for s in chunk_sizes if s >= 1500])
                }
            }
            
            logger.info("Chunk Statistics:")
            logger.info(f"  Total chunks: {stats['total_chunks']}")
            logger.info(f"  Average size: {stats['average_chunk_size']:.0f} chars (~{stats['estimated_avg_tokens_per_chunk']} tokens)")
            logger.info(f"  Size range: {stats['min_chunk_size']} - {stats['max_chunk_size']} chars")
            logger.info(f"  Distribution: <500: {stats['chunk_size_distribution']['under_500']}, "
                       f"500-1000: {stats['chunk_size_distribution']['500_to_1000']}, "
                       f"1000-1500: {stats['chunk_size_distribution']['1000_to_1500']}, "
                       f">1500: {stats['chunk_size_distribution']['over_1500']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating chunk statistics: {e}")
            return {"error": str(e)}
    
    def update_chunk_size(self, new_chunk_size: int, new_overlap: int = None):
        """
        Update chunk size and optionally overlap, and reinitialize the splitter.
        
        :param new_chunk_size: New chunk size.
        :param new_overlap: New overlap size (optional).
        """
        try:
            logger.info(f"Updating chunk size from {self.chunk_size} to {new_chunk_size}")
            
            self.chunk_size = new_chunk_size
            if new_overlap is not None:
                self.chunk_overlap = new_overlap
                logger.info(f"Updating overlap from {self.chunk_overlap} to {new_overlap}")
            
            # Reinitialize splitter with new parameters
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", ", ", " ", ""],
                keep_separator=True
            )
            
            logger.info("✓ Chunker updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating chunk parameters: {e}")
            raise RagException(f"Error updating chunk parameters: {e}", sys)
