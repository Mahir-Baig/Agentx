import chromadb
from chromadb.config import Settings
from langsmith import traceable
from src.logger import logger
from src.exceptions import RagException
import sys
import os
from typing import List, Dict, Optional, Any
import hashlib
from datetime import datetime


class ChromaDBManager:
    """
    Manage ChromaDB vector database for RAG Agent.
    Supports storing embeddings with metadata and automatic cleanup when files are deleted.
    """
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        reset_collection: bool = False
    ):
        """
        Initialize ChromaDB Manager.
        
        :param collection_name: Name of the collection to use.
        :param persist_directory: Directory to persist the database.
        :param reset_collection: If True, delete and recreate the collection.
        """
        logger.info("="*80)
        logger.info("Initializing ChromaDB Manager")
        logger.info("="*80)
        logger.info(f"Collection name: {collection_name}")
        logger.info(f"Persist directory: {persist_directory}")
        logger.info(f"Reset collection: {reset_collection}")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)
            logger.info(f"✓ Persist directory ready: {persist_directory}")
            
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("✓ ChromaDB client initialized")
            
            # Get or create collection
            if reset_collection:
                try:
                    self.client.delete_collection(name=collection_name)
                    logger.info(f"✓ Deleted existing collection: {collection_name}")
                except Exception:
                    pass
            
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "RAG Agent document embeddings with file tracking"}
            )
            
            # Get collection stats
            count = self.collection.count()
            logger.info(f"✓ Collection ready: {collection_name} (current size: {count} documents)")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RagException(f"Failed to initialize ChromaDB: {e}", sys)
    
    def _generate_doc_id(self, text: str, source: str, chunk_index: int) -> str:
        """
        Generate a unique ID for a document chunk.
        
        :param text: Document text content.
        :param source: Source file path.
        :param chunk_index: Index of the chunk in the document.
        :return: Unique document ID.
        """
        # Create unique ID based on source file and chunk index
        unique_string = f"{source}_{chunk_index}_{text[:100]}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def _extract_metadata(self, document: Any) -> Dict[str, Any]:
        """
        Extract and enrich metadata from a LangChain document.
        
        :param document: LangChain Document object.
        :return: Enriched metadata dictionary.
        """
        metadata = document.metadata.copy() if hasattr(document, 'metadata') else {}
        
        # Add timestamp
        metadata['indexed_at'] = datetime.now().isoformat()
        
        # Add text statistics
        metadata['text_length'] = len(document.page_content)
        
        # Ensure source is present
        if 'source' not in metadata:
            metadata['source'] = 'unknown'
        
        # Normalize source path
        if 'source' in metadata:
            metadata['source'] = os.path.normpath(metadata['source'])
            metadata['source_filename'] = os.path.basename(metadata['source'])
            metadata['source_extension'] = os.path.splitext(metadata['source'])[1]
        
        # Convert all values to strings for ChromaDB compatibility
        for key, value in metadata.items():
            if not isinstance(value, (str, int, float, bool)):
                metadata[key] = str(value)
        
        return metadata
    
    @traceable(name="add_documents_to_chromadb", tags=["chromadb", "storage", "batch"])
    def add_documents(
        self,
        documents: List[Any],
        embeddings: List[List[float]],
        batch_size: int = 100
    ) -> int:
        """
        Add documents with embeddings to ChromaDB.
        
        :param documents: List of LangChain Document objects.
        :param embeddings: List of embedding vectors.
        :param batch_size: Number of documents to add per batch.
        :return: Number of documents added.
        """
        try:
            if not documents or not embeddings:
                logger.warning("No documents or embeddings provided")
                return 0
            
            if len(documents) != len(embeddings):
                raise ValueError(f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings")
            
            logger.info("="*80)
            logger.info(f"Adding {len(documents)} documents to ChromaDB")
            logger.info("="*80)
            
            total_added = 0
            source_files = set()
            
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                ids = []
                texts = []
                metadatas = []
                batch_embeddings_list = []
                
                for j, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):
                    # Extract metadata
                    metadata = self._extract_metadata(doc)
                    source_files.add(metadata.get('source', 'unknown'))
                    
                    # Generate unique ID
                    doc_id = self._generate_doc_id(
                        doc.page_content,
                        metadata.get('source', 'unknown'),
                        i + j
                    )
                    
                    ids.append(doc_id)
                    texts.append(doc.page_content)
                    metadatas.append(metadata)
                    batch_embeddings_list.append(embedding)
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=batch_embeddings_list,
                    documents=texts,
                    metadatas=metadatas
                )
                
                total_added += len(batch_docs)
                logger.info(f"✓ Added batch {i//batch_size + 1}: {len(batch_docs)} documents (Total: {total_added}/{len(documents)})")
            
            logger.info("="*80)
            logger.info(f"✓ Successfully added {total_added} documents from {len(source_files)} source files")
            logger.info(f"✓ Collection size now: {self.collection.count()} documents")
            logger.info(f"✓ Source files indexed: {list(source_files)[:5]}..." if len(source_files) > 5 else f"✓ Source files indexed: {list(source_files)}")
            logger.info("="*80)
            
            return total_added
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise RagException(f"Error adding documents to ChromaDB: {e}", sys)
    
    @traceable(name="vector_db_query", tags=["chromadb", "retrieval"])
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Query the vector database with an embedding.
        
        :param query_embedding: Query embedding vector.
        :param n_results: Number of results to return.
        :param where: Metadata filter (e.g., {"source": "file.pdf"}).
        :param where_document: Document content filter.
        :return: Query results with documents, distances, and metadata.
        """
        try:
            logger.info(f"Querying ChromaDB for top {n_results} results")
            if where:
                logger.info(f"Metadata filter: {where}")
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            logger.info(f"✓ Query returned {len(results['ids'][0])} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            raise RagException(f"Error querying ChromaDB: {e}", sys)
    
    def delete_by_source(self, source_path: str) -> int:
        """
        Delete all embeddings associated with a specific source file.
        This is automatically called when a file is deleted.
        
        :param source_path: Path to the source file.
        :return: Number of documents deleted.
        """
        try:
            # Normalize path
            source_path = os.path.normpath(source_path)
            
            logger.info("="*80)
            logger.info(f"Deleting embeddings for source: {source_path}")
            logger.info("="*80)
            
            # Get all documents with this source
            results = self.collection.get(
                where={"source": source_path}
            )
            
            if not results['ids']:
                logger.info(f"⚠ No embeddings found for source: {source_path}")
                logger.info("="*80)
                return 0
            
            # Delete the documents
            self.collection.delete(
                ids=results['ids']
            )
            
            deleted_count = len(results['ids'])
            logger.info(f"✓ Deleted {deleted_count} embeddings for source: {source_path}")
            logger.info(f"✓ Collection size now: {self.collection.count()} documents")
            logger.info("="*80)
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            raise RagException(f"Error deleting documents by source: {e}", sys)
    
    def delete_by_sources(self, source_paths: List[str]) -> int:
        """
        Delete embeddings for multiple source files.
        
        :param source_paths: List of source file paths.
        :return: Total number of documents deleted.
        """
        logger.info(f"Deleting embeddings for {len(source_paths)} source files")
        total_deleted = 0
        
        for source_path in source_paths:
            deleted = self.delete_by_source(source_path)
            total_deleted += deleted
        
        logger.info(f"✓ Total deleted: {total_deleted} embeddings from {len(source_paths)} files")
        return total_deleted
    
    def sync_with_filesystem(self, active_folder: str) -> Dict[str, int]:
        """
        Sync the database with the filesystem.
        Deletes embeddings for files that no longer exist.
        
        :param active_folder: Folder containing active files.
        :return: Dictionary with sync statistics.
        """
        try:
            logger.info("="*80)
            logger.info("Syncing ChromaDB with filesystem")
            logger.info("="*80)
            logger.info(f"Active folder: {active_folder}")
            
            # Get all unique sources in the database
            all_docs = self.collection.get()
            db_sources = set()
            
            if all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    source = metadata.get('source')
                    if source:
                        db_sources.add(source)
            
            logger.info(f"Found {len(db_sources)} unique source files in database")
            
            # Check which files still exist
            missing_files = []
            for source in db_sources:
                if not os.path.exists(source):
                    missing_files.append(source)
                    logger.info(f"⚠ File no longer exists: {source}")
            
            # Delete embeddings for missing files
            deleted_count = 0
            if missing_files:
                logger.info(f"Deleting embeddings for {len(missing_files)} missing files")
                deleted_count = self.delete_by_sources(missing_files)
            else:
                logger.info("✓ All source files still exist")
            
            stats = {
                'total_sources_in_db': len(db_sources),
                'missing_files': len(missing_files),
                'embeddings_deleted': deleted_count,
                'remaining_documents': self.collection.count()
            }
            
            logger.info("="*80)
            logger.info("SYNC SUMMARY")
            logger.info("="*80)
            logger.info(f"Total sources in DB: {stats['total_sources_in_db']}")
            logger.info(f"Missing files: {stats['missing_files']}")
            logger.info(f"Embeddings deleted: {stats['embeddings_deleted']}")
            logger.info(f"Remaining documents: {stats['remaining_documents']}")
            logger.info("="*80)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error syncing with filesystem: {e}")
            raise RagException(f"Error syncing with filesystem: {e}", sys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        :return: Dictionary with collection statistics.
        """
        try:
            count = self.collection.count()
            
            # Get all documents to analyze
            all_docs = self.collection.get()
            
            sources = set()
            file_types = {}
            total_text_length = 0
            
            if all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    source = metadata.get('source', 'unknown')
                    sources.add(source)
                    
                    ext = metadata.get('source_extension', 'unknown')
                    file_types[ext] = file_types.get(ext, 0) + 1
                    
                    text_length = metadata.get('text_length', 0)
                    if isinstance(text_length, (int, float)):
                        total_text_length += text_length
            
            stats = {
                'total_documents': count,
                'unique_sources': len(sources),
                'file_types': file_types,
                'avg_text_length': total_text_length / count if count > 0 else 0,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
            
            logger.info("="*80)
            logger.info("COLLECTION STATISTICS")
            logger.info("="*80)
            logger.info(f"Total documents: {stats['total_documents']}")
            logger.info(f"Unique sources: {stats['unique_sources']}")
            logger.info(f"File types: {stats['file_types']}")
            logger.info(f"Average text length: {stats['avg_text_length']:.2f} characters")
            logger.info("="*80)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise RagException(f"Error getting stats: {e}", sys)
    
    def get_sources(self) -> List[str]:
        """
        Get list of all source files in the collection.
        
        :return: List of source file paths.
        """
        try:
            all_docs = self.collection.get()
            sources = set()
            
            if all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    source = metadata.get('source')
                    if source:
                        sources.add(source)
            
            sources_list = sorted(list(sources))
            logger.info(f"Found {len(sources_list)} unique source files in collection")
            
            return sources_list
            
        except Exception as e:
            logger.error(f"Error getting sources: {e}")
            raise RagException(f"Error getting sources: {e}", sys)
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        :return: True if successful.
        """
        try:
            logger.info("="*80)
            logger.info(f"Clearing collection: {self.collection_name}")
            logger.info("="*80)
            
            count_before = self.collection.count()
            
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG Agent document embeddings with file tracking"}
            )
            
            count_after = self.collection.count()
            
            logger.info(f"✓ Collection cleared")
            logger.info(f"✓ Deleted: {count_before} documents")
            logger.info(f"✓ Current size: {count_after} documents")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise RagException(f"Error clearing collection: {e}", sys)
