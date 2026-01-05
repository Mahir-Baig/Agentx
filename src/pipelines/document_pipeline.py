"""
Document Processing Pipeline
Handles the complete workflow: Upload → Duplicate Check → RAG Processing
"""

import os
import tempfile
from typing import Tuple, Optional
from src.services.azure_blob_service import AzureBlobManager
from src.components.extractor import DocumentExtractor
from src.components.chuncking import TextChunker
from src.components.embedding import EmbeddingGenerator
from src.services.vector_database import ChromaDBManager
from src.logger import logger
from langsmith import traceable


class DocumentPipeline:
    """
    Complete document processing pipeline with Azure Blob Storage integration
    """
    
    def __init__(self):
        """Initialize the pipeline components"""
        self.blob_manager = AzureBlobManager()
        self.extractor = DocumentExtractor()
        self.chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        self.embedder = EmbeddingGenerator()
        
        # ChromaDB configuration
        chromadb_path = os.path.join(os.getcwd(), "data", "chromadb")
        self.db_manager = ChromaDBManager(
            persist_directory=chromadb_path,
            collection_name="documents",
            reset_collection=False
        )
        
        logger.info("Document pipeline initialized")
    
    @traceable(name="check_file_exists_in_accepted", tags=["pipeline", "blob", "duplicate_check"])
    def check_file_exists_in_accepted(self, filename: str) -> bool:
        """
        Check if a file exists in the 'accepted' container
        
        Args:
            filename: Name of the file to check
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            logger.info(f"Checking if '{filename}' exists in accepted container")
            _, blob_files = self.blob_manager.list_blob_names_and_files("accepted")
            
            # Check if filename exists (without any _duplicate suffix)
            base_filename = filename.replace("_duplicate", "").strip()
            exists = any(base_filename in blob_file for blob_file in blob_files)
            
            logger.info(f"{'File exists' if exists else 'File not found'} in accepted container")
            return exists
            
        except Exception as e:
            logger.error(f"Error checking file existence: {str(e)}")
            return False
    
    @traceable(name="upload_to_blob", tags=["pipeline", "blob", "upload"])
    def upload_to_blob(self, file_path: str, container_name: str, blob_name: str):
        """
        Upload a file to Azure Blob Storage
        
        Args:
            file_path: Local path to the file
            container_name: Target container (rawdata, accepted, rejected)
            blob_name: Name for the blob
        """
        try:
            logger.info(f"Uploading '{blob_name}' to '{container_name}'")
            
            container_client = self.blob_manager.storage_account_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"Successfully uploaded to {container_name}/{blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise
    
    @traceable(name="move_blob_between_containers", tags=["pipeline", "blob", "move"])
    def move_blob_between_containers(self, source_container: str, dest_container: str, 
                                     blob_name: str, new_blob_name: Optional[str] = None):
        """
        Move/copy a blob from one container to another
        
        Args:
            source_container: Source container name
            dest_container: Destination container name
            blob_name: Name of the blob to move
            new_blob_name: Optional new name for the blob (for duplicates)
        """
        try:
            logger.info(f"Moving '{blob_name}' from '{source_container}' to '{dest_container}'")
            
            # Get clients
            source_client = self.blob_manager.storage_account_client.get_container_client(source_container)
            dest_client = self.blob_manager.storage_account_client.get_container_client(dest_container)
            
            # Get source blob
            source_blob = source_client.get_blob_client(blob_name)
            
            # Determine destination name
            dest_blob_name = new_blob_name if new_blob_name else blob_name
            dest_blob = dest_client.get_blob_client(dest_blob_name)
            
            # Copy to destination
            dest_blob.start_copy_from_url(source_blob.url)
            
            # Delete from source
            source_blob.delete_blob()
            
            logger.info(f"Successfully moved to {dest_container}/{dest_blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving blob: {str(e)}")
            raise
    
    @traceable(name="process_single_file", tags=["pipeline", "rag", "end_to_end"])
    def process_single_file(self, file_path: str) -> Tuple[bool, str, int]:
        """
        Process a single file through the RAG pipeline
        (Extraction → Chunking → Embedding → Store in ChromaDB)
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Tuple of (success, message, chunks_added)
        """
        try:
            logger.info(f"Starting RAG processing for: {file_path}")
            
            # Step 1: Extract
            logger.info("Step 1/4: Extracting text...")
            documents = self.extractor.extract_from_file(file_path)
            if not documents:
                return False, "No content extracted from file", 0
            logger.info(f"Extracted {len(documents)} document(s)")
            
            # Step 2: Chunk
            logger.info("Step 2/4: Chunking text...")
            chunks = self.chunker.chunk_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 3: Embed
            logger.info("Step 3/4: Generating embeddings...")
            embeddings = self.embedder.embed_documents(chunks)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Step 4: Store
            logger.info("Step 4/4: Storing in ChromaDB...")
            added_count = self.db_manager.add_documents(chunks, embeddings)
            logger.info(f"Stored {added_count} chunks in vector database")
            
            total_docs = self.db_manager.collection.count()
            message = f"Successfully processed! Added {added_count} chunks. Total in DB: {total_docs}"
            
            return True, message, added_count
            
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, 0
    
    @traceable(name="handle_uploaded_file", tags=["pipeline", "workflow", "main"])
    def handle_uploaded_file(self, uploaded_file, file_content: bytes) -> Tuple[bool, str, str]:
        """
        Complete workflow for handling an uploaded file:
        1. Upload to rawdata
        2. Check if exists in accepted
        3. If duplicate → move to rejected with _duplicate suffix
        4. If new → move to accepted and process through RAG pipeline
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            file_content: Bytes content of the file
            
        Returns:
            Tuple of (success, message, final_container)
        """
        filename = uploaded_file.name
        file_ext = os.path.splitext(filename)[1].lower()
        
        logger.info(f"📥 Processing uploaded file: {filename}")
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            # Step 1: Upload to rawdata
            logger.info("📤 Step 1: Uploading to rawdata container...")
            self.upload_to_blob(tmp_path, "rawdata", filename)
            
            # Step 2: Check if file exists in accepted
            logger.info("🔍 Step 2: Checking for duplicates in accepted container...")
            is_duplicate = self.check_file_exists_in_accepted(filename)
            
            if is_duplicate:
                # Step 3a: Handle duplicate - move to rejected
                logger.info("⚠️  Duplicate detected!")
                
                # Add _duplicate suffix before extension
                name_without_ext = os.path.splitext(filename)[0]
                duplicate_filename = f"{name_without_ext}_duplicate{file_ext}"
                
                logger.info(f"🗑️  Moving to rejected as: {duplicate_filename}")
                self.move_blob_between_containers(
                    "rawdata", 
                    "rejected", 
                    filename, 
                    duplicate_filename
                )
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                return False, f"❌ Duplicate file detected! Moved to rejected container as '{duplicate_filename}'", "rejected"
            
            else:
                # Step 3b: Handle new file - move to accepted
                logger.info("New file detected!")
                logger.info("Moving to accepted container...")
                self.move_blob_between_containers("rawdata", "accepted", filename)
                
                # Step 4: Process through RAG pipeline
                logger.info("🤖 Starting RAG processing...")
                success, message, chunks_added = self.process_single_file(tmp_path)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                if success:
                    return True, f"✅ {message}", "accepted"
                else:
                    return False, f"⚠️  File moved to accepted but RAG processing failed: {message}", "accepted"
        
        except Exception as e:
            error_msg = f"❌ Error in pipeline: {str(e)}"
            logger.error(error_msg)
            
            # Clean up temp file if exists
            try:
                if 'tmp_path' in locals():
                    os.unlink(tmp_path)
            except:
                pass
            
            return False, error_msg, "error"