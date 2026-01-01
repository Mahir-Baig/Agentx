from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langsmith import traceable
from src.logger import logger
from src.exceptions import RagException
import sys
import os
from typing import List
from langchain_core.documents import Document


class DocumentExtractor:
    """
    Extract text from PDF and TXT files using LangChain loaders.
    """
    
    def __init__(self):
        """Initialize the DocumentExtractor."""
        logger.info("Initializing DocumentExtractor")
        self.supported_extensions = ['.pdf', '.txt']
    
    @traceable(name="extract_from_file", tags=["extraction", "document"])
    def extract_from_file(self, file_path: str) -> List[Document]:
        """
        Extract text from a single file (PDF or TXT).
        
        :param file_path: Path to the file to extract text from.
        :return: List of LangChain Document objects.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {file_extension}. Supported types: {self.supported_extensions}")
            
            logger.info(f"Extracting text from {file_path}")
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                logger.info(f"Successfully extracted {len(documents)} pages from PDF: {file_path}")
            
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                logger.info(f"Successfully extracted text from TXT: {file_path}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise RagException(f"Error extracting text from {file_path}: {e}", sys)
    
    @traceable(name="extract_from_folder", tags=["extraction", "batch"])
    def extract_from_folder(self, folder_path: str) -> List[Document]:
        """
        Extract text from all PDF and TXT files in a folder.
        
        :param folder_path: Path to the folder containing files.
        :return: List of LangChain Document objects from all files.
        """
        try:
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            
            logger.info(f"Extracting text from all files in folder: {folder_path}")
            
            all_documents = []
            file_count = 0
            
            # Walk through all subdirectories
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_extension = os.path.splitext(file)[1].lower()
                    
                    if file_extension in self.supported_extensions:
                        try:
                            documents = self.extract_from_file(file_path)
                            all_documents.extend(documents)
                            file_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to extract from {file_path}: {e}")
                            continue
            
            logger.info(f"Successfully extracted text from {file_count} files. Total documents: {len(all_documents)}")
            return all_documents
            
        except Exception as e:
            logger.error(f"Error extracting text from folder {folder_path}: {e}")
            raise RagException(f"Error extracting text from folder {folder_path}: {e}", sys)