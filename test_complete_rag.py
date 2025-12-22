"""
Complete RAG Pipeline Test
Phase 1: Index documents from folder into ChromaDB
Phase 2: Query the RAG tool and get answer with citations
"""

import os
from src.components.extractor import DocumentExtractor
from src.components.chuncking import TextChunker
from src.components.embedding import EmbeddingGenerator
from src.services.vector_database import ChromaDBManager
from src.tools.rag import rag
from src.logger import logger


def phase_1_index_documents(folder_path: str, reset_db: bool = False):
    """
    Phase 1: Index documents from folder into ChromaDB
    
    Args:
        folder_path: Path to folder containing documents (PDF, TXT)
        reset_db: If True, clear existing database before indexing
    """
    print("\n" + "="*80)
    print("PHASE 1: INDEXING DOCUMENTS")
    print("="*80)
    
    try:
        # Step 1: Extract documents
        print(f"\n📁 Folder: {folder_path}")
        print("\n🔍 Step 1: Extracting documents...")
        extractor = DocumentExtractor()
        documents = extractor.extract_from_folder(folder_path)
        print(f"✅ Extracted {len(documents)} documents")
        
        if not documents:
            print("❌ No documents found in folder!")
            return False
        
        # Step 2: Chunk documents
        print("\n✂️  Step 2: Chunking documents...")
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.chunk_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings
        print("\n🧮 Step 3: Generating embeddings...")
        embedder = EmbeddingGenerator()
        embeddings = embedder.embed_documents(chunks)
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        # Step 4: Store in ChromaDB
        print("\n💾 Step 4: Storing in ChromaDB...")
        chromadb_path = os.path.join(os.getcwd(), "data", "chromadb")
        
        if reset_db:
            print("⚠️  Clearing existing database...")
        
        db_manager = ChromaDBManager(
            persist_directory=chromadb_path,
            collection_name="documents",
            reset_collection=reset_db
        )
        
        added_count = db_manager.add_documents(chunks, embeddings)
        print(f"✅ Stored {added_count} chunks in ChromaDB")
        
        # Show statistics
        total_docs = db_manager.collection.count()
        print(f"\n📊 Total documents in database: {total_docs}")
        
        print("\n" + "="*80)
        print("PHASE 1 COMPLETE ✅")
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in Phase 1: {str(e)}")
        print(f"\n❌ Error in Phase 1: {str(e)}")
        return False


def phase_2_query_rag(query: str):
    """
    Phase 2: Query the RAG tool and get answer with citations
    
    Args:
        query: Question to ask
    """
    print("\n" + "="*80)
    print("PHASE 2: QUERYING RAG TOOL")
    print("="*80)
    
    try:
        print(f"\n❓ Query: {query}")
        print("\n🔄 Processing with RAG tool...")
        print("-" * 80)
        
        # Use the RAG tool
        result = rag.invoke({"query": query})
        
        print("\n" + "="*80)
        print("FINAL OUTPUT")
        print("="*80)
        print(result)
        print("="*80)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in Phase 2: {str(e)}")
        print(f"\n❌ Error in Phase 2: {str(e)}")
        return None


if __name__ == "__main__":
    print("\n" + "="*80)
    print("RAG PIPELINE TEST")
    print("="*80)
    
    # Configuration
    folder_path = os.path.join(os.getcwd(), "blob")
    query = "what is job description"
    reset_db = True  # Set to False to append to existing embeddings
    
    print(f"\n📁 Folder: {folder_path}")
    print(f"❓ Query: {query}")
    print(f"🗑️  Reset DB: {reset_db}")
    
    # Phase 1: Index documents
    print("\n" + "�"*40)
    success = phase_1_index_documents(folder_path, reset_db)
    
    if not success:
        print("\n❌ Phase 1 failed. Exiting.")
    else:
        # Phase 2: Query
        print("\n" + "-"*80)
        input("Press Enter to continue to Phase 2...")
        phase_2_query_rag(query)
        
        print("\n" + "🎉"*40)
        print("TEST FINISHED!")
        print("🎉"*40)