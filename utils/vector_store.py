"""
Vector database management for storing and retrieving document embeddings.
Supports ChromaDB backend with OpenAI embeddings.
"""
import os

from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import logging
import hashlib

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector store implementations."""
    
    @abstractmethod
    def add_documents(self, chunks: List[Dict], embeddings: List[List[float]]) -> List[str]:
        """Add documents with their embeddings to the store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        pass
    
    @abstractmethod
    def list_collections(self) -> List[str]:
        """List all collections."""
        pass

    @abstractmethod
    def delete_by_metadata(self, filter_dict: Dict[str, Any]) -> int:
        """Delete documents matching metadata filter."""
        pass


class ChromaDBStore(VectorStore):
    """ChromaDB implementation of vector store."""
    
    def __init__(self, persist_directory: str = None, collection_name: str = "study_materials"):
        """
        Initialize ChromaDB store.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
        """
        # Use absolute path from project root if not specified
        if persist_directory is None:
            from pathlib import Path
            project_root = Path(__file__).parent.parent
            persist_directory = str(project_root / "data" / "chromadb")
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "MISM course materials"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_documents(self, chunks: List[Dict], embeddings: List[List[float]]) -> List[str]:
        """
        Add documents with their embeddings to ChromaDB.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            embeddings: List of embedding vectors
            
        Returns:
            List of document IDs
        """
        logger.info(f"Starting ChromaDB add operation for {len(chunks)} chunks")
        
        if len(chunks) != len(embeddings):
            error_msg = f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            logger.error(error_msg)
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Prepare data for ChromaDB
        logger.info("Preparing data for ChromaDB...")
        ids = []
        documents = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = chunk.get('metadata', {})
            # Create a stable ID using file path or document id plus chunk index to avoid collisions across ingestions
            unique_base = metadata.get('file_path') or metadata.get('document_id') or f"doc_{i}"
            hash_input = f"{unique_base}_{chunk.get('chunk_index', i)}".encode("utf-8")
            chunk_id = hashlib.md5(hash_input).hexdigest()
            ids.append(chunk_id)
            documents.append(chunk['text'])
            metadatas.append(metadata)
        
        logger.info(f"Adding {len(documents)} documents to ChromaDB collection '{self.collection.name}'...")
        
        try:
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f" Successfully added {len(chunks)} documents to ChromaDB")
            return ids
        except Exception as e:
            logger.error(f" Error adding documents to ChromaDB: {str(e)}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar documents in ChromaDB.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of result dictionaries with text, metadata, and scores
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from ChromaDB.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {str(e)}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        List all collections in ChromaDB.
        
        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        return [col.name for col in collections]
    
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the current collection.
        
        Returns:
            Number of documents
        """
        return self.collection.count()

    def delete_by_metadata(self, filter_dict: Dict[str, Any]) -> int:
        """
        Delete documents matching a metadata filter.
        
        Args:
            filter_dict: Metadata filter for deletion
            
        Returns:
            Number of documents before deletion (Chroma does not return counts)
        """
        try:
            before_count = self.collection.count()
            self.collection.delete(where=filter_dict)
            return before_count
        except Exception as e:
            logger.error(f"Error deleting documents with filter {filter_dict}: {e}")
            return 0


class EmbeddingGenerator:
    """Generate embeddings for text using OpenAI or other models."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        
        return response.data[0].embedding
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("No texts provided for embedding generation")
            return []
        
        logger.info(f"Starting embedding generation for {len(texts)} texts with batch size {batch_size}")
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Import time for rate limiting
        import time
        
        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch = texts[i:i + batch_size]
            
            # Filter out empty texts
            valid_batch = [text for text in batch if text and text.strip()]
            
            if not valid_batch:
                logger.warning(f"Batch {batch_num}/{total_batches}: No valid texts found, skipping")
                continue
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(valid_batch)} texts)...")
            
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    response = self.client.embeddings.create(
                        input=valid_batch,
                        model=self.model,
                        timeout=60  # 60 second timeout per batch
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    logger.info(f" Batch {batch_num}/{total_batches} completed successfully")
                    
                    # Add small delay between batches to avoid rate limiting
                    if batch_num < total_batches:
                        time.sleep(0.5)
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    
                    if "rate_limit" in error_msg.lower() or "429" in error_msg:
                        wait_time = 2 ** retry_count  # Exponential backoff: 2, 4, 8 seconds
                        logger.warning(f"Rate limit hit in batch {batch_num}/{total_batches}, waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                        time.sleep(wait_time)
                    elif retry_count < max_retries:
                        logger.warning(f"Error in batch {batch_num}/{total_batches}, retry {retry_count}/{max_retries}: {error_msg}")
                        time.sleep(1)
                    else:
                        logger.error(f" Failed batch {batch_num}/{total_batches} after {max_retries} retries: {error_msg}")
                        raise
        
        logger.info(f" Embedding generation completed - generated {len(embeddings)} embeddings")
        return embeddings


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""
    
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
    
    def add_documents(self, chunks: List[Dict]) -> List[str]:
        """
        Add documents to the RAG pipeline.
        
        Args:
            chunks: List of text chunks with metadata
            
        Returns:
            List of document IDs
        """
        if not chunks:
            logger.warning("No chunks to add")
            return []
        
        logger.info(f"RAG Pipeline: Starting to process {len(chunks)} chunks")
        
        # Extract text from chunks
        logger.info("RAG Pipeline: Extracting text from chunks...")
        texts = [chunk['text'] for chunk in chunks]
        logger.info(f"RAG Pipeline: Extracted {len(texts)} text segments")
        
        # Generate embeddings
        logger.info(f"RAG Pipeline: Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_generator.generate_embeddings(texts)
        logger.info(f"RAG Pipeline: Generated {len(embeddings)} embeddings")
        
        # Add to vector store
        logger.info(f"RAG Pipeline: Adding {len(chunks)} chunks to vector store...")
        doc_ids = self.vector_store.add_documents(chunks, embeddings)
        
        logger.info(f" RAG Pipeline: Successfully added {len(doc_ids)} documents")
        return doc_ids
    
    def retrieve(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of relevant document chunks
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k, filter_dict=filter_dict)
        
        logger.info(f"Retrieved {len(results)} documents for query")
        return results
    
    def retrieve_context(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> str:
        """
        Retrieve and format context for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k=top_k, filter_dict=filter_dict)
        
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Context {i}]")
            context_parts.append(result['text'])
            context_parts.append("")  # Empty line for separation
        
        return '\n'.join(context_parts)

    def delete_by_metadata(self, filter_dict: Dict[str, Any]) -> int:
        """
        Delete documents in the vector store by metadata.
        
        Args:
            filter_dict: Metadata filter dict
        
        Returns:
            Number of documents before deletion (best effort)
        """
        return self.vector_store.delete_by_metadata(filter_dict)
