"""
Vector database service for ChromaDB integration
"""
import logging
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import uuid

from app.core.config import settings
from app.models.document import DocumentChunk

logger = logging.getLogger(__name__)


class VectorDBService:
    """Service for managing vector database operations with ChromaDB"""
    
    def __init__(self):
        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None
        self._embedding_model: Optional[SentenceTransformer] = None
        self._collection_name = "document_chunks"
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client
            self._client = chromadb.PersistentClient(
                path=settings.vector_db_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"description": "Document chunks for AI QA system"}
            )
            
            # Initialize embedding model
            self._embedding_model = SentenceTransformer(settings.embedding_model)
            
            logger.info(f"Vector database initialized with collection: {self._collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    async def close(self) -> None:
        """Close database connections"""
        try:
            if self._client:
                # ChromaDB doesn't require explicit closing
                self._client = None
                self._collection = None
                self._embedding_model = None
                logger.info("Vector database connections closed")
        except Exception as e:
            logger.error(f"Error closing vector database: {e}")
    
    def _ensure_initialized(self) -> None:
        """Ensure the service is initialized"""
        if not self._client or not self._collection or not self._embedding_model:
            raise RuntimeError("VectorDBService not initialized. Call initialize() first.")
    
    async def store_document_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Store document chunks with their embeddings in the vector database
        
        Args:
            chunks: List of DocumentChunk objects to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._ensure_initialized()
        
        if not chunks:
            logger.warning("No chunks provided for storage")
            return True
        
        try:
            # Prepare data for ChromaDB
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            for chunk in chunks:
                # Generate embedding if not provided
                if chunk.embedding is None:
                    embedding = self._embedding_model.encode(chunk.content).tolist()
                else:
                    embedding = chunk.embedding
                
                ids.append(chunk.id)
                documents.append(chunk.content)
                embeddings.append(embedding)
                metadatas.append({
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata
                })
            
            # Store in ChromaDB
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully stored {len(chunks)} chunks in vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store document chunks: {e}")
            return False
    
    async def search_similar_chunks(
        self, 
        query: str, 
        document_ids: Optional[List[str]] = None,
        limit: int = 10,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar document chunks using semantic similarity
        
        Args:
            query: Search query text
            document_ids: Optional list of document IDs to filter by
            limit: Maximum number of results to return
            min_similarity: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing chunk data and similarity scores
        """
        self._ensure_initialized()
        
        try:
            # Generate query embedding
            query_embedding = self._embedding_model.encode(query).tolist()
            
            # Prepare where clause for document filtering
            where_clause = None
            if document_ids:
                where_clause = {"document_id": {"$in": document_ids}}
            
            # Search in ChromaDB
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    distance = results["distances"][0][i]
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    if similarity >= min_similarity:
                        search_results.append({
                            "chunk_id": chunk_id,
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "similarity_score": similarity,
                            "document_id": results["metadatas"][0][i]["document_id"],
                            "chunk_index": results["metadatas"][0][i]["chunk_index"]
                        })
            
            logger.info(f"Found {len(search_results)} similar chunks for query")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            return []
    
    async def delete_document_chunks(self, document_id: str) -> bool:
        """
        Delete all chunks for a specific document
        
        Args:
            document_id: ID of the document whose chunks should be deleted
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._ensure_initialized()
        
        try:
            # Query for chunks of this document
            results = self._collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if results["ids"]:
                # Delete the chunks
                self._collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
            else:
                logger.info(f"No chunks found for document {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete chunks for document {document_id}: {e}")
            return False
    
    async def get_document_chunk_count(self, document_id: str) -> int:
        """
        Get the number of chunks for a specific document
        
        Args:
            document_id: ID of the document
            
        Returns:
            int: Number of chunks for the document
        """
        self._ensure_initialized()
        
        try:
            results = self._collection.get(
                where={"document_id": document_id},
                include=[]  # Only need count
            )
            
            return len(results["ids"]) if results["ids"] else 0
            
        except Exception as e:
            logger.error(f"Failed to get chunk count for document {document_id}: {e}")
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on vector database
        
        Returns:
            Dict containing health status information
        """
        try:
            self._ensure_initialized()
            
            # Get collection info
            collection_count = self._collection.count()
            
            return {
                "status": "healthy",
                "collection_name": self._collection_name,
                "total_chunks": collection_count,
                "embedding_model": settings.embedding_model
            }
            
        except Exception as e:
            logger.error(f"Vector database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def reset_database(self) -> bool:
        """
        Reset the vector database (delete all data)
        WARNING: This will delete all stored embeddings
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self._client and self._collection:
                # Delete the collection
                self._client.delete_collection(self._collection_name)
                
                # Recreate the collection
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"description": "Document chunks for AI QA system"}
                )
                
                logger.warning("Vector database has been reset - all data deleted")
                return True
                
        except Exception as e:
            logger.error(f"Failed to reset vector database: {e}")
            return False


# Global instance
vector_db_service = VectorDBService()