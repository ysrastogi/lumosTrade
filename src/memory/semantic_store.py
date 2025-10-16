"""
Semantic Memory Store for LumosTrade Memory System

This module implements the long-term memory store using vector embeddings
for semantic search across agent outputs and experiences.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

from config.settings import settings
from src.llm.client import gemini
from qdrant_client import QdrantClient
from qdrant_client import models

logger = logging.getLogger(__name__)

class SemanticMemoryStore:
    """
    Long-term memory store using vector embeddings for semantic search.
    
    This component handles the storage and retrieval of compressed memory entries,
    allowing for semantic search across agent outputs and experiences.
    """
    
    def __init__(self, memory_core, vector_db_uri: Optional[str] = None):
        """
        Initialize the Semantic Memory Store.
        
        Args:
            memory_core: Reference to the parent MemoryCore instance
            vector_db_uri: Optional URI for vector database connection
        """
        self.memory_core = memory_core
        self.vector_db_uri = vector_db_uri or settings.qdrant_url
        self.collection_name = "lumos_memory"
        self.embedding_dimension: Optional[int] = None
        self.vector_client: Optional[QdrantClient] = None
        
        # Initialize vector database client
        try:
            if self.vector_db_uri and self.vector_db_uri != "memory":
                # Use provided URI for connection
                self.vector_client = QdrantClient(url=self.vector_db_uri)
                logger.info(f"Connected to Qdrant vector database at {self.vector_db_uri}")
            else:
                # Use in-memory database if no URI provided
                self.vector_client = QdrantClient(":memory:")
                logger.info("Using in-memory Qdrant vector database")
                
            # Ensure collection exists (will be created on first embedding)
            self._ensure_collection()
            
        except Exception as e:
            logger.error(f"Error connecting to vector database: {e}")
        
            
        # Initialize components
        logger.info("Semantic Memory Store initialized with Gemini embeddings and Qdrant")
        
    def _extract_text_content(self, data: Any) -> str:
        """
        Extract meaningful text content from memory data for embedding.
        
        Args:
            data: The memory data to extract text from
            
        Returns:
            str: Extracted text content
        """
        if isinstance(data, dict):
            # Try to extract meaningful text from the data
            if "content" in data:
                return str(data["content"])
            elif "text" in data:
                return str(data["text"])
            elif "message" in data:
                return str(data["message"])
            else:
                # Use the JSON representation of the entire data
                return json.dumps(data)
        else:
            return str(data)
    
    def _ensure_collection(self):
        """
        Ensure the collection exists with correct configuration.
        
        Creates the collection if it doesn't exist and embedding dimension is known.
        If collection exists, validates that it has the correct vector configuration.
        """
        if not self.vector_client or not self.embedding_dimension:
            return
            
        try:
            # Check if collection already exists
            collection_info = self.vector_client.get_collection(self.collection_name)
            
            # Validate existing collection configuration
            vectors_config = collection_info.config.params.vectors
            if hasattr(vectors_config, 'size') and vectors_config.size != self.embedding_dimension:
                logger.warning(
                    f"Collection '{self.collection_name}' exists with different vector size "
                    f"({vectors_config.size}) than expected ({self.embedding_dimension}). "
                    f"Consider recreating the collection or updating the embedding model."
                )
            else:
                logger.info(f"Collection '{self.collection_name}' already exists with correct configuration")
                
        except Exception:
            # Collection doesn't exist, create it
            try:
                self.vector_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dimension,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created vector collection '{self.collection_name}' with dimension {self.embedding_dimension}")
            except Exception as e:
                logger.error(f"Failed to create collection '{self.collection_name}': {e}")
                raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text using Gemini.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List[float]: Embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding, using zero vector")
            fallback_dim = self.embedding_dimension or 768
            return [0.0] * fallback_dim
            
        try:
            # Use the gemini client to generate embeddings
            result = gemini.embed(text)
            embedding = result.values
            
            # Validate embedding result
            if not embedding or not isinstance(embedding, list):
                raise ValueError("Embedding result is empty or invalid")
                
            # Ensure all elements are floats
            embedding = [float(x) for x in embedding]
            
            # Set embedding dimension if not already known
            if self.embedding_dimension is None:
                self.embedding_dimension = len(embedding)
                # Now we know the dimension, ensure collection exists
                self._ensure_collection()
                logger.info(f"Determined embedding dimension: {self.embedding_dimension}")
            
            # Validate embedding dimension consistency
            elif len(embedding) != self.embedding_dimension:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                    f"got {len(embedding)}. Using truncated/padded embedding."
                )
                # Truncate or pad to match expected dimension
                if len(embedding) > self.embedding_dimension:
                    embedding = embedding[:self.embedding_dimension]
                else:
                    embedding.extend([0.0] * (self.embedding_dimension - len(embedding)))
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for text '{text[:50]}...': {e}")
            # Return a fallback embedding if there's an error
            fallback_dim = self.embedding_dimension or 768
            return [0.0] * fallback_dim
    
    async def store(self, agent_id: str, content: Dict[str, Any], 
                memory_type: Optional[str] = None, tags: Optional[List[str]] = None) -> str:
        """
        Store memory content in the semantic store.
        
        Args:
            agent_id: ID of the agent storing the memory
            content: Content to store (will be embedded for semantic search)
            memory_type: Optional type of memory (observation, decision, etc.)
            tags: Optional tags for categorization
            
        Returns:
            str: Memory ID of the stored content
        """
        # Generate memory ID
        memory_id = str(uuid.uuid4())
        
        # Get current session
        session_id = self.memory_core.current_session_id
        
        # Add to semantic store
        await self.add_memory(memory_id, agent_id, content, session_id)
        
        logger.info(f"Memory {memory_id} stored in semantic store")
        return memory_id
        
    async def add_memory(self, memory_id: str, agent_id: str,
                       data: Dict[str, Any], session_id: str) -> bool:
        """
        Add a memory to the semantic store.
        
        Args:
            memory_id: ID of the memory entry
            agent_id: ID of the agent
            data: Memory data
            session_id: Session ID
            
        Returns:
            bool: Success status
        """
        if not self.vector_client:
            logger.warning("Vector client not initialized")
            return False
            
        try:
            # Extract text content for embedding
            content_text = self._extract_text_content(data)
            
            # Generate embedding for the text
            embedding = await self.generate_embedding(content_text)
            
            # Create payload with metadata
            payload = {
                "memory_id": memory_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            # Create point for vector database
            point = models.PointStruct(
                id=memory_id,
                vector=embedding,
                payload=payload
            )
            
            # Insert into vector database
            result = self.vector_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Memory {memory_id} added to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding memory {memory_id} to vector store: {e}")
            return False
    
    async def search(self, query: str, agent_id: Optional[str] = None, 
                   memory_types: Optional[List[str]] = None, tags: Optional[List[str]] = None,
                   limit: int = 10, filter_agents: Optional[List[str]] = None,
                   top_k: int = 5, score_threshold: float = 0.7,
                   qdrant_filter: Optional[models.Filter] = None) -> List[Dict[str, Any]]:
        """
        Search for semantically similar memories using Gemini embeddings.
        
        Args:
            query: Natural language or structured query
            agent_id: Optional agent ID to filter by
            memory_types: Optional list of memory types to filter by
            tags: Optional list of tags to filter by
            limit: Maximum number of results to return
            filter_agents: Optional list of agent IDs to filter by (legacy)
            top_k: Number of results to return (legacy)
            score_threshold: Minimum similarity score (0.0 to 1.0)
            qdrant_filter: Optional Qdrant Filter object for complex filtering
            
        Returns:
            List[Dict]: Semantically similar memories
        """
        if not self.vector_client:
            logger.warning("Vector client not initialized")
            return []
            
        try:
            # Use limit if provided, otherwise fallback to top_k
            result_limit = limit if limit is not None else top_k
            
            # Process filter agents - use filter_agents if agent_id is None
            agents_filter = [agent_id] if agent_id else (filter_agents or [])
            
            # Generate embedding for query
            query_embedding = await self.generate_embedding(query)
            
            # Build filter - use provided qdrant_filter or build from legacy parameters
            query_filter = qdrant_filter
            if query_filter is None:
                # Fallback to legacy filter building
                agents_filter = [agent_id] if agent_id else (filter_agents or [])
                if agents_filter:
                    query_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="agent_id",
                                match=models.MatchAny(any=agents_filter)
                            )
                        ]
                    )
                
            # Search for similar vectors
            search_results = self.vector_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=result_limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
            
            # Process and return results
            results = []
            for point in search_results:
                # Extract the memory data from the payload
                memory_data = {
                    "memory_id": point.payload.get("memory_id"),
                    "agent_id": point.payload.get("agent_id"),
                    "session_id": point.payload.get("session_id"),
                    "timestamp": point.payload.get("timestamp"),
                    "data": point.payload.get("data"),
                    "similarity_score": point.score
                }
                results.append(memory_data)
                
            logger.info(f"Found {len(results)} memories similar to '{query[:30]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching semantic store for query '{query[:30]}...': {e}")
            return []
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Dict or None: Memory data if found, None otherwise
        """
        if not self.vector_client:
            return None
            
        try:
            points = self.vector_client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id]
            )
            
            if points and len(points) > 0:
                point = points[0]
                return {
                    "memory_id": point.payload.get("memory_id"),
                    "agent_id": point.payload.get("agent_id"),
                    "session_id": point.payload.get("session_id"),
                    "timestamp": point.payload.get("timestamp"),
                    "data": point.payload.get("data")
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None
            
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from the store.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            bool: True if successfully deleted, False otherwise
        """
        if not self.vector_client:
            return False
            
        try:
            self.vector_client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[memory_id]
                )
            )
            logger.info(f"Memory {memory_id} deleted from semantic store")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False
            
    async def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts at once.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not texts:
            return []
            
        # Filter out empty texts and keep track of indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
            else:
                logger.warning(f"Empty text at index {i}, will use zero vector")
        
        if not valid_texts:
            # All texts were empty
            fallback_dim = self.embedding_dimension or 768
            return [[0.0] * fallback_dim for _ in texts]
        
        try:
            # Use Gemini batch embedding
            result = gemini.embed(valid_texts)
            embeddings = [emb.values for emb in result]
            
            # Validate batch embedding result
            if not embeddings or not isinstance(embeddings, list):
                raise ValueError("Batch embedding result is empty or invalid")
            
            # Ensure all embeddings are lists of floats
            processed_embeddings = []
            for i, emb in enumerate(embeddings):
                if emb and isinstance(emb, list):
                    processed_embeddings.append([float(x) for x in emb])
                else:
                    logger.warning(f"Invalid embedding at index {i}, using zero vector")
                    processed_embeddings.append([])
            
            # Set embedding dimension if not already known
            if self.embedding_dimension is None and processed_embeddings:
                self.embedding_dimension = len(processed_embeddings[0])
                self._ensure_collection()
                
            # Ensure all embeddings have consistent dimensions
            result_embeddings = []
            for i, emb in enumerate(processed_embeddings):
                if len(emb) != self.embedding_dimension:
                    if len(emb) > self.embedding_dimension:
                        emb = emb[:self.embedding_dimension]
                    else:
                        emb.extend([0.0] * (self.embedding_dimension - len(emb)))
                result_embeddings.append(emb)
            
            # Reconstruct the full result list with zero vectors for empty inputs
            final_result = [[0.0] * self.embedding_dimension for _ in texts]
            for valid_idx, embedding in zip(valid_indices, result_embeddings):
                final_result[valid_idx] = embedding
                
            return final_result
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            # Return fallback embeddings if there's an error
            fallback_dim = self.embedding_dimension or 768
            return [[0.0] * fallback_dim for _ in texts]