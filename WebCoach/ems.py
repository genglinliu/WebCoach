#!/usr/bin/env python3
"""
External Memory Store (EMS) for WebCoach Framework

This module implements a simple vector database to store and retrieve condensed trajectories
based on relevance and recency.

Adapted from the original Coach Framework implementation for simplified use.
"""

import os
import json
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExternalMemoryStore:
    """
    External Memory Store (EMS) using simple vector similarity search.
    
    This simplified version uses JSON storage and numpy for similarity calculations,
    avoiding heavy dependencies like FAISS while maintaining core functionality.
    """
    
    def __init__(self, storage_dir: str = "./coach_storage"):
        """
        Initialize the External Memory Store.
        
        Args:
            storage_dir: Directory to store the data
        """
        self.storage_dir = Path(storage_dir)
        self.data_path = self.storage_dir / 'ems_data.json'
        self.embedding_dim = 256  # 256-dimensional embedding for space efficiency
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load the data
        self._initialize_or_load()
    
    def _initialize_or_load(self):
        """Initialize new data or load existing data."""
        if self.data_path.exists():
            logger.info(f"Loading existing data from {self.data_path}")
            try:
                with open(self.data_path, 'r') as f:
                    self.data = json.load(f)
                logger.info(f"Loaded {len(self.data)} experiences from EMS")
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                self.data = []
        else:
            logger.info("Creating new data store")
            self.data = []
    
    def _save(self):
        """Save the data to disk."""
        try:
            with open(self.data_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.debug(f"Data saved to {self.data_path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def add_experience(self, condensed_data: Dict[str, Any]) -> bool:
        """
        Add a condensed experience to the memory store.
        
        Args:
            condensed_data: Condensed trajectory data from the condenser
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if the condensed data has an embedding
            if 'embedding' not in condensed_data:
                logger.warning("No embedding found in condensed data")
                return False
            
            # Validate embedding
            embedding = condensed_data['embedding']
            if not isinstance(embedding, list) or len(embedding) != self.embedding_dim:
                logger.warning(f"Invalid embedding: expected list of {self.embedding_dim} floats")
                return False
            
            # Add timestamp
            experience = {
                'data': condensed_data,
                'timestamp': time.time(),
                'id': len(self.data)  # Simple ID assignment
            }
            
            self.data.append(experience)
            self._save()
            
            logger.info(f"Added experience {experience['id']} to EMS (total: {len(self.data)})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding experience: {e}")
            return False
    
    def get_similar_experiences(self, query_data: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the k most similar experiences based on the query.
        
        Args:
            query_data: Query data with embedding
            k: Number of similar experiences to retrieve
            
        Returns:
            List of similar experiences with similarity scores
        """
        try:
            if not self.data:
                logger.info("No experiences in EMS yet")
                return []
            
            if 'embedding' not in query_data:
                logger.warning("No embedding in query data")
                return []
            
            query_embedding = np.array(query_data['embedding'])
            
            # Calculate similarities
            similarities = []
            for experience in self.data:
                try:
                    exp_embedding = np.array(experience['data']['embedding'])
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, exp_embedding)
                    
                    similarities.append({
                        'experience': experience,
                        'similarity_score': float(similarity)
                    })
                except Exception as e:
                    logger.debug(f"Error calculating similarity for experience {experience.get('id', 'unknown')}: {e}")
                    continue
            
            # Sort by similarity (descending) and recency
            similarities.sort(key=lambda x: (x['similarity_score'], x['experience']['timestamp']), reverse=True)
            
            # Return top k results
            results = similarities[:k]
            
            logger.info(f"Retrieved {len(results)} similar experiences (requested: {k})")
            for i, result in enumerate(results):
                logger.debug(f"  {i+1}. Similarity: {result['similarity_score']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving similar experiences: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return similarity
        except Exception as e:
            logger.debug(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory store.
        
        Returns:
            Dictionary with statistics
        """
        if not self.data:
            return {
                'total_experiences': 0,
                'successful_experiences': 0,
                'failed_experiences': 0,
                'domains': []
            }
        
        total = len(self.data)
        successful = sum(1 for exp in self.data 
                        if exp['data']['meta'].get('final_success', False))
        failed = total - successful
        
        # Extract domains
        domains = set()
        for exp in self.data:
            domain = exp['data']['meta'].get('domain')
            if domain:
                domains.add(domain)
        
        return {
            'total_experiences': total,
            'successful_experiences': successful,
            'failed_experiences': failed,
            'domains': list(domains)
        }
    
    def clear(self):
        """Clear all data from the memory store."""
        self.data = []
        self._save()
        logger.info("Cleared all data from EMS")
    
    def search_by_domain(self, domain: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for experiences by domain.
        
        Args:
            domain: Domain to search for
            k: Maximum number of results
            
        Returns:
            List of experiences from the specified domain
        """
        try:
            results = []
            for experience in self.data:
                exp_domain = experience['data']['meta'].get('domain', '')
                if domain.lower() in exp_domain.lower():
                    results.append(experience)
                    if len(results) >= k:
                        break
            
            logger.info(f"Found {len(results)} experiences for domain: {domain}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching by domain: {e}")
            return []
    
    def get_recent_experiences(self, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get the k most recent experiences.
        
        Args:
            k: Number of recent experiences to retrieve
            
        Returns:
            List of recent experiences
        """
        try:
            # Sort by timestamp (descending)
            sorted_data = sorted(self.data, key=lambda x: x['timestamp'], reverse=True)
            results = sorted_data[:k]
            
            logger.info(f"Retrieved {len(results)} recent experiences")
            return results
            
        except Exception as e:
            logger.error(f"Error getting recent experiences: {e}")
            return []
