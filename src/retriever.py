"""
FAISS RAG Retriever
====================

Retrieves relevant documents from FAISS index for RAG.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import faiss
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from src.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL
)


class RAGRetriever:
    """Retrieves documents from FAISS index for RAG"""
    
    def __init__(self, top_k=TOP_K_RETRIEVAL):
        """
        Initialize FAISS retriever
        
        Args:
            top_k: Number of documents to retrieve
        """
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.top_k = top_k
        self._embedding_cache = {}  # Cache for query embeddings
        
        # Load FAISS index
        print("Loading FAISS index...")
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                f"Run build_rag_index.py first."
            )
        
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        print(f"  ✓ Loaded FAISS index ({self.index.ntotal} vectors)")
        
        # Load metadata
        print("Loading metadata...")
        if not FAISS_METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata not found at {FAISS_METADATA_PATH}"
            )
        
        with open(FAISS_METADATA_PATH, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"  ✓ Loaded {len(self.metadata)} metadata entries\n")
    
    def create_query_embedding(self, query):
        """
        Create embedding for query text with caching
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector (numpy array)
        """
        # Check cache first
        if query in self._embedding_cache:
            return self._embedding_cache[query]
        
        # Generate embedding
        embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Cache it (limit cache size to prevent memory issues)
        if len(self._embedding_cache) < 1000:
            self._embedding_cache[query] = embedding
        
        return embedding
    
    def retrieve(self, query, top_k=None):
        """
        Retrieve top-k most relevant documents
        
        Args:
            query: Query string
            top_k: Override default top_k
            
        Returns:
            List of dicts with 'id', 'score', 'metadata'
        """
        k = top_k or self.top_k
        
        # Create query embedding
        query_embedding = self.create_query_embedding(query)
        
        # Convert to numpy array and normalize
        query_vector = np.array([query_embedding], dtype='float32')
        faiss.normalize_L2(query_vector)
        
        # Search FAISS index
        scores, indices = self.index.search(query_vector, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):  # Valid index
                results.append({
                    'id': self.metadata[idx]['id'],
                    'score': float(scores[0][i]),
                    'metadata': self.metadata[idx]['metadata']
                })
        
        return results
    
    def format_context(self, documents):
        """
        Format retrieved documents as context string
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information found in knowledge base."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc['metadata']
            
            context_part = f"""[Context {i}]
Question: {metadata['instruction']}
Answer: {metadata['response']}"""
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)


def test_retriever():
    """Test the retriever"""
    print("="*80)
    print("TESTING FAISS RETRIEVER")
    print("="*80 + "\n")
    
    retriever = RAGRetriever()
    
    test_queries = [
        "How do I track my order?",
        "What payment methods do you accept?",
        "How can I cancel my subscription?"
    ]
    
    for query in test_queries:
        print(f"\n{'─'*80}")
        print(f"Query: {query}")
        print('─'*80)
        
        results = retriever.retrieve(query)
        
        print(f"\nRetrieved {len(results)} documents:")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Score: {doc['score']:.4f}")
            print(f"   Intent: {doc['metadata']['intent']}")
            print(f"   Q: {doc['metadata']['instruction'][:100]}...")
            print(f"   A: {doc['metadata']['response'][:100]}...")
        
        print(f"\n{'─'*80}")
        print("Formatted Context:")
        print('─'*80)
        context = retriever.format_context(results)
        print(context[:500] + "..." if len(context) > 500 else context)


if __name__ == "__main__":
    test_retriever()
