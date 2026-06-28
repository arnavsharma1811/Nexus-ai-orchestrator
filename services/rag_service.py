import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import time
import logging
from pathlib import Path
import requests
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, chroma_path: str = None, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG service with ChromaDB and embedder.
        """
        if chroma_path is None:
            root_dir = Path(__file__).parent.parent
            chroma_path = str(root_dir / "chroma_db")
        
        logger.info(f"Initializing RAG service with ChromaDB at: {chroma_path}")
        
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection("my_notes")
        self.embedder = SentenceTransformer(model_name)
        self.embedding_dim = 384
        
        # Ollama endpoint
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "llama3.2:3b"  # Changed to Llama 3.2
        self.ollama_available = self._check_ollama()
        
        logger.info(f"RAG service ready. Collection count: {self.collection.count()}")
        logger.info(f"Ollama available: {self.ollama_available}")
        logger.info(f"Ollama model: {self.ollama_model}")
    
    def _check_ollama(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def query(self, query: str, top_k: int = 5, hybrid: bool = True) -> Dict:
        """
        Query the RAG pipeline.
        """
        start_time = time.time()
        
        try:
            # Generate embedding
            embedding = self.embedder.encode(query).tolist()
            
            # Vector search
            vector_results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k * 2 if hybrid else top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Extract results with safe defaults
            documents = vector_results.get("documents", [[]])[0] or []
            metadatas = vector_results.get("metadatas", [[]])[0] or []
            distances = vector_results.get("distances", [[]])[0] or []
            
            # Ensure metadata is a list of dicts (not None)
            metadata_list = []
            for m in metadatas[:top_k]:
                if m is None:
                    metadata_list.append({"source": "unknown"})
                elif isinstance(m, dict):
                    metadata_list.append(m)
                else:
                    metadata_list.append({"source": str(m)})
            
            # Generate answer
            if self.ollama_available:
                answer = self._generate_answer_ollama(query, documents[:top_k])
            else:
                answer = self._generate_answer_fallback(query, documents[:top_k])
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "answer": answer,
                "sources": documents[:top_k] if documents else [],
                "metadata": metadata_list,
                "latency_ms": round(latency_ms, 2),
                "total_documents": len(documents),
                "ollama_used": self.ollama_available
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {str(e)}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "metadata": [],
                "latency_ms": 0,
                "error": str(e)
            }
    
    def _generate_answer_ollama(self, query: str, documents: List[str]) -> str:
        """Generate answer using Ollama."""
        if not documents:
            return "No relevant documents found."
        
        # Build context from retrieved documents
        context = "\n\n".join([f"Source {i+1}: {doc[:500]}" for i, doc in enumerate(documents[:3])])
        
        # Build prompt
        prompt = f"""You are a helpful AI assistant answering questions based on the provided context.

Context:
{context}

Question: {query}

Answer based only on the context above. Be concise and accurate:"""
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "max_tokens": 256
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response from Ollama.").strip()
            else:
                logger.warning(f"Ollama returned {response.status_code}, using fallback")
                return self._generate_answer_fallback(query, documents)
                
        except Exception as e:
            logger.warning(f"Ollama error: {e}, using fallback")
            return self._generate_answer_fallback(query, documents)
    
    def _generate_answer_fallback(self, query: str, documents: List[str]) -> str:
        """Fallback answer generation without Ollama."""
        if not documents:
            return "No relevant documents found."
        
        context = "\n\n".join([f"- {doc[:300]}" for doc in documents[:3]])
        
        return f"""Based on your GATE preparation notes, here's what I found about "{query}":

Key context from your notes:
{context}

Note: This is a fallback response. Start Ollama for full AI-generated answers.

To enable full responses, run in Windows PowerShell:
ollama serve"""
    
    def get_stats(self) -> Dict:
        """Get RAG system statistics."""
        return {
            "total_documents": self.collection.count(),
            "embedding_dim": self.embedding_dim,
            "collection_name": "my_notes",
            "status": "ready",
            "ollama_available": self.ollama_available,
            "ollama_model": self.ollama_model
        }