import logging
import os
from typing import List, Dict, Optional
from services.rag_service import RAGService
from duckduckgo_search import DDGS

# Try to import Tavily, fallback to DuckDuckGo if not available
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

logger = logging.getLogger(__name__)

class AgentService:
    def __init__(self):
        self.rag = RAGService()
        self.tavily_client = None
        self.use_tavily = False

        # Initialize Tavily if API key is set
        api_key = os.getenv("TAVILY_API_KEY")
        if api_key and TAVILY_AVAILABLE:
            try:
                self.tavily_client = TavilyClient(api_key=api_key)
                self.use_tavily = True
                logger.info("[Agent] Tavily search enabled")
            except Exception as e:
                logger.warning(f"[Agent] Tavily init failed: {e}")
        else:
            logger.info("[Agent] Using DuckDuckGo fallback search")

    def run(self, prompt: str, tools: List[str] = None) -> Dict:
        if tools is None:
            tools = ["rag"]

        trace = []
        answer = ""

        try:
            # --- RAG tool ---
            if "rag" in tools:
                logger.info(f"[Agent] Using RAG for: {prompt}")
                result = self.rag.query(prompt, top_k=3)
                answer = result.get("answer", "No RAG answer.")
                trace.append({
                    "step": 1,
                    "thought": f"Querying RAG for: {prompt}",
                    "action": "rag",
                    "observation": f"Retrieved {len(result.get('sources', []))} sources"
                })

            # --- Web Search tool ---
            if "web_search" in tools:
                logger.info(f"[Agent] Using Web Search for: {prompt}")
                if self.use_tavily and self.tavily_client:
                    # Use Tavily: returns direct answer + sources
                    try:
                        response = self.tavily_client.search(query=prompt, search_depth="basic")
                        answer = response.get("answer", "")
                        if not answer:
                            # Fallback to raw snippets if no answer
                            snippets = [r.get("content", "") for r in response.get("results", [])]
                            if snippets:
                                answer = self._generate_from_context(prompt, "\n\n".join(snippets))
                            else:
                                answer = "No clear answer found."
                        trace.append({
                            "step": 2,
                            "thought": f"Searching web via Tavily for: {prompt}",
                            "action": "web_search (Tavily)",
                            "observation": f"Found answer: {answer[:100]}..."
                        })
                    except Exception as e:
                        logger.error(f"[Agent] Tavily error: {e}")
                        answer = self._web_search_fallback(prompt)
                        trace.append({
                            "step": 2,
                            "thought": f"Searching web for: {prompt}",
                            "action": "web_search (DuckDuckGo fallback)",
                            "observation": "Used fallback"
                        })
                else:
                    # Fallback to DuckDuckGo
                    answer = self._web_search_fallback(prompt)
                    trace.append({
                        "step": 2,
                        "thought": f"Searching web for: {prompt}",
                        "action": "web_search (DuckDuckGo)",
                        "observation": "Fallback"
                    })

            # Fallback if no answer
            if not answer:
                answer = "No valid tools selected. Available: rag, web_search"

            return {
                "answer": answer,
                "trace": trace
            }

        except Exception as e:
            logger.error(f"[Agent] Error: {str(e)}")
            return {
                "answer": f"Error: {str(e)}",
                "trace": trace
            }

    def _web_search_fallback(self, query: str, max_results: int = 3) -> str:
        """DuckDuckGo fallback search."""
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                snippets = [r.get('body', '') for r in results if r.get('body')]
                if snippets:
                    combined = "\n\n".join(snippets)
                    return self._generate_from_context(query, combined)
                else:
                    return "No search results found."
        except Exception as e:
            logger.error(f"[Agent] Fallback search error: {e}")
            return "Search failed."

    def _generate_from_context(self, query: str, context: str) -> str:
        """Generate answer using Ollama from raw context."""
        if not context:
            return "No context available."

        # Use RAGService's Ollama call directly
        import requests
        ollama_url = "http://localhost:11434/api/generate"
        prompt = f"""You are a helpful assistant. Answer the question based on the context below. If the context doesn't contain the answer, say "I couldn't find that information."

Question: {query}

Context:
{context}

Answer:"""
        try:
            response = requests.post(
                ollama_url,
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "max_tokens": 256
                    }
                },
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response from Ollama.").strip()
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {e}"