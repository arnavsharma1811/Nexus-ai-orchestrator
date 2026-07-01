import logging
from typing import List, Dict, Optional
from services.rag_service import RAGService
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

class AgentService:
    def __init__(self):
        self.rag = RAGService()
        logger.info("[Agent] Initialized with RAG and Web Search")
    
    def run(self, prompt: str, tools: List[str] = None) -> Dict:
        """
        Run the agent with a prompt and available tools.
        """
        if tools is None:
            tools = ["rag"]
        
        trace = []
        answer = ""
        context = ""

        try:
            # --- RAG tool ---
            if "rag" in tools:
                logger.info(f"[Agent] Using RAG for: {prompt}")
                result = self.rag.query(prompt, top_k=3)
                context = result.get("answer", "No RAG answer.")
                trace.append({
                    "step": 1,
                    "thought": f"Querying RAG for: {prompt}",
                    "action": "rag",
                    "observation": f"Retrieved {len(result.get('sources', []))} sources"
                })
                answer = context  # RAG answer is already generated

            # --- Web Search tool ---
            if "web_search" in tools:
                logger.info(f"[Agent] Using Web Search for: {prompt}")
                search_results = self._web_search(prompt)
                if search_results:
                    trace.append({
                        "step": 2,
                        "thought": f"Searching web for: {prompt}",
                        "action": "web_search",
                        "observation": f"Found {len(search_results)} snippets"
                    })
                    # Generate answer from search results using Ollama
                    answer = self._generate_from_context(prompt, search_results)
                else:
                    answer = "No search results found."
                    trace.append({
                        "step": 2,
                        "thought": f"Searching web for: {prompt}",
                        "action": "web_search",
                        "observation": "No results found"
                    })

            # Fallback if no tool matched
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

    def _web_search(self, query: str, max_results: int = 3) -> str:
        """Search the web using DuckDuckGo and return combined snippets."""
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                snippets = []
                for r in results:
                    body = r.get('body', '')
                    if body:
                        snippets.append(body)
                if snippets:
                    return "\n\n".join(snippets)
                else:
                    return ""
        except Exception as e:
            logger.error(f"[Agent] Web search error: {str(e)}")
            return ""

    def _generate_from_context(self, query: str, context: str) -> str:
        """Generate a concise answer from context using Ollama."""
        if not context:
            return "No context available to generate an answer."

        # Use RAGService's Ollama integration to generate answer
        # We'll create a temporary prompt and use the RAG service's internal method.
        # But RAGService._generate_answer_ollama is private; we can add a public method.
        # For simplicity, we'll call the RAG service's query with a special prompt.
        # Alternatively, we can replicate the call.
        from services.rag_service import RAGService
        rag = RAGService()
        # Build a prompt that includes the context
        prompt = f"Answer the following question based on the provided web search results:\n\nQuestion: {query}\n\nSearch Results:\n{context}\n\nAnswer concisely:"
        # We'll just use the same _generate_answer_ollama logic; but it's private.
        # We'll create a new method or just use the fallback.
        # Quick: use the RAG query but with this prompt as "query" and no retrieval.
        # However, that would trigger retrieval. So we bypass by calling Ollama directly.
        # Let's add a helper method in RAGService to generate from raw context.
        # I'll add a method in RAGService: generate_from_context(self, query, context).
        # But to keep changes minimal, we'll replicate the Ollama call here.
        import requests
        ollama_url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "max_tokens": 256
            }
        }
        try:
            response = requests.post(ollama_url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response from Ollama.").strip()
            else:
                logger.warning(f"Ollama returned {response.status_code}")
                return f"Error: {response.status_code}"
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Error: {e}"