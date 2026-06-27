import logging
from typing import List, Dict, Optional
from services.rag_service import RAGService

logger = logging.getLogger(__name__)

class AgentService:
    def __init__(self):
        self.rag = RAGService()
        logger.info("[Agent] Initialized with RAG")
    
    def run(self, prompt: str, tools: List[str] = None) -> Dict:
        """
        Run the agent with a prompt and available tools.
        """
        if tools is None:
            tools = ["rag"]
        
        try:
            # Simple agent: route to appropriate tool
            if "rag" in tools:
                result = self.rag.query(prompt, top_k=3)
                return {
                    "answer": result.get("answer", "No answer generated."),
                    "trace": [
                        {
                            "step": 1,
                            "thought": f"Querying RAG for: {prompt}",
                            "action": "rag",
                            "observation": f"Retrieved {len(result.get('sources', []))} sources"
                        }
                    ]
                }
            else:
                return {
                    "answer": f"No valid tools selected. Available: {tools}",
                    "trace": []
                }
        except Exception as e:
            logger.error(f"Agent error: {str(e)}")
            return {
                "answer": f"Error: {str(e)}",
                "trace": []
            }