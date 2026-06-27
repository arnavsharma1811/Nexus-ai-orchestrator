from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from services.rag_service import RAGService

router = APIRouter()
rag_service = RAGService()

class RAGQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_hybrid: Optional[bool] = True

class RAGResponse(BaseModel):
    answer: str
    sources: List[str]
    metadata: Optional[List[dict]] = None
    latency_ms: Optional[float] = None

@router.post("/query", response_model=RAGResponse)
async def query(request: RAGQuery):
    try:
        result = rag_service.query(
            query=request.query,
            top_k=request.top_k,
            hybrid=request.use_hybrid
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def stats():
    """Return RAG system stats"""
    return {
        "total_documents": 2159,
        "embedding_dim": 384,
        "retrieval_latency_ms": 200,
        "avg_chunk_size": 256
    }