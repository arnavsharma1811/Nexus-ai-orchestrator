from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import rag, agent, voice, multi_modal

app = FastAPI(
    title="Nexus AI Orchestrator",
    description="Unified API for RAG, ReAct agents, voice, and multi-modal AI systems",
    version="1.0.0"
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
async def root():
    return {
        "status": "alive",
        "systems": ["rag", "agent", "voice", "multi_modal"],
        "message": "Nexus AI Orchestrator is running"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Register routers
app.include_router(rag.router, prefix="/rag", tags=["RAG"])
app.include_router(agent.router, prefix="/agent", tags=["Agent"])
app.include_router(voice.router, prefix="/voice", tags=["Voice"])
app.include_router(multi_modal.router, prefix="/multi-modal", tags=["Multi-Modal"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)