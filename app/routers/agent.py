from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from services.agent_service import AgentService

router = APIRouter()
agent_service = AgentService()

class AgentRequest(BaseModel):
    prompt: str
    tools: Optional[List[str]] = ["calculator", "web_search", "code_execution", "rag"]

class AgentResponse(BaseModel):
    answer: str
    trace: Optional[List[Dict]] = None

@router.post("/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    try:
        result = agent_service.run(
            prompt=request.prompt,
            tools=request.tools
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))