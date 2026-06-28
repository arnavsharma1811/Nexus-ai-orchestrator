from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from services.multimodal_service import MultiModalService

router = APIRouter()
multimodal_service = MultiModalService()

class MultiModalResponse(BaseModel):
    extracted_text: str
    answer: str
    confidence: Optional[float] = None

@router.post("/analyze", response_model=MultiModalResponse)
async def analyze_diagram(
    image: UploadFile = File(...),
    query: Optional[str] = Form(None) 
):
    try:
        result = multimodal_service.process(
            image_bytes=await image.read(),
            query=query
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))