from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from services.voice_service import VoiceService

router = APIRouter()
voice_service = VoiceService()

class VoiceRequest(BaseModel):
    text: str  # For text-to-speech
    wake_word: Optional[str] = "Zenith"

class VoiceResponse(BaseModel):
    text: str
    audio: Optional[str] = None  # Base64 encoded audio

@router.post("/talk", response_model=VoiceResponse)
async def talk(request: VoiceRequest):
    try:
        result = voice_service.process(
            text=request.text
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        text = voice_service.transcribe(await audio.read())
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))