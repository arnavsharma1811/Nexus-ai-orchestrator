import requests
import logging
from typing import Dict, Optional
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class VoiceService:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "llama3.2:3b"
        
        # Initialize Whisper model (runs on GPU)
        logger.info("[Voice] Loading Whisper model...")
        self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        logger.info("[Voice] Whisper ready")
    
    def process(self, text: str) -> Dict:
        """Process text: generate response using Ollama."""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": f"You are a helpful voice assistant named Zenith. Respond concisely to: {text}",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "max_tokens": 128
                    }
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "No response from Ollama.").strip()
                return {
                    "text": answer,
                    "audio": None
                }
            else:
                return {
                    "text": f"Error: Ollama returned {response.status_code}",
                    "audio": None
                }
                
        except Exception as e:
            logger.error(f"[Voice] Error: {str(e)}")
            return {
                "text": f"Error: {str(e)}",
                "audio": None
            }
    
    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text using faster-whisper."""
        try:
            import tempfile
            import os
            
            # Save audio bytes to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            # Transcribe using faster-whisper
            segments, _ = self.whisper_model.transcribe(tmp_path, beam_size=5)
            
            # Combine segments into full text
            transcription = " ".join([seg.text for seg in segments])
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            return transcription if transcription else "No speech detected."
            
        except Exception as e:
            logger.error(f"[Voice] Transcription error: {str(e)}")
            return f"Error: {str(e)}"