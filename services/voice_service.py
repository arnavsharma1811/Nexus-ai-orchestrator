class VoiceService:
    def __init__(self):
        print("[Voice] Initialized")
    
    def process(self, text: str):
        return {"text": text, "audio": None}
    
    def transcribe(self, audio_bytes):
        return "Transcribed text"