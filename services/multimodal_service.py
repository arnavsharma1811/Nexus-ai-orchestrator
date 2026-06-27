import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalService:
    def __init__(self):
        """Initialize EasyOCR reader."""
        logger.info("[MultiModal] Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=True)
        logger.info("[MultiModal] EasyOCR ready")
    
    def process(self, image_bytes: bytes, query: Optional[str] = None) -> Dict:
        """
        Process image: extract text, then optionally answer query.
        """
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert PIL to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image for better OCR
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # OCR
            logger.info("[MultiModal] Running OCR...")
            results = self.reader.readtext(thresh)
            
            # Extract text from results
            extracted_text = " ".join([res[1] for res in results])
            logger.info(f"[MultiModal] Extracted {len(extracted_text)} characters")
            
            # Generate answer if query provided
            answer = self._generate_answer(extracted_text, query) if query else extracted_text
            
            return {
                "extracted_text": extracted_text,
                "answer": answer,
                "confidence": self._calculate_confidence(results)
            }
            
        except Exception as e:
            logger.error(f"[MultiModal] Error: {str(e)}")
            return {
                "extracted_text": "",
                "answer": f"Error: {str(e)}",
                "confidence": 0.0
            }
    
    def _generate_answer(self, extracted_text: str, query: str) -> str:
        """
        Generate answer based on extracted text and query.
        TODO: Replace with actual RAG + Llama 3 integration.
        """
        # Placeholder — in production, send to RAG pipeline
        return f"Based on the diagram, {extracted_text[:200]}..."
    
    def _calculate_confidence(self, results: list) -> float:
        """Calculate average confidence from OCR results."""
        if not results:
            return 0.0
        confidences = [res[2] for res in results]
        return round(sum(confidences) / len(confidences), 2)