import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import Optional, Dict
from services.rag_service import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalService:
    def __init__(self):
        """Initialize EasyOCR reader and RAG service."""
        logger.info("[MultiModal] Initializing EasyOCR...")
        self.reader = easyocr.Reader(
            ['en'],
            gpu=False,  # Set to True if CUDA works
            model_storage_directory='./models'  # Cache models locally
        )
        self.rag = RAGService()
        logger.info("[MultiModal] Ready")

    def process(self, image_bytes: bytes, query: Optional[str] = None) -> Dict:
        """
        Process image: extract text, then answer query using RAG.

        Args:
            image_bytes: Raw image bytes
            query: Optional question about the image

        Returns:
            Dict with extracted_text, answer, confidence
        """
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # --- PREPROCESSING: Improve OCR quality ---
            # 1. Resize if too small
            height, width = image_cv.shape[:2]
            if width < 800:
                scale = 800 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_cv = cv2.resize(
                    image_cv,
                    (new_width, new_height),
                    interpolation=cv2.INTER_CUBIC
                )
                logger.info(f"[MultiModal] Resized: {width}x{height} → {new_width}x{new_height}")

            # 2. Convert to grayscale
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

            # 3. Increase contrast and brightness
            alpha = 1.8  # Contrast
            beta = 30    # Brightness
            enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

            # 4. Adaptive thresholding (better for diagrams with varying lighting)
            thresh = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

            # 5. Denoise (remove small dots)
            denoised = cv2.fastNlMeansDenoising(thresh, h=10)

            logger.info("[MultiModal] Running OCR...")
            results = self.reader.readtext(
                denoised,
                paragraph=False,  # Keep line-by-line for better accuracy
                detail=1
            )

            # Extract text with confidence filtering
            extracted_parts = []
            confidences = []
            for res in results:
                bbox, text, conf = res
                if conf > 0.3:  # Filter low-confidence detections
                    extracted_parts.append(text)
                    confidences.append(conf)

            extracted_text = " ".join(extracted_parts)
            logger.info(f"[MultiModal] Extracted {len(extracted_text)} chars")

            # --- DEBUG: Log the query ---
            logger.info(f"[MultiModal] Query received: '{query}'")

            # --- ANSWER GENERATION with explicit RAG handling ---
            answer = None
            if query and query.strip():
                logger.info("[MultiModal] Query is non-empty, sending to RAG...")
                rag_query = f"{query}\n\nContext extracted from image:\n{extracted_text}"
                logger.info(f"[MultiModal] RAG query preview: {rag_query[:150]}...")
                try:
                    rag_result = self.rag.query(rag_query, top_k=3)
                    answer = rag_result.get("answer", "RAG returned no answer.")
                    logger.info(f"[MultiModal] RAG answer preview: {answer[:100]}...")
                except Exception as e:
                    logger.error(f"[MultiModal] RAG call failed: {str(e)}")
                    answer = f"RAG error: {str(e)}"
            else:
                logger.info("[MultiModal] No query provided, returning extracted text")
                answer = extracted_text

            # If answer is still None (shouldn't happen), fallback
            if answer is None:
                answer = "No answer generated."

            # Calculate average confidence
            avg_confidence = self._calculate_confidence(confidences)

            return {
                "extracted_text": extracted_text,
                "answer": answer,
                "confidence": avg_confidence
            }

        except Exception as e:
            logger.error(f"[MultiModal] Error: {str(e)}")
            return {
                "extracted_text": "",
                "answer": f"Error processing image: {str(e)}",
                "confidence": 0.0
            }

    def _calculate_confidence(self, confidences: list) -> float:
        """Calculate average confidence from OCR results."""
        if not confidences:
            return 0.0
        return round(sum(confidences) / len(confidences), 2)