import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from app.core.config import settings
import logging
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceEngine, cls).__new__(cls)
            cls._instance.model = None
        return cls._instance

    def initialize(self):
        if self.model is None:
            logger.info(f"Initializing InsightFace model: {settings.INSIGHTFACE_MODEL_NAME}...")
            # 'buffalo_l' includes detection (RetinaFace) and recognition (ArcFace)
            self.model = FaceAnalysis(name=settings.INSIGHTFACE_MODEL_NAME, providers=['CPUExecutionProvider'])
            # ctx_id=0 for GPU, -1 for CPU. But providers=['CPUExecutionProvider'] is explicit for ONNXRuntime
            self.model.prepare(ctx_id=settings.CTX_ID, det_thresh=settings.DETECTION_THRESHOLD)
            logger.info("Model loaded successfully.")

    def process_image(self, image_bytes: bytes) -> Tuple[List[any], np.ndarray]:
        """
        Decodes image and detects faces.
        Returns a tuple (faces, image_array).
        """
        if not self.model:
            self.initialize()

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image format")

        # Get faces
        faces = self.model.get(img)
        return faces, img


    def check_liveness(self, face, image) -> Tuple[bool, str]:
        """
        Improved Liveness & Quality Check.
        """
        # 1. Detection Confidence Score
        if face.det_score < 0.70: 
            return False, "Confianza de detección baja"

        # 2. Face Size relative to image
        h, w, _ = image.shape
        bbox = face.bbox
        face_h = bbox[3] - bbox[1]
        face_w = bbox[2] - bbox[0]
        
        if face_h < h * 0.15 or face_w < w * 0.15:
            return False, "Rostro demasiado pequeño, acércate más"

        # 3. Blur Detection (Laplacian Variance) - Very permissive
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_crop = image[y1:y2, x1:x2]
        
        if face_crop.size > 0:
            gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # Reduced to 5 - basically only fails if completely gray/out of focus
            if blur_score < 5: 
                return False, f"Imagen demasiado borrosa (puntaje: {int(blur_score)})"

            # 4. Brightness Check - Very permissive
            avg_brightness = np.mean(gray_face)
            if avg_brightness < 15: # Almost pitch black
                return False, "Imagen demasiado oscura"
            if avg_brightness > 250: # Pure white
                return False, "Demasiada luz"

        return True, "Passed"


    def get_embedding(self, face) -> np.ndarray:
        # Normalizing at source to ensure consistent comparisons
        embedding = face.embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def compute_similarity(self, embed1: np.ndarray, embed2: np.ndarray) -> float:
        """
        Cosine Similarity. 
        Assumes normalized embeddings from get_embedding.
        """
        return np.dot(embed1, embed2)


face_engine = FaceEngine()
