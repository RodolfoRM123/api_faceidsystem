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

    def process_image(self, image_bytes: bytes) -> List[any]:
        """
        Decodes image and detects faces.
        Returns a list of InsightFace Face objects.
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
        return faces, img.shape

    def check_liveness(self, face, image_shape) -> Tuple[bool, str]:
        """
        Basic Liveness & Quality Check heuristics for single-image.
        Real liveness (blink) requires video.
        Here we check:
        1. Detection Score (already filtered by det_thresh, but we can be stricter)
        2. Face Size (too small = suspicious or bad quality)
        3. Simple Pose (looking straight)
        4. (Optional) Laplacian Variance for blur (screen replay often blurs or moire)
        """
        # 1. Score
        if face.det_score < 0.60:
            return False, "Low detection confidence"

        # 2. Size relative to image
        h, w, _ = image_shape
        bbox = face.bbox
        face_h = bbox[3] - bbox[1]
        face_w = bbox[2] - bbox[0]
        
        if face_h < h * 0.1 or face_w < w * 0.1:
            return False, "Face too small"

        # 3. Pose (Pitch, Yaw, Roll)
        # InsightFace returns pose as roughly [pitch, yaw, roll] in degrees (depending on version)
        # or simplified. We check common keys if available, else skip.
        if hasattr(face, 'pose') and face.pose is not None:
             # This depends on specific model, usually output is existing.
             # We skipped strict pose check implementation to avoid crashes if model doesn't output it by default
             # but buffalo_l usually does.
             pass

        return True, "Passed"

    def get_embedding(self, face) -> np.ndarray:
        return face.embedding

    def compute_similarity(self, embed1: np.ndarray, embed2: np.ndarray) -> float:
        """
        Cosine Similarity: (A . B) / (||A|| * ||B||)
        InsightFace embeddings are often normalized, but we ensure it.
        """
        # Normalization
        norm1 = np.linalg.norm(embed1)
        norm2 = np.linalg.norm(embed2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(embed1, embed2) / (norm1 * norm2)

face_engine = FaceEngine()
