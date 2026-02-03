import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "FaceID Backend"
    API_V1_STR: str = "/api/v1"
    
    # Biometrics Model Settings
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"  # 'buffalo_l' is accurate, 'buffalo_s' is fast
    CTX_ID: int = 0  # 0 for GPU, -1 for CPU. Using -1 for safety unless CUDA is known.
    DETECTION_THRESHOLD: float = 0.5
    SIMILARITY_THRESHOLD: float = 0.45  # Tuned for Cosine Similarity (roughly 0.4-0.6 range)
    
    # Database
    DATABASE_URL: str = "sqlite:///./faceid.db"
    
    # Security
    MAX_USERS: int = 10
    
    class Config:
        env_file = ".env"

settings = Settings()
