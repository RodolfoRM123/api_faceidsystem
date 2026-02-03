from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    pass

class UserRead(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class VerificationResponse(BaseModel):
    verified: bool
    username: Optional[str] = None
    similarity: float
    message: str

class EnrollResponse(BaseModel):
    success: bool
    user_id: int
    message: str
    face_count: int
