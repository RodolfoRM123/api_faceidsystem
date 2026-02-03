from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from typing import List
import numpy as np

from app.db.database import get_db
from app.models.domain import User
from app.models.schemas import UserCreate, UserRead, VerificationResponse, EnrollResponse
from app.vision.engine import face_engine
from app.core.config import settings

router = APIRouter()

@router.on_event("startup")
async def startup_event():
    face_engine.initialize()

@router.get("/users", response_model=List[UserRead])
def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users

@router.post("/enroll", response_model=EnrollResponse)
async def enroll_user(
    username: str,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    # Check max users
    count = db.query(User).count()
    if count >= settings.MAX_USERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Max users limit reached ({settings.MAX_USERS})"
        )
    
    # Check if user exists
    existing = db.query(User).filter(User.username == username).first()
    if existing:
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Username already taken"
        )

    embeddings = []
    
    for file in files:
        content = await file.read()
        faces, shape = face_engine.process_image(content)
        
        if not faces:
            # Skip images with no face, or validation could fail strictly
            continue
            
        # Take the largest face if multiple found (assumption)
        faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        target_face = faces[0]
        
        # Quality check
        is_live, msg = face_engine.check_liveness(target_face, shape)
        if not is_live:
            continue # Skip bad quality faces for enrollment
            
        embeddings.append(face_engine.get_embedding(target_face))

    if not embeddings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No valid faces found in uploaded images"
        )

    # Average embeddings (Centroid)
    # Stack and mean along axis 0
    avg_embedding = np.mean(embeddings, axis=0)
    # Normalize again just in case
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    # Save to DB
    new_user = User(username=username, embedding=avg_embedding)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return EnrollResponse(
        success=True,
        user_id=new_user.id,
        message=f"User {username} enrolled successfully",
        face_count=len(embeddings)
    )

@router.post("/verify", response_model=VerificationResponse)
async def verify_user(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    content = await file.read()
    faces, shape = face_engine.process_image(content)
    
    if not faces:
        return VerificationResponse(verified=False, message="No face detected", similarity=0.0)
    
    # Get largest face
    faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    target_face = faces[0]
    
    # Liveness/Quality Check
    is_live, msg = face_engine.check_liveness(target_face, shape)
    if not is_live:
        return VerificationResponse(verified=False, message=f"Liveness Check Failed: {msg}", similarity=0.0)
    
    target_embedding = face_engine.get_embedding(target_face)
    
    # Compare with DB
    users = db.query(User).all()
    best_score = -1.0
    best_match = None
    
    for user in users:
        db_embedding = user.embedding
        # Ensure correct type (pickle might return valid numpy array)
        score = face_engine.compute_similarity(target_embedding, db_embedding)
        
        if score > best_score:
            best_score = score
            best_match = user

    # Decision
    if best_score >= settings.SIMILARITY_THRESHOLD:
        return VerificationResponse(
            verified=True,
            username=best_match.username,
            similarity=float(best_score),
            message="Access Granted"
        )
    else:
        return VerificationResponse(
            verified=False,
            username=None,
            similarity=float(best_score),
            message="No match found"
        )
