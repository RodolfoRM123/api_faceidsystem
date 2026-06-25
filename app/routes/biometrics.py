from typing import List, Annotated
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date
import numpy as np
from datetime import date, datetime

from app.db.database import get_db
from app.models.domain import User, FaceEmbedding, Attendance
from app.models.schemas import UserCreate, UserRead, VerificationResponse, EnrollResponse, AttendanceRead
from app.vision.engine import face_engine
from app.core.config import settings

router = APIRouter()

@router.on_event("startup")
async def startup_event():
    face_engine.initialize()

@router.get("/users", response_model=List[UserRead])
def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    result = []
    for user in users:
        user_dict = {
            "id": user.id,
            "username": user.username,
            "created_at": user.created_at,
            "photo_url": f"/static/profiles/{user.username}.jpg" if user.photo_path else None
        }
        result.append(user_dict)
    return result

@router.get("/attendance", response_model=List[AttendanceRead])
def list_attendance(db: Session = Depends(get_db)):
    # Query attendance with username joined
    results = db.query(
        Attendance.id,
        User.username,
        Attendance.timestamp
    ).join(User, Attendance.user_id == User.id).order_by(Attendance.timestamp.desc()).all()
    return results

@router.post("/enroll", response_model=EnrollResponse)
async def enroll_user(
    username: Annotated[str, Form()],
    files: Annotated[List[UploadFile], File()],
    db: Session = Depends(get_db)
):
    import cv2
    from pathlib import Path
    
    # Check max users
    count = db.query(User).count()
    if count >= settings.MAX_USERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Límite máximo de usuarios alcanzado ({settings.MAX_USERS})"
        )
    
    # Check if user exists
    existing = db.query(User).filter(User.username == username).first()
    if existing:
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="El nombre de usuario ya está en uso"
        )

    embeddings_to_save = []
    first_valid_image = None  # Para guardar la primera foto válida
    
    for file in files:
        content = await file.read()
        faces, img = face_engine.process_image(content)
        
        if not faces:
            continue
            
        # Take the largest face if multiple found
        faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        target_face = faces[0]
        
        # Quality check
        is_valid, msg = face_engine.check_liveness(target_face, img)
        if not is_valid:
            continue 
        
        # Guardar la primera imagen válida
        if first_valid_image is None:
            first_valid_image = img
            
        embeddings_to_save.append(face_engine.get_embedding(target_face))

    if not embeddings_to_save:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No se encontraron rostros de alta calidad en las imágenes subidas. Verifica el enfoque y la iluminación."
        )

    # Guardar la foto de perfil
    photo_path = None
    if first_valid_image is not None:
        profiles_dir = Path("static/profiles")
        profiles_dir.mkdir(parents=True, exist_ok=True)
        photo_filename = f"{username}.jpg"
        photo_path = profiles_dir / photo_filename
        cv2.imwrite(str(photo_path), first_valid_image)
        photo_path = str(photo_path)

    # Save User con la ruta de la foto
    new_user = User(username=username, photo_path=photo_path)
    db.add(new_user)
    db.flush() # Get ID

    # Save all unique embeddings
    for emb in embeddings_to_save:
        new_emb = FaceEmbedding(user_id=new_user.id, embedding=emb)
        db.add(new_emb)
    
    db.commit()
    db.refresh(new_user)

    return EnrollResponse(
        success=True,
        user_id=int(new_user.id),
        message=f"Usuario {username} registrado con {len(embeddings_to_save)} variaciones de rostro",
        face_count=len(embeddings_to_save)
    )

@router.post("/verify", response_model=VerificationResponse)
async def verify_user(
    file: Annotated[UploadFile, File()],
    db: Session = Depends(get_db)
):
    content = await file.read()
    faces, img = face_engine.process_image(content)
    
    if not faces:
        return VerificationResponse(verified=False, message="No se detectó ningún rostro", similarity=0.0)
    
    # Get largest face
    faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    target_face = faces[0]
    
    # Liveness/Quality Check
    is_valid, msg = face_engine.check_liveness(target_face, img)
    if not is_valid:
        return VerificationResponse(verified=False, message=f"Fallo de calidad: {msg}", similarity=0.0)
    
    target_embedding = face_engine.get_embedding(target_face)
    
    # Compare with ALL embeddings in DB
    all_embeddings = db.query(FaceEmbedding, User).join(User).all()
    
    if not all_embeddings:
        return VerificationResponse(verified=False, message="No hay usuarios registrados en la base de datos", similarity=0.0)

    best_score = -1.0
    best_match_username = None
    
    for emb_record, user_record in all_embeddings:
        db_emb = emb_record.embedding
        score = face_engine.compute_similarity(target_embedding, db_emb)
        
        if score > best_score:
            best_score = score
            best_match_username = user_record.username

    # Decision
    if best_score >= settings.SIMILARITY_THRESHOLD:
        # --- SELF-UPDATING LOGIC ---
        matched_user = db.query(User).filter(User.username == best_match_username).first()
        
        # --- REGLA DE NEGOCIO: ENTRADA Y SALIDA (8 HORAS DE DIFERENCIA) ---
        from datetime import timezone, timedelta
        # Configurar zona horaria local (UTC-6 para CDMX/Central Standard Time)
        tz_local = timezone(timedelta(hours=-6))
        ahora_local = datetime.now(tz_local)
        hoy_local = ahora_local.date()
        
        # Consultar todos los registros del usuario
        asistencias_usuario = db.query(Attendance).filter(
            Attendance.user_id == matched_user.id
        ).all()
        
        # Filtrar registros de hoy usando la zona horaria local para la comparación
        asistencias_hoy = []
        for a in asistencias_usuario:
            # Convertir el timestamp de la DB a la zona horaria local
            # Si el timestamp es naive (sin zona), asumimos que la DB está en UTC
            ts_local = a.timestamp.replace(tzinfo=timezone.utc).astimezone(tz_local) if a.timestamp.tzinfo is None else a.timestamp.astimezone(tz_local)
            if ts_local.date() == hoy_local:
                asistencias_hoy.append(a)
        
        asistencias_hoy.sort(key=lambda x: x.timestamp)
        num_registros = len(asistencias_hoy)

        if num_registros == 0:
            # REGISTRAR ENTRADA
            nueva_entrada = Attendance(user_id=matched_user.id, timestamp=ahora_local)
            db.add(nueva_entrada)
            db.commit()
            
            return VerificationResponse(
                verified=True,
                username=best_match_username,
                similarity=float(best_score),
                message=f"ENTRADA REGISTRADA: ¡Bienvenido {best_match_username}!",
                check_in=True,
                check_in_time=ahora_local
            )

        elif num_registros == 1:
            # VERIFICAR SI PUEDE MARCAR SALIDA (8 HORAS)
            # Asegurar que ambos tiempos estén en la misma zona horaria para la resta
            ent_ts = asistencias_hoy[0].timestamp
            hora_entrada_local = ent_ts.replace(tzinfo=timezone.utc).astimezone(tz_local) if ent_ts.tzinfo is None else ent_ts.astimezone(tz_local)
            
            diferencia_segundos = (ahora_local - hora_entrada_local).total_seconds()
            diferencia_horas = diferencia_segundos / 3600

            if diferencia_segundos >= (8 * 3600):
                nueva_salida = Attendance(user_id=matched_user.id, timestamp=ahora_local)
                db.add(nueva_salida)
                db.commit()
                
                return VerificationResponse(
                    verified=True,
                    username=best_match_username,
                    similarity=float(best_score),
                    message=f"SALIDA REGISTRADA: Horas trabajadas: {diferencia_horas:.2f}",
                    check_in=True,
                    check_in_time=ahora_local
                )
            else:
                faltan = 8.0 - diferencia_horas
                return VerificationResponse(
                    verified=False,
                    username=best_match_username,
                    similarity=float(best_score),
                    message=f"ACCESO DENEGADO: Faltan {faltan:.2f} horas para tu salida.",
                    check_in=False,
                    check_in_time=hora_entrada_local
                )
        else:
            ts_exit = asistencias_hoy[1].timestamp
            hora_salida_local = ts_exit.replace(tzinfo=timezone.utc).astimezone(tz_local) if ts_exit.tzinfo is None else ts_exit.astimezone(tz_local)
            return VerificationResponse(
                verified=False,
                username=best_match_username,
                similarity=float(best_score),
                message="ACCESO DENEGADO: Ya registraste entrada y salida por hoy.",
                check_in=False,
                check_in_time=hora_salida_local
            )
    else:
        return VerificationResponse(
            verified=False,
            username=None,
            similarity=float(best_score),
            message=f"ROSTRO NO RECONOCIDO (Similitud: {best_score:.4f})"
        )
