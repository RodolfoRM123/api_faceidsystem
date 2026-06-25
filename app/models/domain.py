from sqlalchemy import Column, Integer, String, PickleType, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base
import numpy as np

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    photo_path = Column(String, nullable=True)  # Ruta de la foto de perfil
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship to multiple embeddings
    embeddings = relationship("FaceEmbedding", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    embedding = Column(PickleType, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="embeddings")

class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # We can store the date separately for easier querying 
    # or just use the timestamp. 
    check_in_date = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")


