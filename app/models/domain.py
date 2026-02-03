from sqlalchemy import Column, Integer, String, PickleType, DateTime
from sqlalchemy.sql import func
from app.db.database import Base
import numpy as np

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Storing embedding as binary pickle or raw bytes. 
    # For better portability in real PG, use ARRAY(Float). 
    # For SQLite, PickleType or BLOB is easiest for numpy arrays.
    embedding = Column(PickleType, nullable=False)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"
