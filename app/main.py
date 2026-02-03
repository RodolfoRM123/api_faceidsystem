from fastapi import FastAPI
from app.routes import biometrics
from app.core.config import settings
from app.db.database import Base, engine
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Create Database Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(biometrics.router, prefix=settings.API_V1_STR, tags=["biometrics"])

@app.get("/health")
def health_check():
    return {"status": "ok", "system": "FaceID Backend Ready"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
