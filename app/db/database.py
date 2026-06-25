from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configura la URL de conexión
#SQLALCHEMY_DATABASE_URL = "postgresql+psycopg2://postgres:123456@localhost:5433/segundacuna"
SQLALCHEMY_DATABASE_URL = "postgresql+psycopg2://master:123456@62.72.1.252:5433/segundacuna"

# Crea el motor de conexión
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,   # Reintenta conexión si se pierde
    pool_size=10,         # Máximo de conexiones activas
    max_overflow=20       # Conexiones extra si se llena el pool
)

# Crea la sesión
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base para tus modelos
Base = declarative_base()


# Dependencia para FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
