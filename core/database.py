# core/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from core.config import DATABASE_URL

# Crear motor solo si hay URL
engine = create_engine(DATABASE_URL, echo=False) if DATABASE_URL else None

# Sesiones de base de datos
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None

# Clase base para modelos ORM
Base = declarative_base()
