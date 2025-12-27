# core/models.py
from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from core.database import Base

class Registro(Base):
    __tablename__ = "registros"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    corriente = Column(Float)
    voltaje = Column(Float)
    potencia_activa = Column(Float)
    temperatura_motor = Column(Float)
    estado = Column(String)  # "Normal" o "Anómalo"

class Metrica(Base):
    __tablename__ = "metricas"

    id = Column(Integer, primary_key=True, index=True)
    total = Column(Integer)
    anomalias = Column(Integer)
    porcentaje = Column(Float)
    estado = Column(String)
    fecha = Column(DateTime, default=datetime.utcnow)
