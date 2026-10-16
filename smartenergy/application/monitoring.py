"""Datos de consulta para las interfaces de monitoreo."""
from datetime import datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from smartenergy.analysis.measurements import prepare_records
from smartenergy.config import load_settings
from smartenergy.infrastructure import persistence
from smartenergy.infrastructure.database import get_engine, get_session_factory


def check_database_connection():
    engine = get_engine()
    if engine is None:
        return False, "DATABASE_URL no configurada en .env"
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True, None
    except SQLAlchemyError as exc:
        return False, f"No se pudo conectar a la base de datos: {exc}"


def get_data_count():
    session_factory = get_session_factory()
    if session_factory is None:
        return 0
    try:
        return persistence.get_record_count(session_factory)
    except SQLAlchemyError:
        return 0


def load_records():
    session_factory = get_session_factory()
    if session_factory is None:
        return None
    df = persistence.load_records(session_factory)
    if df.empty:
        return None
    return prepare_records(df)


def get_last_data_timestamp():
    session_factory = get_session_factory()
    if session_factory is None:
        return None
    try:
        return persistence.get_last_data_timestamp(session_factory)
    except SQLAlchemyError:
        return None


def load_latest_metrics():
    session_factory = get_session_factory()
    if session_factory is None:
        return None
    metric = persistence.load_latest_metric(session_factory)
    if metric is None:
        return None
    return {
        "total": metric.total,
        "anomalias": metric.anomalias,
        "porcentaje": metric.porcentaje,
        "estado": metric.estado,
        "fecha": metric.fecha,
    }


def load_metrics_history(limit=2):
    session_factory = get_session_factory()
    if session_factory is None:
        return []
    try:
        return persistence.load_metrics_history(session_factory, limit)
    except SQLAlchemyError:
        return []


def get_model_artifact_status():
    path = load_settings().model_path
    if not path.is_file():
        return False, None
    return True, datetime.fromtimestamp(path.stat().st_mtime)


def load_model():
    path: Path = load_settings().model_path
    if not path.is_file():
        return None, None, None, None
    try:
        from smartenergy.analysis.measurements import ENERGY_COLUMNS
        from smartenergy.infrastructure.model_artifact import load_model_bundle

        bundle = load_model_bundle(path)
        model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]
        if not callable(getattr(model, "predict", None)) or not callable(
            getattr(scaler, "transform", None)
        ):
            raise ValueError("Modelo o scaler inválido")
        if (
            not isinstance(features, list)
            or not features
            or not set(features).issubset(ENERGY_COLUMNS)
        ):
            raise ValueError("Variables del modelo incompatibles")
    except Exception as exc:
        return None, None, None, exc
    return model, scaler, features, None
