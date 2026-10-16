"""Datos de consulta para las interfaces de monitoreo."""
from datetime import datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from smartenergy.analysis.measurements import prepare_records
from smartenergy.config import load_settings
from smartenergy.infrastructure import persistence
from smartenergy.infrastructure.database import get_engine, get_session_factory


class MonitoringUnavailableError(RuntimeError):
    """Raised when a monitoring query cannot obtain its database session."""


def _require_session_factory():
    session_factory = get_session_factory()
    if session_factory is None:
        raise MonitoringUnavailableError("Monitoring database is unavailable")
    return session_factory


def check_database_connection():
    engine = get_engine()
    if engine is None:
        return False, "DATABASE_URL no configurada en .env"
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True, None
    except SQLAlchemyError:
        return False, "No se pudo conectar a la base de datos."


def get_data_count():
    return persistence.get_record_count(_require_session_factory())


def load_records():
    session_factory = _require_session_factory()
    df = persistence.load_records(session_factory)
    if df.empty:
        return None
    return prepare_records(df)


def get_last_data_timestamp():
    return persistence.get_last_data_timestamp(_require_session_factory())


def load_latest_metrics():
    session_factory = _require_session_factory()
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
    return persistence.load_metrics_history(_require_session_factory(), limit)


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
