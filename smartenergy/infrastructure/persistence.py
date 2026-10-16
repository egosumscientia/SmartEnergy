"""Consultas y escrituras PostgreSQL utilizadas por los casos de uso."""
from datetime import datetime

import pandas as pd

from smartenergy.infrastructure.models import Metrica, Registro


def load_records(session_factory):
    with session_factory() as session:
        query = session.query(
            Registro.id,
            Registro.timestamp,
            Registro.corriente,
            Registro.voltaje,
            Registro.potencia_activa,
            Registro.temperatura_motor,
            Registro.estado,
        ).order_by(Registro.timestamp.asc(), Registro.id.asc())
        return pd.read_sql(query.statement, session.bind)


def get_last_data_timestamp(session_factory):
    with session_factory() as session:
        return (
            session.query(Registro.timestamp)
            .order_by(Registro.timestamp.desc())
            .limit(1)
            .scalar()
        )


def load_latest_metric(session_factory):
    with session_factory() as session:
        return (
            session.query(Metrica)
            .order_by(Metrica.id.desc())
            .limit(1)
            .first()
        )


def load_metrics_history(session_factory, limit=2):
    with session_factory() as session:
        return (
            session.query(Metrica)
            .order_by(Metrica.id.desc())
            .limit(limit)
            .all()
        )


def get_record_count(session_factory):
    with session_factory() as session:
        return session.query(Registro).count()


def load_training_records(session_factory):
    with session_factory() as session:
        query = session.query(
            Registro.corriente,
            Registro.voltaje,
            Registro.potencia_activa,
            Registro.temperatura_motor,
            Registro.estado,
        )
        return pd.read_sql(query.statement, session.bind)


def load_detection_records(session_factory):
    with session_factory() as session:
        query = session.query(
            Registro.id,
            Registro.corriente,
            Registro.voltaje,
            Registro.potencia_activa,
            Registro.temperatura_motor,
            Registro.estado,
        )
        return pd.read_sql(query.statement, session.bind)


def save_detection_metric(
    session_factory,
    total: int,
    anomalies: int,
    percentage: float,
    state: str,
    created_at: datetime,
) -> None:
    with session_factory() as session:
        session.add(
            Metrica(
                total=total,
                anomalias=anomalies,
                porcentaje=percentage,
                estado=state,
                fecha=created_at,
            )
        )
        session.commit()
