"""Coordinacion de entrenamiento y deteccion persistente."""
from datetime import datetime, timezone

from smartenergy.analysis.anomalies import evaluate_model
from smartenergy.analysis.training import train_model as fit_model
from smartenergy.config import load_settings
from smartenergy.infrastructure import persistence
from smartenergy.infrastructure.database import get_session_factory


def train_model():
    from filelock import Timeout

    session_factory = get_session_factory()
    if session_factory is None:
        raise RuntimeError(
            "DATABASE_URL no configurada o engine no inicializado. No se puede entrenar el modelo."
        )

    df = persistence.load_training_records(session_factory)
    if df.empty:
        raise ValueError("No hay datos disponibles en la tabla 'registros' para entrenar el modelo.")

    bundle, metrics = fit_model(df)
    print(f"Registros totales: {metrics['total']}")
    print(f"Anomalias reales: {metrics['true_anomalies']}")
    print(f"Anomalias detectadas: {metrics['detected']}")
    print(f"Verdaderos positivos: {metrics['true_positives']}")
    print(f"Precision aproximada: {metrics['precision']:.2f}%")

    try:
        from smartenergy.infrastructure.model_artifact import save_model_bundle

        model_path = load_settings().model_path
        save_model_bundle(bundle, model_path)
        print("[OK] Modelo guardado en: models/anomaly_detector.pkl")
    except Timeout as exc:
        raise RuntimeError(
            "No se pudo obtener el lock del modelo (otro proceso esta escribiendo)."
        ) from exc


def detect_anomalies():
    from filelock import Timeout

    session_factory = get_session_factory()
    if session_factory is None:
        raise RuntimeError(
            "DATABASE_URL no configurada o engine no inicializado. No se puede detectar anomalias."
        )

    model_path = load_settings().model_path
    if not model_path.is_file():
        raise FileNotFoundError("No se encontro el modelo entrenado. Ejecuta train_model.py primero.")

    try:
        from smartenergy.infrastructure.model_artifact import load_model_bundle

        bundle = load_model_bundle(model_path)
    except Timeout as exc:
        raise RuntimeError(
            "No se pudo obtener el lock del modelo (otro proceso esta escribiendo)."
        ) from exc

    df = persistence.load_detection_records(session_factory)
    if df.empty:
        raise ValueError("No se encontraron registros en la base de datos.")

    model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]
    metrics = evaluate_model(df, model, scaler, features)
    if metrics is None:
        raise ValueError("El bundle no contiene un modelo evaluable.")
    total = metrics["total"]
    detected = metrics["detectadas"]
    true_anomalies = int((df["estado"] == "Anomalo").sum())
    true_positives = metrics["verdaderos_positivos"]

    print(f"Total registros: {total}")
    print(f"Anomalias reales: {true_anomalies}")
    print(f"Detectadas por modelo: {detected}")
    print(f"Verdaderos positivos: {true_positives}")
    print(f"Recall: {metrics['recall']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Coverage (detectadas/total): {metrics['coverage']:.2f}%")

    persistence.save_detection_metric(
        session_factory,
        total=total,
        anomalies=detected,
        percentage=float(round(metrics["coverage"], 2)),
        state=metrics["estado"],
        created_at=datetime.now(timezone.utc),
    )
    print("[OK] METRICAS registradas en la base de datos.")
