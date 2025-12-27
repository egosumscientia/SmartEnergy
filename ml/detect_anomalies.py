# ml/detect_anomalies.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import joblib
from core.database import SessionLocal
from core.models import Registro, Metrica
from datetime import datetime, timezone

MODEL_FILE = "models/anomaly_detector.pkl"

if SessionLocal is None:
    raise RuntimeError("DATABASE_URL no configurada o engine no inicializado. No se puede detectar anomalias.")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("No se encontro el modelo entrenado. Ejecuta train_model.py primero.")

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]

session = SessionLocal()
try:
    query = session.query(
        Registro.id,
        Registro.corriente,
        Registro.voltaje,
        Registro.potencia_activa,
        Registro.temperatura_motor,
        Registro.estado
    )
    df = pd.read_sql(query.statement, session.bind)

    if df.empty:
        raise ValueError("No se encontraron registros en la base de datos.")

    X = df[features].values
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    df["pred_anomalia"] = (preds == -1).astype(int)

    total = int(len(df))
    anomalias_reales = int((df["estado"] == "Anomalo").sum())
    detectadas = int(df["pred_anomalia"].sum())
    verdaderos_positivos = int(((df["estado"] == "Anomalo") & (df["pred_anomalia"] == 1)).sum())
    recall = float(100 * verdaderos_positivos / max(anomalias_reales, 1))
    precision = float(100 * verdaderos_positivos / max(detectadas, 1))
    coverage = float(100 * detectadas / max(total, 1))  # porcentaje detectado sobre total (KPI mostrado en dashboard)

    print(f"Total registros: {total}")
    print(f"Anomalias reales: {anomalias_reales}")
    print(f"Detectadas por modelo: {detectadas}")
    print(f"Verdaderos positivos: {verdaderos_positivos}")
    print(f"Recall: {recall:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Coverage (detectadas/total): {coverage:.2f}%")

    # Estado combinado (alineado con README): considera precision y recall/cobertura
    if precision >= 80 and recall >= 70:
        estado = "Normal"
    elif precision >= 60 and recall >= 40:
        estado = "Precaucion"
    else:
        estado = "Critico"

    metrica = Metrica(
        total=total,
        anomalias=detectadas,
        porcentaje=float(round(coverage, 2)),
        estado=estado,
        fecha=datetime.now(timezone.utc)
    )
    session.add(metrica)
    session.commit()
finally:
    session.close()

print("[OK] METRICAS registradas en la base de datos.")
