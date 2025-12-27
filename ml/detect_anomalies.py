# ml/detect_anomalies.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import joblib
from core.database import SessionLocal
from core.models import Registro, Metrica
from datetime import datetime, timezone

# ============================
# CONFIGURACIÓN
# ============================
MODEL_FILE = "models/anomaly_detector.pkl"

# ============================
# CARGA DEL MODELO
# ============================
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("No se encontró el modelo entrenado. Ejecuta train_model.py primero.")

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]

# ============================
# CARGA DE DATOS DESDE POSTGRESQL
# ============================
session = SessionLocal()
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

# ============================
# DETECCIÓN DE ANOMALÍAS
# ============================
X = df[features].values
X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)
df["pred_anomalia"] = (preds == -1).astype(int)

# ============================
# CÁLCULO DE MÉTRICAS
# ============================
total = int(len(df))
anomalias_reales = int((df["estado"] == "Anómalo").sum())
detectadas = int(df["pred_anomalia"].sum())
verdaderos_positivos = int(((df["estado"] == "Anómalo") & (df["pred_anomalia"] == 1)).sum())
precision = float(100 * verdaderos_positivos / max(detectadas, 1))

print(f"Total registros: {total}")
print(f"Anomalías reales: {anomalias_reales}")
print(f"Detectadas por modelo: {detectadas}")
print(f"Verdaderos positivos: {verdaderos_positivos}")
print(f"Precisión: {precision:.2f}%")

# ============================
# GUARDADO DE MÉTRICAS EN BD
# ============================
metrica = Metrica(
    total=total,
    anomalias=detectadas,
    porcentaje=float(round((detectadas / total) * 100, 2)),
    estado="Crítico" if precision < 50 else "Normal",
    fecha=datetime.now(timezone.utc)
)
session.add(metrica)
session.commit()
session.close()

print("[OK] Métricas registradas en la base de datos.")
