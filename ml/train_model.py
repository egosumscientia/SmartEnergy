# ml/train_model.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # asegura acceso al paquete core

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import random
from core.database import SessionLocal
from core.models import Registro

# ============================
# CONFIGURACION
# ============================
MODEL_FILE = "models/anomaly_detector.pkl"
os.makedirs("models", exist_ok=True)

# ============================
# LECTURA DESDE POSTGRESQL
# ============================
session = SessionLocal()
query = session.query(
    Registro.corriente,
    Registro.voltaje,
    Registro.potencia_activa,
    Registro.temperatura_motor,
    Registro.estado
)

df = pd.read_sql(query.statement, session.bind)
session.close()

if df.empty:
    raise ValueError("No hay datos disponibles en la tabla 'registros' para entrenar el modelo.")

# ============================
# PREPARACION DE DATOS
# ============================
features = ["corriente", "voltaje", "potencia_activa", "temperatura_motor"]
X = df[features].values

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# ENTRENAMIENTO DEL MODELO
# ============================
model = IsolationForest(
    n_estimators=200,
    contamination=random.choice([0.005, 0.02, 0.08]),
    random_state=None,
    n_jobs=-1
)

model.fit(X_scaled)
y_pred = model.predict(X_scaled)
df["pred_anomalia"] = (y_pred == -1).astype(int)

# ============================
# EVALUACION SIMPLE
# ============================
total = len(df)
detected = df["pred_anomalia"].sum()
true_anomalias = (df["estado"] == "Anomalo").sum()
true_positives = ((df["pred_anomalia"] == 1) & (df["estado"] == "Anomalo")).sum()

print(f"Registros totales: {total}")
print(f"Anomalias reales: {true_anomalias}")
print(f"Anomalias detectadas: {detected}")
print(f"Verdaderos positivos: {true_positives}")
print(f"Precision aproximada: {100 * true_positives / max(detected,1):.2f}%")

# ============================
# GUARDADO DEL MODELO
# ============================
joblib.dump({"model": model, "scaler": scaler, "features": features}, MODEL_FILE)
print(f"[OK] Modelo guardado en: {MODEL_FILE}")
