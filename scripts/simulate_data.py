import numpy as np
import random
from datetime import datetime, timedelta
from core.database import SessionLocal, engine
from core.models import Registro, Base

# ============================
# CONFIGURACIÓN GENERAL
# ============================
NUM_SAMPLES = 10000
ANOMALY_RATIO = random.choice([0.0, 0.02, 0.08])  # 0%, 2%, 8%
SAMPLING_INTERVAL = 1  # segundos
MAX_REGISTROS = 20000  # límite total de registros que se mantendrán en BD

# ============================
# INICIALIZACIÓN DE BD
# ============================
Base.metadata.create_all(bind=engine)
session = SessionLocal()

print("=== Generando datos simulados de energía ===")

# ============================
# GENERACIÓN DE DATOS BASE
# ============================
timestamps = [datetime.now() + timedelta(seconds=i * SAMPLING_INTERVAL) for i in range(NUM_SAMPLES)]
corriente = np.random.normal(25, 3, NUM_SAMPLES)
voltaje = np.random.normal(220, 5, NUM_SAMPLES)
temperatura = np.random.normal(45, 2, NUM_SAMPLES)
estado_maquina = np.random.choice([0, 1], NUM_SAMPLES, p=[0.1, 0.9])
potencia = corriente * voltaje * np.random.uniform(0.8, 0.95, NUM_SAMPLES) / 1000

# ============================
# INSERCIÓN DE ANOMALÍAS
# ============================
labels = np.zeros(NUM_SAMPLES)
num_anom = int(NUM_SAMPLES * ANOMALY_RATIO)
anom_indices = np.random.choice(NUM_SAMPLES, num_anom, replace=False)

for idx in anom_indices:
    tipo = random.choice([
        "corriente_alta", "voltaje_bajo", "temperatura_alta",
        "potencia_irregular", "ruido_extremo", "fallo_comb_inicial"
    ])
    if tipo == "corriente_alta":
        corriente[idx] *= np.random.uniform(1.5, 2.0)
    elif tipo == "voltaje_bajo":
        voltaje[idx] *= np.random.uniform(0.6, 0.85)
    elif tipo == "temperatura_alta":
        temperatura[idx] += np.random.uniform(10, 35)
    elif tipo == "potencia_irregular":
        potencia[idx] *= np.random.uniform(1.6, 2.8)
    elif tipo == "ruido_extremo":
        corriente[idx] += np.random.normal(0, 10)
        voltaje[idx] += np.random.normal(0, 20)
    elif tipo == "fallo_comb_inicial":
        corriente[idx] *= np.random.uniform(1.4, 1.8)
        temperatura[idx] += np.random.uniform(8, 20)
        potencia[idx] *= np.random.uniform(1.3, 1.7)
    labels[idx] = 1

# ============================
# CONSTRUCCIÓN DE OBJETOS ORM
# ============================
registros = [
    Registro(
        timestamp=timestamps[i],
        corriente=float(corriente[i]),
        voltaje=float(voltaje[i]),
        potencia_activa=float(potencia[i]),
        temperatura_motor=float(temperatura[i]),
        estado="Anomalo" if labels[i] == 1 else "Normal"
    )
    for i in range(NUM_SAMPLES)
]

# ============================
# GUARDADO EN LA BASE DE DATOS
# ============================
session.bulk_save_objects(registros)
session.commit()

# ============================
# RECORTE AUTOMÁTICO DE DATOS
# ============================
total = session.query(Registro).count()
if total > MAX_REGISTROS:
    exceso = total - MAX_REGISTROS
    print(f"⚠️ Se superó el límite ({MAX_REGISTROS}). Eliminando {exceso} registros antiguos...")
    session.execute(
        f"DELETE FROM registros WHERE id IN (SELECT id FROM registros ORDER BY id ASC LIMIT {exceso});"
    )
    session.commit()

session.close()

print(f"✅ {NUM_SAMPLES} registros insertados en la base de datos.")
print(f"Nivel de anomalías: {ANOMALY_RATIO * 100:.1f}% ({num_anom} de {NUM_SAMPLES})")
print(f"Total actual en base: {min(total, MAX_REGISTROS)} registros.")

