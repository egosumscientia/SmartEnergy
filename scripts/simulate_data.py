import numpy as np
import random
from datetime import datetime, timedelta
from sqlalchemy import text
from core.database import SessionLocal, engine
from core.models import Registro, Base

# ============================
# CONFIGURACION GENERAL
# ============================
NUM_SAMPLES = 10000
ANOMALY_RATIO = random.uniform(0.10, 0.20)  # Variable entre 10% y 20%
SAMPLING_INTERVAL = 1  # segundos
MAX_REGISTROS = 20000  # limite total de registros en BD

# ============================
# INICIALIZACION DE BD
# ============================
Base.metadata.create_all(bind=engine)
session = SessionLocal()

print("=== Generando datos simulados de energia ===")

# ============================
# GENERACION DE DATOS BASE
# ============================
timestamps = [datetime.now() + timedelta(seconds=i * SAMPLING_INTERVAL) for i in range(NUM_SAMPLES)]
corriente = np.random.normal(25, 3, NUM_SAMPLES)
voltaje = np.random.normal(220, 5, NUM_SAMPLES)
temperatura = np.random.normal(45, 2, NUM_SAMPLES)
estado_maquina = np.random.choice([0, 1], NUM_SAMPLES, p=[0.1, 0.9])
# Potencia base consistente
potencia = corriente * voltaje * np.random.uniform(0.8, 0.95, NUM_SAMPLES) / 1000

# ============================
# INSERCION DE ANOMALIAS
# ============================
labels = np.zeros(NUM_SAMPLES)
num_anom = int(NUM_SAMPLES * ANOMALY_RATIO)
anom_indices = np.random.choice(NUM_SAMPLES, num_anom, replace=False)

print(f"Generando {num_anom} anomalias garantizadas...")

for idx in anom_indices:
    tipo = random.choice([
        "corriente_alta", "voltaje_bajo", "temperatura_alta",
        "potencia_irregular", "ruido_extremo", "fallo_comb_inicial"
    ])
    
    if tipo == "corriente_alta":
        # Corriente absurda (2.5x - 4.0x) -> Afecta potencia al alza
        vals_extra = np.random.uniform(2.5, 4.0)
        corriente[idx] *= vals_extra
        potencia[idx] = corriente[idx] * voltaje[idx] * 0.9 / 1000

    elif tipo == "voltaje_bajo":
        # Caida severa de voltaje (40% - 60% del valor)
        vals_extra = np.random.uniform(0.4, 0.6)
        voltaje[idx] *= vals_extra
        potencia[idx] = corriente[idx] * voltaje[idx] * 0.9 / 1000

    elif tipo == "temperatura_alta":
        # Sobrecalentamiento critico (+50-100 grados)
        temperatura[idx] += np.random.uniform(50, 100)

    elif tipo == "potencia_irregular":
        # Desacople fisico: potencia sube mucho pero corriente baja (fisicamente imposible)
        corriente[idx] *= 0.5
        potencia[idx] *= np.random.uniform(2.0, 3.0)

    elif tipo == "ruido_extremo":
        # Ruido aditivo grande
        corriente[idx] += np.random.normal(0, 20)
        voltaje[idx] += np.random.normal(0, 40)
        # Recalcular potencia con los valores ruidosos (usando abs para evitar negativos extraños si el ruido es muy bajo)
        potencia[idx] = abs(corriente[idx] * voltaje[idx] * 0.9 / 1000)

    elif tipo == "fallo_comb_inicial":
        corriente[idx] *= np.random.uniform(2.0, 3.0)
        temperatura[idx] += np.random.uniform(20, 40)
        potencia[idx] = corriente[idx] * voltaje[idx] * 0.9 / 1000
    
    labels[idx] = 1

# ============================
# CONSTRUCCION DE OBJETOS ORM
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
# Borrar datos anteriores para asegurar limpieza (opcional, pero util si queremos ver solo lo nuevo)
# session.query(Registro).delete() 
# No borramos aqui porque reset_db ya lo hace. Solo insertamos.

session.bulk_save_objects(registros)
session.commit()

# ============================
# RECORTE AUTOMATICO DE DATOS
# ============================
total = session.query(Registro).count()
if total > MAX_REGISTROS:
    exceso = total - MAX_REGISTROS
    print(f"Se supero el limite ({MAX_REGISTROS}). Eliminando {exceso} registros antiguos...")
    session.execute(
        text(f"DELETE FROM registros WHERE id IN (SELECT id FROM registros ORDER BY id ASC LIMIT {exceso});")
    )
    session.commit()

session.close()

print(f"{NUM_SAMPLES} registros insertados en la base de datos.")
print(f"Nivel de anomalias: {ANOMALY_RATIO * 100:.1f}% ({num_anom} de {NUM_SAMPLES})")
print(f"Total actual en base: {min(total, MAX_REGISTROS)} registros.")
