# scripts/reset_db.py
import subprocess
import sys
from sqlalchemy import text
from core.database import engine

print("=== Reiniciando base de datos ===")

with engine.connect() as conn:
    conn.execute(text("TRUNCATE TABLE registros, metricas RESTART IDENTITY CASCADE;"))
    conn.commit()

print("Tablas limpiadas correctamente.")
print("Generando dataset inicial de 10 000 registros...")

subprocess.run([sys.executable, "-m", "scripts.simulate_data"], check=True)

print("Base de datos reiniciada y repoblada.")
