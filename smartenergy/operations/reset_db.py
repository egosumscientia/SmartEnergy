"""Mantenimiento destructivo explicito; nunca se ejecuta al importar."""
import subprocess
import sys

from sqlalchemy import text

from smartenergy.config import REPOSITORY_ROOT
from smartenergy.infrastructure.database import get_engine


def main():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("DATABASE_URL no configurada. No se puede reiniciar la base de datos.")

    print("=== Reiniciando base de datos ===")
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE registros, metricas RESTART IDENTITY CASCADE;"))

    print("Tablas limpiadas correctamente.")
    print("Generando dataset inicial de 10 000 registros...")
    subprocess.run(
        [sys.executable, "-m", "smartenergy.operations.simulate_data"],
        check=True,
        cwd=REPOSITORY_ROOT,
    )
    print("Base de datos reiniciada y repoblada.")


if __name__ == "__main__":
    main()
