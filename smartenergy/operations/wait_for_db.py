"""Espera explicita a que PostgreSQL acepte consultas."""
import time

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from smartenergy.config import load_settings, load_wait_settings


def main() -> None:
    settings = load_settings(load_dotenv_file=False)
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL no esta configurada")
    wait_settings = load_wait_settings()

    engine = create_engine(settings.database_url, pool_pre_ping=True)
    try:
        for attempt in range(1, wait_settings.max_attempts + 1):
            try:
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
                print(f"[wait_for_db] PostgreSQL disponible en intento {attempt}.")
                return
            except SQLAlchemyError as exc:
                print(f"[wait_for_db] intento {attempt}/{wait_settings.max_attempts} fallido: {exc}")
                time.sleep(wait_settings.sleep_seconds)
        raise RuntimeError("No se pudo conectar a PostgreSQL dentro del tiempo esperado")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
