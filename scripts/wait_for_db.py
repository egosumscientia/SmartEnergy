import os
import time
from sqlalchemy import create_engine, text


def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL no esta configurada")

    max_attempts = int(os.getenv("DB_MAX_ATTEMPTS", "60"))
    sleep_seconds = float(os.getenv("DB_SLEEP_SECONDS", "2"))

    engine = create_engine(database_url, pool_pre_ping=True)
    for attempt in range(1, max_attempts + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"[wait_for_db] PostgreSQL disponible en intento {attempt}.")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[wait_for_db] intento {attempt}/{max_attempts} fallido: {exc}")
            time.sleep(sleep_seconds)

    raise RuntimeError("No se pudo conectar a PostgreSQL dentro del tiempo esperado")


if __name__ == "__main__":
    main()
