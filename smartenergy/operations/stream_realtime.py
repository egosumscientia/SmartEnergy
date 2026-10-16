"""Generacion continua explicita de mediciones simuladas."""
import argparse
import random
import time
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import text

from smartenergy.infrastructure.database import Base, get_engine, get_session_factory
from smartenergy.infrastructure.models import Registro


def build_sample(ts: datetime, anomaly_ratio: float) -> Registro:
    corriente = np.random.normal(25, 3)
    voltaje = np.random.normal(220, 5)
    temperatura = np.random.normal(45, 2)
    potencia = corriente * voltaje * np.random.uniform(0.8, 0.95) / 1000

    any_anomaly = False
    if random.random() < anomaly_ratio:
        factor = np.random.uniform(2.5, 4.0)
        corriente *= factor
        any_anomaly = True
    if random.random() < anomaly_ratio:
        factor = np.random.uniform(0.4, 0.6)
        voltaje *= factor
        any_anomaly = True
    if random.random() < anomaly_ratio:
        temperatura += np.random.uniform(50, 100)
        any_anomaly = True
    if random.random() < anomaly_ratio:
        corriente *= 0.5
        potencia *= np.random.uniform(2.0, 3.0)
        any_anomaly = True
    if random.random() < anomaly_ratio / 2:
        corriente += np.random.normal(0, 20)
        voltaje += np.random.normal(0, 40)
        any_anomaly = True

    potencia = abs(corriente * voltaje * 0.9 / 1000)
    estado = "Anomalo" if any_anomaly else "Normal"
    return Registro(
        timestamp=ts,
        corriente=float(corriente),
        voltaje=float(voltaje),
        potencia_activa=float(potencia),
        temperatura_motor=float(temperatura),
        estado=estado,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Genera datos en tiempo real y los inserta continuamente."
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Segundos entre ciclos")
    parser.add_argument("--batch-size", type=int, default=5, help="Registros por ciclo")
    parser.add_argument(
        "--anomaly-ratio",
        type=float,
        default=0.12,
        help="Probabilidad de que un registro sea anomalo (0-1)",
    )
    parser.add_argument(
        "--max-registros", type=int, default=20000, help="Limite total de registros"
    )
    args = parser.parse_args()

    engine = get_engine()
    session_factory = get_session_factory()
    if engine is None or session_factory is None:
        raise RuntimeError("DATABASE_URL no esta configurada o el engine no se pudo crear.")

    Base.metadata.create_all(bind=engine)
    session = session_factory()
    inserted = 0
    cycle = 0
    try:
        while True:
            cycle += 1
            start_ts = datetime.now()
            batch = [
                build_sample(
                    start_ts + timedelta(milliseconds=idx * (1000 / max(args.batch_size, 1))),
                    args.anomaly_ratio,
                )
                for idx in range(args.batch_size)
            ]
            session.bulk_save_objects(batch)
            session.commit()
            inserted += len(batch)

            total = session.query(Registro).count()
            if total > args.max_registros:
                excess = total - args.max_registros
                print(
                    f"[{datetime.now().isoformat()}] Limite de {args.max_registros} superado. "
                    f"Eliminando {excess} registros antiguos."
                )
                session.execute(
                    text(
                        "DELETE FROM registros WHERE id IN "
                        "(SELECT id FROM registros ORDER BY id ASC LIMIT :lim)"
                    ),
                    {"lim": excess},
                )
                session.commit()
                total -= excess

            if cycle % 10 == 0:
                print(
                    f"[{datetime.now().isoformat()}] Ciclo {cycle} -> insertados totales "
                    f"en esta sesion: {inserted} | total en BD: {total}"
                )

            time.sleep(max(args.interval, 0.1))
    except KeyboardInterrupt:
        print("Interrumpido por el usuario.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
