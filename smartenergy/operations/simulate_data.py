"""Generacion batch explicita de mediciones simuladas."""
import random
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import text

from smartenergy.infrastructure.database import Base, get_engine, get_session_factory
from smartenergy.infrastructure.models import Registro


def main():
    num_samples = 10000
    anomaly_ratio = random.uniform(0.10, 0.20)
    sampling_interval = 1
    max_records = 20000

    engine = get_engine()
    session_factory = get_session_factory()
    if engine is None or session_factory is None:
        raise RuntimeError("DATABASE_URL no esta configurada o el engine no se pudo crear.")

    Base.metadata.create_all(bind=engine)
    session = session_factory()
    try:
        print("=== Generando datos simulados de energia ===")

        start_time = datetime.now() - timedelta(seconds=(num_samples - 1) * sampling_interval)
        timestamps = [
            start_time + timedelta(seconds=i * sampling_interval) for i in range(num_samples)
        ]
        corriente = np.random.normal(25, 3, num_samples)
        voltaje = np.random.normal(220, 5, num_samples)
        temperatura = np.random.normal(45, 2, num_samples)
        estado_maquina = np.random.choice([0, 1], num_samples, p=[0.1, 0.9])
        potencia = corriente * voltaje * np.random.uniform(0.8, 0.95, num_samples) / 1000

        labels = np.zeros(num_samples)
        num_anom = int(num_samples * anomaly_ratio)
        anom_indices = np.random.choice(num_samples, num_anom, replace=False)
        print(f"Generando {num_anom} anomalias garantizadas...")

        for idx in anom_indices:
            tipo = random.choice(
                [
                    "corriente_alta",
                    "voltaje_bajo",
                    "temperatura_alta",
                    "potencia_irregular",
                    "ruido_extremo",
                    "fallo_comb_inicial",
                ]
            )

            if tipo == "corriente_alta":
                vals_extra = np.random.uniform(2.5, 4.0)
                corriente[idx] *= vals_extra
                potencia[idx] = corriente[idx] * voltaje[idx] * 0.9 / 1000
            elif tipo == "voltaje_bajo":
                vals_extra = np.random.uniform(0.4, 0.6)
                voltaje[idx] *= vals_extra
                potencia[idx] = corriente[idx] * voltaje[idx] * 0.9 / 1000
            elif tipo == "temperatura_alta":
                temperatura[idx] += np.random.uniform(50, 100)
            elif tipo == "potencia_irregular":
                corriente[idx] *= 0.5
                potencia[idx] *= np.random.uniform(2.0, 3.0)
            elif tipo == "ruido_extremo":
                corriente[idx] += np.random.normal(0, 20)
                voltaje[idx] += np.random.normal(0, 40)
                potencia[idx] = abs(corriente[idx] * voltaje[idx] * 0.9 / 1000)
            elif tipo == "fallo_comb_inicial":
                corriente[idx] *= np.random.uniform(2.0, 3.0)
                temperatura[idx] += np.random.uniform(20, 40)
                potencia[idx] = corriente[idx] * voltaje[idx] * 0.9 / 1000

            labels[idx] = 1

        registros = [
            Registro(
                timestamp=timestamps[i],
                corriente=float(corriente[i]),
                voltaje=float(voltaje[i]),
                potencia_activa=float(potencia[i]),
                temperatura_motor=float(temperatura[i]),
                estado="Anomalo" if labels[i] == 1 else "Normal",
            )
            for i in range(num_samples)
        ]

        session.bulk_save_objects(registros)
        session.commit()

        total = session.query(Registro).count()
        if total > max_records:
            excess = total - max_records
            print(f"Se supero el limite ({max_records}). Eliminando {excess} registros antiguos...")
            session.execute(
                text(
                    "DELETE FROM registros WHERE id IN "
                    "(SELECT id FROM registros ORDER BY id ASC LIMIT :lim)"
                ),
                {"lim": excess},
            )
            session.commit()

        print(f"{num_samples} registros insertados en la base de datos.")
        print(f"Nivel de anomalias: {anomaly_ratio * 100:.1f}% ({num_anom} de {num_samples})")
        print(f"Total actual en base: {min(total, max_records)} registros.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
