from datetime import datetime, timezone
import unittest

from sqlalchemy import DateTime, Float, Integer, String

from smartenergy.infrastructure.database import Base
from smartenergy.infrastructure.models import Metrica, Registro


class ModelTests(unittest.TestCase):
    def test_registro_creation(self):
        record = Registro(
            corriente=25.0,
            voltaje=220.0,
            potencia_activa=4.5,
            temperatura_motor=45.0,
            estado="Normal",
        )
        self.assertEqual(record.corriente, 25.0)
        self.assertEqual(record.voltaje, 220.0)
        self.assertEqual(record.potencia_activa, 4.5)
        self.assertEqual(record.temperatura_motor, 45.0)
        self.assertEqual(record.estado, "Normal")

    def test_registro_anomalo(self):
        record = Registro(
            corriente=80.0,
            voltaje=120.0,
            potencia_activa=12.0,
            temperatura_motor=130.0,
            estado="Anomalo",
        )
        self.assertEqual(record.estado, "Anomalo")

    def test_metrica_creation(self):
        metric = Metrica(
            total=10000,
            anomalias=150,
            porcentaje=1.5,
            estado="Normal",
            fecha=datetime.now(timezone.utc),
        )
        self.assertEqual(metric.total, 10000)
        self.assertEqual(metric.anomalias, 150)
        self.assertEqual(metric.porcentaje, 1.5)
        self.assertEqual(metric.estado, "Normal")

    def test_metrica_estados(self):
        for state in ("Normal", "Precaucion", "Critico"):
            with self.subTest(state=state):
                self.assertEqual(Metrica(total=100, anomalias=10, porcentaje=10.0, estado=state).estado, state)

    def test_orm_schema_contract_is_unchanged_and_uses_one_metadata(self):
        self.assertIs(Registro.metadata, Base.metadata)
        self.assertIs(Metrica.metadata, Base.metadata)
        self.assertEqual(Registro.__tablename__, "registros")
        self.assertEqual(Metrica.__tablename__, "metricas")
        self.assertEqual(
            [column.name for column in Registro.__table__.columns],
            [
                "id",
                "timestamp",
                "corriente",
                "voltaje",
                "potencia_activa",
                "temperatura_motor",
                "estado",
            ],
        )
        self.assertEqual(
            [column.name for column in Metrica.__table__.columns],
            ["id", "total", "anomalias", "porcentaje", "estado", "fecha"],
        )
        self.assertIsInstance(Registro.__table__.c.id.type, Integer)
        self.assertTrue(Registro.__table__.c.id.primary_key)
        self.assertTrue(Registro.__table__.c.id.index)
        self.assertIsInstance(Registro.__table__.c.timestamp.type, DateTime)
        self.assertIsNotNone(Registro.__table__.c.timestamp.default)
        for name in ("corriente", "voltaje", "potencia_activa", "temperatura_motor"):
            with self.subTest(column=name):
                self.assertIsInstance(Registro.__table__.c[name].type, Float)
        self.assertIsInstance(Registro.__table__.c.estado.type, String)
        self.assertIsInstance(Metrica.__table__.c.id.type, Integer)
        self.assertTrue(Metrica.__table__.c.id.primary_key)
        self.assertTrue(Metrica.__table__.c.id.index)
        for name in ("total", "anomalias"):
            with self.subTest(column=name):
                self.assertIsInstance(Metrica.__table__.c[name].type, Integer)
        self.assertIsInstance(Metrica.__table__.c.porcentaje.type, Float)
        self.assertIsInstance(Metrica.__table__.c.estado.type, String)
        self.assertIsInstance(Metrica.__table__.c.fecha.type, DateTime)
        self.assertIsNotNone(Metrica.__table__.c.fecha.default)


if __name__ == "__main__":
    unittest.main()
