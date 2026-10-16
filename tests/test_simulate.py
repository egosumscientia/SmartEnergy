from datetime import datetime
import unittest

from smartenergy.infrastructure.models import Registro
from smartenergy.operations.stream_realtime import build_sample


class SimulationTests(unittest.TestCase):
    def test_build_sample_returns_registro(self):
        timestamp = datetime.now()
        sample = build_sample(timestamp, anomaly_ratio=0.0)
        self.assertIsInstance(sample, Registro)
        self.assertEqual(sample.timestamp, timestamp)
        self.assertEqual(sample.estado, "Normal")

    def test_build_sample_normal_ranges(self):
        sample = build_sample(datetime.now(), anomaly_ratio=0.0)
        self.assertGreater(sample.corriente, 0)
        self.assertLess(sample.corriente, 100)
        self.assertGreater(sample.voltaje, 100)
        self.assertLess(sample.voltaje, 350)
        self.assertGreater(sample.potencia_activa, 0)
        self.assertGreater(sample.temperatura_motor, 30)
        self.assertLess(sample.temperatura_motor, 60)

    def test_build_sample_high_anomaly_ratio(self):
        anomaly_count = sum(
            build_sample(datetime.now(), anomaly_ratio=1.0).estado == "Anomalo"
            for _ in range(200)
        )
        self.assertEqual(anomaly_count, 200)


if __name__ == "__main__":
    unittest.main()
