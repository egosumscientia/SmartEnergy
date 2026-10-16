import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from sqlalchemy.exc import OperationalError

from smartenergy.application import monitoring


class MonitoringApplicationTests(unittest.TestCase):
    def test_monitoring_queries_use_infrastructure_and_prepare_records(self):
        rows = pd.DataFrame(
            {
                "id": [1],
                "timestamp": ["2026-10-15 12:00:00"],
                "corriente": [25],
                "voltaje": [220],
                "potencia_activa": [5],
                "temperatura_motor": [45],
                "estado": ["Anomalo"],
            }
        )
        factory = MagicMock()
        metric = SimpleNamespace(
            total=10,
            anomalias=2,
            porcentaje=20.0,
            estado="Precaucion",
            fecha="fecha",
        )
        with (
            patch.object(monitoring, "get_session_factory", return_value=factory),
            patch.object(monitoring.persistence, "load_records", return_value=rows),
            patch.object(monitoring.persistence, "load_latest_metric", return_value=metric),
            patch.object(monitoring.persistence, "get_record_count", return_value=1),
        ):
            records = monitoring.load_records()
            latest_metric = monitoring.load_latest_metrics()
            count = monitoring.get_data_count()

        self.assertEqual(records.estado.tolist(), ["Anomalo"])
        self.assertEqual(records.label_anomalia.tolist(), [1])
        self.assertEqual(latest_metric["total"], 10)
        self.assertEqual(count, 1)

    def test_database_unavailable_states_are_explicit_and_queries_do_not_invent_data(self):
        with patch.object(monitoring, "get_engine", return_value=None):
            self.assertEqual(
                monitoring.check_database_connection(),
                (False, "DATABASE_URL no configurada en .env"),
            )
        with patch.object(monitoring, "get_session_factory", return_value=None):
            self.assertEqual(monitoring.get_data_count(), 0)
            self.assertIsNone(monitoring.load_records())
            self.assertIsNone(monitoring.load_latest_metrics())

    def test_count_failure_keeps_legacy_empty_count_while_connection_reports_failure(self):
        factory = MagicMock()
        with (
            patch.object(monitoring, "get_session_factory", return_value=factory),
            patch.object(
                monitoring.persistence,
                "get_record_count",
                side_effect=OperationalError("count", {}, RuntimeError("offline")),
            ),
        ):
            self.assertEqual(monitoring.get_data_count(), 0)


if __name__ == "__main__":
    unittest.main()
