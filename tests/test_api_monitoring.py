import asyncio
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import httpx
import numpy as np
import pandas as pd
from sqlalchemy.exc import OperationalError

from smartenergy.analysis.measurements import ENERGY_COLUMNS, prepare_records
from smartenergy.application import monitoring
from smartenergy.interfaces.api.main import create_app
from smartenergy.interfaces.api.routes import router


class MonitoringApiTests(unittest.TestCase):
    def setUp(self):
        self.rows = prepare_records(
            pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "timestamp": [
                        "2026-10-15 10:00:00",
                        "2026-10-15 10:00:01",
                        "2026-10-15 10:00:02",
                    ],
                    "corriente": [25.0, 40.0, 20.0],
                    "voltaje": [220.0, 220.0, 220.0],
                    "potencia_activa": [5.0, 8.0, 4.0],
                    "temperatura_motor": [45.0, 46.0, 44.0],
                    "estado": ["Normal", "Anomalo", "Normal"],
                }
            )
        )

    def request(self, path):
        async def run_request():
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                return await client.get(path)

        return asyncio.run(run_request())

    def mock_common(self, record_count=0, records=None, model=False):
        stack = self.enterContext
        stack(patch.object(monitoring, "get_data_count", return_value=record_count))
        stack(
            patch.object(
                monitoring,
                "load_records",
                return_value=records,
            )
        )
        stack(
            patch.object(
                monitoring,
                "get_model_artifact_status",
                return_value=(model, datetime(2026, 10, 15, 12) if model else None),
            )
        )
        stack(patch.object(monitoring, "load_latest_metrics", return_value=None))

    def test_no_data_no_model_returns_clear_empty_state(self):
        self.mock_common()
        response = self.request("/api/v1/monitoring")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"]["status"], "empty")
        self.assertEqual(payload["data"]["record_count"], 0)
        self.assertEqual(payload["model"]["status"], "absent")
        self.assertEqual(payload["model"]["evaluation"]["reason"], "model_absent")
        self.assertEqual(payload["chart"]["points"], [])
        self.assertEqual(payload["table"]["records"], [])

    def test_data_without_model_keeps_measurements_available(self):
        self.mock_common(record_count=3, records=self.rows)
        response = self.request("/api/v1/monitoring")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["data"]["filtered_count"], 3)
        self.assertEqual(payload["model"]["status"], "absent")
        self.assertEqual(payload["chart"]["point_count"], 3)
        self.assertIsNone(payload["population"]["coverage_percent"])
        self.assertEqual(payload["data"]["available_states"], ["Normal", "Anomalo"])

    def test_no_data_with_model_keeps_model_optional(self):
        self.mock_common(model=True)
        model = Mock()
        scaler = Mock()
        with (
            patch.object(
                monitoring,
                "load_model",
                return_value=(model, scaler, ENERGY_COLUMNS, None),
            )
        ):
            response = self.request("/api/v1/monitoring")

        payload = response.json()
        self.assertEqual(payload["data"]["status"], "empty")
        self.assertEqual(payload["model"]["status"], "available")
        self.assertEqual(payload["model"]["evaluation"]["reason"], "no_data")
        model.predict.assert_not_called()

    def test_data_with_model_preserves_original_denominator_and_chart_predictions(self):
        self.mock_common(record_count=3, records=self.rows, model=True)
        model = Mock()
        model.predict.side_effect = lambda values: np.where(values[:, 0] > 30, -1, 1)
        scaler = Mock()
        scaler.transform.side_effect = lambda values: values
        with (
            patch.object(
                monitoring,
                "load_model",
                return_value=(model, scaler, ["corriente"], None),
            )
        ):
            response = self.request("/api/v1/monitoring")

        payload = response.json()
        evaluation = payload["model"]["evaluation"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(evaluation["total"], 3)
        self.assertEqual(evaluation["detected"], 1)
        self.assertEqual(evaluation["coverage"], 100 / 3)
        self.assertEqual(payload["population"]["anomalies_detected"], 1)
        self.assertEqual(payload["chart"]["prediction_status"], "available")
        self.assertEqual(payload["table"]["records"][0]["id"], 3)
        self.assertEqual(payload["chart"]["histograms"]["corriente"][0]["state"], "Normal")

    def test_aggregation_keeps_original_population_denominator(self):
        rows = pd.concat([self.rows] * 600, ignore_index=True)
        rows["id"] = range(1, 1801)
        rows["timestamp"] = pd.date_range("2026-10-15", periods=1800, freq="s")
        self.mock_common(record_count=1800, records=rows, model=True)
        model, scaler = Mock(), Mock()
        scaler.transform.side_effect = lambda values: values
        model.predict.side_effect = lambda values: np.where(values[:, 0] > 30, -1, 1)
        with patch.object(monitoring, "load_model", return_value=(model, scaler, ENERGY_COLUMNS, None)):
            payload = self.request("/api/v1/monitoring").json()
        self.assertEqual(payload["data"]["filtered_count"], 1800)
        self.assertEqual(payload["model"]["evaluation"]["detected"], 600)
        self.assertAlmostEqual(payload["model"]["evaluation"]["coverage"], 100 / 3)
        self.assertLess(payload["chart"]["point_count"], 1800)

    def test_invalid_model_and_null_features_keep_measurements(self):
        self.rows.loc[0, "voltaje"] = None
        self.mock_common(record_count=3, records=self.rows, model=True)
        model, scaler = Mock(), Mock()
        with patch.object(monitoring, "load_model", return_value=(model, scaler, ENERGY_COLUMNS, None)):
            payload = self.request("/api/v1/monitoring").json()
        self.assertEqual(len(payload["table"]["records"]), 3)
        self.assertEqual(payload["model"]["evaluation"]["reason"], "incomplete_features")
        model.predict.assert_not_called()
        with patch.object(monitoring, "load_model", return_value=(None, None, None, ValueError("invalid"))):
            payload = self.request("/api/v1/monitoring").json()
        self.assertEqual(payload["model"]["status"], "unavailable")
        self.assertEqual(len(payload["table"]["records"]), 3)

    def test_null_values_keep_table_without_fictitious_chart_points(self):
        self.rows["timestamp"] = None
        self.rows[ENERGY_COLUMNS] = None
        self.mock_common(record_count=3, records=self.rows)
        payload = self.request("/api/v1/monitoring").json()
        self.assertEqual(len(payload["table"]["records"]), 3)
        self.assertEqual(payload["chart"]["points"], [])

    def test_empty_state_filters_and_validation_errors_are_distinct(self):
        self.mock_common(record_count=3, records=self.rows)
        empty = self.request("/api/v1/monitoring?states=")
        invalid = self.request("/api/v1/monitoring?date_from=2026-10-15")
        unknown_state = self.request("/api/v1/monitoring?states=Unexpected")

        self.assertEqual(empty.status_code, 200)
        self.assertEqual(empty.json()["data"]["filtered_count"], 0)
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.json()["error"]["code"], "validation_error")
        self.assertEqual(
            invalid.json()["error"]["details"][0]["reason"], "required_together"
        )
        self.assertEqual(unknown_state.status_code, 422)
        self.assertEqual(
            unknown_state.json()["error"]["details"][0]["field"], "states"
        )
        self.assertNotIn("detail", invalid.json())

    def test_primary_database_failures_are_not_reported_as_empty_data(self):
        with patch.object(
            monitoring,
            "get_data_count",
            side_effect=OperationalError("count", {}, RuntimeError("offline")),
        ):
            response = self.request("/api/v1/monitoring")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "database_unavailable")
        self.assertNotIn("offline", response.text)

    def test_missing_database_configuration_is_unavailable_not_empty(self):
        with patch.object(
            monitoring,
            "get_data_count",
            side_effect=monitoring.MonitoringUnavailableError(),
        ):
            response = self.request("/api/v1/monitoring")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "database_unavailable")

    def test_optional_metric_failure_does_not_block_monitoring(self):
        self.mock_common(record_count=3, records=self.rows)
        with (
            patch.object(
                monitoring,
                "load_latest_metrics",
                side_effect=OperationalError("metric", {}, RuntimeError("offline")),
            )
        ):
            response = self.request("/api/v1/monitoring")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["persisted_metrics"]["status"], "unavailable"
        )
        self.assertEqual(response.json()["data"]["filtered_count"], 3)

    def test_csv_contains_all_filtered_records_not_only_table_limit(self):
        self.mock_common(record_count=3, records=self.rows)
        response = self.request(
            "/api/v1/monitoring/records.csv?date_from=2026-10-15&date_to=2026-10-15"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/csv; charset=utf-8")
        self.assertIn("label_anomalia", response.text.splitlines()[0])
        self.assertEqual(len(response.text.splitlines()), 4)
        self.assertIn("smartenergy_filtered.csv", response.headers["content-disposition"])

    def test_frontend_is_served_and_api_has_no_write_routes(self):
        response = self.request("/")
        echarts = self.request("/vendor/echarts.min.js")
        license_response = self.request("/vendor/LICENSE")
        notice_response = self.request("/vendor/NOTICE")

        self.assertEqual(response.status_code, 200)
        self.assertIn("SmartEnergy by makeAutomatic", response.text)
        self.assertIn('src="/vendor/echarts.min.js"', response.text)
        self.assertEqual(echarts.status_code, 200)
        self.assertGreater(len(echarts.content), 100_000)
        self.assertEqual(license_response.status_code, 200)
        self.assertIn("Apache License", license_response.text)
        self.assertEqual(notice_response.status_code, 200)
        self.assertIn("Apache ECharts", notice_response.text)
        api_methods = {
            method
            for route in router.routes
            for method in getattr(route, "methods", set())
        }
        self.assertEqual(api_methods, {"GET"})

    def test_monitoring_rejects_writes_without_invoking_administration(self):
        async def requests():
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return [await client.request(method, path)
                        for method in ("POST", "PUT", "PATCH", "DELETE")
                        for path in ("/api/v1/monitoring", "/api/v1/monitoring/records.csv")]
        with (
            patch("smartenergy.application.model_jobs.train_model", side_effect=AssertionError("No training")),
            patch("smartenergy.application.model_jobs.detect_anomalies", side_effect=AssertionError("No writes")),
            patch.object(monitoring, "get_data_count", side_effect=AssertionError("No query")),
        ):
            responses = asyncio.run(requests())
        self.assertTrue(all(response.status_code == 405 for response in responses))

    def test_server_shutdown_closes_database_engine(self):
        app = create_app()
        with patch("smartenergy.interfaces.api.main.dispose_engine") as dispose:
            async def close_lifespan():
                async with app.router.lifespan_context(app):
                    pass

            asyncio.run(close_lifespan())

        dispose.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
