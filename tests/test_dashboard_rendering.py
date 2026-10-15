"""UI con consultas y bundles aislados; no conecta a PostgreSQL ni crea modelos."""
from contextlib import ExitStack
import logging
import os
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app.dashboard_data import ENERGY_COLUMNS


class DashboardRenderingTests(unittest.TestCase):
    def setUp(self):
        self.previous_logging = logging.root.manager.disable
        logging.disable(logging.WARNING)
        self.addCleanup(logging.disable, self.previous_logging)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.rows = pd.DataFrame({
            "id": [1, 2, 3],
            "timestamp": ["2026-10-15 10:00:00", "2026-10-15 10:00:01", "2026-10-15 10:00:02"],
            "corriente": [25.0, 40.0, 20.0], "voltaje": [220.0, 220.0, 220.0],
            "potencia_activa": [5.0, 8.0, 4.0], "temperatura_motor": [45.0, 46.0, 44.0],
            "estado": ["Normal", "Anomalo", "Normal"],
        })
        self.model = Mock()
        self.model.predict.side_effect = lambda values: np.where(values[:, 0] > 30, -1, 1)
        self.scaler = Mock()
        self.scaler.transform.side_effect = lambda values: values

    def render(self, data=False, model=False, rows=None, broken_model=False):
        rows = self.rows.copy() if rows is None else rows.copy()
        query = MagicMock()
        query.order_by.return_value = query
        query.limit.return_value = query
        query.count.return_value = len(rows) if data else 0
        query.scalar.return_value = None
        query.first.return_value = None  # Métricas históricas siempre ausentes.
        query.all.return_value = []
        session = MagicMock()
        session.query.return_value = query
        self.stack.enter_context(patch("core.database.SessionLocal", return_value=session))
        self.stack.enter_context(patch("core.database.engine", MagicMock()))
        self.read_sql = self.stack.enter_context(patch("pandas.read_sql", return_value=rows))
        exists, mtime = os.path.exists, os.path.getmtime
        self.stack.enter_context(patch("os.path.exists", side_effect=lambda p: model if str(p) == "models/anomaly_detector.pkl" else exists(p)))
        self.stack.enter_context(patch("os.path.getmtime", side_effect=lambda p: 0 if str(p) == "models/anomaly_detector.pkl" else mtime(p)))
        self.load = self.stack.enter_context(patch("core.model_artifact.load_model_bundle", return_value={
            "model": self.model, "scaler": self.scaler, "features": ENERGY_COLUMNS.copy(),
        }))
        if broken_model:
            self.load.side_effect = ValueError("Bundle inválido de prueba")
        self.at = AppTest.from_file("app/dashboard_app.py", default_timeout=25).run()
        self.assert_ui()
        return self.at

    def assert_ui(self):
        self.assertFalse(self.at.exception, [x.message for x in self.at.exception])
        self.assertFalse(self.at.error, [x.value for x in self.at.error])
        self.assertEqual([t.label for t in self.at.tabs], [
            "Analisis general", "Variables energeticas", "Comparativos", "Deteccion de anomalias", "Tabla de registros",
        ])
        for label in ("Entrenar modelo", "Recalcular metricas", "Generar dataset historico (Batch)", "Reiniciar base de datos"):
            self.assertTrue(next(b for b in self.at.button if b.label == label).disabled)
        self.model.fit.assert_not_called()

    def assert_empty_views(self):
        self.assertFalse(self.at.get("plotly_chart"))
        self.assertFalse(self.at.dataframe)
        for tab in self.at.tabs:
            self.assertTrue(tab.info)
        self.model.predict.assert_not_called()

    def test_no_data_no_model(self):
        self.render()
        self.assert_empty_views()
        self.read_sql.assert_not_called()
        self.load.assert_not_called()

    def test_data_without_model_keeps_energy_views(self):
        at = self.render(data=True)
        self.assertEqual(len(at.get("plotly_chart")), 4)
        self.assertEqual(len(at.dataframe[0].value), 3)
        self.assertTrue(at.tabs[3].info)
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(metrics["Registros originales filtrados"], "3")
        self.assertEqual(metrics["Anomalías en registros originales"], "N/A")
        self.model.predict.assert_not_called()

    def test_no_data_with_model(self):
        self.render(model=True)
        self.assert_empty_views()
        self.load.assert_called_once()

    def test_data_with_model_preserves_original_and_plot_denominators(self):
        at = self.render(data=True, model=True)
        self.assertEqual(len(at.get("plotly_chart")), 6)
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(metrics["Modelo entrenado"], "Si")
        self.assertEqual(metrics["Registros originales filtrados"], "3")
        self.assertEqual(metrics["Anomalías en registros originales filtrados"], "1")
        self.assertEqual(metrics["Puntos de la gráfica clasificados como anómalos"], "1")
        self.assertEqual(metrics["Registros originales detectados (%)"], "33.33%")

    def test_empty_filters_show_empty_states_even_with_model(self):
        at = self.render(data=True, model=True)
        self.model.predict.reset_mock()
        at.multiselect[0].set_value([])
        next(b for b in at.button if b.label == "Aplicar filtros").click().run()
        self.assert_ui()
        self.assert_empty_views()

    def test_aggregation_does_not_change_original_population_metrics(self):
        rows = pd.concat([self.rows] * 600, ignore_index=True)
        rows["id"] = range(1, 1801)
        rows["timestamp"] = pd.date_range("2026-10-15", periods=1800, freq="s")
        at = self.render(data=True, model=True, rows=rows)
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(metrics["Registros originales filtrados"], "1800")
        self.assertEqual(metrics["Anomalías en registros originales filtrados"], "600")
        self.assertEqual(metrics["Registros originales detectados (%)"], "33.33%")
        self.assertEqual(metrics["Puntos de la gráfica clasificados como anómalos"], "300")

    def test_null_features_do_not_block_monitoring_or_invent_predictions(self):
        rows = self.rows.copy()
        rows.loc[0, "voltaje"] = None
        at = self.render(data=True, model=True, rows=rows)
        self.assertTrue(at.tabs[3].info)
        self.assertTrue(at.get("plotly_chart"))
        self.assertEqual(len(at.dataframe[0].value), 3)
        self.model.predict.assert_not_called()

    def test_null_dates_and_values_keep_table_without_fictitious_charts(self):
        rows = self.rows.copy()
        rows["timestamp"] = None
        rows["estado"] = None
        rows[ENERGY_COLUMNS] = None
        at = self.render(data=True, rows=rows)
        self.assertFalse(at.get("plotly_chart"))
        self.assertEqual(len(at.dataframe[0].value), 3)
        self.assertTrue(at.tabs[1].info)
        self.assertTrue(at.tabs[2].info)

    def test_invalid_model_does_not_hide_energy_views(self):
        at = self.render(data=True, model=True, broken_model=True)
        self.assertTrue(at.warning)
        self.assertEqual(len(at.get("plotly_chart")), 4)
        self.assertEqual(len(at.dataframe[0].value), 3)
        self.model.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
