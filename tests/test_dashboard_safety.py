import ast
from datetime import date
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from app.dashboard_data import filter_records, recent_records, sample_records


class DashboardSafetyTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "id": [3, 1, 2],
            "timestamp": pd.to_datetime(["2026-10-15 12:00", "2026-10-15 11:00", "2026-10-15 12:00"]),
            "estado": ["Normal", "Anomalo", "Normal"],
        })
        self.dates = [date(2026, 10, 15)] * 2

    def test_empty_states_and_incomplete_dates_return_no_records(self):
        self.assertTrue(filter_records(self.df, self.dates, []).empty)
        self.assertTrue(filter_records(self.df, self.dates[:1], ["Normal"]).empty)

    def test_filter_keeps_selected_states_and_dates_without_mutation(self):
        self.assertEqual(filter_records(self.df, self.dates, ["Normal"]).id.tolist(), [2, 3])
        self.assertEqual(len(self.df), 3)
        self.assertTrue(filter_records(self.df, [date(2026, 10, 16)] * 2, ["Normal"]).empty)

    def test_sampling_limit_at_boundary_and_nonmultiples(self):
        for count in (0, 5000, 5001, 9999, 10000, 10001):
            df = pd.DataFrame({"id": range(count), "timestamp": pd.date_range("2026-01-01", periods=count, freq="s")})
            sample = sample_records(df)
            self.assertLessEqual(len(sample), 5000)
            self.assertTrue(sample.timestamp.is_monotonic_increasing)
            self.assertEqual(len(df), count)

    def test_recent_order_and_csv_population_are_independent(self):
        filtered = filter_records(self.df, self.dates, ["Normal", "Anomalo"])
        self.assertEqual(recent_records(filtered, 2).id.tolist(), [3, 2])
        self.assertEqual(len(filtered.to_csv(index=False).splitlines()), 4)

    def test_actual_handler_rejects_all_admin_commands(self):
        # Importar el dashboard ejecutaría la UI; se prueba su manejador real.
        tree = ast.parse(Path("app/dashboard_app.py").read_text())
        handler = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_script")
        namespace = {"st": Mock()}
        exec(compile(ast.Module(body=[handler], type_ignores=[]), "dashboard-handler", "exec"), namespace)
        with patch("subprocess.run", side_effect=AssertionError("No debe ejecutarse ningún proceso")) as process:
            for module in ("ml.train_model", "ml.detect_anomalies", "scripts.simulate_data", "scripts.reset_db"):
                self.assertIsInstance(namespace["run_script"](["python", "-m", module]), PermissionError)
            process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
