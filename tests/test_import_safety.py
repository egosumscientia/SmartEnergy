import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class ImportSafetyTests(unittest.TestCase):
    def test_operational_modules_import_without_running_operations(self):
        script = """
import importlib
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

from sqlalchemy import MetaData
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sklearn.ensemble import IsolationForest
import joblib
import streamlit
import plotly.io as pio
import core.model_artifact as artifact

template_default = pio.templates.default

def fail(*args, **kwargs):
    raise AssertionError("Importó y ejecutó una operación")

with ExitStack() as stack:
    for target, name in (
        (MetaData, "create_all"),
        (Engine, "connect"),
        (Session, "add"),
        (Session, "bulk_save_objects"),
        (Session, "commit"),
        (Session, "execute"),
        (IsolationForest, "fit"),
        (joblib, "dump"),
        (artifact, "load_model_bundle"),
        (artifact, "save_model_bundle"),
        (subprocess, "Popen"),
        (subprocess, "run"),
        (time, "sleep"),
        (streamlit, "set_page_config"),
        (streamlit, "markdown"),
        (streamlit, "title"),
        (streamlit, "tabs"),
    ):
        stack.enter_context(patch.object(target, name, side_effect=fail))
    for module in (
        "scripts.simulate_data",
        "scripts.reset_db",
        "scripts.stream_realtime",
        "scripts.auto_train_loop",
        "scripts.seed_db",
        "scripts.wait_for_db",
        "ml.train_model",
        "ml.detect_anomalies",
        "app.dashboard_app",
    ):
        importlib.import_module(module)

assert pio.templates.default == template_default
assert "dark_custom" not in pio.templates
assert not Path("models").exists()
assert not Path("import_safety.db").exists()
"""
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env.pop("DATABASE_URL", None)
            env["DATABASE_URL"] = "sqlite:///import_safety.db"
            result = subprocess.run(
                [sys.executable, "-B", "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r});\n{script}"],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertFalse((Path(directory) / "models").exists())
            self.assertFalse((Path(directory) / "import_safety.db").exists())

    def test_batch_entrypoint_still_creates_and_inserts_records(self):
        from scripts import simulate_data

        session = MagicMock()
        session.query.return_value.count.return_value = 0
        with (
            patch.object(simulate_data.Base.metadata, "create_all") as create_all,
            patch.object(simulate_data, "SessionLocal", return_value=session),
        ):
            simulate_data.main()

        create_all.assert_called_once_with(bind=simulate_data.engine)
        records = session.bulk_save_objects.call_args.args[0]
        self.assertEqual(len(records), 10000)
        session.commit.assert_called_once()
        session.close.assert_called_once()

    def test_reset_entrypoint_still_truncates_then_runs_batch_command(self):
        from scripts import reset_db

        engine = MagicMock()
        with (
            patch.object(reset_db, "engine", engine),
            patch.object(reset_db.subprocess, "run") as run,
        ):
            reset_db.main()

        connection = engine.connect.return_value.__enter__.return_value
        self.assertIn("TRUNCATE TABLE registros, metricas", str(connection.execute.call_args.args[0]))
        connection.commit.assert_called_once()
        run.assert_called_once_with(
            [reset_db.sys.executable, "-m", "scripts.simulate_data"],
            check=True,
        )

    def test_training_entrypoint_still_fits_and_publishes_bundle(self):
        from ml import train_model

        features = ["corriente", "voltaje", "potencia_activa", "temperatura_motor"]
        data = pd.DataFrame({
            "corriente": [25.0, 40.0],
            "voltaje": [220.0, 210.0],
            "potencia_activa": [5.0, 8.0],
            "temperatura_motor": [45.0, 60.0],
            "estado": ["Normal", "Anomalo"],
        })
        session = MagicMock()
        scaler = MagicMock()
        scaler.fit_transform.return_value = data[features].values
        model = MagicMock()
        model.predict.return_value = np.array([1, -1])

        with (
            patch.object(train_model, "SessionLocal", return_value=session),
            patch.object(train_model.pd, "read_sql", return_value=data),
            patch.object(train_model, "StandardScaler", return_value=scaler),
            patch.object(train_model, "IsolationForest", return_value=model) as forest,
            patch.object(train_model, "save_model_bundle") as save_bundle,
            patch.object(train_model.os, "makedirs") as makedirs,
        ):
            train_model.main()

        makedirs.assert_called_once_with("models", exist_ok=True)
        forest.assert_called_once_with(
            n_estimators=200,
            contamination=0.2,
            random_state=42,
            n_jobs=-1,
        )
        model.fit.assert_called_once()
        save_bundle.assert_called_once()
        self.assertEqual(save_bundle.call_args.args[0]["features"], features)
        session.close.assert_called_once()

    def test_detection_entrypoint_still_persists_metrics(self):
        from ml import detect_anomalies

        features = ["corriente", "voltaje", "potencia_activa", "temperatura_motor"]
        data = pd.DataFrame({
            "id": [1, 2],
            "corriente": [25.0, 40.0],
            "voltaje": [220.0, 210.0],
            "potencia_activa": [5.0, 8.0],
            "temperatura_motor": [45.0, 60.0],
            "estado": ["Anomalo", "Normal"],
        })
        session = MagicMock()
        model = MagicMock()
        model.predict.return_value = np.array([-1, 1])
        scaler = MagicMock()
        scaler.transform.return_value = data[features].values

        with (
            patch.object(detect_anomalies, "SessionLocal", return_value=session),
            patch.object(detect_anomalies.os.path, "exists", return_value=True),
            patch.object(
                detect_anomalies,
                "load_model_bundle",
                return_value={"model": model, "scaler": scaler, "features": features},
            ),
            patch.object(detect_anomalies.pd, "read_sql", return_value=data),
        ):
            detect_anomalies.main()

        metric = session.add.call_args.args[0]
        self.assertIsInstance(metric, detect_anomalies.Metrica)
        self.assertEqual((metric.total, metric.anomalias, metric.porcentaje, metric.estado), (2, 1, 50.0, "Normal"))
        session.commit.assert_called_once()
        session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
