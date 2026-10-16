import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]


class ImportSafetyTests(unittest.TestCase):
    def test_operational_and_interface_modules_import_without_side_effects(self):
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

import smartenergy.config as config
import smartenergy.infrastructure.database as database
import smartenergy.infrastructure.model_artifact as artifact

template_default = pio.templates.default

def fail(*args, **kwargs):
    raise AssertionError("Importó y ejecutó una operación")

with ExitStack() as stack:
    stack.enter_context(patch.object(config, "load_settings", side_effect=fail))
    stack.enter_context(patch.object(database, "load_settings", side_effect=fail))
    stack.enter_context(patch.object(database, "create_engine", side_effect=fail))
    stack.enter_context(patch("dotenv.load_dotenv", side_effect=fail))
    for target, name in (
        (MetaData, "create_all"),
        (Engine, "connect"),
        (Session, "add"),
        (Session, "bulk_save_objects"),
        (Session, "commit"),
        (Session, "execute"),
        (IsolationForest, "fit"),
        (joblib, "dump"),
        (joblib, "load"),
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
        "smartenergy.application.monitoring",
        "smartenergy.application.model_jobs",
        "smartenergy.operations.simulate_data",
        "smartenergy.operations.reset_db",
        "smartenergy.operations.stream_realtime",
        "smartenergy.operations.auto_train_loop",
        "smartenergy.operations.seed_db",
        "smartenergy.operations.wait_for_db",
        "smartenergy.operations.train_model",
        "smartenergy.operations.detect_anomalies",
        "app.dashboard_data",
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
            env["DATABASE_URL"] = "sqlite:///import_safety.db"
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    f"import sys; sys.path.insert(0, {str(ROOT)!r});\n{script}",
                ],
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

    def test_monitoring_application_imports_without_streamlit(self):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
                "import smartenergy.application.monitoring; "
                "assert 'streamlit' not in sys.modules",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dashboard_and_monitoring_import_without_scikit_learn(self):
        script = """
import importlib.abc
import sys

class BlockScikitLearn(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sklearn" or fullname.startswith("sklearn."):
            raise ModuleNotFoundError("scikit-learn is not installed")
        return None

sys.meta_path.insert(0, BlockScikitLearn())
import smartenergy.application.monitoring
import app.dashboard_app
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r});\n{script}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_batch_operation_keeps_generation_and_retention_contract(self):
        from smartenergy.operations import simulate_data

        engine = MagicMock()
        session = MagicMock()
        session.query.return_value.count.return_value = 0
        output = StringIO()
        with (
            patch.object(simulate_data, "get_engine", return_value=engine),
            patch.object(simulate_data, "get_session_factory", return_value=lambda: session),
            patch.object(simulate_data.Base.metadata, "create_all") as create_all,
            redirect_stdout(output),
        ):
            simulate_data.main()

        create_all.assert_called_once_with(bind=engine)
        self.assertEqual(len(session.bulk_save_objects.call_args.args[0]), 10000)
        session.commit.assert_called_once()
        session.close.assert_called_once()
        self.assertIn("10000", output.getvalue())

    def test_reset_operation_truncates_then_delegates_to_canonical_batch_entrypoint(self):
        from smartenergy.config import REPOSITORY_ROOT
        from smartenergy.operations import reset_db

        engine = MagicMock()
        with (
            patch.object(reset_db, "get_engine", return_value=engine),
            patch.object(reset_db.subprocess, "run") as run,
            patch.object(reset_db.sys, "executable", "/venv/bin/python"),
            redirect_stdout(StringIO()),
        ):
            reset_db.main()

        connection = engine.begin.return_value.__enter__.return_value
        self.assertIn("TRUNCATE TABLE registros, metricas", str(connection.execute.call_args.args[0]))
        run.assert_called_once_with(
            ["/venv/bin/python", "-m", "smartenergy.operations.simulate_data"],
            check=True,
            cwd=REPOSITORY_ROOT,
        )


if __name__ == "__main__":
    unittest.main()
