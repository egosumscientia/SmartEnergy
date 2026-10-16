from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from smartenergy.application import model_jobs
from smartenergy.config import Settings


class ModelJobTests(unittest.TestCase):
    def test_training_job_reads_inputs_trains_and_publishes_one_bundle(self):
        data = pd.DataFrame(
            {
                "corriente": [25.0, 40.0],
                "voltaje": [220.0, 210.0],
                "potencia_activa": [5.0, 8.0],
                "temperatura_motor": [45.0, 60.0],
                "estado": ["Normal", "Anomalo"],
            }
        )
        bundle = {"model": object(), "scaler": object(), "features": ["corriente"]}
        metrics = {
            "total": 2,
            "true_anomalies": 1,
            "detected": 1,
            "true_positives": 1,
            "precision": 100.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "anomaly_detector.pkl"
            with (
                patch.object(model_jobs, "get_session_factory", return_value=object()),
                patch.object(model_jobs.persistence, "load_training_records", return_value=data),
                patch.object(model_jobs, "fit_model", return_value=(bundle, metrics)) as fit,
                patch.object(
                    model_jobs, "load_settings", return_value=Settings(None, model_path)
                ),
                patch(
                    "smartenergy.infrastructure.model_artifact.save_model_bundle"
                ) as save_bundle,
                redirect_stdout(StringIO()),
            ):
                model_jobs.train_model()

        fit.assert_called_once_with(data)
        save_bundle.assert_called_once_with(bundle, model_path)

    def test_detection_uses_shared_evaluation_and_persists_one_metric(self):
        data = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "corriente": [10.0, 20.0, 30.0, 40.0],
                "voltaje": [220.0] * 4,
                "potencia_activa": [5.0] * 4,
                "temperatura_motor": [45.0] * 4,
                "estado": ["Anomalo", "Normal", "Anomalo", "Normal"],
            }
        )
        model = Mock()
        model.predict.return_value = np.array([-1, -1, 1, 1])
        scaler = Mock()
        scaler.transform.side_effect = lambda values: values

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pkl"
            model_path.touch()
            settings = Settings(None, model_path)
            factory = object()
            with (
                patch.object(model_jobs, "get_session_factory", return_value=factory),
                patch.object(model_jobs, "load_settings", return_value=settings),
                patch.object(
                    model_jobs.persistence,
                    "load_detection_records",
                    return_value=data,
                ),
                patch(
                    "smartenergy.infrastructure.model_artifact.load_model_bundle",
                    return_value={
                        "model": model,
                        "scaler": scaler,
                        "features": ["corriente"],
                    },
                ),
                patch.object(model_jobs.persistence, "save_detection_metric") as save_metric,
                redirect_stdout(StringIO()),
            ):
                model_jobs.detect_anomalies()

        args = save_metric.call_args.kwargs
        self.assertEqual(
            (args["total"], args["anomalies"], args["percentage"], args["state"]),
            (4, 2, 50.0, "Critico"),
        )
        self.assertIsNotNone(args["created_at"])
        save_metric.assert_called_once()


if __name__ == "__main__":
    unittest.main()
