import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from smartenergy.analysis.anomalies import evaluate_model, predict_anomalies
from smartenergy.analysis.training import train_model


class AnomalyAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "corriente": [10.0, 20.0, 30.0, 40.0],
                "estado": ["Anomalo", "Normal", "Anomalo", "Normal"],
            }
        )
        self.model = Mock()
        self.model.predict.return_value = np.array([-1, -1, 1, 1])
        self.scaler = Mock()
        self.scaler.transform.side_effect = lambda values: values

    def test_evaluation_preserves_metrics_and_original_population_denominators(self):
        result = evaluate_model(self.df, self.model, self.scaler, ["corriente"])
        self.assertEqual(
            result,
            {
                "total": 4,
                "detectadas": 2,
                "verdaderos_positivos": 1,
                "recall": 50.0,
                "precision": 50.0,
                "coverage": 50.0,
                "estado": "Critico",
            },
        )
        self.assertEqual(predict_anomalies(self.df, self.model, self.scaler, ["corriente"]).tolist(), [1, 1, 0, 0])

    def test_zero_predictions_and_empty_population_keep_guarded_denominators(self):
        self.model.predict.return_value = np.array([1, 1, 1, 1])
        result = evaluate_model(self.df, self.model, self.scaler, ["corriente"])
        self.assertEqual(result["detectadas"], 0)
        self.assertEqual(result["verdaderos_positivos"], 0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["coverage"], 0.0)
        self.assertEqual(result["estado"], "Critico")
        self.assertIsNone(evaluate_model(self.df.iloc[:0], self.model, self.scaler, ["corriente"]))
        self.assertIsNone(evaluate_model(self.df, None, self.scaler, ["corriente"]))

    def test_training_parameters_and_feature_order_are_preserved(self):
        data = pd.DataFrame(
            {
                "corriente": [10.0, 20.0],
                "voltaje": [220.0, 210.0],
                "potencia_activa": [5.0, 8.0],
                "temperatura_motor": [45.0, 60.0],
                "estado": ["Normal", "Anomalo"],
            }
        )
        scaled = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        with (
            patch("sklearn.preprocessing.StandardScaler") as scaler_class,
            patch("sklearn.ensemble.IsolationForest") as forest_class,
        ):
            scaler = scaler_class.return_value
            scaler.fit_transform.return_value = scaled
            model = forest_class.return_value
            model.predict.return_value = np.array([-1, 1])
            bundle, metrics = train_model(data)

        forest_class.assert_called_once_with(
            n_estimators=200,
            contamination=0.2,
            random_state=42,
            n_jobs=-1,
        )
        model.fit.assert_called_once_with(scaled)
        self.assertEqual(
            bundle["features"],
            ["corriente", "voltaje", "potencia_activa", "temperatura_motor"],
        )
        self.assertEqual(metrics["true_positives"], 0)


if __name__ == "__main__":
    unittest.main()
