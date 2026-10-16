from datetime import date
import unittest

import pandas as pd

from smartenergy.analysis.measurements import (
    apply_smoothing,
    filter_records,
    ordered_records,
    prepare_plot_records,
    prepare_records,
    recent_records,
    sample_records,
)


class MeasurementAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame(
            {
                "id": [3, 1, 2],
                "timestamp": ["2026-10-15 12:00:02", "2026-10-15 12:00:00", None],
                "corriente": ["3", "1", "inf"],
                "voltaje": [220, 220, "bad"],
                "potencia_activa": [5, 2, 1],
                "temperatura_motor": [45, 40, 30],
                "estado": ["Anomalia", "Normal", None],
            }
        )

    def test_preparation_preserves_null_and_label_rules(self):
        result = prepare_records(self.rows)
        self.assertEqual(result.estado.tolist(), ["Anomalo", "Normal", "Sin estado"])
        self.assertEqual(result.label_anomalia.tolist(), [1, 0, 0])
        self.assertTrue(pd.isna(result.loc[2, "corriente"]))
        self.assertTrue(pd.isna(result.loc[2, "voltaje"]))
        self.assertTrue(pd.isna(result.loc[2, "timestamp"]))

    def test_ordering_and_filters_preserve_stable_id_order_and_empty_behavior(self):
        records = prepare_records(self.rows)
        self.assertEqual(ordered_records(records).id.tolist(), [2, 1, 3])
        selected = filter_records(
            records, [date(2026, 10, 15), date(2026, 10, 15)], ["Normal", "Anomalo"]
        )
        self.assertEqual(selected.id.tolist(), [1, 3])
        self.assertTrue(filter_records(records, None, []).empty)
        self.assertTrue(filter_records(records, [date(2026, 10, 15)], ["Normal"]).empty)

    def test_sampling_limits_and_recent_records_order(self):
        records = prepare_records(self.rows)
        self.assertEqual(sample_records(records, 1).id.tolist(), [2])
        self.assertEqual(recent_records(records, 2).id.tolist(), [3, 1])
        with self.assertRaises(ValueError):
            sample_records(records, 0)

    def test_smoothing_and_window_aggregation_keep_existing_formulas(self):
        records = prepare_records(self.rows.iloc[:2])
        moving = apply_smoothing(records, "corriente", "Media movil", 2, 5)
        self.assertEqual(moving.corriente.tolist(), [1.0, 2.0])

        downsampled = apply_smoothing(records, "corriente", "Downsample por ventana", 2, 1440)
        self.assertEqual(len(downsampled), 1)
        self.assertEqual(downsampled.iloc[0].corriente, 2.0)
        self.assertEqual(downsampled.iloc[0].label_anomalia, 0)

    def test_plot_preparation_keeps_sampling_and_aggregating_separate_from_source(self):
        count = 6000
        source = pd.DataFrame(
            {
                "id": range(count),
                "timestamp": pd.date_range("2026-10-15", periods=count, freq="s"),
                "corriente": range(count),
                "voltaje": [220.0] * count,
                "potencia_activa": [5.0] * count,
                "temperatura_motor": [45.0] * count,
                "estado": ["Normal"] * count,
                "label_anomalia": [0] * count,
            }
        )
        plotted = prepare_plot_records(source, "corriente", True, "Ninguno", 60, 5)
        self.assertLessEqual(len(plotted), 1500)
        self.assertEqual(len(source), count)
        self.assertEqual(plotted.timestamp.is_monotonic_increasing, True)


if __name__ == "__main__":
    unittest.main()
