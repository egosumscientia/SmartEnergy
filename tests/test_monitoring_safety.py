from datetime import date
import unittest

import pandas as pd
from smartenergy.analysis.measurements import filter_records, recent_records, sample_records


class MonitoringSafetyTests(unittest.TestCase):
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



if __name__ == "__main__":
    unittest.main()
