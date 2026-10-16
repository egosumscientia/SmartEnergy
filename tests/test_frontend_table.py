"""Browser regression for the table's empty and populated states, without I/O services."""
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from smartenergy.analysis.measurements import prepare_records
from smartenergy.application import monitoring
from smartenergy.interfaces.api.routes import get_monitoring


ROOT = Path(__file__).resolve().parents[1]
CHROME = shutil.which("google-chrome") or shutil.which("chromium")


def table_fixture_html():
    rows = prepare_records(pd.DataFrame([{
        "id": 1, "timestamp": "2026-10-15 10:00:00", "corriente": 25.0,
        "voltaje": 220.0, "potencia_activa": 5.0, "temperatura_motor": 45.0,
        "estado": "Normal",
    }]))

    def payload(count, states=None):
        with (
            patch.object(monitoring, "get_data_count", return_value=count),
            patch.object(monitoring, "load_records", return_value=rows),
            patch.object(monitoring, "get_model_artifact_status", return_value=(False, None)),
            patch.object(monitoring, "load_latest_metrics", return_value=None),
        ):
            return get_monitoring(
                date_from=None, date_to=None, states=states, variable="corriente",
                sample_plots=True, smoothing_method="Ninguno", rolling_window=60,
                resample_minutes=5, table_limit=1000,
            ).model_dump(mode="json")

    fixtures = {"empty": payload(0), "data": payload(1), "filtered": payload(1, [])}
    page = (ROOT / "frontend/index.html").read_text()
    for name in ("css/dashboard.css", "vendor/echarts.min.js", "js/api.js",
                 "js/charts.js", "js/dashboard.js"):
        page = page.replace(f'"/{name}"', f'"{(ROOT / "frontend" / name).as_uri()}"')
    setup = """<script>
    window.reviewErrors = [];
    window.addEventListener('error', e => reviewErrors.push(e.message));
    window.addEventListener('unhandledrejection', e => reviewErrors.push(String(e.reason)));
    window.tableFixtures = FIXTURES;
    window.tableScenario = 'empty';
    window.lastRequest = '';
    window.fetch = async url => {
      lastRequest = String(url);
      const params = new URL(url, 'http://fixture').searchParams;
      const name = tableScenario === 'data' && params.get('states') === ''
        ? 'filtered' : tableScenario;
      return {ok: true, status: 200, json: async () => tableFixtures[name]};
    };
    </script>""".replace("FIXTURES", json.dumps(fixtures))
    checks = """<script>
    window.addEventListener('DOMContentLoaded', async () => {
      const results = [];
      const wait = () => new Promise(resolve => setTimeout(resolve, 50));
      async function settled() {
        for (let i = 0; i < 40; i++) {
          if (!document.getElementById('refresh-button').disabled) return;
          await wait();
        }
        throw new Error('refresh timeout');
      }
      function snapshot(name) {
        const empty = document.getElementById('records-empty');
        results.push({name, rows: document.querySelectorAll('#records-body tr').length,
          emptyVisible: !empty.hidden && getComputedStyle(empty).display !== 'none',
          message: empty.textContent, panelVisible: !document.getElementById('panel-records').hidden,
          connection: document.getElementById('connection-status').textContent,
          request: lastRequest});
      }
      try {
        await settled();
        document.getElementById('tab-records').click();
        snapshot('no_records');
        tableScenario = 'data';
        document.getElementById('refresh-button').click();
        await settled();
        snapshot('with_records');
        for (const box of document.querySelectorAll('input[name="states"]')) box.checked = false;
        document.getElementById('filters-form').requestSubmit();
        await settled();
        snapshot('filters_without_results');
        for (const box of document.querySelectorAll('input[name="states"]')) box.checked = true;
        document.getElementById('filters-form').requestSubmit();
        await settled();
        snapshot('records_restored');
      } catch (error) { reviewErrors.push(error.message); }
      const pre = document.createElement('pre');
      pre.id = 'table-test-result';
      pre.textContent = JSON.stringify({results, errors: reviewErrors});
      document.body.append(pre);
    });
    </script>"""
    return page.replace("<head>", "<head>" + setup).replace("</body>", checks + "</body>")


@unittest.skipUnless(CHROME, "Chrome/Chromium is required for the browser regression")
class FrontendTableTests(unittest.TestCase):
    def test_table_empty_data_and_filtered_states_in_browser(self):
        with tempfile.TemporaryDirectory(prefix="smartenergy-table-") as directory:
            fixture = Path(directory) / "table.html"
            fixture.write_text(table_fixture_html())
            result = subprocess.run(
                [CHROME, "--headless", "--disable-gpu", "--no-sandbox",
                 f"--user-data-dir={directory}/browser", "--virtual-time-budget=5000",
                 "--dump-dom", fixture.as_uri()],
                capture_output=True, text=True, timeout=30, check=True,
            )
        match = re.search(r'<pre id="table-test-result">(.*?)</pre>', result.stdout, re.S)
        self.assertIsNotNone(match, "Browser did not complete the table regression")
        outcome = json.loads(html.unescape(match.group(1)))
        self.assertEqual(outcome["errors"], [])
        self.assertEqual([item["name"] for item in outcome["results"]], [
            "no_records", "with_records", "filters_without_results", "records_restored",
        ])
        for item, expected_rows in zip(outcome["results"], [0, 1, 0, 1]):
            with self.subTest(state=item["name"]):
                self.assertTrue(item["panelVisible"])
                self.assertEqual(item["connection"], "Servicio disponible")
                self.assertEqual(item["rows"], expected_rows)
                self.assertEqual(item["emptyVisible"], expected_rows == 0)
                if expected_rows == 0:
                    self.assertTrue(item["message"].strip(), "Empty table needs a clear message")
        self.assertIn("states=", outcome["results"][2]["request"])


if __name__ == "__main__":
    unittest.main()
