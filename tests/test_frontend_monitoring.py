"""Five-view browser coverage replacing the retired Streamlit AppTest cases."""
import html
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from smartenergy.analysis.measurements import ENERGY_COLUMNS, prepare_records
from smartenergy.application import monitoring
from smartenergy.interfaces.api.routes import get_monitoring
from tests.test_frontend_table import CHROME, table_fixture_html


def monitoring_fixtures():
    rows = prepare_records(pd.DataFrame({
        "id": [1, 2, 3], "timestamp": pd.date_range("2026-10-15", periods=3, freq="s"),
        "corriente": [25., 40., 20.], "voltaje": [220.] * 3,
        "potencia_activa": [5., 8., 4.], "temperatura_motor": [45., 46., 44.],
        "estado": ["Normal", "Anomalo", "Normal"],
    }))
    fixtures = {}
    for data in (False, True):
        for model_available in (False, True):
            model, scaler = Mock(), Mock()
            model.predict.side_effect = lambda values: np.where(values[:, 0] > 30, -1, 1)
            scaler.transform.side_effect = lambda values: values
            with (
                patch.object(monitoring, "get_data_count", return_value=3 if data else 0),
                patch.object(monitoring, "load_records", return_value=rows if data else None),
                patch.object(monitoring, "get_model_artifact_status", return_value=(model_available, None)),
                patch.object(monitoring, "load_latest_metrics", return_value=None),
                patch.object(monitoring, "load_model", return_value=(model, scaler, ENERGY_COLUMNS, None)),
            ):
                fixtures[f"d{int(data)}m{int(model_available)}"] = get_monitoring(
                    date_from=None, date_to=None, states=None, variable="corriente",
                    sample_plots=True, smoothing_method="Ninguno", rolling_window=60,
                    resample_minutes=5, table_limit=1000,
                ).model_dump(mode="json")
    return fixtures


CHECKS = """<script>
window.addEventListener('DOMContentLoaded', async () => {
  const results = [], filters = [];
  const el = id => document.getElementById(id);
  const wait = () => new Promise(resolve => setTimeout(resolve, 50));
  async function settled() {
    for (let i=0; i<60; i++) {
      if (!el('refresh-button').disabled) return;
      await wait();
    }
    throw new Error('refresh timeout');
  }
  try {
    await settled();
    for (const name of ['d0m0', 'd0m1', 'd1m0', 'd1m1']) {
      tableScenario = name;
      el('refresh-button').click(); await settled();
      const panels = [];
      for (const tab of document.querySelectorAll('[role=tab]')) {
        tab.click();
        panels.push({selected: tab.getAttribute('aria-selected'),
          visible: !el(tab.getAttribute('aria-controls')).hidden});
      }
      const chartIds = ['overview-chart', 'energy-chart', 'histogram-chart',
        'comparison-chart', 'anomaly-chart'];
      results.push({name, panels, rows: el('records-body').children.length,
        model: el('model-status').textContent,
        charts: chartIds.map(id => !el(id).hidden && !!echarts.getInstanceByDom(el(id))),
        anomaly: el('anomaly-count').textContent,
        emptyMessages: ['overview-empty','energy-empty','comparison-empty','anomaly-empty','records-empty']
          .map(id => !el(id).hidden && !!el(id).textContent.trim())});
    }
    el('variable').value = 'voltaje';
    el('sample-plots').checked = false;
    el('date-from').value = '2026-10-15'; el('date-to').value = '2026-10-15';
    for (const method of ['Media movil', 'Downsample por ventana']) {
      el('smoothing-method').value = method;
      el('filters-form').requestSubmit(); await settled();
      filters.push(lastRequest);
    }
    const csv = el('csv-download').getAttribute('href');
    // A query error must remain visibly different from an empty response.
    window.fetch = async () => ({ok:false, status:503,
      json:async () => ({error:{message:'Consulta no disponible'}})});
    el('refresh-button').click(); await settled();
    results.push({name:'error', connection:el('connection-status').textContent,
      notice:el('request-message').textContent, noticeVisible:!el('request-message').hidden,
      rows:el('records-body').children.length});
    const pre=document.createElement('pre'); pre.id='monitoring-test-result';
    pre.textContent=JSON.stringify({results, filters, csv, errors:reviewErrors,
      version:echarts.version, tabs:document.querySelectorAll('[role=tab]').length,
      overflow:document.documentElement.scrollWidth > innerWidth,
      adminControls:[...document.querySelectorAll('button')].filter(b =>
        /Entrenar|Recalcular|Generar dataset|Reiniciar/.test(b.textContent)).length});
    document.body.append(pre);
  } catch(error) {
    const pre=document.createElement('pre'); pre.id='monitoring-test-result';
    pre.textContent=JSON.stringify({errors:[String(error)]}); document.body.append(pre);
  }
});
</script>"""


@unittest.skipUnless(CHROME, "Chrome/Chromium is required for browser verification")
class FrontendMonitoringTests(unittest.TestCase):
    def test_five_views_availability_filters_and_errors(self):
        page = table_fixture_html(monitoring_fixtures(), CHECKS)
        for width in (1365, 390):
            with self.subTest(width=width), tempfile.TemporaryDirectory(prefix="smartenergy-monitoring-") as directory:
                fixture = Path(directory) / "monitoring.html"
                fixture.write_text(page)
                result = subprocess.run([
                    CHROME, "--headless", "--disable-gpu", "--no-sandbox",
                    f"--user-data-dir={directory}/browser", f"--window-size={width},1000",
                    "--virtual-time-budget=10000", "--dump-dom", fixture.as_uri(),
                ], capture_output=True, text=True, timeout=45, check=True)
                match = re.search(r'<pre id="monitoring-test-result">(.*?)</pre>', result.stdout, re.S)
                self.assertIsNotNone(match, "Browser did not finish verification")
                outcome = json.loads(html.unescape(match.group(1)))
                self.assertEqual(outcome["errors"], [])
                self.assertEqual(outcome["tabs"], 5)
                self.assertEqual(outcome["version"], "6.1.0")
                self.assertFalse(outcome["overflow"])
                self.assertEqual(outcome["adminControls"], 0)
                for item in outcome["results"][:4]:
                    self.assertTrue(all(p["visible"] and p["selected"] == "true" for p in item["panels"]))
                    data, model = item["name"][1] == '1', item["name"][3] == '1'
                    self.assertEqual(item["rows"], 3 if data else 0)
                    self.assertEqual(item["charts"], [data] * 4 + [data and model])
                    self.assertEqual(item["model"], "Disponible" if model else "No disponible")
                    self.assertEqual(item["anomaly"], "1" if data and model else "N/A")
                    if not data:
                        self.assertTrue(all(item["emptyMessages"]))
                for request in outcome["filters"]:
                    self.assertIn('variable=voltaje', request)
                    self.assertIn('sample_plots=false', request)
                    self.assertIn('date_from=2026-10-15', request)
                    self.assertIn('smoothing_method=', request)
                self.assertIn('/api/v1/monitoring/records.csv?', outcome["csv"])
                self.assertIn('date_to=2026-10-15', outcome["csv"])
                self.assertNotIn('smoothing_method', outcome["csv"])
                failure = outcome["results"][-1]
                self.assertEqual(failure["connection"], 'Servicio no disponible')
                self.assertTrue(failure["noticeVisible"])
                self.assertIn('Consulta no disponible', failure["notice"])
                self.assertEqual(failure["rows"], 3)
