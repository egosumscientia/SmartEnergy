from datetime import datetime
from scripts.stream_realtime import build_sample
from core.models import Registro


def test_build_sample_returns_registro():
    ts = datetime.now()
    sample = build_sample(ts, anomaly_ratio=0.0)
    assert isinstance(sample, Registro)
    assert sample.timestamp == ts
    assert sample.estado == "Normal"


def test_build_sample_normal_ranges():
    ts = datetime.now()
    sample = build_sample(ts, anomaly_ratio=0.0)
    assert 0 < sample.corriente < 100
    assert 100 < sample.voltaje < 350
    assert sample.potencia_activa > 0
    assert 30 < sample.temperatura_motor < 60


def test_build_sample_high_anomaly_ratio():
    ts = datetime.now()
    anomaly_count = 0
    n = 200
    for _ in range(n):
        sample = build_sample(ts, anomaly_ratio=1.0)
        if sample.estado == "Anomalo":
            anomaly_count += 1
    assert anomaly_count == n
