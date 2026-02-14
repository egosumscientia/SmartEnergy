from datetime import datetime, timezone
from core.models import Registro, Metrica


def test_registro_creation():
    r = Registro(
        corriente=25.0,
        voltaje=220.0,
        potencia_activa=4.5,
        temperatura_motor=45.0,
        estado="Normal",
    )
    assert r.corriente == 25.0
    assert r.voltaje == 220.0
    assert r.potencia_activa == 4.5
    assert r.temperatura_motor == 45.0
    assert r.estado == "Normal"


def test_registro_anomalo():
    r = Registro(
        corriente=80.0,
        voltaje=120.0,
        potencia_activa=12.0,
        temperatura_motor=130.0,
        estado="Anomalo",
    )
    assert r.estado == "Anomalo"


def test_metrica_creation():
    m = Metrica(
        total=10000,
        anomalias=150,
        porcentaje=1.5,
        estado="Normal",
        fecha=datetime.now(timezone.utc),
    )
    assert m.total == 10000
    assert m.anomalias == 150
    assert m.porcentaje == 1.5
    assert m.estado == "Normal"


def test_metrica_estados():
    for estado in ("Normal", "Precaucion", "Critico"):
        m = Metrica(total=100, anomalias=10, porcentaje=10.0, estado=estado)
        assert m.estado == estado
