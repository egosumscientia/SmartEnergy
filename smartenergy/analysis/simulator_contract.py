"""Construcción S03 del perfil instrumental S06, sin admisión ni I/O.

No es un validador general de dispositivos. Sólo traduce Reading del perfil S05
ya normalizado; no consulta verdad física ni define canonicalización.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json

from smartenergy.analysis.simulator_instrumentation import (
    COUNTERS, OUTPUT_UNITS, UNITS, InstrumentCapture, Reading, _text,
)
from smartenergy.analysis.simulator_physics import _finite_real


QUANTITIES = dict(zip(
    ("U", "ULL", "I", "IN", "P", "Q", "S", "FP", "f", "EP+", "EP-", "EQ+", "EQ-"),
    ("voltage_rms", "voltage_rms", "current_rms", "current_rms", "active_power",
     "reactive_power", "apparent_power", "power_factor", "frequency",
     "active_energy_import", "active_energy_export", "reactive_energy_inductive",
     "reactive_energy_capacitive"),
))


@dataclass(frozen=True)
class ObservationConfig:
    installation_id: str
    device_id: str
    topology: str
    scenario_id: str
    scenario_version: str
    run_id: str
    configuration_ref: str
    producer_version: str
    origin_utc: datetime
    counter_epoch: str | None = None
    resolution_s: float | None = None
    uncertainty_s: float | None = None

    def __post_init__(self):
        for name in ("installation_id", "device_id", "scenario_id", "scenario_version",
                     "run_id", "configuration_ref", "producer_version"):
            _text(getattr(self, name), name)
        if self.topology not in ("1P2W", "3P4W"):
            raise ValueError("Topología desconocida")
        if not isinstance(self.origin_utc, datetime) or self.origin_utc.tzinfo is None:
            raise ValueError("Origen UTC consciente requerido")
        object.__setattr__(self, "origin_utc", self.origin_utc.astimezone(timezone.utc))
        if self.counter_epoch is not None:
            _text(self.counter_epoch, "Época")
        for name in ("resolution_s", "uncertainty_s"):
            value = getattr(self, name)
            if value is not None:
                value = _finite_real(value, name)
                if value < 0 or (name == "resolution_s" and value == 0):
                    raise ValueError("Resolución/incertidumbre fuera de dominio")
                object.__setattr__(self, name, value)

    def timestamp(self, seconds):
        seconds = _finite_real(seconds, "Tiempo relativo")
        if seconds < 0:
            raise ValueError("Tiempo negativo")
        # No redondear silenciosamente un calendario fuera de la resolución del DTO.
        delta = timedelta(seconds=seconds)
        if delta.total_seconds() != seconds:
            raise ValueError("Tiempo no representable exactamente en microsegundos")
        return (self.origin_utc + delta).isoformat().replace("+00:00", "Z")


def _channel_context(channel, config):
    phases = ("L1",) if config.topology == "1P2W" else ("L1", "L2", "L3")
    allowed = {"f": (None, "supply", ()), "c1/IN": ("c1", "phase", ("N",))}
    for phase in phases:
        allowed[f"bus-a/{phase}-N/U"] = (None, "voltage_pair", (phase, "N"))
        for variable in ("I", "P", "Q", "S", "FP", *COUNTERS):
            allowed[f"c1/{phase}/{variable}"] = ("c1", "phase", (phase,))
    for variable in ("P", "Q", "S", "FP", *COUNTERS):
        allowed[f"c1/{variable}"] = ("c1", "circuit_total", ())
    if len(phases) == 3:
        for pair in (("L1", "L2"), ("L2", "L3"), ("L3", "L1")):
            allowed[f"bus-a/{'-'.join(pair)}/ULL"] = (None, "voltage_pair", pair)
    if channel not in allowed:
        raise ValueError("Canal incompatible con el perfil/topología")
    circuit, kind, conductors = allowed[channel]
    return dict(installation_id=config.installation_id, device_id=config.device_id,
                circuit_id=circuit, voltage_reference_id="bus-a", topology=config.topology,
                scope=dict(kind=kind, conductors=list(conductors)))


def observation_from_reading(capture, reading, config, *, lineage_ref):
    """DTO de origen nuevo e independiente por llamada; referencias opacas explícitas.

    El llamador mantiene el linaje privado. No se infieren measured_at ni soporte
    a partir de producción/recepción cuando el instrumento declara tiempo unknown.
    """
    if not isinstance(config, ObservationConfig) or not isinstance(capture, InstrumentCapture):
        raise ValueError("Configuración/captura requerida")
    if not isinstance(reading, Reading) or capture.device != config.device_id:
        raise ValueError("Lectura/dispositivo incompatible")
    if reading not in capture.readings:
        raise ValueError("Lectura ajena a captura")
    if capture.capture_id != f"{config.run_id}/{capture.index}":
        raise ValueError("Identidad de ejecución incompatible con captura S05")
    _text(lineage_ref, "Linaje")
    context = _channel_context(reading.channel, config)
    variable = reading.channel.split("/")[-1]
    if reading.unit != UNITS[variable]:
        raise ValueError("Unidad canónica incompatible")
    if (reading.original_unit not in OUTPUT_UNITS or
            OUTPUT_UNITS[reading.original_unit][0] != reading.unit):
        raise ValueError("Unidad original incompatible")
    value = reading.value
    if value is not None:
        _finite_real(value, "Valor")
        _finite_real(reading.original_value, "Valor original")
        if reading.original_value * OUTPUT_UNITS[reading.original_unit][1] != value:
            raise ValueError("Original y valor normalizado incompatibles")
        if reading.validity not in ("valid", "invalid", "unknown"):
            raise ValueError("Validez incompatible con valor presente")
        availability = "present"
    elif (variable == "FP" and reading.reason == "undefined_zero_denominator"
          and reading.validity == "not_evaluated" and reading.original_value is None):
        availability = "not_applicable"
    else:
        raise ValueError("Ausencia sin semántica S05 aprobada")
    support = reading.support_s
    if support is None:
        start = end = None
        kind = "unknown"
    else:
        a, b = support
        if not 0 <= a <= b <= capture.produced_s:
            raise ValueError("Soporte incompatible con producción")
        start, end = config.timestamp(a), config.timestamp(b)
        kind = "point" if a == b else "interval"
    provenance = dict(origin="synthetic", producer_version=config.producer_version,
                      configuration_ref=config.configuration_ref, method_ref=reading.method_ref,
                      source_observation_ids=[],
                      original=dict(value=reading.original_value, unit=reading.original_unit,
                                    code=reading.channel),
                      scenario_ref=dict(id=config.scenario_id, version=config.scenario_version,
                                        run_id=config.run_id), lineage_ref=lineage_ref)
    if variable in COUNTERS:
        if config.counter_epoch is None:
            raise ValueError("Contador requiere época explícita")
        provenance["counter_epoch"] = config.counter_epoch
    _text(reading.method_ref, "Método")
    return dict(
        contract_version="1.0.0",
        # Codificación inyectiva de identidad, no canonicalización de contenido.
        observation_id=json.dumps([capture.capture_id, reading.channel], ensure_ascii=True,
                                  separators=(",", ":")),
        context=context, quantity=QUANTITIES[variable], value=value, unit=reading.unit,
        quality=dict(availability=availability, validity=reading.validity,
                     freshness="unknown" if value is not None else "not_applicable",
                     method="calculated", reason=reading.reason,
                     evidence_ref=config.configuration_ref),
        time=dict(event_at=None, support=dict(kind=kind, start=start, end=end),
                  measured_at=end,
                  clock=dict(basis="simulated" if support is not None else "unknown",
                             resolution_s=config.resolution_s if support is not None else None,
                             uncertainty_s=config.uncertainty_s if support is not None else None),
                  assessment=dict(at=None, max_age_s=None, policy_ref=None)),
        provenance=provenance)
