"""S05 aislada: transformaciones instrumentales en memoria, sin adquisición/S03.

Las capturas son lotes simultáneos de canales, no DTO de observaciones. Trace
contiene verdad de ensayo privada; Reading sólo evidencia declarada. Un contador
requiere época/iniciales explícitos y una trayectoria determinista continua:
el ruido definido sólo por índice de captura no define una integral entre puntos.
"""

from dataclasses import dataclass, replace
import hashlib
import json
import math
import random
import sys

from smartenergy.analysis.simulator_physics import _finite_real
from smartenergy.analysis.simulator_energy import EnergyCounters, CircuitEnergy
from smartenergy.analysis.simulator_scenarios import PhysicalScenario


UNITS = {"U": "V", "I": "A", "P": "W", "Q": "var", "S": "VA", "FP": "1",
         "f": "Hz", "EP+": "Wh", "EP-": "Wh", "EQ+": "varh", "EQ-": "varh",
         "IN": "A", "ULL": "V"}
OUTPUT_UNITS = {"V": ("V", 1), "A": ("A", 1), "W": ("W", 1),
                "kW": ("W", 1000), "var": ("var", 1), "kvar": ("var", 1000),
                "VA": ("VA", 1), "kVA": ("VA", 1000), "1": ("1", 1),
                "Hz": ("Hz", 1), "Wh": ("Wh", 1), "kWh": ("Wh", 1000),
                "varh": ("varh", 1), "kvarh": ("varh", 1000)}
COUNTERS = dict(zip(("EP+", "EP-", "EQ+", "EQ-"), EnergyCounters.__dataclass_fields__))


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name}: string no vacío requerido")
    return value


@dataclass(frozen=True)
class RandomIdentity:
    master: int
    run_id: str
    device: str
    replicate: str = "0"

    def __post_init__(self):
        if isinstance(self.master, bool) or not isinstance(self.master, int) or not 0 <= self.master < 2**64:
            raise ValueError("Master fuera del dominio uint64")
        for name in ("run_id", "device", "replicate"):
            _text(getattr(self, name), name)

    def seed(self, channel, profile):
        payload = ["s05-rng-v1", str(self.master), self.run_id, "instrumentation",
                   self.device, _text(channel, "canal"), _text(profile, "perfil"), self.replicate]
        data = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        return int.from_bytes(hashlib.sha256(data).digest(), "big")


@dataclass(frozen=True)
class TransformerRatios:
    real: float = 1.0
    configured: float = 1.0
    scale_at: str = "device"

    def __post_init__(self):
        for name in ("real", "configured"):
            value = _finite_real(getattr(self, name), name)
            if not 1 <= value <= 1000:
                raise ValueError("Relación CT/VT fuera de 1…1000")
            object.__setattr__(self, name, value)
        if self.scale_at not in ("device", "adapter"):
            raise ValueError("Escala única: device o adapter; doble escala rechazada")


@dataclass(frozen=True)
class ErrorProfile:
    profile_ref: str = "ideal@1"
    reference: float = 1.0
    gain_bias: float = 0.0
    offset: float = 0.0
    drift_rate: float = 0.0
    drift_start_s: float = 0.0
    drift_duration_s: float = 100.0
    noise_amplitude: float = 0.0
    # Control analítico I01-AC: u prescritos, sin consumir PRNG.
    prescribed_u: tuple[float, ...] = ()

    def __post_init__(self):
        _text(self.profile_ref, "perfil")
        for name in ("reference", "gain_bias", "offset", "drift_rate", "drift_start_s",
                     "drift_duration_s", "noise_amplitude"):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))
        if (self.reference <= 0 or not -.5 <= self.gain_bias <= .5
                or abs(self.offset) > .1 * self.reference or self.noise_amplitude < 0
                or self.drift_start_s < 0 or self.drift_duration_s <= 0
                or abs(self.drift_rate) * self.drift_duration_s > .1
                or 1 + self.gain_bias + min(0, self.drift_rate * self.drift_duration_s) <= 0):
            raise ValueError("Perfil fuera de dominios I01/I02")
        us = tuple(_finite_real(u, "u") for u in self.prescribed_u)
        if any(not 0 <= u < 1 for u in us) or (us and not self.noise_amplitude):
            raise ValueError("Innovaciones prescritas requieren amplitud y 0≤u<1")
        object.__setattr__(self, "prescribed_u", us)

    def apply(self, value, time_s, innovation=0):
        drift = self.drift_rate * min(max(time_s - self.drift_start_s, 0), self.drift_duration_s)
        result = (1 + self.gain_bias + drift) * value + self.offset + innovation
        return _finite_real(result, "Resultado instrumental")


@dataclass(frozen=True)
class ChannelConfig:
    """Primarios U/I/f: error en secundario; resultados: error explícito en SI.

    shared_noise identifica una fuente latente c=1; sólo comparte innovación,
    no ganancia ni offsets. La salida convierte SI a original, cuantiza en esa
    unidad y normaliza una vez sin volver a aplicar CT/VT.
    """

    key: str
    error: ErrorProfile = ErrorProfile()
    ratios: TransformerRatios = TransformerRatios()
    output_unit: str | None = None
    quantum: float | None = None
    shared_noise: str | None = None

    def __post_init__(self):
        _text(self.key, "canal")
        variable = self.key.split("/")[-1]
        if variable not in UNITS or not isinstance(self.error, ErrorProfile) or not isinstance(self.ratios, TransformerRatios):
            raise ValueError("Canal/perfil/relaciones desconocidos")
        if variable not in ("U", "I") and self.ratios != TransformerRatios():
            raise ValueError("CT/VT sólo en U/I")
        unit = self.output_unit or UNITS[variable]
        if unit not in OUTPUT_UNITS or OUTPUT_UNITS[unit][0] != UNITS[variable]:
            raise ValueError("Unidad incompatible/desconocida")
        object.__setattr__(self, "output_unit", unit)
        if self.quantum is not None:
            quantum = _finite_real(self.quantum, "Resolución")
            if quantum <= 0:
                raise ValueError("Resolución requiere q>0")
            object.__setattr__(self, "quantum", quantum)
        if self.shared_noise is not None:
            _text(self.shared_noise, "fuente latente")


@dataclass(frozen=True)
class Retention:
    start_s: float
    end_s: float
    source_index: int
    channels: tuple[str, ...]
    clock_policy: str = "original"

    def __post_init__(self):
        a, b = _finite_real(self.start_s, "Inicio"), _finite_real(self.end_s, "Fin")
        if not 0 <= a < b <= 86400:
            raise ValueError("Intervalo de retención inválido")
        if isinstance(self.source_index, bool) or not isinstance(self.source_index, int) or self.source_index < 0:
            raise ValueError("Índice retenido inválido")
        channels = tuple(self.channels)
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("Grupo de retención explícito requerido")
        for key in channels:
            _text(key, "canal retenido")
        if self.clock_policy not in ("original", "current", "unknown"):
            raise ValueError("Política temporal desconocida")
        object.__setattr__(self, "start_s", a)
        object.__setattr__(self, "end_s", b)
        object.__setattr__(self, "channels", channels)


@dataclass(frozen=True)
class InstrumentConfig:
    device: str
    channels: tuple[ChannelConfig, ...]
    random_identity: RandomIdentity
    retentions: tuple[Retention, ...] = ()
    silent_indices: tuple[int, ...] = ()
    counter_epoch: str | None = None
    initial_counters: CircuitEnergy | None = None
    energy_route: str = "primary"

    def __post_init__(self):
        _text(self.device, "instrumento")
        channels = tuple(self.channels)
        if not channels or not all(isinstance(c, ChannelConfig) for c in channels):
            raise ValueError("Canales explícitos requeridos")
        if len({c.key for c in channels}) != len(channels):
            raise ValueError("Canales duplicados")
        if not isinstance(self.random_identity, RandomIdentity) or self.device != self.random_identity.device:
            raise ValueError("Identidad RNG incompatible")
        if self.energy_route not in ("primary", "reported_power"):
            raise ValueError("Ruta de energía desconocida")
        if (self.counter_epoch is None) != (self.initial_counters is None):
            raise ValueError("Época e iniciales instrumentales deben declararse juntos")
        if self.counter_epoch is not None:
            _text(self.counter_epoch, "época")
            if not isinstance(self.initial_counters, CircuitEnergy):
                raise ValueError("Iniciales instrumentales inválidos")
        retentions = tuple(self.retentions)
        if not all(isinstance(r, Retention) for r in retentions):
            raise ValueError("Retenciones inválidas")
        keys = {c.key for c in channels}
        for r in retentions:
            if not set(r.channels) <= keys:
                raise ValueError("Retención de canal no configurado")
        for i, r in enumerate(retentions):
            for s in retentions[:i]:
                if set(r.channels) & set(s.channels) and max(r.start_s, s.start_s) < min(r.end_s, s.end_s):
                    raise ValueError("Retenciones ambiguas")
        silent = tuple(self.silent_indices)
        if any(isinstance(j, bool) or not isinstance(j, int) or j < 0 for j in silent) or len(set(silent)) != len(silent):
            raise ValueError("Máscara de silencio inválida")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "retentions", retentions)
        object.__setattr__(self, "silent_indices", silent)


@dataclass(frozen=True)
class Reading:
    channel: str
    value: float | None
    unit: str
    original_value: float | None
    original_unit: str
    support_s: tuple[float, float] | None
    validity: str
    reason: str
    method_ref: str = "s05-instrumentation@1"

    def freshness_at(self, evaluation_s, max_age_s=5):
        """Sólo evaluación sobre evidencia temporal; no recepción/adquisición."""
        now = _finite_real(evaluation_s, "Evaluación")
        age = _finite_real(max_age_s, "Edad máxima")
        if age < 0:
            raise ValueError("Edad máxima negativa")
        if self.value is None:
            return "not_applicable"
        if self.support_s is None:
            return "unknown"
        if now < self.support_s[1]:
            raise ValueError("Evaluación anterior a medición")
        return "stale" if now - self.support_s[1] > age else "current"


@dataclass(frozen=True)
class _OriginalReading:
    channel: ChannelConfig
    value: float | None
    support_s: tuple[float, float] | None
    temporal_unknown: bool = False


@dataclass(frozen=True)
class InstrumentCapture:
    capture_id: str
    device: str
    index: int
    scheduled_s: float
    produced_s: float
    readings: tuple[Reading, ...]

    def channel(self, key):
        return next(r for r in self.readings if r.channel == key)


@dataclass(frozen=True)
class ChannelTrace:
    """Privado del ensayo: valores anteriores a salida y soporte físico real."""
    channel: str
    primary: float | None
    secondary_ideal: float | None
    secondary_instrumental: float | None
    indicated: float | None
    innovation: float
    real_support_s: tuple[float, float]
    retained_from_index: int | None = None


@dataclass(frozen=True)
class CaptureTrace:
    index: int
    scheduled_s: float
    produced: InstrumentCapture | None
    channels: tuple[ChannelTrace, ...]
    seeds: tuple[tuple[str, int], ...]
    python_version: str


class VirtualInstrument:
    """Ensayo independiente. capture(times) reinicia flujos locales reproducibles.

    No se modifican escenario/configuración. La lista completa asigna índices;
    incluye silencios y retenciones. No admite ruido independiente de resultados
    simultáneamente al ruido primario que ya se propaga a esos resultados.
    """

    def __init__(self, scenario: PhysicalScenario, config: InstrumentConfig):
        if not isinstance(scenario, PhysicalScenario) or not isinstance(config, InstrumentConfig):
            raise ValueError("Se requiere escenario físico y configuración instrumental")
        self.scenario, self.config = scenario, config
        self._channels = {c.key: c for c in config.channels}
        count = len(scenario.initial_energy.phases)
        allowed = {"f", "c1/IN"}
        for k in range(1, count + 1):
            allowed.update(f"c1/L{k}/{v}" for v in ("I", "P", "Q", "S", "FP", *COUNTERS))
            allowed.add(f"bus-a/L{k}-N/U")
        allowed.update(f"c1/{v}" for v in ("P", "Q", "S", "FP", *COUNTERS))
        if count == 3:
            allowed.update(f"bus-a/{pair}/ULL" for pair in ("L1-L2", "L2-L3", "L3-L1"))
        if not set(self._channels) <= allowed:
            raise ValueError("Alcance/canal incompatible con escenario")
        if config.initial_counters is not None and len(config.initial_counters.phases) != count:
            raise ValueError("Contadores de topología incompatible")
        if any(c.key.split("/")[-1] in COUNTERS for c in config.channels) and config.counter_epoch is None:
            raise ValueError("Contador requiere época e iniciales explícitos")
        noisy_primary = any(c.error.noise_amplitude and c.key.split("/")[-1] in ("U", "I") for c in config.channels)
        if noisy_primary and any(c.error.noise_amplitude and c.key.split("/")[-1] in ("P", "Q", "S", "FP") for c in config.channels):
            raise ValueError("Ruido derivado adicional al primario rechazado")
        shared = {}
        for c in config.channels:
            if c.shared_noise:
                signature = (c.error.profile_ref, c.error.prescribed_u)
                if c.shared_noise in shared and shared[c.shared_noise] != signature:
                    raise ValueError("Fuente común requiere perfil/innovaciones idénticos")
                shared[c.shared_noise] = signature
        # No inventar ángulos de corriente a carga cero, ni sólo en nodos.
        for c in config.channels:
            if c.key.endswith("/I") and (c.error.offset or c.error.noise_amplitude):
                k = int(c.key.split("/")[1][1:]) - 1
                for s in scenario.segments:
                    a, b = s.left.phases[k], s.right.phases[k]
                    if a.displacement_rad is None or b.displacement_rad is None:
                        raise ValueError("Offset/ruido de I con ángulo físico indefinido")
                    y0, y1 = s.left.admittances_s[k], s.right.admittances_s[k]
                    d = y1 - y0
                    if d and 0 <= -(y0.conjugate() * d).real / abs(d)**2 <= 1:
                        f = -(y0.conjugate() * d).real / abs(d)**2
                        if abs(y0 + f * d) == 0:
                            raise ValueError("Trayectoria cruza corriente cero sin ángulo")

    def _profile(self, key):
        return self._channels.get(key, ChannelConfig(key))

    def _values(self, state, t, innovations):
        values, traces, us, currents = {}, {}, [], []
        def primary(key, x):
            c = self._profile(key)
            secondary = x / c.ratios.real
            eta = innovations.get(key, 0)
            z = c.error.apply(secondary, t, eta)
            if z < 0:
                raise ValueError("Módulo instrumental negativo; no recortar/resortear")
            value = _finite_real(z * c.ratios.configured, "Indicado")
            values[key] = value
            traces[key] = (x, secondary, z, value, eta)
            return value
        for k, phase in enumerate(state.phases, 1):
            u = primary(f"bus-a/L{k}-N/U", phase.voltage_rms_v)
            i = primary(f"c1/L{k}/I", phase.current_rms_a)
            if phase.displacement_rad is None:
                if i != 0:
                    raise ValueError("Ángulo físico de corriente no definido")
                p = q = s = 0.0
            else:
                s = u * i
                # cos/sin físicos como razones: preserva ceros exactos en
                # reactancia pura, sin el residuo de cos(atan2(Q,0)).
                p = s * (phase.active_power_w / phase.apparent_power_va)
                q = s * (phase.reactive_power_var / phase.apparent_power_va)
            scope = f"c1/L{k}"
            values.update({f"{scope}/P": p, f"{scope}/Q": q, f"{scope}/S": s,
                           f"{scope}/FP": p / s if s else None})
            angle = (0, -2 * math.pi / 3, 2 * math.pi / 3)[k - 1]
            us.append(u * complex(math.cos(angle), math.sin(angle)))
            currents.append(i * complex(math.cos(angle - (phase.displacement_rad or 0)),
                                        math.sin(angle - (phase.displacement_rad or 0))))
        p = math.fsum(values[f"c1/L{k}/P"] for k in range(1, len(us) + 1))
        q = math.fsum(values[f"c1/L{k}/Q"] for k in range(1, len(us) + 1))
        s = math.fsum(values[f"c1/L{k}/S"] for k in range(1, len(us) + 1))
        values.update({"c1/P": p, "c1/Q": q, "c1/S": s, "c1/FP": p / s if s else None,
                       "f": state.frequency_hz, "c1/IN": abs(sum(currents))})
        if len(us) == 3:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                values[f"bus-a/L{a+1}-L{b+1}/ULL"] = abs(us[a] - us[b])
        return values, traces

    def energy(self, start_s, end_s):
        """Incremento instrumental, previo a cuantización/retención de pantalla.

        Integración adaptativa determinista de P/Q con cortes en fronteras físicas
        y mesetas de deriva. No integra lecturas ni ruido sin trayectoria continua
        declarada. Falla explícitamente si no converge; tolerancia 1e-9 Wh/varh.
        """
        a, b = self.scenario._time(start_s), self.scenario._time(end_s)
        if a > b:
            raise ValueError("Ventana negativa")
        route = self.config.energy_route
        used = [c for c in self.config.channels if c.key.split("/")[-1] in
                (("U", "I", "P", "Q") if route == "reported_power" else ("U", "I"))]
        if any(c.error.noise_amplitude for c in used):
            raise ValueError("S05 no define trayectoria entre innovaciones: contador ruidoso no implementado")
        count = len(self.scenario.initial_energy.phases)
        if a == b:
            return CircuitEnergy((EnergyCounters(),) * count)
        cuts = {a, b}
        cuts.update(s.start_s for s in self.scenario.segments if a < s.start_s < b)
        cuts.update(e.time_s for e in self.scenario.events if a < e.time_s < b)
        for c in used:
            for t in (() if not c.error.drift_rate else
                      (c.error.drift_start_s, c.error.drift_start_s + c.error.drift_duration_s)):
                if a < t < b:
                    cuts.add(t)
        cuts = sorted(cuts)
        totals = [[0.0] * 4 for _ in range(count + 1)]
        for lo, hi in zip(cuts, cuts[1:]):
            segment = self.scenario.segment_at(lo)
            def powers(t):
                vals, _ = self._values(segment.state_at(t), t, {})
                if route == "reported_power":
                    for key in tuple(vals):
                        if key.endswith(("/P", "/Q")):
                            vals[key] = self._profile(key).error.apply(vals[key], t)
                return vals
            # Cached vector integration also keeps phase/circuit sign separation.
            def f(t):
                vals = powers(t)
                result = []
                for scope in (*(f"c1/L{k}" for k in range(1, count + 1)), "c1"):
                    p, q = vals[f"{scope}/P"], vals[f"{scope}/Q"]
                    result.extend((max(p, 0), max(-p, 0), max(q, 0), max(-q, 0)))
                return tuple(x / 3600 for x in result)
            def simpson(x, y, fx, fm, fy):
                return tuple((y-x)/6*(u+4*v+w) for u, v, w in zip(fx, fm, fy))
            def refine(x, y, fx, fm, fy, area, tolerance, depth):
                mid = (x+y)/2
                fl, fr = f((x+mid)/2), f((mid+y)/2)
                left, right = simpson(x, mid, fx, fl, fm), simpson(mid, y, fm, fr, fy)
                delta = tuple(l+r-old for l, r, old in zip(left, right, area))
                if all(abs(d) <= 15*tolerance for d in delta):
                    return tuple(max(0, l+r+d/15) for l, r, d in zip(left, right, delta))
                if depth == 0:
                    raise ValueError("Integración instrumental no convergió")
                l = refine(x, mid, fx, fl, fm, left, tolerance/2, depth-1)
                r = refine(mid, y, fm, fr, fy, right, tolerance/2, depth-1)
                return tuple(u+v for u, v in zip(l, r))
            fx, fm, fy = f(lo), f((lo+hi)/2), f(hi)
            result = refine(lo, hi, fx, fm, fy, simpson(lo, hi, fx, fm, fy), 1e-9/(len(cuts)-1), 48)
            for k in range(count + 1):
                totals[k] = [u+v for u, v in zip(totals[k], result[4*k:4*k+4])]
        counters = tuple(EnergyCounters(*scope) for scope in totals)
        return CircuitEnergy(counters[:-1], counters[-1])

    def counters_at(self, time_s):
        if self.config.initial_counters is None:
            raise ValueError("Contador requiere época e iniciales explícitos")
        return self.config.initial_counters.plus(self.energy(0, time_s))

    def _quantize(self, c, value, t):
        variable = c.key.split("/")[-1]
        factor = OUTPUT_UNITS[c.output_unit][1]
        original = None if value is None else value / factor
        if original is not None and c.quantum is not None:
            quotient = _finite_real(original / c.quantum, "Cociente de cuantización")
            original = _finite_real(c.quantum * round(quotient), "Cuantizado")
        support = (0.0, t) if variable in COUNTERS else (t, t)
        return _OriginalReading(c, original, support)

    def _normalize(self, original_reading, nominal):
        c = original_reading.channel
        original = original_reading.value
        variable = c.key.split("/")[-1]
        factor = OUTPUT_UNITS[c.output_unit][1]
        canonical = None if original is None else _finite_real(original * factor, "Salida")
        invalid = False
        if canonical is not None:
            if variable in ("U", "ULL"):
                invalid = not 0 <= canonical <= 1.2 * nominal * (math.sqrt(3) if variable == "ULL" else 1)
            elif variable in ("I", "IN"):
                invalid = not 0 <= canonical <= (300 if variable == "IN" else 100)
            elif variable == "f":
                invalid = not 50 <= canonical <= 65
            elif variable == "FP":
                invalid = not 0 <= canonical <= 1
            elif variable in ("P", "Q", "S"):
                count = 1 if "/L" in c.key else len(self.scenario.initial_energy.phases)
                limit = 1.2 * nominal * 100 * count
                invalid = abs(canonical) > limit or (variable in ("P", "S") and canonical < 0)
            elif variable in COUNTERS:
                invalid = canonical < 0
        return Reading(c.key, canonical, UNITS[variable], original, c.output_unit, original_reading.support_s,
                       "not_evaluated" if canonical is None else "invalid" if invalid else
                       "unknown" if original_reading.temporal_unknown else "valid",
                       "undefined_zero_denominator" if canonical is None else "range_violation" if invalid else
                       "time_unknown" if original_reading.temporal_unknown else "no_diagnostic")

    def _reading(self, c, value, t, nominal):
        """Frontera de salida usada también por fixtures negativos explícitos."""
        return self._normalize(self._quantize(c, value, t), nominal)

    def capture(self, times):
        times = tuple(self.scenario._time(t) for t in times)
        if not times or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("Capturas deben estar ordenadas sin duplicación temporal")
        if any(j >= len(times) for j in self.config.silent_indices):
            raise ValueError("Silencio fuera del calendario")
        for r in self.config.retentions:
            if r.end_s > self.scenario.duration_s or r.source_index >= len(times) or times[r.source_index] >= r.start_s:
                raise ValueError("Retención sin antecedente o fuera del soporte")
        streams, seeds = {}, {}
        for c in self.config.channels:
            if c.error.noise_amplitude and not c.error.prescribed_u:
                source = c.shared_noise or c.key
                seed = self.config.random_identity.seed(source, c.error.profile_ref)
                seeds[source] = seed
                streams.setdefault(source, random.Random(seed))
            if c.error.prescribed_u and len(c.error.prescribed_u) != len(times):
                raise ValueError("Innovaciones prescritas no cubren calendario")
        produced, traces = {}, []
        for j, t in enumerate(times):
            innovations, common = {}, {}
            for c in self.config.channels:
                if not c.error.noise_amplitude:
                    continue
                source = c.shared_noise or c.key
                if source not in common:
                    common[source] = c.error.prescribed_u[j] if c.error.prescribed_u else streams[source].random()
                innovations[c.key] = c.error.noise_amplitude * (2*common[source]-1)
            state = self.scenario.state_at(t)
            values, primary_trace = self._values(state, t, innovations)
            if any(c.key.split("/")[-1] in COUNTERS for c in self.config.channels):
                energy = self.counters_at(t)
                for scope, counters in zip((*(f"c1/L{k}" for k in range(1, len(energy.phases)+1)), "c1"), (*energy.phases, energy.circuit)):
                    values.update({f"{scope}/{v}": getattr(counters, attr) for v, attr in COUNTERS.items()})
            readings, channel_traces, originals = [], [], {}
            for c in self.config.channels:
                value = values[c.key]
                variable = c.key.split("/")[-1]
                if variable not in ("U", "I") and value is not None:
                    value = c.error.apply(value, t, innovations.get(c.key, 0))
                original = self._quantize(c, value, t)
                raw = primary_trace.get(c.key, (values[c.key], None, None, value, innovations.get(c.key, 0)))
                trace = ChannelTrace(c.key, *raw, original.support_s)
                for r in self.config.retentions:
                    if c.key in r.channels and r.start_s <= t < r.end_s:
                        if r.source_index not in produced:
                            raise ValueError("Retención sin lectura producida anterior")
                        old_originals, old_traces = produced[r.source_index]
                        old = old_originals[c.key]
                        old_trace = next(x for x in old_traces if x.channel == c.key)
                        support = old.support_s if r.clock_policy == "original" else original.support_s if r.clock_policy == "current" else None
                        original = replace(old, support_s=support,
                                           temporal_unknown=(old.temporal_unknown if r.clock_policy == "original"
                                                             else r.clock_policy == "unknown"))
                        # Candidate (including hidden noise) remains in trace; support/link identifies held output.
                        trace = replace(trace, real_support_s=old_trace.real_support_s,
                                        retained_from_index=r.source_index)
                originals[c.key] = original
                # Una sola normalización del original seleccionado, después de I03.
                if j not in self.config.silent_indices:
                    readings.append(self._normalize(original, state.nominal_voltage_v))
                channel_traces.append(trace)
            capture = None if j in self.config.silent_indices else InstrumentCapture(
                f"{self.config.random_identity.run_id}/{j}", self.config.device, j, t, t, tuple(readings))
            if capture is not None:
                produced[j] = originals, tuple(channel_traces)
            traces.append(CaptureTrace(j, t, capture, tuple(channel_traces), tuple(sorted(seeds.items())), sys.version))
        return tuple(traces)
