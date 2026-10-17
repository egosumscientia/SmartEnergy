"""Transporte S05 A01/A02 puro, sobre capturas instrumentales ya producidas.

Tiempos relativos en segundos; horizonte inclusivo. Un lote pertenece a un
dispositivo y calendario instrumental. El resultado completo es privado del
ensayo; sólo deliveries representa lo recibido. No hay admisión, deduplicación,
diagnóstico, DTO S03, reloj, persistencia ni integración con infraestructura.
"""

from dataclasses import dataclass
import hashlib
import json
import random
import sys

from smartenergy.analysis.simulator_physics import _finite_real
from smartenergy.analysis.simulator_instrumentation import (
    CaptureTrace, InstrumentCapture, RandomIdentity, _text,
)


def _index(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Índice entero no negativo requerido")
    return value


def _time(value):
    value = _finite_real(value, "Tiempo")
    if value < 0:
        raise ValueError("Tiempo negativo")
    return value


def _calendar(entries):
    result = tuple((_index(j), _time(t)) for j, t in entries)
    if len(set(result)) != len(result):
        raise ValueError("Entrada de calendario duplicada")
    return result


@dataclass(frozen=True)
class FifoPolicy:
    """B, H, K y calendario explícitos; nunca reintenta automáticamente.

    retry_times contiene (índice de captura, tiempo de envío). Sólo se ejecuta
    si esa captura sigue en la cola y es su cabeza; edad desde producción.
    """
    capacity: int
    retention_s: float
    max_attempts: int
    retry_times: tuple[tuple[int, float], ...]

    def __post_init__(self):
        if not 1 <= _index(self.capacity) <= 10000 or not 1 <= _index(self.max_attempts) <= 10:
            raise ValueError("B/K fuera de dominio S05")
        h = _time(self.retention_s)
        if h == 0:
            raise ValueError("H debe ser positivo")
        object.__setattr__(self, "retention_s", h)
        object.__setattr__(self, "retry_times", _calendar(self.retry_times))


@dataclass(frozen=True)
class AcquisitionConfig:
    horizon_s: float
    lost_indices: tuple[int, ...] = ()
    delays: tuple[tuple[int, float], ...] = ()
    redeliveries: tuple[tuple[int, float], ...] = ()
    outages: tuple[tuple[float, float], ...] = ()
    fifo: FifoPolicy | None = None
    loss_probability: float | None = None
    random_identity: RandomIdentity | None = None
    random_channel: str | None = None
    random_profile: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "horizon_s", _time(self.horizon_s))
        lost = tuple(_index(j) for j in self.lost_indices)
        if len(set(lost)) != len(lost):
            raise ValueError("Índices perdidos duplicados")
        delays = _calendar(self.delays)
        if len({j for j, _ in delays}) != len(delays):
            raise ValueError("Retrasos ambiguos")
        outages = tuple((_time(a), _time(b)) for a, b in self.outages)
        if any(a >= b for a, b in outages):
            raise ValueError("Caída requiere intervalo positivo [a,b)")
        outages = tuple(sorted(outages))
        if any(a[1] > b[0] for a, b in zip(outages, outages[1:])):
            raise ValueError("Caídas superpuestas")
        if self.fifo is not None and not isinstance(self.fifo, FifoPolicy):
            raise ValueError("Política FIFO inválida")
        identity = self.random_identity
        if self.loss_probability is None:
            if any(x is not None for x in (identity, self.random_channel, self.random_profile)):
                raise ValueError("RNG sin perfil probabilístico")
        else:
            p = _finite_real(self.loss_probability, "Probabilidad")
            if not 0 <= p <= 1 or not isinstance(identity, RandomIdentity):
                raise ValueError("Bernoulli requiere p e identidad explícitos")
            _text(self.random_channel, "Canal RNG")
            _text(self.random_profile, "Perfil RNG")
            object.__setattr__(self, "loss_probability", p)
        for name, value in (("lost_indices", lost), ("delays", delays),
                            ("redeliveries", _calendar(self.redeliveries)), ("outages", outages)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class AcquisitionAttempt:
    capture: InstrumentCapture
    attempt_id: str
    number: int
    ordinal: int
    attempted_s: float
    kind: str
    outcome: str
    reason: str
    arrival_s: float | None


@dataclass(frozen=True)
class Delivery:
    """Recepción sin evaluación/admisión; capture conserva el origen completo.

    Cada recepción tiene ID independiente. kind y source_attempt_id enlazan
    reentregas explícitas sin clasificar contenidos ni deduplicarlos.
    """
    delivery_id: str
    capture: InstrumentCapture
    received_s: float
    attempt_id: str
    ordinal: int
    kind: str
    source_attempt_id: str


@dataclass(frozen=True)
class QueueEvent:
    index: int
    time_s: float
    reason: str


@dataclass(frozen=True)
class AcquisitionResult:
    """Artefacto privado: distingue silencio, pérdida, cola y tránsito al cierre."""
    config: AcquisitionConfig
    instrument_traces: tuple[CaptureTrace, ...]
    attempts: tuple[AcquisitionAttempt, ...]
    deliveries: tuple[Delivery, ...]
    never_received: tuple[int, ...]
    pending: tuple[int, ...]
    not_produced: tuple[int, ...]
    queue_events: tuple[QueueEvent, ...]
    loss_indices: tuple[int, ...]
    seed: int | None
    python_version: str


def acquire(traces, config: AcquisitionConfig):
    """Reproduce un calendario completo sin tocar física ni instrumentación.

    Cada captura producida inicia un envío en produced_s. Los retrasos por índice
    se aplican a cada envío; las reentregas fijan tiempos de envío adicionales.
    La caída se evalúa al intentar enviar, antes del retraso de transporte.
    Bernoulli consume un u por captura producida, en orden de índice, sin sorteos
    de reintento. Se conserva la decisión por captura. No se envía un silencio.
    """
    if not isinstance(config, AcquisitionConfig):
        raise ValueError("Configuración de adquisición requerida")
    traces = tuple(traces)
    captures, indices, ids, devices = {}, set(), set(), set()
    previous_index = previous_time = -1
    for trace in traces:
        if not isinstance(trace, CaptureTrace):
            raise ValueError("Se requieren trazas instrumentales")
        j, t = _index(trace.index), _time(trace.scheduled_s)
        if j <= previous_index or t <= previous_time or t > config.horizon_s:
            raise ValueError("Calendario instrumental desordenado o fuera del horizonte")
        previous_index, previous_time = j, t
        indices.add(j)
        c = trace.produced
        if c is None:
            continue
        if (not isinstance(c, InstrumentCapture) or c.index != j or c.scheduled_s != t
                or _time(c.produced_s) < t or c.produced_s > config.horizon_s):
            raise ValueError("Captura incompatible con calendario/producción")
        _text(c.capture_id, "ID captura")
        _text(c.device, "Dispositivo")
        if c.capture_id in ids:
            raise ValueError("Identidad de captura repetida/conflictiva")
        ids.add(c.capture_id)
        devices.add(c.device)
        keys = set()
        for r in c.readings:
            if r.channel in keys:
                raise ValueError("Canal instrumental repetido")
            keys.add(r.channel)
            if r.support_s is not None:
                a, b = map(_time, r.support_s)
                if a > b or b > c.produced_s:
                    raise ValueError("Producción anterior al soporte de medición")
        captures[j] = c
    if len(devices) > 1:
        raise ValueError("Un calendario de adquisición por dispositivo")
    if config.random_identity and devices and devices != {config.random_identity.device}:
        raise ValueError("Dispositivo RNG incompatible")
    fifo = config.fifo
    retries = () if fifo is None else fifo.retry_times
    refs = set(config.lost_indices) | {j for j, _ in (*config.delays, *config.redeliveries, *retries)}
    if not refs <= indices:
        raise ValueError("Índice fuera del calendario")
    # Una máscara A puede incluir un silencio para composición; no crea pérdida.
    if any(j not in captures for j, _ in (*config.redeliveries, *retries)):
        raise ValueError("No se puede reenviar una captura no producida")
    extra = [(t, j, "redelivery") for j, t in config.redeliveries]
    extra += [(t, j, "retry") for j, t in retries]
    if any(t <= captures[j].produced_s for t, j, _ in extra):
        raise ValueError("Reenvío debe ser posterior a producción/intento inicial")
    if len({(t, j) for t, j, _ in extra}) != len(extra):
        raise ValueError("Reenvíos ambiguos")
    seed = None
    losses = set(config.lost_indices)
    if config.loss_probability is not None:
        identity = config.random_identity
        payload = ["s05-rng-v1", str(identity.master), identity.run_id, "acquisition",
                   identity.device, config.random_channel, config.random_profile, identity.replicate]
        seed = int.from_bytes(hashlib.sha256(json.dumps(
            payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).digest(), "big")
        rng = random.Random(seed)
        for j in captures:
            if rng.random() < config.loss_probability:
                losses.add(j)

    # Ordinal estable de calendario: empates iniciales antes de reenvíos y luego
    # orden de índice. Recepción ordena por (tiempo, ordinal de intento ejecutado).
    events = [(c.produced_s, j, "initial") for j, c in captures.items()] + extra
    events.sort(key=lambda e: (e[0], e[2] != "initial", e[1]))
    delays = dict(config.delays)
    queue, queue_events, attempts, deliveries, counts = [], [], [], [], {}

    def discard(j, t, reason):
        queue.remove(j)
        queue_events.append(QueueEvent(j, t, reason))

    def expire(t):
        if fifo:
            for j in tuple(queue):
                if t - captures[j].produced_s > fifo.retention_s:
                    discard(j, t, "expired")

    for t, j, kind in events:
        if t > config.horizon_s:
            break
        expire(t)
        if kind == "retry":
            if j not in queue:
                continue
            if queue[0] != j:
                raise ValueError("Calendario de reintento viola FIFO")
        if fifo and counts.get(j, 0) >= fifo.max_attempts:
            if j in queue:
                discard(j, t, "attempt_limit")
            continue
        c = captures[j]
        number = counts[j] = counts.get(j, 0) + 1
        ordinal = len(attempts)
        attempt_id = f"{c.capture_id}/acquisition-{number}"
        blocked = any(a <= t < b for a, b in config.outages)
        reason = "link_down" if blocked else "loss" if j in losses else "delivered"
        arrival = None if reason != "delivered" else _time(t + delays.get(j, 0))
        if arrival is not None and delays.get(j, 0) > 0 and arrival <= t:
            raise ValueError("Retraso fuera de resolución numérica")
        outcome = "failed" if arrival is None else "received" if arrival <= config.horizon_s else "in_flight"
        attempts.append(AcquisitionAttempt(c, attempt_id, number, ordinal, t, kind, outcome, reason, arrival))
        if arrival is not None:
            if j in queue:
                discard(j, t, "sent")
            if outcome == "received":
                deliveries.append(Delivery(f"{c.capture_id}/attempt-{number}", c, arrival,
                                           attempt_id, ordinal, kind, f"{c.capture_id}/acquisition-1"))
        elif fifo:
            if number >= fifo.max_attempts:
                if j in queue:
                    discard(j, t, "attempt_limit")
                else:
                    queue_events.append(QueueEvent(j, t, "attempt_limit"))
            elif j not in queue:
                if len(queue) == fifo.capacity:
                    discard(queue[0], t, "overflow")
                queue.append(j)
                queue_events.append(QueueEvent(j, t, "retained"))
    expire(config.horizon_s)
    # Un calendario agotado no implica reintentos implícitos tras el horizonte.
    for j in tuple(queue):
        if not any(k == j and t > config.horizon_s for k, t in retries):
            discard(j, config.horizon_s, "calendar_exhausted")
    received = {d.capture.index for d in deliveries}
    outstanding = set(queue) | {a.capture.index for a in attempts if a.outcome == "in_flight"}
    pending = outstanding - received
    return AcquisitionResult(config, traces, tuple(attempts),
                             tuple(sorted(deliveries, key=lambda d: (d.received_s, d.ordinal))),
                             tuple(sorted(set(captures) - received - pending)), tuple(sorted(pending)),
                             tuple(sorted(indices - set(captures))), tuple(queue_events),
                             tuple(sorted(losses & set(captures))), seed, sys.version)
