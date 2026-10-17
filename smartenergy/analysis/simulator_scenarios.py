"""Catálogo físico S04@1 en memoria; sin observaciones, adquisición ni diagnóstico.

Tiempo relativo al origen fijo 2026-10-17T12:00:00Z. Estados en [0,T),
consulta terminal T = límite izquierdo (no hay continuación implícita).
Los perfiles admiten U fija y Y afín; C13 usa Y cuadrática monótona y trapecios
explícitos. Esto garantiza el dominio entre extremos, sin interpolar U o I.
"""

from dataclasses import dataclass

from smartenergy.analysis.simulator_physics import CircuitState, _finite_real
from smartenergy.analysis.simulator_energy import (
    CircuitEnergy, EnergyCounters, PowerInterval, integrate_intervals, integrate_smooth,
)


@dataclass(frozen=True)
class PhysicalSegment:
    """Tramo con U fija, Y afín o cuadrática en la fracción temporal local.

    right es el límite izquierdo final; quadratic sólo representa el control C13.
    La trayectoria de Y es un segmento convexo: I no supera sus extremos.
    """

    start_s: float
    end_s: float
    left: CircuitState
    right: CircuitState
    operating_state: str = "operación"
    quadratic: bool = False

    def __post_init__(self):
        interval = PowerInterval(self.start_s, self.end_s, self.left, self.right)
        if (self.left.voltages_v != self.right.voltages_v
                or self.left.frequency_hz != self.right.frequency_hz):
            raise ValueError("Cada tramo requiere fuente fija")
        if self.operating_state not in ("reposo", "operación", "rampa", "arranque", "parada"):
            raise ValueError("Estado operativo no aprobado")
        if not isinstance(self.quadratic, bool):
            raise ValueError("Perfil cuadrático debe ser booleano")
        object.__setattr__(self, "start_s", interval.start_s)
        object.__setattr__(self, "end_s", interval.end_s)

    def state_at(self, time_s):
        t = _finite_real(time_s, "Tiempo físico")
        if not self.start_s <= t <= self.end_s:
            raise ValueError("Tiempo fuera del tramo")
        if t == self.start_s:
            return self.left
        if t == self.end_s:
            return self.right
        fraction = (t - self.start_s) / (self.end_s - self.start_s)
        if self.quadratic:
            fraction *= fraction
        ys = tuple((1 - fraction) * a + fraction * b for a, b in zip(
            self.left.admittances_s, self.right.admittances_s
        ))
        return CircuitState(self.left.topology, self.left.voltages_v, ys,
                            self.left.nominal_voltage_v, self.left.frequency_hz)


@dataclass(frozen=True)
class PhysicalEvent:
    """Frontera puntual; los cambios simultáneos comparten un único evento."""

    time_s: float
    changes: tuple[str, ...]
    subjects: tuple[str, ...]


@dataclass(frozen=True)
class PhysicalCondition:
    """Verdad de ensayo; confirmada tras 10 s, nunca etiqueta del producto.

    Las banderas conservan la desigualdad estricta en un cruce continuo.
    El soporte siempre termina antes del estado nuevo en end_s.
    """

    kind: str
    subject: str
    start_s: float
    end_s: float
    start_inclusive: bool
    confirmed_at_s: float


def sustained_condition(kind, subject, spans, *, threshold, above=True):
    """Evaluar señal escalar afín (a,b,x(a),x(b-)) con umbral estricto y 10 s.

    No muestrea ni aplica tolerancias al umbral. Sólo une tramos contiguos
    cuando el punto de unión pertenece a la condición. Devuelve intervalos
    completos confirmados, preservando el inicio físico previo a confirmación.
    """
    threshold = _finite_real(threshold, "Umbral")
    if not isinstance(above, bool):
        raise ValueError("Dirección de umbral inválida")
    runs = []
    previous_end = None
    for a, b, first, last in spans:
        a, b = _finite_real(a, "Inicio"), _finite_real(b, "Fin")
        if not 0 <= a < b <= 86400 or (previous_end is not None and a != previous_end):
            raise ValueError("Tramos escalares inválidos")
        previous_end = b
        x = _finite_real(first, "Valor inicial") - threshold
        y = _finite_real(last, "Límite final") - threshold
        if not above:
            x, y = -x, -y
        if x <= 0 and y <= 0:
            continue
        lo, hi, inclusive = a, b, x > 0
        if x <= 0:
            lo = a + (b - a) * (-x) / (y - x)
        elif y < 0:
            hi = a + (b - a) * x / (x - y)
        if runs and runs[-1][1] == lo and inclusive:
            old = runs.pop()
            runs.append((old[0], hi, old[2]))
        else:
            runs.append((lo, hi, inclusive))
    return tuple(PhysicalCondition(kind, subject, a, b, inclusive, a + 10)
                 for a, b, inclusive in runs if b - a >= 10)


@dataclass(frozen=True)
class PhysicalScenario:
    """Definición física, no manifiesto S03. Usar build_scenario para IDs S04.

    energy requiere ventana positiva; counters_at permite t=0 y añade E0.
    state_at(t, side='left') consulta el límite anterior a una frontera interna.
    En 0 ambos lados devuelven el estado inicial definido; en T el terminal.
    Fuera de [0,T] todas las consultas fallan. Sin reloj, RNG o I/O.
    """

    scenario_id: str
    segments: tuple[PhysicalSegment, ...]
    initial_energy: CircuitEnergy
    conditions: tuple[PhysicalCondition, ...] = ()
    version: int = 1

    def __post_init__(self):
        segments = tuple(self.segments)
        if not segments or not all(isinstance(s, PhysicalSegment) for s in segments):
            raise ValueError("Escenario sin tramos físicos")
        if segments[0].start_s != 0:
            raise ValueError("El soporte empieza en cero")
        integrate_intervals(tuple(PowerInterval(s.start_s, s.end_s, s.left, s.right)
                                  for s in segments), start_s=0, end_s=0)
        if not isinstance(self.initial_energy, CircuitEnergy) or len(
            self.initial_energy.phases
        ) != len(segments[0].left.phases):
            raise ValueError("Iniciales incompatibles")
        if not isinstance(self.scenario_id, str) or not self.scenario_id or self.version != 1:
            raise ValueError("Identidad inválida")
        conditions = tuple(self.conditions)
        for c in conditions:
            if (not isinstance(c, PhysicalCondition) or not
                    0 <= c.start_s < c.confirmed_at_s <= c.end_s <= segments[-1].end_s
                    or c.confirmed_at_s != c.start_s + 10):
                raise ValueError("Condición inválida")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "conditions", conditions)

    @property
    def reference(self):
        return f"{self.scenario_id}@{self.version}"

    @property
    def duration_s(self):
        return self.segments[-1].end_s

    @property
    def configuration_ref(self):
        return "s04-1p@1" if len(self.initial_energy.phases) == 1 else "s04-3p@1"

    def _time(self, time_s):
        t = _finite_real(time_s, "Tiempo físico")
        if not 0 <= t <= self.duration_s:
            raise ValueError("Tiempo fuera del soporte")
        return t

    def segment_at(self, time_s, *, side="right"):
        t = self._time(time_s)
        if side not in ("left", "right"):
            raise ValueError("Lado inválido")
        for segment in self.segments:
            if t < segment.end_s or (t == segment.end_s and side == "left"):
                return segment
        return self.segments[-1]

    def state_at(self, time_s, *, side="right"):
        return self.segment_at(time_s, side=side).state_at(time_s)

    def energy(self, start_s=0, end_s=None, *, subintervals=400):
        """Incrementos, no lecturas acumuladas. C13: N trapecios por tramo recortado.

        El N explícito permite verificar convergencia; 400 es el control aprobado.
        Cualquier discontinuidad se divide antes de llamar al núcleo.
        """
        a = self._time(start_s)
        b = self.duration_s if end_s is None else self._time(end_s)
        if a >= b:
            raise ValueError("Ventana requiere duración positiva")
        if isinstance(subintervals, bool) or not isinstance(subintervals, int) or not 1 <= subintervals <= 1_000_000:
            raise ValueError("Subintervalos inválidos")
        result = CircuitEnergy((EnergyCounters(),) * len(self.initial_energy.phases))
        for s in self.segments:
            lo, hi = max(a, s.start_s), min(b, s.end_s)
            if lo >= hi:
                continue
            if s.quadratic:
                delta = integrate_smooth(s.state_at, lo, hi, subintervals)
            else:
                delta = integrate_intervals((PowerInterval(s.start_s, s.end_s, s.left, s.right),),
                                            start_s=lo, end_s=hi)
            result = result.plus(delta)
        return result

    def counters_at(self, time_s, *, subintervals=400):
        t = self._time(time_s)
        return self.initial_energy if t == 0 else self.initial_energy.plus(
            self.energy(0, t, subintervals=subintervals))

    @property
    def events(self):
        result = [PhysicalEvent(0, ("inicio",), ("c1",))]
        for old, new in zip(self.segments, self.segments[1:]):
            changes = []
            if old.right.voltages_v != new.left.voltages_v:
                changes.append("fuente")
            if old.right.admittances_s != new.left.admittances_s:
                changes.append("carga")
            if old.operating_state != new.operating_state:
                changes.append("operación")
            # Una frontera continua puede cambiar la pendiente del perfil.
            changes.append("perfil")
            result.append(PhysicalEvent(new.start_s, tuple(changes),
                                        ("bus-a", "c1") if "fuente" in changes else ("c1",)))
        for segment in self.segments:
            crossings = {}
            for subject, left, right in zip(
                (*(f"c1/L{k + 1}" for k in range(len(segment.left.phases))), "c1"),
                (*segment.left.phases, segment.left),
                (*segment.right.phases, segment.right),
            ):
                q0, q1 = left.reactive_power_var, right.reactive_power_var
                if (q0 > 0 > q1) or (q0 < 0 < q1):
                    fraction = abs(q0) / (abs(q0) + abs(q1))
                    if segment.quadratic:
                        fraction = fraction ** 0.5
                    t = segment.start_s + (segment.end_s - segment.start_s) * fraction
                    crossings.setdefault(t, []).append(subject)
            result.extend(PhysicalEvent(t, ("cruce-reactivo",), tuple(subjects))
                          for t, subjects in crossings.items())
        result.append(PhysicalEvent(self.duration_s, ("fin",), ("c1",)))
        return tuple(sorted(result, key=lambda event: event.time_s))


def _state(y, u=120):
    ys = y if isinstance(y, tuple) else (y,)
    us = u if isinstance(u, tuple) else (u,) * len(ys)
    return CircuitState("1P2W" if len(ys) == 1 else "3P4W", us, ys)


# Sólo identidades con parámetros fijos; no overrides silenciosos bajo un ID @1.
SCENARIO_IDS = (
    "F01-REP", "F01-REP-24h", "F01-EST", "F01-EST-inductiva",
    "F01-EST-capacitiva", "F01-EST-3P4W", "F01-ESC", "F01-RAM", "F01-Q",
    "F01-TUR", "F02-SOB", "F03-SUB", "F03-SUB-interrupcion", "F04-CAR",
    "F04-TEN", "F04-COM", "F05-ARR", "F02-F03-JUN",
    "F01-ESC-C09", "F01-ESC-C10", "F01-Q-C13",
)


def build_scenario(scenario_id):
    """Construir uno de SCENARIO_IDS, versiones @1, sin parámetros libres.

    F01-EST es resistiva. C09/C10/C13 son controles S02 citados por S04,
    conservan entradas y soporte; sus sufijos son nombres locales explícitos.
    No genera run_id, manifiestos, serializaciones ni observaciones S03.
    """
    if not isinstance(scenario_id, str) or scenario_id not in SCENARIO_IDS:
        raise ValueError("Escenario/variante no aprobado")
    r, rest = _state(1 / 12), _state(0)
    ind, cap = _state(complex(1200, -900) / 14400), _state(complex(1200, 900) / 14400)
    segments = []
    initial = EnergyCounters()

    def add(end, left, right=None, op="operación", quadratic=False):
        segments.append(PhysicalSegment(segments[-1].end_s if segments else 0,
                                       end, left, left if right is None else right, op, quadratic))

    if scenario_id.startswith("F01-REP"):
        add(86400 if scenario_id.endswith("24h") else 600, rest, op="reposo")
        if scenario_id.endswith("24h"):
            initial = EnergyCounters(100, 7, 20, 30)
    elif scenario_id.startswith("F01-EST"):
        ys = {"F01-EST": 1 / 12, "F01-EST-inductiva": complex(1 / 15, -1 / 20),
              "F01-EST-capacitiva": complex(1 / 15, 1 / 20),
              "F01-EST-3P4W": (complex(1 / 15, -1 / 20),) * 3}
        add(600, _state(ys[scenario_id]))
    elif scenario_id == "F01-ESC":
        add(600, ind)
        add(900, _state(complex(600, 900) / 14400))
    elif scenario_id in ("F01-ESC-C09", "F01-ESC-C10"):
        add(1800, ind)
        add(3600, cap)
        if scenario_id.endswith("C10"):
            initial = EnergyCounters(100, 7, 20, 30)
    elif scenario_id == "F01-RAM":
        add(60, _state(1 / 24), r, "rampa")
        add(120, r)
    elif scenario_id == "F01-Q":
        add(3600, ind, cap)
    elif scenario_id == "F01-Q-C13":
        add(3600, r, _state(1 / 8), quadratic=True)
    elif scenario_id == "F01-TUR":
        for end, state, op in ((3600, rest, "reposo"), (14400, ind, "operación"),
                               (18000, rest, "reposo"), (25200, cap, "operación")):
            add(end, state, op=op)
        add(25260, cap, rest, "parada")
        add(28800, rest, op="reposo")
    elif scenario_id in ("F02-SOB", "F03-SUB", "F03-SUB-interrupcion", "F02-F03-JUN"):
        middle = {"F02-SOB": _state(1 / 6), "F03-SUB": _state(1 / 12, 60),
                  "F03-SUB-interrupcion": _state(1 / 12, 0),
                  "F02-F03-JUN": _state(1 / 3, 60)}[scenario_id]
        add(30, r)
        add(90, middle)
        add(120, r)
    elif scenario_id.startswith("F04"):
        if scenario_id == "F04-CAR":
            state = _state((1 / 12, 1 / 6, 0))
        elif scenario_id == "F04-TEN":
            state = _state((1 / 12,) * 3, (120, 100, 120))
        else:
            state = _state((complex(1 / 15, -1 / 20), complex(1 / 15, 1 / 20), 1 / 12))
        add(3600, state)
    else:  # F05-ARR: envolvente RMS prescrita, no motor dinámico.
        base, peak = _state(complex(1 / 15, -1 / 20)), _state(complex(2 / 15, -1 / 10))
        add(10, rest, peak, "arranque")
        add(20, peak, op="arranque")
        add(30, peak, base, "arranque")
        add(90, base)
        add(100, base, rest, "parada")
        add(120, rest, op="reposo")

    conditions = []

    def condition(kind, subject, getter, threshold, above=True):
        spans = tuple((s.start_s, s.end_s, getter(s.left), getter(s.right)) for s in segments)
        conditions.extend(sustained_condition(kind, subject, spans, threshold=threshold, above=above))

    if scenario_id in ("F02-SOB", "F02-F03-JUN"):
        condition("sobrecorriente", "c1/L1", lambda s: s.phases[0].current_rms_a, 15)
    if scenario_id in ("F03-SUB", "F03-SUB-interrupcion", "F02-F03-JUN"):
        condition("subtensión", "bus-a/L1-N", lambda s: s.voltages_v[0], 108, False)
    if scenario_id in ("F04-CAR", "F04-TEN"):
        condition("dispersión-tensión", "bus-a", lambda s: (max(s.voltages_v) - min(s.voltages_v)) / 120, 0.10)
        condition("dispersión-corriente", "c1", lambda s: (
            max(p.current_rms_a for p in s.phases) - min(p.current_rms_a for p in s.phases)) / 10, 0.20)
    count = len(segments[0].left.phases)
    return PhysicalScenario(scenario_id, tuple(segments), CircuitEnergy((initial,) * count, initial), tuple(conditions))
