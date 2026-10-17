"""Integración pura S02 en segundos relativos y contadores Wh/varh por alcance.

Tramos [a,b) con P/Q constantes o lineales declaradas, límite izquierdo en b
explícito y cortes en cruces de signo. No interpola U/I ni define escenarios.
La aproximación suave usa trapecios sobre estados; sus errores de discretización
se verifican aparte de E06. No integra observaciones instrumentales o entregas.
"""

from dataclasses import dataclass
import math
from typing import Callable, Iterable

from smartenergy.analysis.simulator_physics import CircuitState, _finite_real


@dataclass(frozen=True)
class EnergyCounters:
    """Cuatro contadores no negativos; netos derivados, sin reset ni rollover."""

    active_import_wh: float = 0.0
    active_export_wh: float = 0.0
    reactive_inductive_varh: float = 0.0
    reactive_capacitive_varh: float = 0.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = _finite_real(getattr(self, name), name)
            if value < 0:
                raise ValueError("Contador inicial negativo")
            object.__setattr__(self, name, value)

    @property
    def active_net_wh(self):
        return self.active_import_wh - self.active_export_wh

    @property
    def reactive_net_varh(self):
        return self.reactive_inductive_varh - self.reactive_capacitive_varh

    def plus(self, increment: "EnergyCounters") -> "EnergyCounters":
        """Acumular sin mutación; conservar y verificar el incremento por separado."""
        return EnergyCounters(*(
            getattr(self, name) + getattr(increment, name)
            for name in self.__dataclass_fields__
        ))


@dataclass(frozen=True)
class CircuitEnergy:
    """Alcances distintos: fases en orden L1,L2,L3 y circuito ya agregado."""

    phases: tuple[EnergyCounters, ...]
    circuit: EnergyCounters = EnergyCounters()

    def __post_init__(self):
        phases = tuple(self.phases)
        if len(phases) not in (1, 3) or not all(
            isinstance(counter, EnergyCounters) for counter in (*phases, self.circuit)
        ):
            raise ValueError("Contadores incompletos para 1P2W/3P4W")
        object.__setattr__(self, "phases", phases)

    def plus(self, increment: "CircuitEnergy") -> "CircuitEnergy":
        if len(self.phases) != len(increment.phases):
            raise ValueError("No mezclar topologías")
        return CircuitEnergy(
            tuple(a.plus(b) for a, b in zip(self.phases, increment.phases)),
            self.circuit.plus(increment.circuit),
        )


def _bounds(start_s, end_s):
    start = _finite_real(start_s, "Inicio")
    end = _finite_real(end_s, "Fin")
    if not 0 <= start < end <= 86400:
        raise ValueError("Se requiere 0≤inicio<fin≤86400 s")
    return start, end


@dataclass(frozen=True)
class PowerInterval:
    """P/Q lineales en [start_s,end_s); right es el límite anterior al salto final.

    left/right son estados físicos válidos. La linealidad de sus potencias es
    una hipótesis explícita del llamador (p. ej. U fija, Y lineal). Un tramo
    constante usa el mismo estado en ambos extremos. Un estado nuevo en end_s
    pertenece al tramo siguiente y nunca sustituye right del tramo anterior.
    """

    start_s: float
    end_s: float
    left: CircuitState
    right: CircuitState

    def __post_init__(self):
        start, end = _bounds(self.start_s, self.end_s)
        if not isinstance(self.left, CircuitState) or not isinstance(self.right, CircuitState):
            raise ValueError("Los extremos deben ser estados físicos S02")
        if (self.left.topology, self.left.nominal_voltage_v) != (
            self.right.topology, self.right.nominal_voltage_v
        ):
            raise ValueError("No mezclar topologías ni nominales")
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)


def _positive_area(first, last, duration):
    """Área positiva de una recta, dividiendo en su raíz cuando cruza cero."""
    if first >= 0 and last >= 0:
        area = (first / 2 + last / 2) * duration / 3600
    elif first <= 0 and last <= 0:
        return 0.0
    else:
        positive = max(first, last)
        fraction = positive / (abs(first) + abs(last))
        area = positive * (duration / 3600) * fraction / 2
    if max(first, last) > 0 and area == 0:
        raise ValueError("Underflow de energía: no representar incremento no nulo como cero")
    return area


def _integrate_scope(p0, p1, q0, q1, duration):
    return EnergyCounters(
        _positive_area(p0, p1, duration),
        0.0,  # P≥0 en todo tramo consumidor aprobado.
        _positive_area(q0, q1, duration),
        _positive_area(-q0, -q1, duration),
    )


def integrate_intervals(
    intervals: Iterable[PowerInterval], *, start_s=None, end_s=None,
) -> CircuitEnergy:
    """Incrementos de ventana; tramos contiguos ordenados, sin huecos/solapes.

    Por defecto usa todo el soporte. Consulta a=b dentro de él devuelve cero
    sin crear un tramo de duración cero. Se rechazan ventanas fuera del soporte.
    Iniciales explícitos se añaden con CircuitEnergy.plus; el incremento sigue
    disponible para verificarlo incluso con un contador inicial muy grande.
    """
    intervals = tuple(intervals)
    if not intervals or len(intervals) > 1_000_000:
        raise ValueError("Se requieren 1…1000000 tramos")
    if not all(isinstance(item, PowerInterval) for item in intervals):
        raise ValueError("Se requieren tramos físicos")
    first = intervals[0]
    signature = (first.left.topology, first.left.nominal_voltage_v)
    for previous, item in zip(intervals, intervals[1:]):
        if previous.end_s != item.start_s:
            raise ValueError("Tramos desordenados, con huecos o solapes")
        if (item.left.topology, item.left.nominal_voltage_v) != signature:
            raise ValueError("Cambio de topología o nominal dentro del ensayo")
    start = first.start_s if start_s is None else _finite_real(start_s, "Inicio ventana")
    end = intervals[-1].end_s if end_s is None else _finite_real(end_s, "Fin ventana")
    if not first.start_s <= start <= end <= intervals[-1].end_s:
        raise ValueError("Ventana negativa o fuera del soporte")
    count = len(first.left.phases)
    increments = [[] for _ in range(count + 1)]
    for item in intervals:
        lo, hi = max(start, item.start_s), min(end, item.end_s)
        if lo >= hi:
            continue
        width = item.end_s - item.start_s
        a, b = (lo - item.start_s) / width, (hi - item.start_s) / width
        # El alcance total usa Q ya agregada; nunca suma contadores de fases.
        for index, (left, right) in enumerate(zip(
            (*item.left.phases, item.left), (*item.right.phases, item.right)
        )):
            p0, p1 = left.active_power_w, right.active_power_w
            q0, q1 = left.reactive_power_var, right.reactive_power_var
            increments[index].append(_integrate_scope(
                (1 - a) * p0 + a * p1, (1 - b) * p0 + b * p1,
                (1 - a) * q0 + a * q1, (1 - b) * q0 + b * q1, hi - lo,
            ))
    totals = tuple(EnergyCounters(*(
        math.fsum(getattr(value, name) for value in scope)
        for name in EnergyCounters.__dataclass_fields__
    )) for scope in increments)
    return CircuitEnergy(totals[:-1], totals[-1])


def integrate_smooth(
    state_at: Callable[[float], CircuitState], start_s: float, end_s: float,
    subintervals: int,
) -> CircuitEnergy:
    """Trapecio compuesto sobre P/Q de un perfil suave suministrado por el llamador.

    No detecta eventos ni certifica dominios entre nodos. El llamador debe dividir
    discontinuidades y proveer el límite izquierdo al final; tampoco la convergencia
    acredita por sí sola precisión. Contrastar contra integral independiente E06.
    """
    start, end = _bounds(start_s, end_s)
    if isinstance(subintervals, bool) or not isinstance(subintervals, int) or not (
        1 <= subintervals <= 1_000_000
    ):
        raise ValueError("Número de subintervalos fuera de 1…1000000")
    intervals = []
    previous_t, previous_state = start, state_at(start)
    for index in range(1, subintervals + 1):
        t = end if index == subintervals else start + (end - start) * index / subintervals
        state = state_at(t)
        intervals.append(PowerInterval(previous_t, t, previous_state, state))
        previous_t, previous_state = t, state
    return integrate_intervals(intervals)
