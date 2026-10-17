"""Núcleo puro S02: fasores RMS, fuente/neutro ideales y admitancias de consumo.

Unidades SI: V, A, S (admitancia), W, var, VA, Hz y radianes. No representa
observaciones S03 ni un instrumento. Los estados sólo se construyen desde U/Y;
no hay RNG, configuración de entorno, reloj, persistencia ni modelo térmico.
"""

from dataclasses import dataclass, field
import math
from numbers import Real


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name}: se requiere un real finito")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name}: fuera del dominio numérico") from error
    if not math.isfinite(result):
        raise ValueError(f"{name}: se requiere un real finito")
    return result


@dataclass(frozen=True)
class PhaseState:
    """Resultado inmutable por rama; ángulos físicos indefinidos se expresan None."""

    voltage_phasor_v: complex
    admittance_s: complex
    current_phasor_a: complex
    voltage_rms_v: float
    current_rms_a: float
    voltage_angle_rad: float | None
    displacement_rad: float | None
    active_power_w: float
    reactive_power_var: float
    apparent_power_va: float
    power_factor: float | None


@dataclass(frozen=True)
class CircuitState:
    """Entrada U/Y y salida calculada; orden L1,L2,L3 y pares L1-L2,L2-L3,L3-L1.

    El constructor valida S02.6, calcula las ramas y sus agregados. La fuente
    mantiene ángulos ABC fijos; no acepta ángulos arbitrarios ni exportación.
    No existe corriente total escalar ni tensión total. En 1P2W no hay pares L-L.
    """

    topology: str
    voltages_v: tuple[float, ...]
    admittances_s: tuple[complex, ...]
    nominal_voltage_v: float = 120.0
    frequency_hz: float = 60.0
    phases: tuple[PhaseState, ...] = field(init=False)
    active_power_w: float = field(init=False)
    reactive_power_var: float = field(init=False)
    apparent_power_va: float = field(init=False)
    vector_apparent_power_va: float = field(init=False)
    power_factor: float | None = field(init=False)
    neutral_phasor_a: complex = field(init=False)
    neutral_current_a: float = field(init=False)
    line_voltages_v: tuple[float, ...] = field(init=False)
    motor_temperature_c: None = field(init=False, default=None)

    def __post_init__(self):
        if self.topology not in ("1P2W", "3P4W"):
            raise ValueError("Topología no aprobada por S02")
        nominal = _finite_real(self.nominal_voltage_v, "U nominal")
        frequency = _finite_real(self.frequency_hz, "Frecuencia")
        if nominal <= 0 or not math.isfinite(1.2 * nominal):
            raise ValueError("U nominal fuera de dominio")
        if not 50 <= frequency <= 65:
            raise ValueError("Frecuencia fuera de 50…65 Hz")
        voltages = tuple(self.voltages_v)
        admittances = tuple(self.admittances_s)
        count = 1 if self.topology == "1P2W" else 3
        if len(voltages) != count or len(admittances) != count:
            raise ValueError("U/Y incompletas para la topología")
        angles = (0.0, -2 * math.pi / 3, 2 * math.pi / 3)
        phases = []
        for voltage, admittance, angle in zip(voltages, admittances, angles):
            voltage = _finite_real(voltage, "U")
            if isinstance(admittance, bool) or not isinstance(admittance, (Real, complex)):
                raise ValueError("Y debe ser una admitancia finita")
            try:
                admittance = complex(admittance)
            except OverflowError as error:
                raise ValueError("Y fuera del dominio numérico") from error
            g, b = admittance.real, admittance.imag
            if not (math.isfinite(g) and math.isfinite(b)) or g < 0:
                raise ValueError("Y requiere G≥0 y B finita")
            if not 0 <= voltage <= 1.2 * nominal:
                raise ValueError("U fuera de 0…1,2 U nominal")
            # Escalar antes de hypot evita overflow de |Y| con Y finita y U=0.
            current_g, current_b = voltage * g, voltage * b
            current = math.hypot(current_g, current_b)
            p = voltage * current_g
            q = -voltage * current_b
            apparent = voltage * current
            if not all(math.isfinite(x) for x in (current, p, q, apparent)):
                raise ValueError("Estado fuera del dominio numérico")
            if current > 100:
                raise ValueError("I excede 100 A de ensayo; no se satura")
            if voltage > 0 and (
                (admittance != 0 and (current == 0 or apparent == 0))
                or (g > 0 and p == 0) or (b != 0 and q == 0)
            ):
                raise ValueError("Underflow: no representar carga no nula como cero")
            u_phasor = voltage * complex(math.cos(angle), math.sin(angle))
            i_phasor = admittance * u_phasor
            phases.append(PhaseState(
                u_phasor, admittance, i_phasor, voltage, current,
                angle if voltage else None,
                math.atan2(q, p) if apparent else None,
                p, q, apparent, p / apparent if apparent else None,
            ))
        phases = tuple(phases)
        try:
            p = math.fsum(phase.active_power_w for phase in phases)
            q = math.fsum(phase.reactive_power_var for phase in phases)
            apparent = math.fsum(phase.apparent_power_va for phase in phases)
        except OverflowError as error:
            raise ValueError("Agregado fuera del dominio numérico") from error
        if not all(math.isfinite(x) for x in (p, q, apparent)):
            raise ValueError("Agregado fuera del dominio numérico")
        neutral = -complex(
            math.fsum(phase.current_phasor_a.real for phase in phases),
            math.fsum(phase.current_phasor_a.imag for phase in phases),
        )
        lines = () if count == 1 else tuple(
            abs(phases[k].voltage_phasor_v - phases[l].voltage_phasor_v)
            for k, l in ((0, 1), (1, 2), (2, 0))
        )
        if not all(math.isfinite(value) for value in lines):
            raise ValueError("Tensión L-L fuera del dominio numérico")
        for name, value in (
            ("voltages_v", tuple(phase.voltage_rms_v for phase in phases)),
            ("admittances_s", tuple(phase.admittance_s for phase in phases)),
            ("nominal_voltage_v", nominal), ("frequency_hz", frequency),
            ("phases", phases), ("active_power_w", p), ("reactive_power_var", q),
            ("apparent_power_va", apparent), ("vector_apparent_power_va", math.hypot(p, q)),
            ("power_factor", p / apparent if apparent else None),
            ("neutral_phasor_a", neutral), ("neutral_current_a", abs(neutral)),
            ("line_voltages_v", lines),
        ):
            object.__setattr__(self, name, value)
