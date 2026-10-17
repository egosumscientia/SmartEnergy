"""Oráculos S02 independientes: constantes, fracciones, radicales e integrales.

Sólo biblioteca estándar y nuevo núcleo. No importa generadores ni aplicación.
No se ejecutan ensayos estadísticos: aún no se implementa instrumentación.
"""

from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import subprocess
import sys
import unittest

from smartenergy.analysis.simulator_physics import CircuitState
from smartenergy.analysis.simulator_energy import (
    CircuitEnergy, EnergyCounters, PowerInterval, integrate_intervals, integrate_smooth,
)


def mono(y, voltage=120):
    return CircuitState("1P2W", (voltage,), (y,))


def three(ys, voltages=(120, 120, 120)):
    return CircuitState("3P4W", voltages, ys)


class S02Assertions(unittest.TestCase):
    def numeric(self, actual, reference, absolute=1e-6, relative=1e-9):
        self.assertTrue(math.isfinite(actual))
        self.assertLessEqual(abs(actual - reference), absolute + relative * abs(reference))

    def power(self, state, p, q, s, fp):
        self.numeric(state.active_power_w, p)
        self.numeric(state.reactive_power_var, q)
        self.numeric(state.apparent_power_va, s)
        if fp is None:
            self.assertIsNone(state.power_factor)
        else:
            self.numeric(state.power_factor, fp, 1e-12)
            self.assertGreaterEqual(state.power_factor, 0)
            self.assertLessEqual(state.power_factor, 1)

    def counters(self, actual, expected):
        for name, reference in zip(EnergyCounters.__dataclass_fields__, expected):
            self.numeric(getattr(actual, name), reference)
        self.assertEqual(actual.active_export_wh, expected[1])


class PhysicsOracles(S02Assertions):
    def test_c01_c02_c03_monophase(self):
        for y, p, q, fp, phi in (
            (1 / 12, 1200, 0, 1, 0),
            (complex(1 / 15, -1 / 20), 960, 720, 4 / 5, math.atan(3 / 4)),
            (complex(1 / 15, 1 / 20), 960, -720, 4 / 5, -math.atan(3 / 4)),
        ):
            with self.subTest(q=q):
                state = mono(y)
                self.power(state, p, q, 1200, fp)
                self.power(state.phases[0], p, q, 1200, fp)
                self.numeric(state.phases[0].current_rms_a, 10, 1e-9)
                self.numeric(state.neutral_current_a, 10, 1e-9)
                self.numeric(state.phases[0].displacement_rad, phi, 1e-9)
                self.assertEqual(state.line_voltages_v, ())
                self.assertIsNone(state.motor_temperature_c)

    def test_c04_balanced_phasors(self):
        state = three((complex(1 / 15, -1 / 20),) * 3)
        self.power(state, 2880, 2160, 3600, 4 / 5)
        self.numeric(state.vector_apparent_power_va, 3600)
        self.numeric(state.neutral_current_a, 0, 1e-9)
        for phase in state.phases:
            self.power(phase, 960, 720, 1200, 4 / 5)
        for line in state.line_voltages_v:
            self.numeric(line, 120 * math.sqrt(3), 1e-9)
        for phase, real, imaginary in zip(
            state.phases, (120, -60, -60), (0, -60 * math.sqrt(3), 60 * math.sqrt(3))
        ):
            self.numeric(phase.voltage_phasor_v.real, real, 1e-9)
            self.numeric(phase.voltage_phasor_v.imag, imaginary, 1e-9)

    def test_c05_unbalanced_neutral(self):
        state = three((1 / 12, 1 / 6, 0))
        self.power(state, 3600, 0, 3600, 1)
        for phase, p in zip(state.phases, (1200, 2400, 0)):
            self.power(phase, p, 0, p, 1 if p else None)
        self.numeric(state.neutral_phasor_a.real, 0, 1e-9)
        self.numeric(state.neutral_phasor_a.imag, 10 * math.sqrt(3), 1e-9)
        self.numeric(state.neutral_current_a, 10 * math.sqrt(3), 1e-9)

    def test_c06_explicit_rest_is_exact(self):
        for state in (mono(0), three((0, 0, 0))):
            self.assertEqual(state.active_power_w, 0)
            self.assertEqual(state.reactive_power_var, 0)
            self.assertEqual(state.apparent_power_va, 0)
            self.assertIsNone(state.power_factor)
            self.assertEqual(state.neutral_current_a, 0)
            self.assertEqual(state.frequency_hz, 60)
            for phase in state.phases:
                self.assertEqual(phase.current_rms_a, 0)
                self.assertIsNone(phase.displacement_rad)
            result = integrate_intervals([PowerInterval(0, 600, state, state)])
            self.assertEqual(result.circuit, EnergyCounters())

    def test_c11_arithmetic_aggregation_and_phase_counters(self):
        state = three((complex(1 / 15, -1 / 20), complex(1 / 15, 1 / 20), 1 / 12))
        self.power(state, 3120, 0, 3600, 13 / 15)
        self.numeric(state.vector_apparent_power_va, 3120)
        self.numeric(state.neutral_current_a, 6 * math.sqrt(3) - 2, 1e-9)
        result = integrate_intervals([PowerInterval(0, 3600, state, state)])
        self.counters(result.circuit, (3120, 0, 0, 0))
        self.counters(result.phases[0], (960, 0, 720, 0))
        self.counters(result.phases[1], (960, 0, 0, 720))
        self.counters(result.phases[2], (1200, 0, 0, 0))
        self.assertEqual(result.circuit.reactive_net_varh, 0)
        self.numeric(sum(e.active_net_wh for e in result.phases), result.circuit.active_net_wh)
        self.numeric(sum(e.reactive_net_varh for e in result.phases), result.circuit.reactive_net_varh)

    def test_c14_fixed_admittance_voltage_and_reactive_axes(self):
        for voltage, current, p in ((120, 10, 1200), (60, 5, 300), (0, 0, 0)):
            state = mono(1 / 12, voltage)
            self.power(state, p, 0, p, 1 if voltage else None)
            self.numeric(state.phases[0].current_rms_a, current, 1e-9)
            if not voltage:
                self.assertIsNone(state.phases[0].voltage_angle_rad)
        for y, q in ((-1j / 12, 1200), (1j / 12, -1200)):
            state = mono(y)
            self.power(state, 0, q, 1200, 0)
            result = integrate_intervals([PowerInterval(0, 3600, state, state)])
            self.counters(result.circuit, (0, 0, max(q, 0), max(-q, 0)))

    def test_pure_reactive_compensation_keeps_total_pf_defined(self):
        state = three((-1j / 12, 1j / 12, 0))
        self.power(state, 0, 0, 2400, 0)
        self.assertEqual(state.vector_apparent_power_va, 0)
        self.assertEqual(integrate_intervals((PowerInterval(0, 3600, state, state),)).circuit, EnergyCounters())

    def test_unbalanced_line_voltage_uses_phasor_difference(self):
        state = three((1 / 12,) * 3, (120, 100, 120))
        for actual, reference in zip(state.line_voltages_v, (math.sqrt(36400), math.sqrt(36400), 120 * math.sqrt(3))):
            self.numeric(actual, reference, 1e-9)
        self.numeric(state.neutral_current_a, 5 / 3, 1e-9)
        self.power(state, 9700 / 3, 0, 9700 / 3, 1)

    def test_e06_near_zero_both_reactive_signs(self):
        for q in (6e-7, -6e-7):
            state = mono(complex(8e-7 / 14400, -q / 14400))
            phase = state.phases[0]
            for actual, expected in (
                (phase.active_power_w, 8e-7), (phase.reactive_power_var, q),
                (phase.apparent_power_va, 1e-6), (phase.current_rms_a, 1e-6 / 120),
            ):
                self.assertGreater(actual * expected, 0)
                self.assertLessEqual(abs((actual - expected) / expected), 1e-6)
            self.numeric(state.power_factor, 0.8, 1e-12)
        tiny = mono(1e-200)
        self.assertGreater(tiny.apparent_power_va, 0)
        self.assertEqual(tiny.power_factor, 1)


class EnergyOracles(S02Assertions):
    def setUp(self):
        self.inductive = mono(complex(1200 / 14400, -900 / 14400))
        self.capacitive = mono(complex(600 / 14400, 900 / 14400))

    def test_c07_constant_and_partition_independence(self):
        whole = integrate_intervals([PowerInterval(0, 600, self.inductive, self.inductive)])
        self.counters(whole.circuit, (200, 0, 150, 0))
        self.counters(whole.phases[0], (200, 0, 150, 0))
        parts = integrate_intervals([
            PowerInterval(0, 137, self.inductive, self.inductive),
            PowerInterval(137, 600, self.inductive, self.inductive),
        ])
        self.counters(parts.circuit, (200, 0, 150, 0))

    def test_c08_steps_and_half_open_boundaries(self):
        intervals = (
            PowerInterval(0, 600, self.inductive, self.inductive),
            PowerInterval(600, 900, self.capacitive, self.capacitive),
        )
        whole = integrate_intervals(intervals)
        self.counters(whole.circuit, (250, 0, 150, 75))
        self.numeric(whole.circuit.reactive_net_varh, 75)
        for start, end, expected in (
            (590, 600, (10 / 3, 0, 5 / 2, 0)),
            (600, 610, (5 / 3, 0, 0, 5 / 2)),
            (590, 610, (5, 0, 5 / 2, 5 / 2)),
            (600, 600, (0, 0, 0, 0)),
            (0, 0, (0, 0, 0, 0)), (900, 900, (0, 0, 0, 0)),
        ):
            with self.subTest(start=start, end=end):
                self.counters(integrate_intervals(intervals, start_s=start, end_s=end).circuit, expected)

    def test_c09_c10_direction_changes_initials_and_resume(self):
        capacitive = mono(complex(1200 / 14400, 900 / 14400))
        first = PowerInterval(0, 1800, self.inductive, self.inductive)
        second = PowerInterval(1800, 3600, capacitive, capacitive)
        increment = integrate_intervals((first, second))
        self.counters(increment.circuit, (1200, 0, 450, 450))
        initial = CircuitEnergy((EnergyCounters(100, 7, 20, 30),), EnergyCounters(100, 7, 20, 30))
        result = initial.plus(increment)
        self.counters(result.circuit, (1300, 7, 470, 480))
        self.numeric(result.circuit.active_net_wh, 1293)
        self.numeric(result.circuit.reactive_net_varh, -10)
        rest = mono(0)
        resumed = result.plus(integrate_intervals((PowerInterval(3600, 4200, rest, rest),)))
        self.assertEqual(resumed, result)
        self.assertEqual(initial.circuit, EnergyCounters(100, 7, 20, 30))
        early = initial.plus(integrate_intervals((first,)))
        for name in EnergyCounters.__dataclass_fields__:
            self.assertGreaterEqual(getattr(result.circuit, name), getattr(early.circuit, name))

    def test_c12_linear_reactive_zero_crossing(self):
        capacitive = mono(complex(1200 / 14400, 900 / 14400))
        interval = PowerInterval(0, 3600, self.inductive, capacitive)
        self.counters(integrate_intervals((interval,)).circuit, (1200, 0, 225, 225))
        self.counters(integrate_intervals((interval,), start_s=900, end_s=2700).circuit, (600, 0, 225 / 4, 225 / 4))
        # Asymmetric crossing catches an accidental split at the midpoint.
        short = mono(complex(1200 / 14400, 300 / 14400))
        asymmetric = PowerInterval(0, 3600, self.inductive, short)
        self.counters(integrate_intervals((asymmetric,)).circuit, (1200, 0, 675 / 2, 75 / 2))

    def test_c13_numerical_convergence_against_integral(self):
        def state_at(t):
            return mono((1200 + 600 * (t / 3600) ** 2) / 14400)

        errors = []
        for n, reference in ((100, 1400.01), (200, 1400.0025), (400, 1400.000625)):
            result = integrate_smooth(state_at, 0, 3600, n)
            self.counters(result.circuit, (reference, 0, 0, 0))
            errors.append(result.circuit.active_import_wh - 1400)
        for ratio in (errors[0] / errors[1], errors[1] / errors[2]):
            self.assertGreaterEqual(ratio, 3.5)
            self.assertLessEqual(ratio, 4.5)
        self.assertLessEqual(errors[-1], 1e-3 + 1e-6 * 1400)

    def test_linear_ramp_and_multiple_transitions_without_scenarios(self):
        low, high, rest = mono(1 / 24), mono(1 / 12), mono(0)
        ramp = PowerInterval(0, 60, low, high)
        self.counters(integrate_intervals((ramp,)).circuit, (15, 0, 0, 0))
        self.counters(integrate_intervals((ramp,), start_s=30).circuit, (35 / 4, 0, 0, 0))
        intervals = (ramp, PowerInterval(60, 120, high, high), PowerInterval(120, 180, rest, rest))
        self.counters(integrate_intervals(intervals).circuit, (35, 0, 0, 0))

    def test_large_initial_counter_does_not_hide_increment(self):
        state = mono(1 / 12)
        increment = integrate_intervals((PowerInterval(0, 1, state, state),))
        self.counters(increment.circuit, (1 / 3, 0, 0, 0))
        # The API exposes increment independently of final binary64 rounding.
        initial = EnergyCounters(1e18, 0, 0, 0)
        self.assertEqual(initial.plus(increment.circuit).active_import_wh, 1e18)
        self.numeric(increment.circuit.active_import_wh, 1 / 3)


class DomainAndIsolationTests(unittest.TestCase):
    def test_reject_invalid_physical_inputs(self):
        cases = (
            ("3P3W", (120,), (0,), {}),
            ("3P4W", (120,), (0,), {}),
            ("1P2W", (120, 120), (0, 0), {}),
            ("1P2W", (-1,), (0,), {}),
            ("1P2W", (145,), (0,), {}),
            ("1P2W", (float("nan"),), (0,), {}),
            ("1P2W", (120,), (-1,), {}),
            ("1P2W", (120,), (complex(0, float("inf")),), {}),
            ("1P2W", (120,), (1,), {}),
            ("1P2W", (120,), (True,), {}),
            ("1P2W", (True,), (0,), {}),
            ("1P2W", (120,), (0,), {"frequency_hz": 49}),
            ("1P2W", (120,), (0,), {"frequency_hz": float("inf")}),
            ("1P2W", (120,), (0,), {"nominal_voltage_v": 0}),
            ("1P2W", (1e-200,), (1e-200,), {}),
        )
        for topology, voltages, ys, kwargs in cases:
            with self.subTest(topology=topology, voltages=voltages, ys=ys, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    CircuitState(topology, voltages, ys, **kwargs)

    def test_alternative_nominal_and_domain_boundaries(self):
        state = CircuitState("1P2W", (277,), (10 / 277,), nominal_voltage_v=277, frequency_hz=50)
        self.assertAlmostEqual(state.active_power_w, 2770)
        state = CircuitState("1P2W", (144,), (100 / 144,), frequency_hz=65)
        self.assertEqual(state.phases[0].current_rms_a, 100)

    def test_numeric_overflow_and_underflow_are_explicit(self):
        # Finite Y is valid at U=0 even if its unscaled modulus overflows.
        self.assertEqual(mono(complex(1.7e308, 1.7e308), 0).apparent_power_va, 0)
        with self.assertRaises(ValueError):
            three((1e-306,) * 3, (1e308,) * 3)
        with self.assertRaises(ValueError):
            EnergyCounters(1e308).plus(EnergyCounters(1e308))
        tiny = mono(1e-320)
        with self.assertRaises(ValueError):
            integrate_intervals((PowerInterval(0, 1e-10, tiny, tiny),))

    def test_reject_invalid_time_windows_and_partitions(self):
        state = mono(0)
        for start, end in ((0, 0), (2, 1), (-1, 1), (0, 86401), (0, float("inf")), (float("nan"), 1)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                PowerInterval(start, end, state, state)
        interval = PowerInterval(0, 1, state, state)
        for intervals in (
            (), (interval, interval),
            (interval, PowerInterval(2, 3, state, state)),
            (interval, PowerInterval(1, 2, three((0, 0, 0)), three((0, 0, 0)))),
        ):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                integrate_intervals(intervals)
        for start, end in ((-1, 1), (0, 2), (1, 0), (float("nan"), 1)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                integrate_intervals((interval,), start_s=start, end_s=end)
        for n in (0, -1, 1_000_001, 2.0, True):
            with self.subTest(n=n), self.assertRaises(ValueError):
                integrate_smooth(lambda t: state, 0, 1, n)

    def test_counters_validate_and_states_are_immutable(self):
        for value in (-1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EnergyCounters(value)
        state = mono(1 / 12)
        with self.assertRaises(FrozenInstanceError):
            state.active_power_w = 0
        with self.assertRaises(FrozenInstanceError):
            state.phases[0].current_rms_a = 0
        inputs = [120]
        state = CircuitState("1P2W", inputs, [1 / 12])
        inputs[0] = 0
        self.assertEqual(state.voltages_v, (120,))
        with self.assertRaises(ValueError):
            CircuitEnergy(())
        with self.assertRaises(ValueError):
            CircuitEnergy((EnergyCounters(),)).plus(CircuitEnergy((EnergyCounters(),) * 3))

    def test_reproducibility_and_no_mutation_during_energy(self):
        a, b = mono(complex(1 / 15, -1 / 20)), mono(complex(1 / 15, 1 / 20))
        first = PowerInterval(0, 3600, a, b)
        self.assertEqual(a, mono(complex(1 / 15, -1 / 20)))
        self.assertEqual(integrate_intervals((first,)), integrate_intervals((first,)))
        self.assertEqual(a.active_power_w, 960)
        self.assertEqual(a.reactive_power_var, 720)

    def test_isolated_import_and_calculation_forbid_external_effects(self):
        root = Path(__file__).resolve().parents[1]
        script = r'''
import importlib.abc
import math
import random
import sys
import time
from unittest.mock import patch

class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'sqlalchemy', 'psycopg2', 'dotenv', 'numpy', 'pandas', 'sklearn', 'joblib'
        } or fullname.startswith((
            'smartenergy.config', 'smartenergy.infrastructure',
            'smartenergy.operations', 'smartenergy.application', 'smartenergy.interfaces'
        )):
            raise AssertionError('Dependencia no autorizada: ' + fullname)

def audit(event, args):
    if event.startswith(('socket.', 'subprocess.', 'os.system')):
        raise AssertionError('Efecto externo: ' + event)
    if event == 'open':
        path, mode, flags = args
        if str(path).endswith('.env') or flags & (1 | 2 | 64 | 512 | 1024):
            raise AssertionError('Acceso de configuración o escritura')

sys.meta_path.insert(0, BlockExternal())
sys.addaudithook(audit)
initial_rng = random.getstate()
with patch.object(time, 'sleep', side_effect=AssertionError('sleep')), \
     patch.object(time, 'time', side_effect=AssertionError('reloj')):
    from smartenergy.analysis.simulator_physics import CircuitState
    from smartenergy.analysis.simulator_energy import PowerInterval, integrate_intervals
    state = CircuitState('1P2W', (120,), (1/12,))
    result = integrate_intervals((PowerInterval(0, 600, state, state),))
    assert result.circuit.active_import_wh == 200
assert initial_rng == random.getstate()
'''
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=root,
            capture_output=True, text=True, timeout=20, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
