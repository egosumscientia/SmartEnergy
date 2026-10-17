"""Oráculos S04 independientes, exclusivamente unitarios y en memoria."""

from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import subprocess
import sys
import unittest

from smartenergy.analysis.simulator_physics import CircuitState
from smartenergy.analysis.simulator_energy import CircuitEnergy, EnergyCounters
from smartenergy.analysis.simulator_scenarios import (
    SCENARIO_IDS, PhysicalScenario, PhysicalSegment, build_scenario, sustained_condition,
)


class ScenarioOracles(unittest.TestCase):
    def numeric(self, actual, expected, absolute=1e-6):
        self.assertTrue(math.isfinite(actual))
        self.assertLessEqual(abs(actual - expected), absolute + 1e-9 * abs(expected))

    def counters(self, actual, expected):
        for field, value in zip(EnergyCounters.__dataclass_fields__, expected):
            self.numeric(getattr(actual, field), value)
        self.assertEqual(actual.active_export_wh, expected[1])

    def test_all_catalog_energy_oracles(self):
        expected = {
            "F01-REP": (0, 0, 0, 0), "F01-REP-24h": (0, 0, 0, 0),
            "F01-EST": (200, 0, 0, 0), "F01-EST-inductiva": (160, 0, 120, 0),
            "F01-EST-capacitiva": (160, 0, 0, 120), "F01-EST-3P4W": (480, 0, 360, 0),
            "F01-ESC": (250, 0, 150, 75), "F01-RAM": (35, 0, 0, 0),
            "F01-Q": (1200, 0, 225, 225), "F01-TUR": (6010, 0, 2700, 1807.5),
            "F02-SOB": (60, 0, 0, 0), "F03-SUB": (25, 0, 0, 0),
            "F03-SUB-interrupcion": (20, 0, 0, 0), "F04-CAR": (3600, 0, 0, 0),
            "F04-TEN": (9700 / 3, 0, 0, 0), "F04-COM": (3120, 0, 0, 0),
            "F05-ARR": (88 / 3, 0, 22, 0), "F02-F03-JUN": (40, 0, 0, 0),
            "F01-ESC-C09": (1200, 0, 450, 450), "F01-ESC-C10": (1200, 0, 450, 450),
            "F01-Q-C13": (1400.000625, 0, 0, 0),
        }
        self.assertEqual(set(expected), set(SCENARIO_IDS))
        for name, values in expected.items():
            with self.subTest(name=name):
                scenario = build_scenario(name)
                result = scenario.energy()
                self.counters(result.circuit, values)
                if len(result.phases) == 1:
                    self.counters(result.phases[0], values)
                self.assertEqual(scenario.reference, name + "@1")
                for counter in (*result.phases, result.circuit):
                    self.assertEqual(counter.active_export_wh, 0)

    def test_rest_and_initial_counters(self):
        scenario = build_scenario("F01-REP-24h")
        for t in (0, 1, 600, 86399, 86400):
            self.assertEqual(scenario.counters_at(t), scenario.initial_energy)
            state = scenario.state_at(t)
            self.assertEqual((state.active_power_w, state.reactive_power_var,
                              state.apparent_power_va, state.neutral_current_a), (0, 0, 0, 0))
            self.assertIsNone(state.power_factor)
        self.counters(scenario.counters_at(86400).circuit, (100, 7, 20, 30))
        self.counters(build_scenario("F01-ESC-C10").counters_at(3600).circuit,
                      (1300, 7, 470, 480))
        self.assertEqual(build_scenario("F01-Q").initial_energy.circuit, EnergyCounters())

    def test_stable_all_variants(self):
        for name, p, q, fp in (("F01-EST", 1200, 0, 1),
                                ("F01-EST-inductiva", 960, 720, .8),
                                ("F01-EST-capacitiva", 960, -720, .8)):
            state = build_scenario(name).state_at(100)
            self.numeric(state.active_power_w, p)
            self.numeric(state.reactive_power_var, q)
            self.numeric(state.apparent_power_va, 1200)
            self.numeric(state.phases[0].current_rms_a, 10, 1e-9)
            self.numeric(state.power_factor, fp, 1e-12)
        scenario = build_scenario("F01-EST-3P4W")
        state = scenario.state_at(0)
        for field, value in (("active_power_w", 2880), ("reactive_power_var", 2160),
                             ("apparent_power_va", 3600), ("power_factor", .8)):
            self.numeric(getattr(state, field), value, 1e-12 if field == "power_factor" else 1e-6)
        self.numeric(state.neutral_current_a, 0, 1e-9)
        for phase in scenario.energy().phases:
            self.counters(phase, (160, 0, 120, 0))

    def test_step_boundaries_and_windows(self):
        s = build_scenario("F01-ESC")
        self.numeric(s.state_at(600, side="left").active_power_w, 1200)
        self.numeric(s.state_at(math.nextafter(600, 0)).reactive_power_var, 900)
        self.numeric(s.state_at(600).active_power_w, 600)
        self.numeric(s.state_at(600).reactive_power_var, -900)
        self.numeric(s.state_at(600).apparent_power_va, 300 * math.sqrt(13))
        for a, b, expected in ((590, 600, (10 / 3, 0, 2.5, 0)),
                                (600, 610, (5 / 3, 0, 0, 2.5)),
                                (590, 610, (5, 0, 2.5, 2.5))):
            self.counters(s.energy(a, b).circuit, expected)
        self.numeric(s.energy(590, 610).circuit.active_net_wh * 3600 / 20, 900)
        self.numeric(s.energy(590, 610).circuit.reactive_net_varh, 0)
        event = s.events[1]
        self.assertEqual(event.time_s, 600)
        self.assertIn("carga", event.changes)
        self.assertNotIn("operación", event.changes)

    def test_ramp_and_reactive_crossing(self):
        s = build_scenario("F01-RAM")
        for t, p in ((0, 600), (15, 750), (30, 900), (60, 1200), (120, 1200)):
            state = s.state_at(t)
            self.numeric(state.active_power_w, p)
            self.numeric(state.phases[0].current_rms_a, p / 120, 1e-9)
        self.assertEqual(s.state_at(60, side="left"), s.state_at(60))
        self.counters(s.energy(30, 90).circuit, (37.5 / 2, 0, 0, 0))
        s = build_scenario("F01-Q")
        for t, q in ((0, 900), (900, 450), (1800, 0), (2700, -450), (3600, -900)):
            self.numeric(s.state_at(t).reactive_power_var, q)
        self.numeric(s.state_at(1800).power_factor, 1, 1e-12)
        self.counters(s.energy(900, 2700).circuit, (600, 0, 225 / 4, 225 / 4))
        crossing = s.events[1]
        self.assertEqual((crossing.time_s, crossing.changes, crossing.subjects),
                         (1800, ("cruce-reactivo",), ("c1/L1", "c1")))

    def test_shift_resume_multiple_boundaries(self):
        s = build_scenario("F01-TUR")
        self.counters(s.energy(14390, 18010).circuit, (20 / 3, 0, 2.5, 2.5))
        self.numeric(s.energy(14390, 18010).circuit.active_net_wh * 3600 / 3620, 1200 / 181)
        self.assertEqual(s.counters_at(14400), s.counters_at(18000))
        self.assertGreater(s.counters_at(18001).circuit.active_import_wh,
                           s.counters_at(18000).circuit.active_import_wh)
        for t, op in ((0, "reposo"), (3600, "operación"), (14400, "reposo"),
                       (18000, "operación"), (25200, "parada"), (25260, "reposo")):
            self.assertEqual(s.segment_at(t).operating_state, op)
        self.numeric(s.state_at(25230).active_power_w, 600)
        self.numeric(s.state_at(25230).reactive_power_var, -450)
        self.assertIsNone(s.state_at(25260).power_factor)

    def test_voltage_load_and_joint_events(self):
        for name, u, i, p in (("F02-SOB", 120, 20, 2400), ("F03-SUB", 60, 5, 300),
                              ("F03-SUB-interrupcion", 0, 0, 0), ("F02-F03-JUN", 60, 20, 1200)):
            s = build_scenario(name)
            for t in (30, 60, math.nextafter(90, 0)):
                state = s.state_at(t)
                self.numeric(state.voltages_v[0], u, 1e-9)
                self.numeric(state.phases[0].current_rms_a, i, 1e-9)
                self.numeric(state.active_power_w, p)
                self.assertEqual(state.frequency_hz, 60)
            self.numeric(s.state_at(90).active_power_w, 1200)
            self.numeric(s.state_at(30, side="left").active_power_w, 1200)
        s = build_scenario("F02-F03-JUN")
        self.assertEqual(len(s.conditions), 2)
        for event in s.events[1:-1]:
            self.assertIn("fuente", event.changes)
            self.assertIn("carga", event.changes)
            self.assertEqual(event.subjects, ("bus-a", "c1"))
        self.assertIsNone(build_scenario("F03-SUB-interrupcion").state_at(60).power_factor)

    def test_three_phase_neutral_and_lines(self):
        s = build_scenario("F04-CAR")
        state = s.state_at(0)
        self.numeric(state.neutral_current_a, 10 * math.sqrt(3), 1e-9)
        for phase, p, i, counter in zip(state.phases, (1200, 2400, 0), (10, 20, 0), s.energy().phases):
            self.numeric(phase.active_power_w, p)
            self.numeric(phase.current_rms_a, i, 1e-9)
            self.counters(counter, (p, 0, 0, 0))
        self.assertIsNone(state.phases[2].power_factor)
        s = build_scenario("F04-TEN")
        state = s.state_at(0)
        self.numeric(state.neutral_current_a, 5 / 3, 1e-9)
        for line, expected in zip(state.line_voltages_v, (math.sqrt(36400), math.sqrt(36400), 120 * math.sqrt(3))):
            self.numeric(line, expected, 1e-9)
        for counter, p in zip(s.energy().phases, (1200, 2500 / 3, 1200)):
            self.counters(counter, (p, 0, 0, 0))

    def test_reactive_compensation_keeps_scopes_distinct(self):
        s = build_scenario("F04-COM")
        state = s.state_at(0)
        self.numeric(state.active_power_w, 3120)
        self.numeric(state.reactive_power_var, 0)
        self.numeric(state.apparent_power_va, 3600)
        self.numeric(state.power_factor, 13 / 15, 1e-12)
        self.numeric(state.neutral_current_a, 6 * math.sqrt(3) - 2, 1e-9)
        for counter, values in zip(s.energy().phases, ((960, 0, 720, 0), (960, 0, 0, 720), (1200, 0, 0, 0))):
            self.counters(counter, values)
        self.assertEqual(s.conditions, ())

    def test_start_stop_envelope_continuity(self):
        s = build_scenario("F05-ARR")
        for t, a in ((0, 0), (5, 1), (10, 2), (20, 2), (25, 1.5), (30, 1),
                     (90, 1), (95, .5), (100, 0), (120, 0)):
            state = s.state_at(t)
            self.numeric(state.active_power_w, 960 * a)
            self.numeric(state.reactive_power_var, 720 * a)
            self.numeric(state.apparent_power_va, 1200 * a)
            self.numeric(state.phases[0].current_rms_a, 10 * a, 1e-9)
            if a:
                self.numeric(state.power_factor, .8, 1e-12)
            else:
                self.assertIsNone(state.power_factor)
        for t in (10, 20, 30, 90, 100):
            self.assertEqual(s.state_at(t), s.state_at(t, side="left"))
        self.counters(s.energy(0, 10).circuit, (8 / 3, 0, 2, 0))
        self.counters(s.energy(95, 105).circuit, (1 / 3, 0, 1 / 4, 0))
        for t, op in ((0, "arranque"), (30, "operación"), (90, "parada"), (100, "reposo")):
            self.assertEqual(s.segment_at(t).operating_state, op)
        self.assertEqual(s.conditions, ())  # el arranque no activa F02 automáticamente

    def test_conditions_scope_duration_confirmation(self):
        for name, kinds, start, end in (
            ("F02-SOB", ("sobrecorriente",), 30, 90),
            ("F03-SUB", ("subtensión",), 30, 90),
            ("F03-SUB-interrupcion", ("subtensión",), 30, 90),
            ("F02-F03-JUN", ("sobrecorriente", "subtensión"), 30, 90),
            ("F04-CAR", ("dispersión-corriente",), 0, 3600),
            ("F04-TEN", ("dispersión-tensión",), 0, 3600),
        ):
            s = build_scenario(name)
            self.assertEqual(tuple(c.kind for c in s.conditions), kinds)
            for c in s.conditions:
                self.assertEqual((c.start_s, c.end_s, c.confirmed_at_s, c.start_inclusive),
                                 (start, end, start + 10, True))

    def test_strict_thresholds_and_minimum_duration(self):
        for threshold, above in ((15, True), (108, False), (.10, True), (.20, True)):
            good = math.nextafter(threshold, math.inf if above else -math.inf)
            bad = math.nextafter(threshold, -math.inf if above else math.inf)
            for value, seconds, expected in ((threshold, 10, 0), (bad, 10, 0),
                                              (good, 9, 0), (good, 10, 1)):
                result = sustained_condition("test", "L1", ((0, seconds, value, value),),
                                             threshold=threshold, above=above)
                self.assertEqual(len(result), expected)
        # Cruce continuo: empieza después del punto de igualdad, no en el nodo de muestreo.
        c, = sustained_condition("test", "L1", ((0, 30, 0, 30),), threshold=15)
        self.assertEqual((c.start_s, c.end_s, c.confirmed_at_s, c.start_inclusive), (15, 30, 25, False))
        self.assertEqual(len(sustained_condition("test", "L1", ((0, 5, 20, 20), (5, 10, 20, 20)), threshold=15)), 1)
        self.assertEqual(sustained_condition("test", "L1", ((0, 9, 20, 20), (9, 10, 15, 15), (10, 19, 20, 20)), threshold=15), ())

    def test_c13_convergence_with_independent_integral(self):
        s = build_scenario("F01-Q-C13")
        errors = []
        for n, reference in ((100, 1400.01), (200, 1400.0025), (400, 1400.000625)):
            value = s.energy(subintervals=n).circuit.active_import_wh
            self.numeric(value, reference)
            errors.append(value - 1400)
        for ratio in (errors[0] / errors[1], errors[1] / errors[2]):
            self.assertGreaterEqual(ratio, 3.5)
            self.assertLessEqual(ratio, 4.5)
        self.assertLessEqual(errors[-1], .0024)
        self.numeric(s.state_at(1800).active_power_w, 1350)

    def test_partition_energy_monotonicity_and_invariants(self):
        for name in SCENARIO_IDS:
            if name.endswith("C13"):
                continue  # aproximación dependiente de discretización, ver control de convergencia
            s = build_scenario(name)
            cuts = sorted({0, s.duration_s, s.duration_s / 2, *(x.end_s for x in s.segments)})
            parts = [s.energy(a, b) for a, b in zip(cuts, cuts[1:])]
            for index, scope in enumerate((*s.energy().phases, s.energy().circuit)):
                for field in EnergyCounters.__dataclass_fields__:
                    self.numeric(sum(getattr((*x.phases, x.circuit)[index], field) for x in parts), getattr(scope, field))
            times = sorted({0, s.duration_s, *(x.start_s for x in s.segments),
                            *((x.start_s + x.end_s) / 2 for x in s.segments)})
            previous = s.initial_energy
            for t in times:
                state = s.state_at(t)
                self.assertIsNone(state.motor_temperature_c)
                self.numeric(state.active_power_w, sum(p.active_power_w for p in state.phases))
                self.numeric(state.reactive_power_var, sum(p.reactive_power_var for p in state.phases))
                self.numeric(state.apparent_power_va, sum(p.apparent_power_va for p in state.phases))
                for phase in state.phases:
                    self.numeric(phase.apparent_power_va, math.hypot(phase.active_power_w, phase.reactive_power_var))
                    self.assertGreaterEqual(phase.current_rms_a, 0)
                    self.assertLessEqual(phase.current_rms_a, 100)
                current = s.counters_at(t)
                for a, b in zip((*previous.phases, previous.circuit), (*current.phases, current.circuit)):
                    for field in EnergyCounters.__dataclass_fields__:
                        self.assertGreaterEqual(getattr(b, field) + 1e-9, getattr(a, field))
                previous = current

    def test_reproducibility_and_immutability(self):
        for name in SCENARIO_IDS:
            s = build_scenario(name)
            self.assertEqual(s, build_scenario(name))
            self.assertEqual(s.events, build_scenario(name).events)
            self.assertEqual(s.energy(), build_scenario(name).energy())
            with self.assertRaises(FrozenInstanceError):
                s.version = 2
            with self.assertRaises(FrozenInstanceError):
                s.segments[0].left.frequency_hz = 50
            with self.assertRaises(FrozenInstanceError):
                s.segments[0].end_s = 999
        segments = list(s.segments)
        copy = PhysicalScenario("copy", segments, s.initial_energy)
        segments.clear()
        self.assertEqual(copy.segments, s.segments)

    def test_invalid_queries_and_profiles(self):
        s = build_scenario("F01-ESC")
        for name in (None, [], "F01-EST-resistiva-otra", "F01-ESC@2", "C01-MAS"):
            with self.assertRaises(ValueError):
                build_scenario(name)
        for t in (-1, 901, float("nan"), float("inf"), True, "0"):
            with self.assertRaises(ValueError):
                s.state_at(t)
            with self.assertRaises(ValueError):
                s.counters_at(t)
        for a, b in ((0, 0), (900, 900), (10, 9), (-1, 1), (0, 901), (0, float("nan"))):
            with self.assertRaises(ValueError):
                s.energy(a, b)
        with self.assertRaises(ValueError):
            s.state_at(0, side="otro")
        for n in (0, -1, True, 2.0, 1_000_001):
            with self.assertRaises(ValueError):
                s.energy(subintervals=n)
        state = s.state_at(0)
        for a, b in ((0, 0), (2, 1), (-1, 1), (0, 86401)):
            with self.assertRaises(ValueError):
                PhysicalSegment(a, b, state, state)
        with self.assertRaises(ValueError):
            PhysicalSegment(0, 1, state, CircuitState("1P2W", (60,), (1 / 12,)))
        for segments in ((), (PhysicalSegment(1, 2, state, state),),
                         (PhysicalSegment(0, 1, state, state), PhysicalSegment(2, 3, state, state)),
                         (PhysicalSegment(0, 2, state, state), PhysicalSegment(1, 3, state, state))):
            with self.assertRaises(ValueError):
                PhysicalScenario("bad", segments, s.initial_energy)
        with self.assertRaises(ValueError):
            PhysicalScenario("bad", s.segments, CircuitEnergy((EnergyCounters(),) * 3))
        for spans in (((0, 0, 20, 20),), ((0, 10, float("nan"), 20),), ((0, 5, 20, 20), (6, 10, 20, 20))):
            with self.assertRaises(ValueError):
                sustained_condition("test", "L1", spans, threshold=15)

    def test_isolation_import_and_every_scenario(self):
        script = r'''
import importlib.abc
import random
import sys
import time
from unittest.mock import patch
class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sqlalchemy', 'psycopg2', 'dotenv', 'numpy', 'pandas', 'sklearn', 'joblib'} or fullname.startswith(('smartenergy.config', 'smartenergy.infrastructure', 'smartenergy.operations', 'smartenergy.application', 'smartenergy.interfaces')):
            raise AssertionError(fullname)
def audit(event, args):
    if event.startswith(('socket.', 'subprocess.', 'os.system')):
        raise AssertionError(event)
    if event == 'open':
        path, mode, flags = args
        if str(path).endswith('.env') or flags & (1 | 2 | 64 | 512 | 1024):
            raise AssertionError('I/O no autorizado')
sys.meta_path.insert(0, BlockExternal())
sys.addaudithook(audit)
rng = random.getstate()
with patch.object(time, 'time', side_effect=AssertionError('reloj')), patch.object(time, 'sleep', side_effect=AssertionError('sleep')):
    from smartenergy.analysis.simulator_scenarios import SCENARIO_IDS, build_scenario
    for name in SCENARIO_IDS:
        s = build_scenario(name)
        s.state_at(0)
        s.state_at(s.duration_s)
        s.energy()
        s.counters_at(s.duration_s)
        s.events
assert random.getstate() == rng
'''
        result = subprocess.run([sys.executable, "-B", "-c", script],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True,
                                text=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
