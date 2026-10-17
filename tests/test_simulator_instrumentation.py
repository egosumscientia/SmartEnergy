"""Oráculos independientes S05. Sólo núcleo/escenarios/instrumentación y stdlib."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

from smartenergy.analysis.simulator_energy import CircuitEnergy, EnergyCounters
from smartenergy.analysis.simulator_scenarios import build_scenario
from smartenergy.analysis.simulator_physics import CircuitState
from smartenergy.analysis.simulator_instrumentation import (
    RandomIdentity, TransformerRatios, ErrorProfile, ChannelConfig, Retention,
    InstrumentConfig, VirtualInstrument,
)

I = "c1/L1/I"
U = "bus-a/L1-N/U"
P, Q, S, FP = ("c1/" + v for v in ("P", "Q", "S", "FP"))


def instrument(scenario="F01-EST", *, channels=(), error=None, ratios=None,
               retentions=(), silent=(), counters=False, route="primary", replicate="0", device="sim-a"):
    s = build_scenario(scenario)
    cs = (ChannelConfig(I, error or ErrorProfile(), ratios or TransformerRatios()),
          ChannelConfig(U), *(ChannelConfig(k) for k in (P, Q, S, FP)), *channels)
    cfg = InstrumentConfig(device, cs, RandomIdentity(20261017, "s05-i01-uni", device, replicate),
                           retentions, silent, "epoch-0" if counters else None,
                           CircuitEnergy((EnergyCounters(),)*len(s.initial_energy.phases)) if counters else None, route)
    return VirtualInstrument(s, cfg)


class InstrumentationOracles(unittest.TestCase):
    def numeric(self, value, expected, absolute=1e-6, relative=1e-9):
        self.assertTrue(math.isfinite(value))
        self.assertLessEqual(abs(value-expected), absolute+relative*abs(expected))

    def values(self, capture, i, p, q, s, fp):
        for key, value in ((I, i), (P, p), (Q, q), (S, s), (FP, fp)):
            if value is None:
                self.assertIsNone(capture.channel(key).value)
            else:
                self.numeric(capture.channel(key).value, value, 1e-12 if key == FP else 1e-6)

    def counters(self, value, expected):
        for field, ref in zip(EnergyCounters.__dataclass_fields__, expected):
            self.numeric(getattr(value, field), ref)

    def test_i01_ac_prescribed_innovations(self):
        v = instrument(error=ErrorProfile("I01-AC@1", 10, noise_amplitude=.1,
                                         prescribed_u=(0, .25, .5, .75)))
        for trace, i, p in zip(v.capture((0, 1, 2, 3)), (9.9, 9.95, 10, 10.05), (1188, 1194, 1200, 1206)):
            self.values(trace.produced, i, p, 0, p, 1)
            self.assertEqual(trace.seeds, ())
        self.counters(v.scenario.energy().circuit, (200, 0, 0, 0))
        noisy_result = ChannelConfig(P, ErrorProfile("result@1", 1200, noise_amplitude=1))
        cfg = replace(v.config, channels=tuple(c for c in v.config.channels if c.key != P)+(noisy_result,))
        with self.assertRaisesRegex(ValueError, "Ruido derivado"):
            VirtualInstrument(v.scenario, cfg)

    def test_i01_uni_predeclared_simultaneous_controls(self):
        v = instrument(error=ErrorProfile("I01-UNI@1", 10, noise_amplitude=.1))
        n, amplitude, alpha = 10000, .1, .001
        traces = v.capture(tuple(600*j/n for j in range(n)))
        residuals = [x.produced.channel(I).value - 10 for x in traces]
        bound = math.sqrt(math.log(26/alpha)/(2*n))
        margin = 1e-12  # S05: fijado antes de observar las muestras.
        self.assertEqual(len(traces), n)
        self.assertTrue(all(math.isfinite(x) and abs(x) <= amplitude+margin for x in residuals))
        self.assertLessEqual(abs(math.fsum(residuals)/n), 2*amplitude*bound+margin)
        self.assertLessEqual(abs(math.fsum(x*x for x in residuals)/n-amplitude**2/3), amplitude**2*bound+margin)
        self.assertLessEqual(abs(sum(x < 0 for x in residuals)/n-.5), bound+margin)
        frequencies = [0]*10
        for x in residuals:
            frequencies[min(9, int((x+amplitude)/(2*amplitude)*10))] += 1
        for count in frequencies:
            self.assertLessEqual(abs(count/n-.1), bound+margin)
        for trace in traces:
            i = trace.produced.channel(I).value
            self.numeric(trace.produced.channel(P).value, 120*i)
            self.numeric(trace.produced.channel(S).value, 120*i)
            self.assertEqual(trace.produced.channel(Q).value, 0)
        self.counters(v.scenario.energy().circuit, (200, 0, 0, 0))

    def test_i02_gan_inductive_and_capacitive(self):
        for scenario, sign in (("F01-EST-inductiva", 1), ("F01-EST-capacitiva", -1)):
            v = instrument(scenario, error=ErrorProfile(reference=10, gain_bias=.1), counters=True)
            for trace in v.capture((0, 60, 120)):
                self.values(trace.produced, 11, 1056, sign*792, 1320, .8)
                self.assertEqual(trace.produced.channel(U).value, 120)
            self.counters(v.counters_at(600).circuit, (176, 0, 132 if sign > 0 else 0, 132 if sign < 0 else 0))
            self.counters(v.scenario.energy().circuit, (160, 0, 120 if sign > 0 else 0, 120 if sign < 0 else 0))

    def test_i02_der_integrates_gain_before_plateau(self):
        v = instrument(error=ErrorProfile(reference=10, drift_rate=.001), counters=True)
        for trace, i, p in zip(v.capture((0, 50, 100, 150)), (10, 10.5, 11, 11), (1200, 1260, 1320, 1320)):
            self.values(trace.produced, i, p, 0, p, 1)
        self.counters(v.counters_at(600).circuit, (655/3, 0, 0, 0))
        self.counters(v.scenario.counters_at(600).circuit, (200, 0, 0, 0))
        self.assertNotAlmostEqual(v.counters_at(600).circuit.active_import_wh, 220)

    def test_i02_qnt_round_even_then_normalize_once(self):
        v = instrument()
        channel = ChannelConfig(P, ErrorProfile(reference=1200, offset=5), output_unit="kW", quantum=.01)
        v = VirtualInstrument(v.scenario, replace(v.config, channels=tuple(c for c in v.config.channels if c.key != P)+(channel,)))
        trace, = v.capture((0,))
        reading = trace.produced.channel(P)
        self.numeric(next(c for c in trace.channels if c.channel == P).indicated, 1205)
        self.assertEqual(reading.original_value, 1.2)
        self.assertEqual(reading.original_unit, "kW")
        self.assertEqual(reading.value, 1200)
        for bias, expected in ((5, 1200), (15, 1220), (-5, 1200)):
            c = replace(channel, error=ErrorProfile(reference=1200, offset=bias), output_unit="W", quantum=10)
            cfg = replace(v.config, channels=(c,))
            self.assertEqual(VirtualInstrument(v.scenario, cfg).capture((0,))[0].produced.channel(P).value, expected)

    def test_result_fault_is_not_repaired_energy_route_explicit(self):
        v = instrument("F01-EST-inductiva", counters=True)
        cfg = replace(v.config, channels=tuple(c for c in v.config.channels if c.key != P)+
                      (ChannelConfig(P, ErrorProfile(reference=1200, offset=12)),))
        v = VirtualInstrument(v.scenario, cfg)
        self.values(v.capture((0,))[0].produced, 10, 972, 720, 1200, .8)
        self.counters(v.counters_at(600).circuit, (160, 0, 120, 0))
        alternate = VirtualInstrument(v.scenario, replace(cfg, energy_route="reported_power"))
        self.counters(alternate.counters_at(600).circuit, (162, 0, 120, 0))
        self.counters(v.scenario.energy().circuit, (160, 0, 120, 0))

    def test_i03_ret_original_support_and_half_open_boundaries(self):
        retention = Retention(20, 40, 1, (I, P))
        v = instrument("F01-RAM", retentions=(retention,))
        traces = v.capture((0, 10, 20, 30, 40))
        for trace, i, support in zip(traces, (5, 35/6, 35/6, 35/6, 25/3), (0, 10, 10, 10, 40)):
            self.numeric(trace.produced.channel(I).value, i, 1e-9)
            self.assertEqual(trace.produced.channel(I).support_s, (support, support))
            raw = next(c for c in trace.channels if c.channel == I)
            self.assertEqual(raw.real_support_s, (support, support))
            self.assertEqual(trace.produced.produced_s, trace.scheduled_s)
        self.assertEqual(len({t.produced.capture_id for t in traces}), 5)
        for trace in traces[2:4]:
            self.numeric(trace.produced.channel(P).value, 700)
            self.assertEqual(trace.produced.channel(I).freshness_at(trace.scheduled_s), "stale")
        self.numeric(v.scenario.state_at(20).phases[0].current_rms_a, 20/3)
        self.numeric(v.scenario.state_at(30).phases[0].current_rms_a, 15/2)

    def test_i03_current_and_unknown_use_only_declared_evidence(self):
        for policy, freshness, validity, support in (("current", "current", "valid", (20, 20)),
                                                     ("unknown", "unknown", "unknown", None)):
            v = instrument("F01-RAM", retentions=(Retention(20, 40, 1, (I,), policy),))
            trace = v.capture((0, 10, 20))[2]
            r = trace.produced.channel(I)
            self.assertEqual((r.freshness_at(20), r.validity, r.support_s), (freshness, validity, support))
            self.assertNotIn("frozen", r.reason)
            self.assertEqual(next(c for c in trace.channels if c.channel == I).real_support_s, (10, 10))
        # Retener después una lectura cuyo soporte declarado ya era desconocido
        # no crea evidencia temporal ni restaura validez por accidente.
        v = instrument("F01-RAM", retentions=(Retention(20, 30, 1, (I,), "unknown"),
                                              Retention(30, 40, 2, (I,), "original")))
        r = v.capture((0, 10, 20, 30))[3].produced.channel(I)
        self.assertEqual((r.support_s, r.validity, r.freshness_at(30)), (None, "unknown", "unknown"))

    def test_i03_int_silence_is_no_production(self):
        v = instrument(silent=(2, 3, 4))
        traces = v.capture(range(6))
        self.assertEqual(len(traces), 6)
        self.assertEqual([t.produced.index for t in traces if t.produced is not None], [0, 1, 5])
        self.assertEqual(sum(t.produced is None for t in traces), 3)
        self.counters(v.scenario.energy().circuit, (200, 0, 0, 0))

    def test_i03_est_equal_values_distinct_supports(self):
        for scenario, i in (("F01-EST", 10), ("F01-REP", 0)):
            v = instrument(scenario, counters=True)
            traces = v.capture(range(6))
            for j, trace in enumerate(traces):
                r = trace.produced.channel(I)
                self.numeric(r.value, i)
                self.assertEqual(r.support_s, (j, j))
                self.assertEqual(r.freshness_at(j), "current")
                self.assertEqual(r.validity, "valid")
                if i == 0:
                    self.assertIsNone(trace.produced.channel(FP).value)
                    self.counters(v.counters_at(j).circuit, (0, 0, 0, 0))
            self.assertEqual(len({t.produced.capture_id for t in traces}), 6)

    def test_counter_retention_does_not_stop_internal_accumulation(self):
        key = "c1/EP+"
        v = instrument(channels=(ChannelConfig(key, quantum=.1),), counters=True,
                       retentions=(Retention(20, 40, 1, (key,)),), silent=(3,))
        traces = v.capture((0, 10, 20, 30, 40))
        a, b = traces[1].produced.channel(key), traces[2].produced.channel(key)
        self.assertEqual(a.value, b.value)
        self.assertEqual(b.support_s, (0, 10))
        self.assertEqual(traces[4].produced.channel(key).value, 13.3)
        self.numeric(v.counters_at(40).circuit.active_import_wh, 40/3)
        self.assertEqual(v.config.counter_epoch, "epoch-0")

    def test_i04_ct_secondary_and_primary_indication(self):
        for location in ("device", "adapter"):
            v = instrument("F01-EST-inductiva", ratios=TransformerRatios(20, 40, location), counters=True)
            for trace in v.capture((0, 60)):
                self.values(trace.produced, 20, 1920, 1440, 2400, .8)
                raw = next(x for x in trace.channels if x.channel == I)
                self.assertEqual(raw.primary, 10)
                self.assertEqual(raw.secondary_ideal, .5)
                self.assertEqual(trace.produced.channel(I).original_value, 20)
            self.counters(v.counters_at(600).circuit, (320, 0, 240, 0))
            self.counters(v.scenario.energy().circuit, (160, 0, 120, 0))

    def test_i04_vt_invalid_value_preserved(self):
        v = instrument("F01-EST-inductiva")
        cfg = replace(v.config, channels=tuple(replace(c, ratios=TransformerRatios(2, 3)) if c.key == U else c for c in v.config.channels))
        v = VirtualInstrument(v.scenario, cfg)
        trace, = v.capture((0,))
        self.values(trace.produced, 10, 1440, 1080, 1800, .8)
        r = trace.produced.channel(U)
        self.assertEqual((r.value, r.validity, r.reason), (180, "invalid", "range_violation"))
        self.assertEqual(next(c for c in trace.channels if c.channel == U).secondary_ideal, 60)
        self.assertEqual(v.scenario.state_at(0).voltages_v, (120,))

    def test_i04_double_scale_rejected_negative_fixture_detected(self):
        with self.assertRaises(ValueError):
            TransformerRatios(20, 20, "device+adapter")
        v = instrument("F01-EST-inductiva", ratios=TransformerRatios(20, 20))
        trace, = v.capture((0,))
        self.assertEqual(trace.produced.channel(I).value, 10)
        # Fixture adverso deliberado en la frontera de salida, no configuración aceptada.
        candidate = v._reading(ChannelConfig(I), 200, 0, 120)
        self.assertEqual((candidate.value, candidate.validity, candidate.reason), (200, "invalid", "range_violation"))
        self.assertEqual(v.scenario.state_at(0).frequency_hz, 60)
        self.assertEqual(v.scenario.state_at(0).phases[0].current_rms_a, 10)

    def test_three_phase_compensation_neutral_lines_and_energy(self):
        v = instrument("F04-COM", counters=True)
        cs = tuple(ChannelConfig(f"c1/L{k}/I", ErrorProfile(reference=10, gain_bias=.1)) for k in (1, 2, 3))
        cs += tuple(c for c in v.config.channels if c.key != I)
        cs += (ChannelConfig("c1/IN"), ChannelConfig("bus-a/L1-L2/ULL"))
        v = VirtualInstrument(v.scenario, replace(v.config, channels=cs))
        r = v.capture((0,))[0].produced
        for key, ref in ((P, 3432), (Q, 0), (S, 3960), (FP, 13/15),
                         ("c1/IN", 1.1*(6*math.sqrt(3)-2)), ("bus-a/L1-L2/ULL", 120*math.sqrt(3))):
            self.numeric(r.channel(key).value, ref)
        energy = v.counters_at(3600)
        self.counters(energy.circuit, (3432, 0, 0, 0))
        self.counters(energy.phases[0], (1056, 0, 792, 0))
        self.counters(energy.phases[1], (1056, 0, 0, 792))

    def test_window_boundary_energy_and_initials(self):
        v = instrument("F01-ESC", error=ErrorProfile(reference=10, gain_bias=.1), counters=True)
        self.counters(v.energy(590, 610).circuit, (11/2, 0, 11/4, 11/4))
        self.counters(v.energy(600, 600).circuit, (0, 0, 0, 0))
        initial = CircuitEnergy((EnergyCounters(100, 7, 20, 30),), EnergyCounters(100, 7, 20, 30))
        v = VirtualInstrument(v.scenario, replace(v.config, initial_counters=initial))
        self.counters(v.counters_at(0).circuit, (100, 7, 20, 30))
        self.counters(v.counters_at(600).circuit, (320, 7, 185, 30))
        self.counters(v.energy(0, 600).circuit, (220, 0, 165, 0))

    def test_product_of_voltage_and_current_drift_integrated(self):
        v = instrument(counters=True)
        drift = ErrorProfile(reference=120, drift_rate=.001)
        cfg = replace(v.config, channels=tuple(replace(c, error=drift) if c.key in (I, U) else c
                                               for c in v.config.channels))
        v = VirtualInstrument(v.scenario, cfg)
        # [0,100]: 1200*(1+.001t)^2; [100,600]: 1452 W.
        expected = (1200*(100+10+1/3)+500*1452)/3600
        self.counters(v.counters_at(600).circuit, (expected, 0, 0, 0))
        self.numeric(v.capture((100,))[0].produced.channel(P).value, 1452)

    def test_smooth_physical_profile_and_directional_crossing(self):
        v = instrument("F01-Q", error=ErrorProfile(reference=10, gain_bias=.1), counters=True)
        self.counters(v.counters_at(3600).circuit, (1320, 0, 247.5, 247.5))
        v = instrument("F01-Q-C13", error=ErrorProfile(reference=10, gain_bias=.1), counters=True)
        self.counters(v.counters_at(3600).circuit, (1540, 0, 0, 0))
        reactive = CircuitState("1P2W", (120,), (-1j/12,))
        base = instrument(counters=True)
        segment = replace(base.scenario.segments[0], left=reactive, right=reactive)
        scenario = replace(base.scenario, scenario_id="reactive-control", segments=(segment,))
        v = VirtualInstrument(scenario, base.config)
        self.assertEqual(v.capture((0,))[0].produced.channel(P).value, 0)
        self.counters(v.counters_at(600).circuit, (0, 0, 200, 0))

    def test_secondary_offset_before_configured_ratio(self):
        v = instrument(error=ErrorProfile(reference=.5, offset=.05), ratios=TransformerRatios(20, 40))
        trace, = v.capture((0,))
        raw = next(c for c in trace.channels if c.channel == I)
        self.assertEqual(raw.secondary_ideal, .5)
        self.assertEqual(raw.secondary_instrumental, .55)
        self.assertEqual(trace.produced.channel(I).value, 22)
        self.numeric(trace.produced.channel(P).value, 2640)

    def test_retention_selects_quantized_original_before_single_normalization(self):
        v = instrument("F01-RAM", retentions=(Retention(20, 40, 1, (P,)),))
        cfg = replace(v.config, channels=(ChannelConfig(P, output_unit="kW", quantum=.01),),
                      retentions=(Retention(20, 40, 1, (P,)),))
        v = VirtualInstrument(v.scenario, cfg)
        with patch.object(v, "_normalize", wraps=v._normalize) as normalize:
            traces = v.capture((0, 10, 20))
        self.assertEqual(normalize.call_count, 3)
        self.numeric(normalize.call_args_list[-1].args[0].value, .7)
        self.assertEqual(traces[2].produced.channel(P).original_unit, "kW")
        self.numeric(traces[2].produced.channel(P).value, 700)

    def test_result_drift_keeps_declared_counter_path(self):
        v = instrument(counters=True)
        cfg = replace(v.config, channels=tuple(ChannelConfig(P, ErrorProfile(reference=1200, drift_rate=.001))
                                               if c.key == P else c for c in v.config.channels))
        v = VirtualInstrument(v.scenario, cfg)
        self.numeric(v.capture((150,))[0].produced.channel(P).value, 1320)
        self.counters(v.counters_at(600).circuit, (200, 0, 0, 0))
        v = VirtualInstrument(v.scenario, replace(cfg, energy_route="reported_power"))
        self.counters(v.counters_at(600).circuit, (655/3, 0, 0, 0))

    def test_sha256_tuple_and_rng_reproduction_global_independence(self):
        v = instrument(error=ErrorProfile("I01-UNI@1", 10, noise_amplitude=.1))
        payload = '["s05-rng-v1","20261017","s05-i01-uni","instrumentation","sim-a","c1/L1/I","I01-UNI@1","0"]'
        expected = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")
        self.assertEqual(v.config.random_identity.seed(I, "I01-UNI@1"), expected)
        state = random.getstate()
        first = v.capture(range(10))
        self.assertEqual(first, v.capture(range(10)))
        self.assertEqual(random.getstate(), state)
        rng = random.Random(expected)
        for trace in first:
            eta = .1*(2*rng.random()-1)
            self.numeric(trace.produced.channel(I).value, 10+eta, 1e-12)
            self.assertEqual(trace.seeds, ((I, expected),))
        other = VirtualInstrument(v.scenario, replace(v.config, random_identity=replace(v.config.random_identity, replicate="1")))
        self.assertNotEqual([t.produced for t in first], [t.produced for t in other.capture(range(10))])
        self.assertEqual(random.getstate(), state)

    def test_independent_channels_and_common_bus_source(self):
        v = instrument("F01-EST-3P4W")
        err = ErrorProfile("noise@1", 10, noise_amplitude=.1)
        cs = (ChannelConfig("c1/L1/I", err), ChannelConfig("c1/L2/I", err))
        v = VirtualInstrument(v.scenario, replace(v.config, channels=cs))
        traces = v.capture(range(10))
        self.assertEqual(len(traces[0].seeds), 2)
        self.assertNotEqual(traces[0].seeds[0][1], traces[0].seeds[1][1])
        self.assertNotEqual([t.channels[0].innovation for t in traces], [t.channels[1].innovation for t in traces])
        common = replace(v.config, channels=tuple(replace(c, shared_noise="ct-common") for c in cs))
        traces = VirtualInstrument(v.scenario, common).capture(range(10))
        for trace in traces:
            self.assertEqual(trace.channels[0].innovation, trace.channels[1].innovation)
            self.assertEqual(len(trace.seeds), 1)
        # U bus se instrumenta una sola vez, no hay sensores por circuito duplicados.
        cfg = replace(v.config, channels=(ChannelConfig(U, err, shared_noise="bus-a/noise"), ChannelConfig(P)))
        trace = VirtualInstrument(v.scenario, cfg).capture((0,))[0]
        self.assertEqual(len(trace.seeds), 1)

    def test_noise_draws_are_consumed_under_retention_and_silence(self):
        v = instrument(error=ErrorProfile("noise@1", 10, noise_amplitude=.1))
        base = v.capture(range(6))
        cfg = replace(v.config, silent_indices=(3,), retentions=(Retention(2, 5, 1, (I,)),))
        altered = VirtualInstrument(v.scenario, cfg).capture(range(6))
        for a, b in zip(base, altered):
            self.assertEqual(a.channels[0].innovation, b.channels[0].innovation)
        self.assertEqual(base[5].produced.channel(I), altered[5].produced.channel(I))
        self.assertIsNone(altered[3].produced)

    def test_same_truth_two_instruments_no_mutation(self):
        a = instrument(counters=True)
        s, state, energy = a.scenario, a.scenario.state_at(0), a.scenario.energy()
        cfg = replace(a.config, device="sim-b", random_identity=replace(a.config.random_identity, device="sim-b"),
                      channels=tuple(replace(c, error=ErrorProfile(reference=10, gain_bias=.1)) if c.key == I else c for c in a.config.channels))
        b = VirtualInstrument(s, cfg)
        self.numeric(a.capture((0,))[0].produced.channel(I).value, 10)
        self.numeric(b.capture((0,))[0].produced.channel(I).value, 11)
        self.assertEqual(s.state_at(0), state)
        self.assertEqual(s.energy(), energy)
        self.counters(s.energy().circuit, (200, 0, 0, 0))
        for obj, field, value in ((s, "version", 2), (state, "frequency_hz", 50), (cfg, "device", "x")):
            with self.assertRaises(FrozenInstanceError):
                setattr(obj, field, value)

    def test_freshness_age_boundary_and_negative_evaluation(self):
        r = instrument().capture((0,))[0].produced.channel(I)
        self.assertEqual(r.freshness_at(5), "current")
        self.assertEqual(r.freshness_at(5.001), "stale")
        for now, age in ((-1, 5), (1, -1), (float("nan"), 5)):
            with self.assertRaises(ValueError):
                r.freshness_at(now, age)

    def test_reject_invalid_parameters_units_and_manifests(self):
        for kwargs in ({"reference": 0}, {"gain_bias": .6}, {"offset": .2}, {"noise_amplitude": -.1},
                       {"drift_rate": .002}, {"drift_start_s": -1}, {"drift_duration_s": 0},
                       {"reference": float("nan")}, {"gain_bias": float("inf")}, {"noise_amplitude": True},
                       {"noise_amplitude": .1, "prescribed_u": (1,)}, {"prescribed_u": (.5,)}):
            with self.assertRaises(ValueError):
                ErrorProfile(**kwargs)
        for ratios in ((0, 1), (1, 1001), (float("inf"), 1), (True, 1)):
            with self.assertRaises(ValueError):
                TransformerRatios(*ratios)
        for c in (lambda: ChannelConfig(I, output_unit="kW"), lambda: ChannelConfig(I, output_unit="unknown"),
                  lambda: ChannelConfig(P, ratios=TransformerRatios(20, 20)),
                  *(lambda q=q: ChannelConfig(P, quantum=q) for q in (0, -1, float("nan")))):
            with self.assertRaises(ValueError):
                c()
        for master in (-1, 2**64, True, 1.5):
            with self.assertRaises(ValueError):
                RandomIdentity(master, "run", "device")
        v = instrument()
        for cfg in (replace(v.config, channels=(ChannelConfig("c2/L1/I"),)),
                    replace(v.config, channels=(ChannelConfig("c1/EP+"),))):
            with self.assertRaises(ValueError):
                VirtualInstrument(v.scenario, cfg)
        for times in ((), (1, 0), (0, 0), (-1,), (601,), (True,)):
            with self.assertRaises(ValueError):
                v.capture(times)
        for r in (Retention(0, 2, 0, (I,)), Retention(1, 2, 4, (I,))):
            cfg = replace(v.config, retentions=(r,))
            with self.assertRaises(ValueError):
                VirtualInstrument(v.scenario, cfg).capture((0, 1))
        cfg = replace(v.config, retentions=(Retention(1, 2, 0, (I,)),), silent_indices=(0,))
        with self.assertRaisesRegex(ValueError, "anterior"):
            VirtualInstrument(v.scenario, cfg).capture((0, 1))
        with self.assertRaises(ValueError):
            instrument("F01-REP", error=ErrorProfile(reference=10, noise_amplitude=.1))
        negative = instrument(error=ErrorProfile(reference=100, offset=-10, gain_bias=-.5))
        with self.assertRaisesRegex(ValueError, "negativo"):
            negative.capture((0,))

    def test_noisy_counter_requires_undefined_trajectory_rejected(self):
        v = instrument(error=ErrorProfile(reference=10, noise_amplitude=.1), counters=True)
        with self.assertRaisesRegex(ValueError, "trayectoria"):
            v.counters_at(600)
        self.counters(v.scenario.counters_at(600).circuit, (200, 0, 0, 0))

    def test_isolation_import_capture_and_counters_in_separate_process(self):
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
    from smartenergy.analysis.simulator_instrumentation import *
    from smartenergy.analysis.simulator_scenarios import build_scenario
    from smartenergy.analysis.simulator_energy import CircuitEnergy, EnergyCounters
    s = build_scenario('F01-EST')
    cfg = InstrumentConfig('sim-a', (ChannelConfig('c1/L1/I', ErrorProfile(reference=10, noise_amplitude=.1)),), RandomIdentity(1, 'isolated', 'sim-a'))
    v = VirtualInstrument(s, cfg)
    assert v.capture((0, 1)) == v.capture((0, 1))
    cfg = InstrumentConfig('sim-a', (ChannelConfig('c1/EP+'),), cfg.random_identity, counter_epoch='zero', initial_counters=CircuitEnergy((EnergyCounters(),)))
    v = VirtualInstrument(s, cfg)
    assert abs(v.capture((0, 600))[1].produced.readings[0].value-200) < 1e-6
assert random.getstate() == rng
'''
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
