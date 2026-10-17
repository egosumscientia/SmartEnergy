"""Oráculos A01/A02 S05; sin admisión/deduplicación ni composición S03."""

from collections import Counter
from dataclasses import FrozenInstanceError, replace
import hashlib
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

from smartenergy.analysis.simulator_acquisition import AcquisitionConfig, FifoPolicy, acquire
from smartenergy.analysis.simulator_energy import CircuitEnergy, EnergyCounters
from smartenergy.analysis.simulator_instrumentation import (
    ChannelConfig, ErrorProfile, InstrumentConfig, RandomIdentity, Retention, VirtualInstrument,
)
from smartenergy.analysis.simulator_scenarios import build_scenario

P, I = "c1/P", "c1/L1/I"


def instrument(*, scenario="F01-RAM", channels=None, silent=(), retentions=()):
    return VirtualInstrument(build_scenario(scenario), InstrumentConfig(
        "sim-a", channels or (ChannelConfig(P),), RandomIdentity(20261017, "A01-CAL", "sim-a"),
        silent_indices=silent, retentions=retentions))


def fifo_config(b=3, h=5, k=2, times=((1, 6), (2, 7), (3, 8)), horizon=8):
    return AcquisitionConfig(horizon, outages=((1, 4),), fifo=FifoPolicy(b, h, k, times))


class AcquisitionOracles(unittest.TestCase):
    def setUp(self):
        self.v = instrument()
        self.traces = self.v.capture(range(6))

    def counts(self, r, produced, attempts, failed, received, never, pending=0, flight=0):
        self.assertEqual(sum(t.produced is not None for t in r.instrument_traces), produced)
        self.assertEqual(len(r.attempts), attempts)
        self.assertEqual(sum(a.outcome == "failed" for a in r.attempts), failed)
        self.assertEqual(len(r.deliveries), received)
        self.assertEqual(len(r.never_received), never)
        self.assertEqual(len(r.pending), pending)
        self.assertEqual(sum(a.outcome == "in_flight" for a in r.attempts), flight)
        self.assertEqual(produced, len({d.capture.index for d in r.deliveries}) + never + pending)
        self.assertEqual(attempts, received + failed + flight)
        self.assertEqual(len({d.delivery_id for d in r.deliveries}), received)
        for d in r.deliveries:
            self.assertNotEqual(d.delivery_id, d.attempt_id)
            self.assertIs(d.capture, r.instrument_traces[d.capture.index].produced)
            a = next(a for a in r.attempts if a.attempt_id == d.attempt_id)
            self.assertEqual(a.arrival_s, d.received_s)
            self.assertGreaterEqual(d.received_s, a.attempted_s)
            self.assertGreaterEqual(a.attempted_s, d.capture.produced_s)

    def order(self, r, indices, times):
        self.assertEqual([d.capture.index for d in r.deliveries], indices)
        self.assertEqual([d.received_s for d in r.deliveries], times)

    def test_normal_and_negative_control_equal_values_different_ids(self):
        r = acquire(self.traces, AcquisitionConfig(5))
        self.counts(r, 6, 6, 0, 6, 0)
        self.order(r, list(range(6)), list(range(6)))
        self.assertEqual(r.queue_events, ())
        self.assertIsNone(r.seed)
        for scenario, expected in (("F01-EST", 1200), ("F01-REP", 0)):
            ts = instrument(scenario=scenario).capture(range(6))
            rr = acquire(ts, AcquisitionConfig(5))
            self.counts(rr, 6, 6, 0, 6, 0)
            for d in rr.deliveries:
                self.assertAlmostEqual(d.capture.channel(P).value, expected, delta=1e-6+1e-9*expected)
            self.assertEqual(len({d.capture.capture_id for d in rr.deliveries}), 6)

    def test_a01_cal_exact_oracle(self):
        cfg = AcquisitionConfig(8, lost_indices=(2,), delays=((1, 6),), redeliveries=((3, 8),))
        r = acquire(self.traces, cfg)
        self.counts(r, 6, 7, 1, 6, 1)
        self.order(r, [0, 3, 4, 5, 1, 3], [0, 3, 4, 5, 7, 8])
        for d, expected in zip(r.deliveries, (600, 630, 640, 650, 610, 630)):
            self.assertAlmostEqual(d.capture.channel(P).value, expected, delta=1e-6+1e-9*expected)
        self.assertEqual(r.never_received, (2,))
        self.assertEqual([d.delivery_id for d in r.deliveries], [
            "A01-CAL/0/attempt-1", "A01-CAL/3/attempt-1", "A01-CAL/4/attempt-1",
            "A01-CAL/5/attempt-1", "A01-CAL/1/attempt-1", "A01-CAL/3/attempt-2"])
        self.assertEqual(Counter(d.capture.index for d in r.deliveries), {0: 1, 1: 1, 3: 2, 4: 1, 5: 1})
        first, duplicate = r.deliveries[1], r.deliveries[-1]
        self.assertIs(first.capture, duplicate.capture)
        self.assertNotEqual(first.delivery_id, duplicate.delivery_id)
        self.assertEqual(duplicate.kind, "redelivery")
        self.assertEqual(duplicate.source_attempt_id, first.attempt_id)
        self.assertEqual(r.deliveries[-2].capture.channel(P).freshness_at(7), "stale")
        self.assertEqual(duplicate.capture.channel(P).freshness_at(8), "current")
        # Comparación del soporte por el ensayo, sin implementar selección/admisión.
        self.assertEqual(max(r.deliveries, key=lambda d: d.capture.channel(P).support_s[1]).capture.index, 5)
        self.assertAlmostEqual(self.v.scenario.energy(0, 120).circuit.active_import_wh, 35, delta=1e-6)
        self.assertEqual(r, acquire(self.traces, cfg))

    def test_a02_discard_exact_oracle(self):
        r = acquire(self.traces, AcquisitionConfig(8, outages=((1, 4),)))
        self.counts(r, 6, 6, 3, 3, 3)
        self.order(r, [0, 4, 5], [0, 4, 5])
        self.assertEqual(r.never_received, (1, 2, 3))
        self.assertEqual([a.capture.index for a in r.attempts if a.reason == "link_down"], [1, 2, 3])
        self.assertEqual(r.queue_events, ())
        self.assertTrue(all(d.capture.channel(P).value is not None for d in r.deliveries))

    def test_a02_fifo_exact_oracle(self):
        r = acquire(self.traces, fifo_config())
        self.counts(r, 6, 9, 3, 6, 0)
        self.order(r, [0, 4, 5, 1, 2, 3], [0, 4, 5, 6, 7, 8])
        self.assertEqual([a.number for a in r.attempts], [1]*6 + [2]*3)
        for d in r.deliveries[-3:]:
            self.assertEqual(d.received_s-d.capture.channel(P).support_s[1], 5)
            self.assertEqual(d.capture.channel(P).freshness_at(d.received_s), "current")
        self.assertEqual(r.deliveries[-1].capture.index, 3)
        self.assertEqual(r, acquire(self.traces, fifo_config()))

    def test_a02_fifo_capacity_two_exact_oracle(self):
        r = acquire(self.traces, fifo_config(b=2, times=((2, 6), (3, 7))))
        self.counts(r, 6, 8, 3, 5, 1)
        self.order(r, [0, 4, 5, 2, 3], [0, 4, 5, 6, 7])
        self.assertEqual(r.never_received, (1,))
        self.assertEqual([(e.index, e.time_s) for e in r.queue_events if e.reason == "overflow"], [(1, 3)])
        self.assertEqual([d.received_s-d.capture.produced_s for d in r.deliveries[-2:]], [4, 4])

    def test_a02_fifo_expiration_exact_oracle(self):
        r = acquire(self.traces, fifo_config(b=2, h=3, times=((2, 6), (3, 7))))
        self.counts(r, 6, 6, 3, 3, 3)
        self.order(r, [0, 4, 5], [0, 4, 5])
        self.assertEqual({e.index for e in r.queue_events if e.reason == "expired"}, {2, 3})

    def test_i03_int_not_produced_is_not_transport_loss(self):
        ts = instrument(scenario="F01-EST", silent=(2, 3, 4)).capture(range(6))
        r = acquire(ts, AcquisitionConfig(5, lost_indices=(2, 3, 4)))
        self.counts(r, 3, 3, 0, 3, 0)
        self.order(r, [0, 1, 5], [0, 1, 5])
        self.assertEqual(r.not_produced, (2, 3, 4))
        self.assertEqual(r.loss_indices, ())
        for cfg in (AcquisitionConfig(8, redeliveries=((2, 7),)),
                    AcquisitionConfig(8, fifo=FifoPolicy(3, 5, 2, ((2, 7),)))):
            with self.assertRaisesRegex(ValueError, "no producida"):
                acquire(ts, cfg)

    def test_all_silent_and_empty_calendar(self):
        r = acquire(instrument(silent=tuple(range(6))).capture(range(6)), AcquisitionConfig(5))
        self.counts(r, 0, 0, 0, 0, 0)
        self.assertEqual(r.not_produced, tuple(range(6)))
        self.counts(acquire((), AcquisitionConfig(0)), 0, 0, 0, 0, 0)

    def test_receipt_ties_use_attempt_ordinal(self):
        r = acquire(self.traces, AcquisitionConfig(6, delays=((0, 5), (1, 4), (2, 3), (3, 2), (4, 1))))
        self.order(r, list(range(6)), [5]*6)
        self.assertEqual([d.ordinal for d in r.deliveries], list(range(6)))

    def test_outage_and_horizon_boundaries(self):
        r = acquire(self.traces, AcquisitionConfig(5, outages=((0, 1), (3, 5))))
        self.counts(r, 6, 6, 3, 3, 3)
        self.order(r, [1, 2, 5], [1, 2, 5])
        # La caída impide envío; la latencia ya programada no se reinterpreta.
        r = acquire(self.traces, AcquisitionConfig(5, delays=((0, 2),), outages=((1, 4),)))
        self.order(r, [0, 4, 5], [2, 4, 5])

    def test_in_flight_and_pending_accounting(self):
        r = acquire(self.traces, AcquisitionConfig(5, delays=((1, 7),)))
        self.counts(r, 6, 6, 0, 5, 0, pending=1, flight=1)
        self.assertEqual(r.pending, (1,))
        r = acquire(self.traces, fifo_config(horizon=5))
        self.counts(r, 6, 6, 3, 3, 0, pending=3)
        self.assertEqual(r.pending, (1, 2, 3))
        r = acquire(self.traces, fifo_config(h=10, times=()))
        self.counts(r, 6, 6, 3, 3, 3)
        self.assertEqual({e.index for e in r.queue_events if e.reason == "calendar_exhausted"}, {1, 2, 3})

    def test_retry_limit_and_fifo_order(self):
        r = acquire(self.traces, fifo_config(k=1))
        self.counts(r, 6, 6, 3, 3, 3)
        r = acquire(self.traces, AcquisitionConfig(8, outages=((1, 8),),
                    fifo=FifoPolicy(5, 10, 2, ((1, 6), (1, 8), (2, 7), (3, 8)))))
        self.assertEqual(sum(a.capture.index == 1 for a in r.attempts), 2)
        self.assertTrue(any(e.index == 1 and e.reason == "attempt_limit" for e in r.queue_events))
        with self.assertRaisesRegex(ValueError, "viola FIFO"):
            acquire(self.traces, fifo_config(times=((2, 6), (1, 7), (3, 8))))

    def test_k_limits_initial_fifo_and_explicit_redeliveries_together(self):
        # S05.5: K intentos TOTALES; una reentrega consume el mismo presupuesto.
        ts = self.traces[:1]
        for retries, redeliveries in (((), ((0, 2), (0, 3))),
                                      (((0, 2),), ((0, 3),))):
            with self.subTest(retries=retries):
                r = acquire(ts, AcquisitionConfig(3, outages=((0, 1),),
                    fifo=FifoPolicy(1, 5, 2, retries), redeliveries=redeliveries))
                self.assertEqual([(a.attempted_s, a.number) for a in r.attempts], [(0, 1), (2, 2)])
                self.assertEqual([d.received_s for d in r.deliveries], [2])
        r = acquire(ts, AcquisitionConfig(3, fifo=FifoPolicy(1, 5, 1, ()), redeliveries=((0, 2),)))
        self.assertEqual(len(r.attempts), 1)
        self.assertEqual(len(r.deliveries), 1)

    def test_retries_do_not_share_random_draws(self):
        cfg = AcquisitionConfig(8, loss_probability=.5,
                                random_identity=RandomIdentity(20261017, "a01-bernoulli-control", "sim-a"),
                                random_channel=P, random_profile="A01-BER-control@1")
        seen = []
        original = random.Random.random
        def draw(rng):
            u = original(rng)
            seen.append(u)
            return u
        with patch.object(random.Random, "random", draw):
            r = acquire(self.traces, cfg)
        self.assertEqual(len(seen), 6)
        with patch.object(random.Random, "random", draw):
            rr = acquire(self.traces, replace(cfg, redeliveries=((0, 7), (1, 8))))
        self.assertEqual(len(seen), 12)
        self.assertEqual(seen[:6], seen[6:])
        self.assertEqual([a.outcome for a in r.attempts], [a.outcome for a in rr.attempts[:6]])

    def test_bernoulli_seed_threshold_and_exact_reproduction(self):
        identity = RandomIdentity(20261017, "a01-bernoulli-control", "sim-a")
        cfg = AcquisitionConfig(8, loss_probability=.5, random_identity=identity,
                                random_channel=P, random_profile="A01-BER-control@1")
        payload = b'["s05-rng-v1","20261017","a01-bernoulli-control","acquisition","sim-a","c1/P","A01-BER-control@1","0"]'
        expected_seed = int(hashlib.sha256(payload).hexdigest(), 16)
        r = acquire(self.traces, cfg)
        self.assertEqual(r.seed, expected_seed)
        self.assertNotEqual(r.seed, identity.seed(P, cfg.random_profile))
        self.assertEqual(r, acquire(self.traces, cfg))
        # Oráculo analítico de transformación u<p: empate u=p no pierde.
        with patch.object(random.Random, "random", side_effect=(0, .25, .5, .75, .99, .5)):
            rr = acquire(self.traces, cfg)
        self.assertEqual(rr.never_received, (0, 1))
        self.assertEqual(rr.loss_indices, (0, 1))
        for p, failed in ((0, 0), (1, 6)):
            self.counts(acquire(self.traces, replace(cfg, loss_probability=p)), 6, 6, failed, 6-failed, failed)
        changed = acquire(self.traces, replace(cfg, random_identity=replace(identity, replicate="1")))
        self.assertNotEqual(r.seed, changed.seed)

    def test_channel_and_device_rng_streams_are_separate(self):
        cfg = AcquisitionConfig(8, loss_probability=.5, random_identity=RandomIdentity(1, "streams", "sim-a"),
                                random_channel=P, random_profile="loss@1")
        r = acquire(self.traces, cfg)
        other = acquire(self.traces, replace(cfg, random_channel=I))
        self.assertNotEqual(r.seed, other.seed)
        self.assertEqual(r, acquire(self.traces, cfg))
        device_traces = tuple(replace(t, produced=replace(t.produced, device="sim-b")) for t in self.traces)
        b = acquire(device_traces, replace(cfg, random_identity=replace(cfg.random_identity, device="sim-b")))
        self.assertNotEqual(r.seed, b.seed)

    def test_r1_noise_trajectory_rejection_and_transport_independence(self):
        v = instrument(scenario="F01-EST", channels=(ChannelConfig(I, ErrorProfile(
            reference=10, noise_amplitude=.1)), ChannelConfig(P)))
        ts = v.capture(range(6))
        before_state, before_energy, before_events = v.scenario.state_at(5), v.scenario.energy(), v.scenario.events
        rng = random.getstate()
        for cfg in (AcquisitionConfig(8), AcquisitionConfig(8, lost_indices=(1, 2), delays=((3, 3),),
                    redeliveries=((4, 8),)), fifo_config()):
            r = acquire(ts, cfg)
            self.assertEqual(r.instrument_traces, ts)
            self.assertEqual(v.capture(range(6)), ts)
        self.assertEqual(random.getstate(), rng)
        self.assertEqual(v.scenario.state_at(5), before_state)
        self.assertEqual(v.scenario.energy(), before_energy)
        self.assertEqual(v.scenario.events, before_events)
        with self.assertRaisesRegex(ValueError, "trayectoria"):
            v.energy(0, 5)

    def test_counters_unchanged_by_loss_delay_duplicate_and_link(self):
        v = instrument(scenario="F01-EST")
        v = VirtualInstrument(v.scenario, replace(v.config, counter_epoch="zero",
                              channels=(ChannelConfig("c1/EP+"),),
                              initial_counters=CircuitEnergy((EnergyCounters(),))))
        ts = v.capture((0, 60, 120, 600))
        before = v.counters_at(600)
        r = acquire(ts, AcquisitionConfig(610, lost_indices=(1,), delays=((2, 4),),
                    redeliveries=((3, 610),), outages=((1, 61),)))
        self.assertEqual(v.counters_at(600), before)
        self.assertAlmostEqual(before.circuit.active_import_wh, 200, delta=1e-6)
        self.assertEqual([d.capture.channel("c1/EP+").support_s for d in r.deliveries],
                         [(0, 0), (0, 120), (0, 600), (0, 600)])
        self.assertEqual(ts, v.capture((0, 60, 120, 600)))

    def test_r2_retention_declared_and_real_support_survive(self):
        for policy, declared, freshness in (("original", (10, 10), "stale"),
                                            ("current", (20, 20), "current"),
                                            ("unknown", None, "unknown")):
            v = instrument(retentions=(Retention(20, 40, 1, (P,), policy),))
            ts = v.capture((0, 10, 20, 30))
            r = acquire(ts, AcquisitionConfig(35, redeliveries=((2, 25),)))
            d = next(d for d in r.deliveries if d.capture.index == 2)
            self.assertEqual(d.capture.channel(P).support_s, declared)
            self.assertEqual(d.capture.channel(P).freshness_at(d.received_s), freshness)
            trace = r.instrument_traces[2].channels[0]
            self.assertEqual(trace.real_support_s, (10, 10))
            self.assertEqual(trace.retained_from_index, 1)
            self.assertEqual(d.capture.scheduled_s, 20)
            self.assertEqual(d.capture.produced_s, 20)
            self.assertEqual(d.capture.channel(P).reason, ts[2].produced.channel(P).reason)

    def test_r3_result_incoherence_preserved_without_diagnosis(self):
        v = instrument(scenario="F01-EST", channels=(ChannelConfig(P, ErrorProfile(reference=1200, offset=12)),
                       ChannelConfig("c1/S"), ChannelConfig("c1/FP")))
        ts = v.capture((0,))
        r = acquire(ts, AcquisitionConfig(1, redeliveries=((0, 1),)))
        for d in r.deliveries:
            self.assertEqual([x.value for x in d.capture.readings], [1212, 1200, 1])
            self.assertEqual({x.validity for x in d.capture.readings}, {"valid"})
            self.assertEqual({x.reason for x in d.capture.readings}, {"no_diagnostic"})

    def test_inputs_and_results_immutable(self):
        losses, delays = [2], [[1, 6]]
        cfg = AcquisitionConfig(8, lost_indices=losses, delays=delays)
        r = acquire(self.traces, cfg)
        losses.append(3)
        delays[0][1] = 0
        self.assertEqual(cfg.lost_indices, (2,))
        self.assertEqual(cfg.delays, ((1, 6),))
        self.assertEqual(acquire(self.traces, cfg), r)
        with self.assertRaises(FrozenInstanceError):
            r.deliveries[0].received_s = 1

    def test_production_is_separate_from_schedule_and_measurement(self):
        ts = (replace(self.traces[0], produced=replace(self.traces[0].produced, produced_s=2)),)
        r = acquire(ts, AcquisitionConfig(6, delays=((0, 3),), redeliveries=((0, 3),)))
        self.assertEqual([a.attempted_s for a in r.attempts], [2, 3])
        self.assertEqual([d.received_s for d in r.deliveries], [5, 6])
        for d in r.deliveries:
            self.assertEqual(d.capture.scheduled_s, 0)
            self.assertEqual(d.capture.produced_s, 2)
            self.assertEqual(d.capture.channel(P).support_s, (0, 0))
        self.assertEqual(r.deliveries[0].capture.channel(P).freshness_at(5), "current")
        self.assertEqual(r.deliveries[1].capture.channel(P).freshness_at(6), "stale")

    def test_bernoulli_still_draws_under_link_interruption_and_silence(self):
        ts = instrument(silent=(2, 3, 4)).capture(range(6))
        cfg = AcquisitionConfig(8, loss_probability=.5, random_identity=RandomIdentity(1, "control", "sim-a"),
                                random_channel=P, random_profile="loss@1", outages=((0, 8),))
        with patch.object(random.Random, "random", side_effect=(0, .5, .25)) as draws:
            r = acquire(ts, cfg)
        self.assertEqual(draws.call_count, 3)
        self.assertEqual(r.loss_indices, (0, 5))
        self.assertEqual({a.reason for a in r.attempts}, {"link_down"})
        self.counts(r, 3, 3, 3, 0, 3)

    def test_positive_delay_must_not_underflow_or_overflow(self):
        for delta in (5e-324, 1e308):
            ts = (replace(self.traces[0], produced=replace(self.traces[0].produced, produced_s=1e308)),)
            with self.assertRaises(ValueError):
                acquire(ts, AcquisitionConfig(1e308, delays=((0, delta),)))

    def test_invalid_configurations(self):
        for kwargs in ({"horizon_s": -1}, {"horizon_s": True}, {"horizon_s": float("nan")},
                       {"lost_indices": (True,)}, {"lost_indices": (-1,)}, {"lost_indices": (1, 1)},
                       {"delays": ((1, -1),)}, {"delays": ((1, float("inf")),)},
                       {"delays": ((1, 0), (1, 1))}, {"redeliveries": ((1, 7), (1, 7))},
                       {"outages": ((1, 1),)}, {"outages": ((1, 4), (3, 5))},
                       {"fifo": "auto"}, {"loss_probability": .5}, {"random_channel": P},
                       {"loss_probability": -.1}, {"loss_probability": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AcquisitionConfig(**({"horizon_s": 8} | kwargs))
        for args in ((0, 5, 2), (10001, 5, 2), (3, 0, 2), (3, float("nan"), 2),
                     (3, 5, 0), (3, 5, 11), (True, 5, 2)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                FifoPolicy(*args, ())

    def test_invalid_calendar_and_identity(self):
        for cfg in (AcquisitionConfig(4), AcquisitionConfig(8, lost_indices=(6,)),
                    AcquisitionConfig(8, redeliveries=((2, 1),)), AcquisitionConfig(8, redeliveries=((2, 2),)),
                    AcquisitionConfig(8, redeliveries=((1, 7),), fifo=FifoPolicy(3, 5, 2, ((1, 7),)))):
            with self.assertRaises(ValueError):
                acquire(self.traces, cfg)
        for ts in (self.traces[::-1], (self.traces[0],)*2, ("bad",),
                   (replace(self.traces[0], produced=replace(self.traces[0].produced, produced_s=-1)),),
                   (replace(self.traces[0], produced=replace(self.traces[0].produced, scheduled_s=1)),),
                   (self.traces[0], replace(self.traces[1], produced=replace(self.traces[1].produced,
                     capture_id=self.traces[0].produced.capture_id))),
                   (self.traces[0], replace(self.traces[1], produced=replace(self.traces[1].produced, device="sim-b")))):
            with self.assertRaises(ValueError):
                acquire(ts, AcquisitionConfig(8))
        future = replace(self.traces[0].produced.readings[0], support_s=(1, 1))
        ts = (replace(self.traces[0], produced=replace(self.traces[0].produced, readings=(future,))),)
        with self.assertRaisesRegex(ValueError, "soporte"):
            acquire(ts, AcquisitionConfig(8))
        with self.assertRaises(ValueError):
            acquire(self.traces, None)

    def test_isolation_in_separate_process(self):
        script = r'''
import importlib.abc
import random
import sys
import time
from unittest.mock import patch
class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'smartenergy.application', 'smartenergy.application.simulator'}:
            return None
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
state = random.getstate()
with patch.object(time, 'time', side_effect=AssertionError('reloj')), patch.object(time, 'sleep', side_effect=AssertionError('sleep')):
    from smartenergy.analysis.simulator_acquisition import *
    from smartenergy.analysis.simulator_instrumentation import *
    from smartenergy.analysis.simulator_scenarios import build_scenario
    from smartenergy.analysis.simulator_contract import ObservationConfig
    from smartenergy.application.simulator import compose_received
    from datetime import datetime, timezone
    s = build_scenario('F01-RAM')
    identity = RandomIdentity(20261017, 'isolated', 'sim-a')
    v = VirtualInstrument(s, InstrumentConfig('sim-a', (ChannelConfig('c1/P'),), identity))
    ts = v.capture(range(6))
    cfg = AcquisitionConfig(8, outages=((1, 4),), fifo=FifoPolicy(3, 5, 2, ((1, 6), (2, 7), (3, 8))))
    assert len(acquire(ts, cfg).deliveries) == 6
    ds = acquire(ts, cfg).deliveries
    oc = ObservationConfig('plant-a', 'sim-a', '1P2W', 'F01-RAM', '1', 'isolated',
                           'cfg@1', 's06@1', datetime(2026, 10, 17, 12, tzinfo=timezone.utc))
    refs = {(d.capture.capture_id, r.channel): 'opaque-lineage' for d in ds for r in d.capture.readings}
    assert compose_received(ds, oc, lineage_refs=refs) == compose_received(ds, oc, lineage_refs=refs)
    cfg = AcquisitionConfig(8, loss_probability=.5, random_identity=identity, random_channel='c1/P', random_profile='loss@1')
    assert acquire(ts, cfg) == acquire(ts, cfg)
assert state == random.getstate()
'''
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
