"""Composición parcial S03; oráculos S05, sin admisión canónica."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import unittest

from smartenergy.analysis.simulator_acquisition import AcquisitionConfig, acquire
from smartenergy.analysis.simulator_contract import ObservationConfig
from smartenergy.analysis.simulator_energy import CircuitEnergy, EnergyCounters
from smartenergy.analysis.simulator_instrumentation import (
    ChannelConfig, ErrorProfile, InstrumentConfig, RandomIdentity, Retention, VirtualInstrument,
)
from smartenergy.analysis.simulator_scenarios import build_scenario
from smartenergy.application.simulator import compose_received

P = "c1/P"


def fixture(scenario="F01-RAM", channels=(ChannelConfig(P),), times=range(6), **kwargs):
    return VirtualInstrument(build_scenario(scenario), InstrumentConfig(
        "sim-a", channels, RandomIdentity(20261017, "run-1", "sim-a"), **kwargs)).capture(times)


def config(**kwargs):
    return replace(ObservationConfig(
        "plant-a", "sim-a", "1P2W", "F01-RAM", "1", "run-1", "cfg@1", "s06@1",
        datetime(2026, 10, 17, 12, tzinfo=timezone.utc)), **kwargs)


def compose(ds, cfg=None):
    refs = {(d.capture.capture_id, r.channel): f"opaque:{d.capture.index}:{r.channel}"
            for d in ds for r in d.capture.readings}
    return compose_received(ds, cfg or config(), lineage_refs=refs)


class CompositionTests(unittest.TestCase):
    def test_a01_cal_origin_order_and_age_boundary(self):
        ds = acquire(fixture(), AcquisitionConfig(
            8, lost_indices=(2,), delays=((1, 6),), redeliveries=((3, 8),))).deliveries
        rows = compose(ds)
        for row, delivery, expected in zip(rows, ds, (600, 630, 640, 650, 610, 630)):
            value = row["observation"]["value"]
            self.assertAlmostEqual(value, expected, delta=1e-6 + 1e-9 * expected)
            self.assertEqual(value, delivery.capture.readings[0].value)
        self.assertEqual([r["evaluation"]["freshness"] for r in rows],
                         ["current"] * 4 + ["stale", "current"])
        self.assertEqual(rows[1]["observation"], rows[-1]["observation"])
        self.assertNotEqual(rows[1]["reception"]["delivery_id"], rows[-1]["reception"]["delivery_id"])
        late = rows[-2]
        self.assertEqual(late["observation"]["time"]["support"], dict(
            kind="point", start="2026-10-17T12:00:01Z", end="2026-10-17T12:00:01Z"))
        self.assertEqual(late["reception"]["received_at"], "2026-10-17T12:00:07Z")
        self.assertIsNone(late["reception"]["recorded_at"])
        self.assertEqual(late["observation"]["quality"]["freshness"], "unknown")
        self.assertEqual(late["evaluation"]["reason"], "age_exceeded")
        self.assertNotIn("status", late["reception"])
        self.assertEqual(rows, compose(ds))
        json.dumps(rows, allow_nan=False)

    def test_units_scopes_and_private_truth(self):
        cs = tuple(ChannelConfig(k, output_unit="kW" if k == P else None)
                   for k in ("bus-a/L1-N/U", "c1/L1/I", "c1/L1/P", P, "f"))
        ds = acquire(fixture(channels=cs, times=(0,)), AcquisitionConfig(0)).deliveries
        rows = compose(ds)
        self.assertEqual(len({r["observation"]["observation_id"] for r in rows}), 5)
        voltage, current, phase, total, frequency = [r["observation"] for r in rows]
        self.assertIsNone(voltage["context"]["circuit_id"])
        self.assertEqual(voltage["context"]["scope"], dict(kind="voltage_pair", conductors=["L1", "N"]))
        self.assertEqual(current["context"]["circuit_id"], "c1")
        self.assertEqual(phase["context"]["scope"]["kind"], "phase")
        self.assertEqual(total["context"]["scope"]["kind"], "circuit_total")
        self.assertEqual((total["value"], total["unit"]), (600, "W"))
        self.assertEqual(total["provenance"]["original"], dict(value=.6, unit="kW", code=P))
        self.assertEqual(frequency["context"]["scope"]["kind"], "supply")
        payload = json.dumps(rows)
        for private in ("primary", "innovation", "retained_from_index", "real_support_s", "seeds"):
            self.assertNotIn(private, payload)

    def test_retention_declared_time_only(self):
        for clock, expected, end in (("original", "stale", "00"),
                                     ("current", "current", "10"), ("unknown", "unknown", None)):
            with self.subTest(clock=clock):
                ts = fixture(times=(0, 10), retentions=(Retention(1, 11, 0, (P,), clock),))
                row = compose(acquire(ts, AcquisitionConfig(10)).deliveries)[-1]
                self.assertEqual(row["observation"]["value"], 600)
                self.assertEqual(row["evaluation"]["freshness"], expected)
                time = row["observation"]["time"]
                self.assertEqual(time["support"]["end"], None if end is None else f"2026-10-17T12:00:{end}Z")
                self.assertEqual(time["measured_at"], time["support"]["end"])
                if clock == "unknown":
                    self.assertEqual(time["clock"], dict(basis="unknown", resolution_s=None, uncertainty_s=None))

    def test_null_fp_zero_and_silence_loss(self):
        ts = fixture("F01-REP", (ChannelConfig(P), ChannelConfig("c1/FP")), (0,))
        zero, fp = compose(acquire(ts, AcquisitionConfig(0)).deliveries, config(scenario_id="F01-REP"))
        self.assertEqual(zero["observation"]["value"], 0)
        self.assertEqual(zero["observation"]["quality"]["availability"], "present")
        self.assertIsNone(fp["observation"]["value"])
        self.assertEqual(fp["observation"]["quality"]["availability"], "not_applicable")
        self.assertEqual(fp["evaluation"]["freshness"], "not_applicable")
        ts = fixture(silent_indices=(2, 3, 4))
        rows = compose(acquire(ts, AcquisitionConfig(5, lost_indices=(1,))).deliveries)
        self.assertEqual(len(rows), 2)
        for row, expected in zip(rows, (600, 650)):
            self.assertAlmostEqual(row["observation"]["value"], expected, delta=1e-6 + 1e-9 * expected)
        self.assertEqual(compose(()), ())

    def test_incoherence_and_invalid_values_not_repaired(self):
        cs = (ChannelConfig(P, error=ErrorProfile(reference=12000, offset=1200)),
              ChannelConfig("c1/S"), ChannelConfig("c1/FP"))
        ds = acquire(fixture(channels=cs, times=(0,)), AcquisitionConfig(6, delays=((0, 6),))).deliveries
        rows = compose(ds)
        self.assertEqual([r["observation"]["value"] for r in rows], [1800, 600, 1])
        self.assertEqual(rows[0]["observation"]["quality"]["reason"], "no_diagnostic")
        reading = replace(ds[0].capture.readings[0], value=20000, original_value=20000,
                          validity="invalid", reason="range_violation")
        row = compose((replace(ds[0], capture=replace(ds[0].capture, readings=(reading,))),))[0]
        self.assertEqual(row["observation"]["value"], 20000)
        self.assertEqual(row["observation"]["quality"]["reason"], "range_violation")
        self.assertEqual(row["evaluation"]["reason"], "age_exceeded")

    def test_no_overwrite_or_admission_and_independent_outputs(self):
        d = acquire(fixture(times=(0,)), AcquisitionConfig(0)).deliveries[0]
        other = replace(d, delivery_id="adverse", capture=replace(d.capture, readings=(
            replace(d.capture.readings[0], value=601, original_value=601),)))
        rows = compose((d, other))
        self.assertEqual(rows[0]["observation"]["observation_id"], rows[1]["observation"]["observation_id"])
        self.assertEqual([r["observation"]["value"] for r in rows], [600, 601])
        self.assertTrue(all("status" not in r["reception"] for r in rows))
        rows[0]["observation"]["value"] = -1
        self.assertEqual(compose((d,))[0]["observation"]["value"], 600)
        self.assertEqual(d.capture.readings[0].value, 600)

    def test_counter_support_epoch_and_redelivery(self):
        initial = EnergyCounters(10, 0, 0, 0)
        ts = fixture("F01-EST", (ChannelConfig("c1/EP+", output_unit="kWh"),), (0, 60),
                     counter_epoch="epoch-1", initial_counters=CircuitEnergy((initial,), initial))
        ds = acquire(ts, AcquisitionConfig(65, redeliveries=((1, 65),))).deliveries
        rows = compose(ds, config(counter_epoch="epoch-1", scenario_id="F01-EST"))
        self.assertEqual([r["observation"]["value"] for r in rows], [10, 30, 30])
        self.assertEqual(rows[1]["observation"], rows[2]["observation"])
        self.assertEqual(rows[1]["observation"]["time"]["support"]["kind"], "interval")
        self.assertEqual(rows[1]["observation"]["provenance"]["counter_epoch"], "epoch-1")
        with self.assertRaisesRegex(ValueError, "época"):
            compose(ds)

    def test_three_phase_scope_and_wrong_device(self):
        ts = fixture("F04-CAR", (ChannelConfig("bus-a/L2-L3/ULL"), ChannelConfig("c1/L2/P")), (0,))
        ds = acquire(ts, AcquisitionConfig(0)).deliveries
        rows = compose(ds, config(topology="3P4W", scenario_id="F04-CAR"))
        self.assertEqual(rows[0]["observation"]["context"]["scope"]["conductors"], ["L2", "L3"])
        for cfg in (config(), config(device_id="other")):
            with self.assertRaises(ValueError):
                compose(ds, cfg)
        with self.assertRaisesRegex(ValueError, "ejecución"):
            compose(ds, config(topology="3P4W", run_id="other"))

    def test_invalid_inputs_and_exact_time_representation(self):
        ds = acquire(fixture(times=(0,)), AcquisitionConfig(0)).deliveries
        with self.assertRaisesRegex(ValueError, "linaje"):
            compose_received(ds, config(), lineage_refs={})
        for kwargs in (dict(origin_utc=datetime(2026, 10, 17)), dict(topology="other"),
                       dict(resolution_s=0), dict(uncertainty_s=-1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                config(**kwargs)
        with self.assertRaises(ValueError):
            config().timestamp(1e-7)
        self.assertEqual(config().timestamp(.125), "2026-10-17T12:00:00.125000Z")
        bad = replace(ds[0].capture.readings[0], value=float("nan"))
        with self.assertRaises(ValueError):
            compose((replace(ds[0], capture=replace(ds[0].capture, readings=(bad,))),))
        for bad in (replace(ds[0].capture.readings[0], original_value=float("inf")),
                    replace(ds[0].capture.readings[0], original_unit="A"),
                    replace(ds[0].capture.readings[0], original_value=601)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compose((replace(ds[0], capture=replace(ds[0].capture, readings=(bad,))),))


if __name__ == "__main__":
    unittest.main()
