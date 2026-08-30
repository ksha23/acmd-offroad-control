"""Contracts on the harness's metric-extraction and plant-configuration paths.

The numbers in every results.csv pass through a handful of parser and
aggregation helpers that previously had no coverage at all: a silent change to
any of them would move published values while every simulation stayed
identical. These tests pin the extraction semantics, the rule that failed runs
never contribute to metric means, and the rule that a driver spawning the
simulator itself must apply the HIL_SIM_EXTRA plant flags that run.py exports.
"""
from __future__ import annotations

import math
import os
import re
import tempfile
import unittest
from pathlib import Path

import benchmarking.common as common
from benchmarking.common import (RunResult, parse_diag_csv, parse_shield_csv,
                                 parse_sim_diag_csv, summarize_by_variant)

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarking"


def _write(tmp: str, name: str, text: str) -> Path:
    p = Path(tmp) / name
    p.write_text(text)
    return p


class ParseDiagCsvTest(unittest.TestCase):
    def test_metrics_from_known_trace(self):
        # 3 samples past metric_start with cte 0.3/0.4/0.5, odometer legs
        # 1.0 + 1.0; one pre-start sample that must be excluded from cte.
        text = ("sim_time,crosstrack_err,u_true,solve_time_ms,x_fa_true,y_fa_true\n"
                "0.0,9.9,0.0,5.0,0.0,0.0\n"
                "2.0,0.3,4.0,5.0,1.0,0.0\n"
                "2.1,0.4,4.0,5.0,2.0,0.0\n"
                "2.2,0.5,4.0,5.0,2.0,1.0\n")
        with tempfile.TemporaryDirectory() as tmp:
            out = parse_diag_csv(_write(tmp, "diag.csv", text), "mpc", 4.0)
        self.assertEqual(out["n_samples"], 4)
        expected_rms = math.sqrt((0.3 ** 2 + 0.4 ** 2 + 0.5 ** 2) / 3.0)
        self.assertAlmostEqual(out["rms_cte_m"], expected_rms, places=9)
        self.assertAlmostEqual(out["max_abs_cte_m"], 0.5, places=9)
        # progress is the odometer path length over the whole trace, not the
        # displacement: 1 + 1 + 1 = 3, while final_x is 2.
        self.assertAlmostEqual(out["progress_m"], 3.0, places=9)
        self.assertAlmostEqual(out["final_x_m"], 2.0, places=9)
        self.assertAlmostEqual(out["mean_speed_mps"], 4.0, places=9)
        self.assertAlmostEqual(out["speed_ratio"], 1.0, places=9)

    def test_header_only_and_empty_are_no_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            hdr = parse_diag_csv(
                _write(tmp, "h.csv", "sim_time,crosstrack_err\n"), "mpc", 4.0)
            empty = parse_diag_csv(_write(tmp, "e.csv", ""), "mpc", 4.0)
        self.assertEqual(hdr, {"n_samples": 0})
        self.assertEqual(empty, {"n_samples": 0})


class ParseShieldCsvTest(unittest.TestCase):
    def test_deviation_means(self):
        text = ("steer_in,steer_out,throttle_in,throttle_out\n"
                "0.0,0.1,0.5,0.5\n"
                "0.0,-0.1,0.5,0.1\n")
        with tempfile.TemporaryDirectory() as tmp:
            out = parse_shield_csv(_write(tmp, "s.csv", text))
        self.assertAlmostEqual(out["mean_abs_dsteer"], 0.1, places=9)
        self.assertAlmostEqual(out["mean_abs_dthrottle"], 0.2, places=9)

    def test_missing_or_malformed_returns_empty(self):
        self.assertEqual(parse_shield_csv(None), {})
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(parse_shield_csv(_write(tmp, "s.csv", "")), {})
            self.assertEqual(
                parse_shield_csv(_write(tmp, "t.csv", "a,b\n1,2\n")), {})


class ParseSimDiagCsvTest(unittest.TestCase):
    def test_collision_counts_and_clearance(self):
        text = ("time,nearest_clearance_m,collisions,near_misses\n"
                "0.0,0.5,0,0\n"          # pre-start clearance excluded
                "2.0,3.0,0,0\n"
                "2.1,2.0,1,2\n")
        with tempfile.TemporaryDirectory() as tmp:
            out = parse_sim_diag_csv(_write(tmp, "d.csv", text))
        self.assertAlmostEqual(out["min_clearance_m"], 2.0, places=9)
        self.assertEqual(out["collisions"], 1)   # final cumulative count
        self.assertEqual(out["near_misses"], 2)


class SummarizeByVariantTest(unittest.TestCase):
    @staticmethod
    def _result(variant: str, status: str, value: float) -> RunResult:
        return RunResult(
            experiment="e", variant=variant, controller_mode="m",
            mpc_model="x", nn_model="n", terrain="clay", path="s",
            speed_mps=5.0, bumpiness=0, seed=1, run_dir="/tmp/none",
            status=status, extra={}, collisions=value)

    def test_failed_runs_counted_but_excluded_from_means(self):
        rows = [self._result("a", "ok", 1.0),
                self._result("a", "ok", 3.0),
                # A crashed run carrying a wild metric value must not move the
                # mean; it shows up only as attrition in n_ok.
                self._result("a", "exit_1", 1000.0)]
        df = summarize_by_variant(rows, ["collisions"])
        row = df[df["variant"] == "a"].iloc[0]
        self.assertEqual(int(row["n_runs"]), 3)
        self.assertEqual(int(row["n_ok"]), 2)
        self.assertAlmostEqual(float(row["collisions_mean"]), 2.0, places=9)

    def test_variant_with_no_ok_runs_has_nan_metrics(self):
        rows = [self._result("a", "exit_1", 7.0),
                self._result("a", "timeout", 7.0)]
        df = summarize_by_variant(rows, ["collisions"])
        row = df.iloc[0]
        self.assertEqual(int(row["n_ok"]), 0)
        self.assertTrue(math.isnan(float(row["collisions_mean"])))


class HilSimExtraContractTest(unittest.TestCase):
    """A driver that spawns chrono_sim_node itself must apply HIL_SIM_EXTRA.

    run.py exports the deployed plant flags through this variable for every
    study it does not explicitly exempt; common.launch_and_collect appends it
    for the studies that go through common.py. A driver that builds its own
    simulator command and drops the variable silently runs a different plant
    than the matrix its rows are read against -- that is exactly how a 60-run
    convoy/latency generation was invalidated on 2026-08-22.
    """

    # Drivers allowed to spawn the simulator without HIL_SIM_EXTRA, with the
    # reason the exemption is sound.
    EXEMPT = {
        "teleop_failure_modes.py":   # declares its own 5G battery plant;
            "run.py maps teleop_battery -> None and unsets HIL_SIM_EXTRA",
        "numerical_convergence.py":  # solver-convergence diagnostic;
            "not a paper contact study and not launched through run.py",
        "paper_provenance.py":       # hashes source files, spawns nothing
            "references the path only to hash it",
    }

    def test_every_direct_spawner_applies_the_variable(self):
        for path in sorted(BENCH.glob("*.py")):
            src = path.read_text()
            if "chrono_sim_node" not in src or path.name in self.EXEMPT:
                continue
            self.assertIn(
                "HIL_SIM_EXTRA", src,
                f"{path.name} spawns chrono_sim_node but never applies "
                f"HIL_SIM_EXTRA; append os.environ HIL_SIM_EXTRA to its sim "
                f"command or add a justified exemption here")

    # The deployed HIL_SIM_EXTRA mixes simulator and controller flags; a
    # replay driver must apply exactly the simulator's share. Appending the
    # whole string made the simulator's argparser exit(2) instantly, which is
    # how the 2026-08-26 convoy and latency generations lost all 180 runs.
    DEPLOYED = "--simple-powertrain --ff-drag-surrogate --dob-ki 0 --dob-max 0"

    def test_helper_filters_to_the_simulator_share(self):
        from benchmarking.common import sim_node_flags_from_hil_extra
        os.environ["HIL_SIM_EXTRA"] = self.DEPLOYED
        try:
            self.assertEqual(sim_node_flags_from_hil_extra(),
                             ["--simple-powertrain"])
        finally:
            del os.environ["HIL_SIM_EXTRA"]

    def test_helper_refuses_unclassified_flags(self):
        from benchmarking.common import sim_node_flags_from_hil_extra
        os.environ["HIL_SIM_EXTRA"] = "--some-new-flag"
        try:
            with self.assertRaisesRegex(ValueError, "unclassified"):
                sim_node_flags_from_hil_extra()
        finally:
            del os.environ["HIL_SIM_EXTRA"]

    def test_convoy_build_cmd_appends_simulator_flags(self):
        import benchmarking.convoy_counterfactual_eval as convoy
        t = convoy.Task(idx=0, filter_name="none", delay=0.0, sim_port=1,
                        run_dir="/tmp/r", trace="/tmp/t.csv", convoy="jam",
                        terrain="dirt", time_s=1.0, mesh=0.1, buffer=1.0,
                        horizon=18, timeout=1.0)
        os.environ["HIL_SIM_EXTRA"] = self.DEPLOYED
        try:
            cmd = convoy._build_cmd(t)
        finally:
            del os.environ["HIL_SIM_EXTRA"]
        self.assertEqual(cmd[-1], "--simple-powertrain")
        self.assertNotIn("--dob-ki", cmd)
        self.assertNotIn("--ff-drag-surrogate", cmd)

    def test_latency_build_cmd_appends_simulator_flags(self):
        import benchmarking.latency_awareness_ablation as lat
        t = lat.Task(idx=0, trace="/tmp/t.csv", convoy="jam", variant="none",
                     delay=0.3, throttle=0.5, run_dir="/tmp/r", sim_port=1,
                     terrain="dirt", time_s=1.0, mesh=0.1, buffer=1.0,
                     timeout=1.0, cell="c")
        os.environ["HIL_SIM_EXTRA"] = self.DEPLOYED
        try:
            cmd = lat._build_cmd(t)
        finally:
            del os.environ["HIL_SIM_EXTRA"]
        self.assertEqual(cmd[-1], "--simple-powertrain")
        self.assertNotIn("--dob-max", cmd)


if __name__ == "__main__":
    unittest.main()
