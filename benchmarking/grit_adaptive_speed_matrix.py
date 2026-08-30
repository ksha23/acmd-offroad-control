#!/usr/bin/env python3
"""Terrain-adaptive speed matrix: what an online soil estimate buys in speed.

This study produces Fig. 3 of the manuscript and the closed-loop terrain-estimation
benefit cited in Sec. 3.2: the mean forward speed each arm sustains on each soil,
and the mean per-run peak crosstrack error that the over-optimistic fixed prior
incurs. It is the study in which the estimate acts through the speed channel,
complementing the fixed-reference conditioning study, where it acts through the
cornering model alone.

The terrain-adaptive g--g speed cap is live: ``terrain_grip_limits`` keys the
acceleration envelope off whichever soil the controller currently assumes or
estimates. Four ways of supplying that soil are compared on every terrain, so
that an online estimate can be separated from any single fixed assumption:

  * ``oracle``       -- the matched true soil, a per-terrain reference rather
                        than a bound, since no deployable controller has it.
  * ``grit``         -- the live GRIT estimate feeds the cap.
  * ``conservative`` -- a fixed low-grip clay assumption: safe everywhere, and
                        correspondingly slow on firm soil.
  * ``aggressive``   -- a fixed high-grip sand assumption: fast on firm soil,
                        and over-driving the vehicle on soft soil.

The over-optimistic arm is what makes the comparison informative: a set of
assumptions that are all either correct or conservative cannot demonstrate that
identifying the soil matters, because none of them ever leaves the path.

The matrix crosses 4 arms with {clay, dirt, sand}, {sinusoidal, lane_change,
right_left}, {0, 4, 8} bumpiness, and 5 seeds, giving 540 cells. The speed
setpoint is set high enough that the cap, not the setpoint, is what limits
speed. Each cell is scored over a fixed distance window per path, identical
across arms and beginning after acquisition, and reports mean forward speed and
peak crosstrack error. Runs are truth-free (``--no-tire-forces``) and use ROS 2
through Chrono::ROS on the deployed powertrain, with no obstacles, safety filter,
or transport delay, so the speed channel is the only mechanism in play. Parallel
workers each own a fixed port, and therefore a unique modulo-101 DDS domain, and
run their cells serially.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarking"))
sys.path.insert(0, str(ROOT / "simulation" / "shared"))
try:
    from common import (launch_and_collect, estimator_runtime_args,
                        timestamped_result_dir)
    from paper_provenance import downstream_repository_provenance
except ModuleNotFoundError:
    from benchmarking.common import (launch_and_collect, estimator_runtime_args,
                                    timestamped_result_dir)
    from benchmarking.paper_provenance import downstream_repository_provenance

BACKEND = "grit"
DEPLOYED = "--simple-powertrain --ff-drag-surrogate --dob-ki 0 --dob-max 0"
# Scoring window per path, in forward distance. It starts after the online arm
# has had room to acquire and ends before the course does, and it is identical
# across arms so that every arm is scored on the same stretch of ground.
WINDOW = {"sinusoidal": (42.0, 90.0), "lane_change": (42.0, 60.0),
          "right_left": (42.0, 73.0)}


def _arms() -> dict[str, list[str]]:
    grit = ["--terrain-estimator", "--terrain-estimator-backend", BACKEND,
            "--terrain-estimator-mode", "n"] + estimator_runtime_args(BACKEND)
    return {"oracle": [], "grit": grit,
            "conservative": ["--controller-prior-terrain", "clay"],
            "aggressive": ["--controller-prior-terrain", "sand"]}


def _window_metrics(diag_csv: str, path: str) -> tuple[float, float, float, int]:
    x0, x1 = WINDOW[path]
    d = pd.read_csv(diag_csv)
    s = d[(d.x_fa_meas >= x0) & (d.x_fa_meas <= x1)]
    if len(s) < 5:
        return (float("nan"), float("nan"), float(d.x_fa_meas.max()), 0)
    return (float(s.u_meas.mean()), float(s.crosstrack_err.abs().max()),
            float(d.x_fa_meas.max()), len(s))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--terrains", nargs="+", default=["clay", "dirt", "sand"])
    ap.add_argument("--paths", nargs="+", default=["sinusoidal", "lane_change", "right_left"])
    ap.add_argument("--bumpiness", nargs="+", type=int, default=[0, 4, 8])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=400)
    ap.add_argument("--arms", nargs="+", default=list(_arms()))
    ap.add_argument("--joint-model-dir", default=None,
                    help="Evaluate a candidate design: run the grit arm's "
                         "estimator on this force model rather than on the one "
                         "named by its frozen contract. Not for published runs.")
    ap.add_argument("--joint-r-ay", type=float, default=None,
                    help="Evaluate a candidate design: override the grit arm's "
                         "lateral-acceleration scale. Not for published runs.")
    ap.add_argument("--speed", type=float, default=9.0)
    ap.add_argument("--time", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--base-port", type=int, default=35000)
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--aggregate-only", default=None,
                    help="Existing run dir to re-aggregate (skip the sims).")
    args = ap.parse_args()

    if not args.aggregate_only:
        # One top-level ROS benchmark at a time: concurrent sweeps can collide
        # on a DDS domain and silently mix runs. The handle is kept on
        # ``args`` so the exclusive flock is held until this process exits.
        from paper_provenance import acquire_paper_ros_lease
        args._ros_lease = acquire_paper_ros_lease("grit_adaptive_speed_matrix")

    provenance = downstream_repository_provenance()
    out = (Path(args.aggregate_only) if args.aggregate_only
           else timestamped_result_dir("grit_adaptive_speed_matrix"))
    arms = _arms()
    if args.joint_model_dir:
        arms["grit"] = arms["grit"] + ["--te-joint-model-dir", str(args.joint_model_dir)]
    if args.joint_r_ay is not None:
        arms["grit"] = arms["grit"] + ["--te-joint-r-ay", str(args.joint_r_ay)]
    seeds = [args.base_seed + i for i in range(args.seeds)]
    cells = [(arm, terr, path, bump, seed)
             for arm in args.arms for terr in args.terrains
             for path in args.paths for bump in args.bumpiness for seed in seeds]

    # Every cell runs the deployed plant configuration; the launcher appends
    # HIL_SIM_EXTRA to each launch, so setting it once configures the matrix.
    os.environ["HIL_SIM_EXTRA"] = DEPLOYED
    rows: list[dict] = []
    lock = threading.Lock()

    def run_cell(cell, sim_port):
        variant, terr, path, bump, seed = cell
        base = f"{variant}_{terr}_{path}_b{bump}_s{seed}"
        row = dict(variant=variant, terrain=terr, path=path, bumpiness=bump, seed=seed,
                   status="unrun", mean_u_window=float("nan"),
                   max_cte_window=float("nan"), x_max_m=float("nan"),
                   window_rows=0, sim_port=sim_port, attempts=0)
        for attempt in range(args.retries + 1):     # retry transient launch failures
            rd = out / (base if attempt == 0 else f"{base}_retry{attempt}")
            rd.mkdir(parents=True, exist_ok=True)
            row["attempts"] = attempt + 1
            try:
                r = launch_and_collect(
                    experiment="grit_adaptive_speed_matrix", variant=variant,
                    controller_mode="standard", mpc_model="nn", nn_model="tire_force_static",
                    terrain=terr, path=path, speed=args.speed, bumpiness=bump, seed=seed,
                    run_dir=rd, sim_port=sim_port, ctrl_port=sim_port + 1,
                    sim_time=args.time, timeout=args.timeout,
                    extra_args=arms[variant] + ["--no-tire-forces"])
                if r.status == "ok" and r.diag_csv:
                    mu, mcte, xmax, n = _window_metrics(r.diag_csv, path)
                    row.update(status="ok", mean_u_window=mu, max_cte_window=mcte,
                               x_max_m=xmax, window_rows=n)
                    if n >= 5:
                        break                        # a scorable window: accept the cell
                    row["status"] = "short_window"
                else:
                    row["status"] = r.status
            except Exception as exc:  # noqa: BLE001
                row["status"] = f"error:{type(exc).__name__}"
        with lock:
            rows.append(row)
            done = len(rows)
        print(f"[{done}/{len(cells)}] {row['status']:8s} {variant:12s} {terr:5s} "
              f"{path:12s} b{bump} s{seed}: u={row['mean_u_window']:.2f} "
              f"maxcte={row['max_cte_window']:.2f} (att {row['attempts']})", flush=True)

    if args.aggregate_only:                            # rescore existing runs in place
        for cell in cells:
            variant, terr, path, bump, seed = cell
            rd = out / f"{variant}_{terr}_{path}_b{bump}_s{seed}"
            diags = sorted(rd.rglob("diag_*.csv"), key=lambda p: p.stat().st_mtime)
            if diags:
                mu, mcte, xmax, n = _window_metrics(str(diags[-1]), path)
                st = "ok" if n >= 5 else "short_window"
            else:
                mu = mcte = xmax = float("nan"); n = 0; st = "missing"
            rows.append(dict(variant=variant, terrain=terr, path=path, bumpiness=bump,
                             seed=seed, status=st, mean_u_window=mu, max_cte_window=mcte,
                             x_max_m=xmax, window_rows=n, sim_port=0, attempts=1))
    else:
        # One port per worker gives one DDS domain per worker, since the
        # launcher derives the domain from the port modulo 101. The 250-port
        # stride keeps those domains distinct for every supported worker count.
        ports = [args.base_port + 250 * w for w in range(args.workers)]
        run_cell(cells[0], ports[0])                   # warm the solver cache alone
        remaining = cells[1:]

        def worker(wid: int):
            for i in range(wid, len(remaining), args.workers):
                run_cell(remaining[i], ports[wid])
        threads = [threading.Thread(target=worker, args=(w,)) for w in range(args.workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    results = pd.DataFrame(rows).sort_values(["variant", "terrain", "path", "bumpiness", "seed"])
    results.to_csv(out / "results.csv", index=False)
    ok = results[results.status == "ok"]
    summary = ok.groupby(["variant", "terrain"]).agg(
        mean_u=("mean_u_window", "mean"), mean_u_sd=("mean_u_window", "std"),
        max_cte=("max_cte_window", "mean"), max_cte_sd=("max_cte_window", "std"),
        max_cte_worst=("max_cte_window", "max"),
        n=("seed", "size")).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    manifest = dict(
        study="grit_adaptive_speed_matrix", tire_model="tire_force_static",
        terrain_estimator=BACKEND, reference_policy="prior_conditioned_live_gg_cap",
        longitudinal_stack="torque_command_simple_powertrain_surrogate_drag_ff_no_throttle_dob",
        speed_setpoint_mps=args.speed, time_s=args.time,
        distance_windows_m={k: list(v) for k, v in WINDOW.items()},
        arms=list(args.arms), terrains=args.terrains, paths=args.paths,
        bumpiness=args.bumpiness, seeds=seeds, n_cells=len(cells),
        n_ok=int(len(ok)), tire_force_truth_enabled=False, transport="ros",
        launch_identity_contract="path_speed_seed_ports_domain", **provenance)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out}  ({len(ok)}/{len(cells)} ok)")
    print(summary.to_string(index=False))
    print("\nGRIT_MATRIX_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
