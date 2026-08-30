#!/usr/bin/env python3
"""Solve cost of the predictive safety filter against its obstacle capacity.

This study produces the predictive-filter solve-time figures the manuscript
quotes in Sec. 4.2, where the predictive filter is described as costing a
heavier solve than the reactive backstop while remaining well inside the message
period out to sixteen obstacles. It measures the per-call filter solve time
directly, so the claim rests on a recorded quantity rather than on inspection of
the formulation.

MPSF sizes its optimal control problem for a fixed obstacle count
(``--mpsf-n-obstacles``, deployed value 3) and selects that many
most-threatening obstacles from the scene, so its cost should follow the
capacity rather than the size of the field. The sweep therefore holds the rock
field fixed at sixteen rocks and varies the filter's capacity alone, and the
reactive DOB-CBF filter is measured on the identical scene so that the
comparison between the two is like for like.

Runs execute one at a time. Filter timing measured under benchmark parallelism
is inflated roughly twofold and does not represent deployment cost.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarking"))
try:
    from common import launch_and_collect, timestamped_result_dir, write_manifest
except ModuleNotFoundError:
    from benchmarking.common import (launch_and_collect, timestamped_result_dir,
                                     write_manifest)

# A field large enough that the filter's capacity, not the scene, is the limit.
ROCKS = ["--rocks", "16", "--rock-zone-x", "18", "70",
         "--rock-zone-y", "-6", "6", "--rock-size", "0.8", "1.2",
         "--rock-min-spacing", "5", "--rock-centerline-clear", "0.0",
         "--rock-spawn-clear", "12", "--rock-seed", "11"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", nargs="+", type=int, default=[3, 6, 10, 16])
    ap.add_argument("--baseline-flavor", default="dob_cbf",
                    help="Reactive filter measured on the identical scene, which "
                         "turns the manuscript's statement that predictive "
                         "safety costs a heavier solve into a measured "
                         "comparison. Pass an empty value to skip it.")
    ap.add_argument("--terrain", default="dirt")
    ap.add_argument("--path", default="straight")
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--time", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--base-port", type=int, default=24000)
    ap.add_argument("--timeout", type=float, default=500.0)
    args = ap.parse_args()

    out = timestamped_result_dir("mpsf_solve_scaling")
    write_manifest(out, args, "MPSF filter solve cost against its obstacle count")

    rows = []
    for i, n in enumerate(args.counts):
        rd = out / f"n{n}"
        rd.mkdir(parents=True, exist_ok=True)
        r = launch_and_collect(
            experiment="mpsf_solve_scaling", variant=f"n_obstacles_{n}",
            controller_mode="standard", mpc_model="nn", nn_model="tire_force_static",
            terrain=args.terrain, path=args.path, speed=args.speed, bumpiness=0,
            seed=args.seed, run_dir=rd, sim_port=args.base_port + 4 * i,
            ctrl_port=args.base_port + 4 * i + 1, sim_time=args.time,
            timeout=args.timeout, metric_start=5.0,
            extra_args=["--safety-filter", "--safety-flavor", "mpsf",
                        "--mpsf-n-obstacles", str(n), *ROCKS])
        row = dict(n_obstacles=n, status=r.status,
                   filter_solve_ms=float(getattr(r, "filter_solve_ms", float("nan"))),
                   mpc_solve_ms=float(r.mean_solve_ms),
                   intervention_rate_pct=float(r.intervention_rate_pct),
                   collisions=int(r.collisions) if r.collisions == r.collisions else -1)
        rows.append(row)
        print(f"  n_obstacles={n:2d}  status={row['status']:8s} "
              f"filter={row['filter_solve_ms']:7.3f} ms  "
              f"nmpc={row['mpc_solve_ms']:6.2f} ms", flush=True)

    if args.baseline_flavor:
        rd = out / f"baseline_{args.baseline_flavor}"
        rd.mkdir(parents=True, exist_ok=True)
        r = launch_and_collect(
            experiment="mpsf_solve_scaling", variant=f"baseline_{args.baseline_flavor}",
            controller_mode="standard", mpc_model="nn", nn_model="tire_force_static",
            terrain=args.terrain, path=args.path, speed=args.speed, bumpiness=0,
            seed=args.seed, run_dir=rd, sim_port=args.base_port + 400,
            ctrl_port=args.base_port + 401, sim_time=args.time,
            timeout=args.timeout, metric_start=5.0,
            extra_args=["--safety-filter", "--safety-flavor", args.baseline_flavor, *ROCKS])
        solve_ms = float(getattr(r, "filter_solve_ms", float("nan")))
        if solve_ms != solve_ms:          # reactive filters log one row per call
            log = rd / "cbf_filter_log.csv"
            if log.is_file():
                import csv as _csv
                vals = [float(x["solve_ms"]) for x in _csv.DictReader(log.open())
                        if x.get("solve_ms") not in (None, "")]
                if vals:
                    solve_ms = sum(vals) / len(vals)
        rows.append(dict(n_obstacles=None, flavor=args.baseline_flavor, status=r.status,
                         filter_solve_ms=solve_ms,
                         mpc_solve_ms=float(r.mean_solve_ms),
                         intervention_rate_pct=float(r.intervention_rate_pct),
                         collisions=int(r.collisions) if r.collisions == r.collisions else -1))
        print(f"  {args.baseline_flavor:16s} status={r.status:8s} "
              f"filter={rows[-1]['filter_solve_ms']:7.3f} ms", flush=True)

    (out / "scaling.json").write_text(json.dumps(
        {"schema_version": 1, "study": "mpsf_solve_scaling",
         "note": "run one at a time; parallel execution inflates solve times ~2x",
         "rocks_in_scene": 16, "rows": rows}, indent=2) + "\n")
    ok = [r for r in rows if r["status"] == "ok" and r["filter_solve_ms"] == r["filter_solve_ms"]]
    if ok:
        worst = max(ok, key=lambda r: r["filter_solve_ms"])
        print(f"\nworst filter solve: {worst['filter_solve_ms']:.3f} ms at "
              f"n_obstacles={worst['n_obstacles']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
