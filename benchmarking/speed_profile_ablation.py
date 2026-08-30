#!/usr/bin/env python3
"""Component study of the terrain-aware g-g speed reference.

This study produces the known-terrain speed-reference component check reported
in Sec. 3.3 of the manuscript, which credits the terrain-aware reference a
reduction in mean RMS crosstrack error at a stated mean-speed cost. It is a
paired comparison of two speed references at matched terrain, path, target
speed, and seed:

  * ``baseline`` -- the static curvature-only reference, which sets speed from
                    path geometry alone and therefore asks for more than
                    deformable soil can deliver on the tighter sections.
  * ``terrain``  -- the reference additionally capped by a friction-circle (g-g)
                    bound, whose combined longitudinal and lateral acceleration
                    budget is the envelope the tire surrogate predicts for the
                    applied soil.

Both arms are supplied the known soil preset, so this measures the speed
reference itself and not online terrain identification; the value of a live
estimate in the speed channel is measured by ``grit_adaptive_speed_matrix.py``.
Mean RMS crosstrack error and mean achieved speed are reported for every
(terrain, speed) cell so that the tracking gain is always read against its
speed cost, and ``speed_profile_paired.csv`` carries the per-cell pairing that
the manuscript's figures and statistics consume.
"""
from __future__ import annotations
import argparse, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_NN_MODEL, launch_and_collect, summarize_by_variant,
    timestamped_result_dir, write_results_csv, write_manifest, RunResult,
)

VARIANTS = {
    "baseline": ["--legacy-speed-ref"],   # curvature-only reference
    "terrain":  [],                        # curvature reference capped by the g-g bound
}


@dataclass(frozen=True)
class _Task:
    idx: int
    variant: str
    extra: tuple
    terrain: str
    path: str
    speed: float
    seed: int
    run_dir_str: str
    sim_port: int
    ctrl_port: int
    sim_time: float
    timeout: float


def _run_one(task: _Task) -> RunResult:
    return launch_and_collect(
        experiment="speed_profile_ablation", variant=task.variant,
        controller_mode="standard", mpc_model="nn", nn_model=DEFAULT_NN_MODEL,
        terrain=task.terrain, path=task.path, speed=task.speed,
        bumpiness=0, seed=task.seed, run_dir=Path(task.run_dir_str),
        sim_port=task.sim_port, ctrl_port=task.ctrl_port,
        sim_time=task.sim_time, timeout=task.timeout, rocks=0, lead_in=5.0,
        extra_args=list(task.extra),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrains", nargs="+", default=["clay", "dirt", "sand"])
    ap.add_argument("--paths", nargs="+", default=["sinusoidal"])
    ap.add_argument("--speeds", nargs="+", type=float, default=[5.0, 7.0])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--time", type=float, default=18.0)
    ap.add_argument("--timeout", type=float, default=220.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-port", type=int, default=9000)
    args = ap.parse_args()

    # One top-level ROS benchmark at a time: concurrent sweeps can collide on
    # a DDS domain and silently mix runs. The handle is kept on ``args`` so
    # the exclusive flock is held until this process exits.
    from paper_provenance import acquire_paper_ros_lease
    args._ros_lease = acquire_paper_ros_lease("speed_profile_ablation")

    out_dir = timestamped_result_dir("speed_profile_ablation")
    write_manifest(out_dir, args,
                   "Terrain-aware speed-reference component study.")
    print(f"Output: {out_dir}")
    tasks, idx = [], 0
    for variant, extra in VARIANTS.items():
        for terr in args.terrains:
            for path in args.paths:
                for sp in args.speeds:
                    for si in range(args.seeds):
                        port = args.base_port + 2 * idx
                        rd = out_dir / "raw" / f"{idx:03d}_{variant}_{terr}_{path}_v{sp:g}_s{si}"
                        tasks.append(_Task(idx, variant, tuple(extra), terr, path, sp,
                                           950 + si, str(rd), port, port + 1,
                                           args.time, args.timeout))
                        idx += 1
    print(f"{len(tasks)} runs")

    results = [_run_one(tasks[0])]
    write_results_csv(out_dir / "results.csv", results)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks[1:]}
        for fut in as_completed(futs):
            results.append(fut.result())
            write_results_csv(out_dir / "results.csv", results)

    write_results_csv(out_dir / "results.csv", results)
    summarize_by_variant(results, ["rms_cte_m", "speed_ratio", "mean_speed_mps"]).to_csv(
        out_dir / "summary_by_variant.csv", index=False)
    import pandas as pd
    ok = pd.read_csv(out_dir / "results.csv")
    ok = ok[ok["status"] == "ok"]
    for m in ["rms_cte_m", "mean_speed_mps"]:
        print(f"\n=== {m} (terrain,speed x variant) ===")
        p = ok.pivot_table(index=["terrain", "speed_mps"], columns="variant", values=m, aggfunc="mean")
        cols = [c for c in ["baseline", "terrain"] if c in p.columns]
        print(p[cols].round(3).to_string())

    # One row per (path, terrain, speed) cell holding both arms' mean crosstrack
    # error, mean speed, and across-seed standard deviation. Keeping the pairing
    # explicit is what allows the manuscript to quote a paired reduction rather
    # than a difference of independent group means. publish_paper_figures
    # republishes this file as my_paper/paper_figures/speed_profile_ablation.csv.
    def _piv(col, agg):
        return (ok.groupby(["path", "terrain", "speed_mps", "variant"])[col]
                  .agg(agg).unstack("variant"))
    mc, sc, ms = _piv("rms_cte_m", "mean"), _piv("rms_cte_m", "std"), _piv("mean_speed_mps", "mean")
    paired = pd.DataFrame({
        "cte_baseline": mc.get("baseline"), "speed_baseline": ms.get("baseline"),
        "cte_terrain": mc.get("terrain"), "speed_terrain": ms.get("terrain"),
        "cte_baseline_sd": sc.get("baseline"), "cte_terrain_sd": sc.get("terrain"),
    }).reset_index().rename(columns={"speed_mps": "speed"})
    paired = paired[["path", "terrain", "speed", "cte_baseline", "speed_baseline",
                     "cte_terrain", "speed_terrain", "cte_baseline_sd", "cte_terrain_sd"]]
    paired.to_csv(out_dir / "speed_profile_paired.csv", index=False)
    print(f"wrote {out_dir / 'speed_profile_paired.csv'}")

    print(f"\nSPEED_PROFILE_ABLATION_DONE {out_dir}")


if __name__ == "__main__":
    main()
