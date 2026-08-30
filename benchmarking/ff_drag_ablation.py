#!/usr/bin/env python3
"""Ablation of the NMPC's longitudinal channel over its motion-resistance terms.

This study produces the paired drag-feedforward credit reported in Sec. 3.3 of
the manuscript: the ``ffdragsurr`` arm against the ``off`` arm, cell by cell,
credits the surrogate-drag feedforward a +0.067 gain in achieved-to-target speed
ratio at a tracking change within a centimeter. The wider variant set below
establishes the accompanying claim that the deployed longitudinal channel stays
kinematic rather than becoming an open-loop force integrator.

On deformable soil the NMPC's kinematic longitudinal channel
(``u_dot = ax + du_dot_resid``) over-predicts speed, because it carries no
compaction resistance. The principal arms differ in how that missing resistance
is supplied:

  dob             reactive throttle disturbance observer, an integrator that
                  learns the deficit online without a model of it.
  off             neither observer nor feedforward, exposing the raw deficit.
  ffdrag          a static drag feedforward, du_dot_resid = -c_drag(n_hat),
                  calibrated per soil by calibrate_motion_resistance.py.
  ffdragsurr      the surrogate's own zero-slip compaction drag evaluated live
                  at the current operating point: a model-based term rather than
                  a calibrated table, and the deployed choice.
  ffthrottle      the same correction applied in throttle units through a
                  one-dimensional map indexed by the soil estimate.
  ffthrottle2d    the throttle map extended to depend on speed as well as soil,
                  capturing the operating-point dependence the 1-D map omits.
  forcebalance    the longitudinal channel replaced outright by the surrogate's
                  force balance, u_dot = sum Fx(kappa) / M.

Several of the model-based arms have a stacked counterpart that also enables the
reactive observer, so that a term's contribution can be read both on its own and
in combination with the integrator.

Results are reported as RMS crosstrack error, speed ratio, and mean speed per
(variant, terrain), because the terms differ sharply by soil: compaction
resistance dominates on firm soil, and an arm that recovers speed there may
still leave a deficit on soft soil. Sensor noise is enabled in every run.
"""
from __future__ import annotations
import argparse, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_NN_MODEL, launch_and_collect, summarize_by_variant,
    timestamped_result_dir, write_results_csv, RunResult,
)

VARIANTS = {
    "dob":          [],
    "off":          ["--dob-ki", "0.0", "--dob-max", "0.0"],
    "ffdrag":       ["--dob-ki", "0.0", "--dob-max", "0.0", "--ff-drag"],
    "ffdrag+dob":   ["--ff-drag"],
    # Longitudinal force balance: u_dot = sum Fx(kappa) / M taken directly from
    # the tire surrogate, replacing the kinematic channel outright. It tests
    # whether a first-principles integrator recovers the observer's speed
    # without any reactive integrator or calibrated feedforward.
    "forcebalance": ["--dob-ki", "0.0", "--dob-max", "0.0",
                     "--longitudinal-force-balance"],
    "forcebalance+dob": ["--longitudinal-force-balance"],
    # The same correction applied in throttle rather than acceleration units: a
    # static offset indexed by the soil estimate, holding the value the reactive
    # observer converges to, with the observer disabled. It carries no
    # integrator state, so it isolates whether a per-soil lookup alone
    # reproduces the observer's speed and tracking.
    "ffthrottle": ["--dob-ki", "0.0", "--dob-max", "0.0", "--ff-throttle"],
    "ffthrottle+dob": ["--ff-throttle"],
    # The same static offset scaled up, which separates a residual speed
    # shortfall of the throttle map from a structural limitation of the map.
    "ffthrottle13": ["--dob-ki", "0.0", "--dob-max", "0.0", "--ff-throttle",
                     "--ff-throttle-scale", "1.3"],
    # Two-dimensional throttle offset d(n_hat, u), adding the speed dependence
    # the one-dimensional map omits, since compaction resistance varies with the
    # operating point and not with soil alone.
    "ffthrottle2d": ["--dob-ki", "0.0", "--dob-max", "0.0", "--ff-throttle-2d"],
    # Deployed arm: the tire surrogate's own zero-slip compaction drag evaluated
    # live at the current operating point, following the explicit drag term of
    # Dallas et al. and applied at torque level, with the observer disabled. It
    # replaces the calibrated table with the model the controller already
    # carries, and is the arm the manuscript credits in Sec. 3.3.
    "ffdragsurr": ["--dob-ki", "0.0", "--dob-max", "0.0", "--ff-drag-surrogate"],
}


@dataclass(frozen=True)
class _Task:
    idx: int
    variant: str
    extra: tuple
    terrain: str
    speed: float
    seed: int
    run_dir_str: str
    sim_port: int
    ctrl_port: int
    sim_time: float
    timeout: float


def _run_one(task: _Task) -> RunResult:
    res = launch_and_collect(
        experiment="ff_drag_ablation", variant=task.variant,
        controller_mode="standard", mpc_model="nn", nn_model=DEFAULT_NN_MODEL,
        terrain=task.terrain, path="sinusoidal", speed=task.speed,
        bumpiness=0, seed=task.seed, run_dir=Path(task.run_dir_str),
        sim_port=task.sim_port, ctrl_port=task.ctrl_port,
        sim_time=task.sim_time, timeout=task.timeout, rocks=0, lead_in=5.0,
        extra_args=list(task.extra),
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrains", nargs="+", default=["clay", "dirt", "sand"])
    ap.add_argument("--speeds", nargs="+", type=float, default=[5.0, 7.0])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--time", type=float, default=18.0)
    ap.add_argument("--timeout", type=float, default=220.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-port", type=int, default=8600)
    ap.add_argument("--variants", nargs="+", default=None,
                    help="Subset of VARIANTS to run (default: all).")
    ap.add_argument("--common-extra", type=str, default="",
                    help="Space-separated launch arguments appended to every "
                         "variant, which is how the whole ablation is placed on "
                         "one plant configuration. Requires the equals form, "
                         "for example --common-extra='--simple-powertrain'.")
    args = ap.parse_args()

    _ce = args.common_extra.split() if args.common_extra else []
    variants = ({v: VARIANTS[v] + _ce for v in args.variants} if args.variants
                else {v: e + _ce for v, e in VARIANTS.items()})

    out_dir = timestamped_result_dir("ff_drag_ablation")
    print(f"Output: {out_dir}")
    tasks, idx = [], 0
    for variant, extra in variants.items():
        for terr in args.terrains:
            for sp in args.speeds:
                for si in range(args.seeds):
                    port = args.base_port + 2 * idx
                    rd = out_dir / "raw" / f"{idx:03d}_{variant}_{terr}_v{sp:g}_s{si}"
                    tasks.append(_Task(idx, variant, tuple(extra), terr, sp,
                                       800 + si, str(rd), port, port + 1,
                                       args.time, args.timeout))
                    idx += 1
    print(f"{len(tasks)} runs across {len(variants)} variants")

    results = [_run_one(tasks[0])]
    print(f"  warmup {tasks[0].variant}/{tasks[0].terrain}: "
          f"cte={results[0].rms_cte_m:.3f} sr={results[0].speed_ratio:.2f}")
    write_results_csv(out_dir / "results.csv", results)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks[1:]}
        for fut in as_completed(futs):
            r = fut.result(); results.append(r)
            write_results_csv(out_dir / "results.csv", results)

    write_results_csv(out_dir / "results.csv", results)
    summ = summarize_by_variant(results, ["rms_cte_m", "speed_ratio", "mean_speed_mps"])
    summ.to_csv(out_dir / "summary_by_variant.csv", index=False)
    # Report per (terrain, variant): the terms differ sharply by soil, so a
    # pooled mean would hide which soils a given term actually corrects.
    import pandas as pd
    df = pd.read_csv(out_dir / "results.csv")
    ok = df[df["status"] == "ok"]
    piv_sr = ok.pivot_table(index="terrain", columns="variant", values="speed_ratio", aggfunc="mean")
    piv_cte = ok.pivot_table(index="terrain", columns="variant", values="rms_cte_m", aggfunc="mean")
    print("\n=== speed_ratio by terrain x variant ===")
    print(piv_sr.round(3).to_string())
    print("\n=== rms_cte (m) by terrain x variant ===")
    print(piv_cte.round(3).to_string())
    piv_sr.to_csv(out_dir / "speed_ratio_by_terrain.csv")
    piv_cte.to_csv(out_dir / "rms_cte_by_terrain.csv")
    paper = piv_sr.add_prefix("sr_").join(piv_cte.add_prefix("cte_"))
    paper.to_csv(out_dir / "paper_summary.csv")
    print(f"\nFF_DRAG_ABLATION_DONE {out_dir}")


if __name__ == "__main__":
    main()
