#!/usr/bin/env python3
"""Paired ablation separating delay compensation from the presence of a filter.

This study produces the delay-awareness result reported in Sec. 4.2 of the
manuscript: the paired change in closest approach between the delay-blind and
delay-aware filter, the count of paired cells in which awareness is better, and
the accompanying cost in forward progress.

The command path carries the same delay d in every run, so the delay itself is
never the variable. What varies is whether the filter is told about it:

  * ``none``        -- no filter, the reference outcome.
  * ``dob_blind``   -- DOB-CBF with ``teleop_delay = 0``, screening the stale
                       command as though it were current.
  * ``dob_aware``   -- DOB-CBF with ``teleop_delay = d``, inflating each obstacle
                       radius by the distance covered during the round trip and
                       predicting forward over the queue of past commands before
                       it acts.

Comparing the blind and aware arms is what isolates delay compensation, since
both arms have a filter and differ in one input to it. Every cell replays the
same intent on the same (scenario, delay) under a deterministic simulation, so
the difference within a cell is attributable to that input. The design fixes in
advance which scenarios are primary: awareness inflates the obstacle radius in
proportion to speed, so it can only act where the ego is moving toward the
hazard, and the stationary and rear-approach scenarios are therefore negative
controls that should show no effect.

Reproduce:
  python benchmarking/latency_awareness_ablation.py
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style  # noqa: E402
paper_style.apply()
from common import (run_process, timestamped_result_dir, save_summary_markdown,  # noqa: E402
                    sim_node_flags_from_hil_extra, write_manifest)
# Reuse the deterministic-replay machinery from the counterfactual eval.
from convoy_counterfactual_eval import (  # noqa: E402
    generate_reckless_trace, _metrics_from_run, SIM_NODE,
)

VARIANTS = ("none", "dob_blind", "dob_aware")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--convoy", nargs="+",
                   default=["lead_brake", "convoy", "platoon", "rear_approach", "stalled"])
    p.add_argument("--delays", nargs="+", type=float, default=[0.15, 0.30],
                   help="Command-path delays at which delay-awareness can matter "
                        "(at 0 s blind==aware, so it is omitted by default).")
    p.add_argument("--reckless-throttle", nargs="+", type=float, default=[0.4, 0.6, 0.8])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS),
                   help="Arms to run. Restricting to one arm calibrates the "
                        "difficulty of a scenario set without exposing the "
                        "blind-against-aware contrast: run '--variants "
                        "dob_blind' alone, choose the scenarios from where its "
                        "margins fall, then run all three arms once.")
    p.add_argument("--terrain", default="clay")
    p.add_argument("--time", type=float, default=20.0)
    p.add_argument("--mesh-resolution", type=float, default=0.08,
                   help="SCM grid spacing. The published value, matching the "
                        "teleoperation battery so the two are comparable.")
    p.add_argument("--safety-buffer", type=float, default=0.25)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=float, default=400.0)
    p.add_argument("--base-port", type=int, default=11800)
    return p.parse_args()


@dataclass(frozen=True)
class Task:
    idx: int
    variant: str
    convoy: str
    delay: float
    throttle: float
    sim_port: int
    run_dir: str
    trace: str
    terrain: str
    time_s: float
    mesh: float
    buffer: float
    timeout: float
    cell: str


def _build_cmd(t: Task) -> list[str]:
    cmd = [
        sys.executable, "-u", str(SIM_NODE),
        "--transport", "ros",
        "--terrain", t.terrain, "--time", str(t.time_s), "--vis-mode", "none",
        "--sim-port", str(t.sim_port), "--mesh-resolution", str(t.mesh), "--no-noise",
        "--rocks", "0", "--convoy", t.convoy,
        "--replay-cmds", t.trace,
        "--sim-diag-csv", str(Path(t.run_dir) / "sim_diag.csv"),
        # The latency is ALWAYS present on the command path.
        "--manual-input-delay", str(t.delay),
    ]
    if t.variant != "none":
        cmd += ["--safety-filter", "--safety-flavor", "dob_cbf",
                "--safety-buffer", str(t.buffer)]
        # The single difference between the blind and aware arms: the delay the
        # filter is told about. The command path is delayed identically in both.
        cmd += ["--teleop-delay", str(t.delay) if t.variant == "dob_aware" else "0"]
    # Plant-configuration contract: run.py exports HIL_SIM_EXTRA (the deployed
    # plant flags) for this study, and launch_and_collect appends it for the
    # studies that go through common.py. This driver spawns the simulator
    # itself, so it must append the simulator's share of those flags or its
    # runs execute a different plant (full nonlinear powertrain) than the
    # safety matrix its rows are read against; the controller-side flags are
    # filtered out because a replay run starts no controller process and the
    # simulator's argparser exits on them.
    cmd += sim_node_flags_from_hil_extra()
    return cmd


def _run_one(task: Task) -> dict:
    run_dir = Path(task.run_dir)
    rc, wall, _ = run_process(_build_cmd(task), run_dir, task.timeout)
    row = {"idx": task.idx, "cell": task.cell, "variant": task.variant,
           "convoy": task.convoy, "delay_s": task.delay, "throttle": task.throttle,
           "rc": rc, "wall_s": round(wall, 1)}
    row.update(_metrics_from_run(run_dir))
    if rc != 0 and row["status"] == "ok":
        row["status"] = f"exit_{rc}"
    return row


def plot_figures(per_delay: pd.DataFrame, out_dir: Path) -> None:
    if per_delay.empty:
        return
    delays = sorted(per_delay["delay_s"].unique())
    variants = [v for v in VARIANTS if v in set(per_delay["variant"])]
    # Muted grays for the reference arms and the navy accent for the delay-aware
    # arm under test. Distinct markers and dash patterns keep the muted lines
    # legible in print and readable in grayscale.
    colors = {"none": paper_style.GRAYS[2], "dob_blind": paper_style.GRAYS[0],
              "dob_aware": paper_style.ACCENT}
    lstyle = {"none": (0, (4, 2)), "dob_blind": (0, (1, 1.5)), "dob_aware": "-"}
    marker = {"none": "o", "dob_blind": "s", "dob_aware": "D"}
    labels = {"none": "no filter", "dob_blind": "DOB-CBF, delay-blind",
              "dob_aware": "DOB-CBF, delay-aware"}
    # Two-delay comparison: collision rate, signed geometric clearance, and
    # intrusiveness.  Negative clearance denotes overlap in the diagnostic
    # geometry; native body contact remains the collision truth.
    panels = [("collision_rate", "collision rate", "Collisions vs delay"),
              ("mean_clearance_m", "signed mean clearance (m)",
               "Clearance vs delay"),
              ("dsteer", "mean $|\\Delta\\mathrm{steer}|$", "Intrusiveness vs delay")]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.96))
    for ax, (col, ylab, ttl) in zip(axes, panels):
        if col not in per_delay.columns:
            continue
        for v in variants:
            sub = per_delay[per_delay["variant"] == v].set_index("delay_s").reindex(delays)
            ax.plot(delays, sub[col].values, marker=marker.get(v, "o"),
                    ls=lstyle.get(v, "-"), lw=1.6, ms=5,
                    color=colors.get(v, paper_style.INK), label=labels.get(v, v))
        ax.set_xlabel("command delay $\\Delta_\\mathrm{cmd}$ (s)"); ax.set_ylabel(ylab)
        ax.set_title(ttl)
    axes[0].legend()
    fig.suptitle(
        "Latency awareness at two delays: aware vs blind DOB-CBF "
        "(same replayed intent)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "latency_awareness_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    # One top-level ROS benchmark at a time: the launcher maps ports onto a
    # finite range of DDS domain identifiers, and a concurrent sweep can
    # collide on a domain and silently mix runs. Held for the whole matrix.
    from paper_provenance import acquire_paper_ros_lease
    lease = acquire_paper_ros_lease("latency_awareness_ablation")
    try:
        _main(args)
    finally:
        lease.close()


def _main(args) -> None:
    out_dir = timestamped_result_dir("latency_awareness_ablation")
    write_manifest(out_dir, args,
                   "Paired delay-awareness ablation over convoy scenarios.")
    print(f"Output: {out_dir}")

    traces = {}
    for thr in args.reckless_throttle:
        tp = str(out_dir / f"reckless_trace_t{thr:.2f}.csv")
        generate_reckless_trace(Path(tp), args.time, thr)
        traces[thr] = tp

    tasks: list[Task] = []
    idx = 0
    for thr in args.reckless_throttle:
        for preset in args.convoy:
            for delay in args.delays:
                for variant in args.variants:
                    run_dir = out_dir / "raw" / f"{idx:03d}_{preset}_{variant}_d{delay:.2f}_t{thr:.2f}"
                    tasks.append(Task(idx, variant, preset, delay, thr,
                                      args.base_port + 2 * idx, str(run_dir), traces[thr],
                                      args.terrain, args.time, args.mesh_resolution,
                                      args.safety_buffer, args.timeout,
                                      cell=f"{preset}@d{delay:.2f}@t{thr:.2f}"))
                    idx += 1

    rows = []
    print(f"[1/{len(tasks)}] prewarm ({tasks[0].variant})")
    rows.append(_run_one(tasks[0]))
    if len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(_run_one, t): t for t in tasks[1:]}
            for fut in as_completed(futs):
                rows.append(fut.result())
                pd.DataFrame(rows).sort_values("idx").to_csv(out_dir / "results.csv", index=False)
    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    df.to_csv(out_dir / "results.csv", index=False)

    ok = df[df["status"] == "ok"].copy()
    if "mean_abs_dsteer" not in ok.columns:
        ok["mean_abs_dsteer"] = float("nan")
    # Per (variant, delay): collision rate + clearance margin + intrusiveness
    # (mean |Delta steer| the filter applies; the per-step correction magnitude).
    per_delay = (ok.groupby(["variant", "delay_s"])
                 .agg(n=("collided", "size"), collisions=("collided", "sum"),
                      mean_clearance_m=("min_clearance_m", "mean"),
                      dsteer=("mean_abs_dsteer", "mean"))
                 .reset_index())
    per_delay["collision_rate"] = (per_delay["collisions"] / per_delay["n"]).round(3)
    per_delay["mean_clearance_m"] = per_delay["mean_clearance_m"].round(3)
    per_delay["dsteer"] = per_delay["dsteer"].round(3)
    per_delay.to_csv(out_dir / "summary_by_delay.csv", index=False)

    # Per variant overall, with blind->aware harm-prevented matched by cell.
    summary_rows = []
    blind = ok[ok["variant"] == "dob_blind"].drop_duplicates("cell").set_index("cell")
    for v in VARIANTS:
        sub = ok[ok["variant"] == v]
        if sub.empty:
            continue
        rec = {"variant": v, "n": len(sub), "collisions": int(sub["collided"].sum()),
               "collision_rate": round(sub["collided"].mean(), 3),
               "mean_clearance_m": round(sub["min_clearance_m"].mean(), 3)}
        if v == "dob_aware":
            extra = 0  # collisions the blind filter had that aware prevented
            for _, r in sub.iterrows():
                if r["cell"] in blind.index and int(blind.loc[r["cell"], "collided"]) == 1 \
                        and r["collided"] == 0:
                    extra += 1
            rec["blind_collisions_prevented_by_awareness"] = extra
        summary_rows.append(rec)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    plot_figures(per_delay, out_dir)
    save_summary_markdown(out_dir, "Latency-Awareness Ablation", summary, [
        f"Convoy {args.convoy} x delays {args.delays} x intents {args.reckless_throttle}.",
        "The command path carries delay d in every run; the filter's teleop_delay "
        "is the single toggled input (blind = 0, aware = d). The replay is "
        "deterministic, so the blind-against-aware difference is attributable to "
        "that input. 'blind_collisions_prevented_by_awareness' counts the cells "
        "in which the delay-blind filter recorded a body contact and the "
        "delay-aware filter did not.",
    ])
    print(f"\nDone: {out_dir}")
    print("Per (variant, delay):")
    print(per_delay.to_string(index=False))
    print("\nOverall:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
