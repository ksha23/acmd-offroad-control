#!/usr/bin/env python3
"""Open-loop numerical convergence of the multibody/SCM plant under refinement.

This study produces the two monotone refinement sequences the manuscript cites
in Sec. 2 as evidence that the deployed discretization is converged: RMS and
maximum planar deviation against the finest member of the physics-step family
and of the SCM-grid family.

A closed-loop tracking metric cannot isolate discretization error: the controller
runs asynchronously over ROS 2, so its effective bandwidth and the achieved speed
both shift with wall-clock cost, mixing transport timing into the comparison.
This study therefore removes the controller entirely and replays one fixed,
pre-recorded command trace (steering sweep plus constant drive torque) through
the plant while varying one discretization knob at a time about the deployed
operating point (physics step 3 ms, SCM grid 0.08 m):

  * step size {6.0, 3.0, 1.5, 0.75} ms at the deployed 0.08 m grid
  * SCM grid  {0.16, 0.12, 0.08, 0.04} m at the deployed 3 ms step

With no controller, no sensor noise, and an identical command sequence, the only
difference between runs is the discretization, so the trajectory spread is
numerical error. Each family is compared against its own finest member as the
reference solution, reporting the RMS and maximum position deviation over the
common time grid; convergence means those deviations shrink under refinement and
that the deployed setting already sits near the reference.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "simulation" / "runtime" / "chrono_sim_node.py"
sys.path.insert(0, str(ROOT / "benchmarking"))
sys.path.insert(0, str(ROOT / "simulation" / "shared"))
try:
    from common import timestamped_result_dir
    from paper_provenance import downstream_repository_provenance
except ModuleNotFoundError:
    from benchmarking.common import timestamped_result_dir
    from benchmarking.paper_provenance import downstream_repository_provenance

BASE_STEP, BASE_MESH = 3.0e-3, 0.08
STEP_FAMILY = [6.0e-3, 3.0e-3, 1.5e-3, 0.75e-3]
MESH_FAMILY = [0.16, 0.12, 0.08, 0.04]


def write_command_trace(path: Path, duration: float, throttle: float,
                        amplitude: float, period: float) -> None:
    """Fixed open-loop command trace: steering sweep at constant drive torque."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "steering_op", "throttle_op", "braking_op"])
        t = 0.0
        while t <= duration + 2.0:
            # hold straight briefly so the vehicle is rolling before steering
            steer = 0.0 if t < 3.0 else amplitude * math.sin(2 * math.pi * (t - 3.0) / period)
            w.writerow([f"{t:.3f}", f"{steer:.6f}", f"{throttle:.3f}", "0.000"])
            t += 0.05


def run_case(label: str, step: float, mesh: float, out: Path, cmds: Path,
             terrain: str, duration: float, port: int, timeout: float) -> dict:
    rd = out / label
    rd.mkdir(parents=True, exist_ok=True)
    diag = rd / "sim_diag.csv"
    cmd = [sys.executable, "-u", str(SIM), "--transport", "ros",
           "--terrain", terrain, "--time", str(duration), "--vis-mode", "none",
           "--sim-port", str(port), "--path", "straight",
           "--no-noise", "--no-tire-forces", "--simple-powertrain",
           "--replay-cmds", str(cmds),
           "--step-size", str(step), "--mesh-resolution", str(mesh),
           "--sim-diag-csv", str(diag)]
    env = dict(os.environ)
    env.setdefault("ACADOS_SOURCE_DIR", str(Path.home() / "Documents/sbel/acados"))
    env["HIL_RUN_LOG_DIR"] = str(rd)
    with (rd / "run.log").open("w") as log:
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                            env=env, timeout=timeout).returncode
    row = dict(config=label, step_s=step, mesh_m=mesh, returncode=rc,
               diag=str(diag), rows=0, final_x=float("nan"), final_y=float("nan"))
    if diag.is_file():
        d = pd.read_csv(diag)
        row.update(rows=len(d), final_x=float(d.x.iloc[-1]), final_y=float(d.y.iloc[-1]))
    print(f"  {label:12s} step={step*1e3:5.2f}ms mesh={mesh:.2f}m rc={rc} "
          f"rows={row['rows']} final=({row['final_x']:.2f},{row['final_y']:.2f})",
          flush=True)
    return row


def deviation(ref_csv: str, test_csv: str) -> tuple[float, float, int]:
    """RMS and max planar position deviation on the common time grid."""
    a, b = pd.read_csv(ref_csv), pd.read_csv(test_csv)
    t0, t1 = max(a.time.min(), b.time.min()), min(a.time.max(), b.time.max())
    grid = np.arange(t0, t1, 0.05)
    if len(grid) < 10:
        return (float("nan"), float("nan"), 0)
    dx = np.interp(grid, a.time, a.x) - np.interp(grid, b.time, b.x)
    dy = np.interp(grid, a.time, a.y) - np.interp(grid, b.time, b.y)
    r = np.hypot(dx, dy)
    return (float(np.sqrt(np.mean(r**2))), float(np.max(r)), len(grid))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--terrain", default="dirt")
    ap.add_argument("--time", type=float, default=20.0)
    ap.add_argument("--throttle", type=float, default=0.55)
    ap.add_argument("--amplitude", type=float, default=0.25)
    ap.add_argument("--period", type=float, default=8.0)
    ap.add_argument("--base-port", type=int, default=41000)
    ap.add_argument("--timeout", type=float, default=2400.0)
    args = ap.parse_args()

    provenance = downstream_repository_provenance()
    out = timestamped_result_dir("numerical_convergence")
    cmds = out / "command_trace.csv"
    write_command_trace(cmds, args.time, args.throttle, args.amplitude, args.period)

    # The deployed operating point belongs to both families, so it is run once
    # and reused, under the label step_3ms, as the grid family's 0.08 m member.
    cases = ([(f"step_{s*1e3:g}ms", s, BASE_MESH) for s in STEP_FAMILY]
             + [(f"mesh_{m:g}m", BASE_STEP, m)
                for m in MESH_FAMILY if m != BASE_MESH])
    print(f"Open-loop plant convergence: {len(cases)} cases, "
          f"{args.time}s replay on {args.terrain}")
    rows = [run_case(lbl, st, me, out, cmds, args.terrain, args.time,
                     args.base_port + 4 * i, args.timeout)
            for i, (lbl, st, me) in enumerate(cases)]      # serial: identical machine load

    results = pd.DataFrame(rows)
    results.to_csv(out / "results.csv", index=False)
    by = {r["config"]: r for r in rows}

    def family(labels: list[str], ref_label: str, knob: str) -> list[dict]:
        ref = by.get(ref_label, {})
        recs = []
        for lbl in labels:
            r = by.get(lbl)
            if not r or not Path(r["diag"]).is_file() or not ref.get("diag"):
                continue
            rms, mx, n = deviation(ref["diag"], r["diag"])
            recs.append(dict(family=knob, config=lbl, step_s=r["step_s"],
                             mesh_m=r["mesh_m"], rms_dev_m=rms, max_dev_m=mx,
                             samples=n, reference=ref_label))
        return recs

    step_labels = [f"step_{s*1e3:g}ms" for s in STEP_FAMILY]
    mesh_labels = [f"mesh_{m:g}m" if m != BASE_MESH else "step_3ms" for m in MESH_FAMILY]
    conv = pd.DataFrame(family(step_labels, step_labels[-1], "step")
                        + family(mesh_labels, mesh_labels[-1], "mesh"))
    conv.to_csv(out / "convergence.csv", index=False)

    (out / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "study": "numerical_convergence_open_loop",
        "design": "controller removed; one fixed replayed command trace per case",
        "terrain": args.terrain, "duration_s": args.time,
        "command_trace": {"throttle": args.throttle, "steer_amplitude_rad": args.amplitude,
                          "steer_period_s": args.period},
        "deployed_point": {"step_s": BASE_STEP, "mesh_m": BASE_MESH},
        "step_family_s": STEP_FAMILY, "mesh_family_m": MESH_FAMILY,
        "reference": {"step": step_labels[-1], "mesh": mesh_labels[-1]},
        "metric": "planar position deviation vs finest member on common time grid",
        "n_cases": len(cases), "code_provenance": provenance,
    }, indent=2) + "\n")

    print("\n" + conv.to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
