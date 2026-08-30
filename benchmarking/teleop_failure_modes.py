#!/usr/bin/env python3
"""Literature-grounded teleoperation failure-mode benchmark for safety filters.

This study produces Table 4 of the manuscript, the failure-mode battery, and the
per-scenario trajectories behind Fig. 4: the minimum signed clearance of each
filter on each failure mode, which modes produce a body contact, and the
prevention tallies quoted in Sec. 4.3.

Ten seeded scenarios encode failure modes documented in the teleoperated-driving
literature -- pilot-induced oscillation, delayed hazard response, a missed
obstacle, a late-revealed peripheral hazard, connection-loss freeze, overspeed
for the conditions, and four modes involving other traffic. Each scenario's
``cite`` field records the source its operator-failure parameters are drawn
from. Parameterizing the operator's failure from the literature rather than
tuning it is what places the filter, not the human, under test, and hazard
positions are simulator truth so that the battery isolates command-channel
failures from perception.

Every filter receives the same true-soil belief and otherwise runs at its own
defaults, so an outcome difference is the filter's and not an artifact of
asymmetric configuration. Runs execute one at a time, recording each run's
``sim_diag.csv`` and ``obstacles.json`` and, under ``--render``, a top-down
animation and a closest-approach still for visual inspection. The result is a
failure-mode by filter matrix of contact, clearance, intrusion, oscillation, and
progress.

Env (ROS 2 / Chrono::ROS, scm-terrain):
    export PYTHONPATH=$HOME/Documents/sbel/chrono_fork/build/bin:$PYTHONPATH
    export ACADOS_SOURCE_DIR=$HOME/Documents/sbel/acados
    source /opt/ros/jazzy/setup.bash && source "$HOME/packages/chrono_ros_ws/install/setup.bash"
    conda run -n scm-terrain python benchmarking/teleop_failure_modes.py --render
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import write_manifest  # noqa: E402

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "simulation" / "runtime" / "chrono_sim_node.py"
# Measurement-grounded 5G environment. A neural hierarchical-interpolation
# (N-HiTS) generator trained on packet captures from a commercial 5G network
# produces cloud-gaming traffic, which a queue-load link model maps to one-way
# delay: the uplink series drives the operator command channel and the downlink
# series the camera channel. The generator is trained outside the control and
# estimation stack and touches no vehicle or terrain quantity.
LAT5G = "data/latency_profiles/5g_nhits_geforce.json"
FILTERS = ("none", "dob_cbf", "mpsf")


def rocks(n, x0, x1, y0, y1, size0, size1, spacing, clear, seed):
    return ["--rocks", str(n), "--rock-zone-x", str(x0), str(x1),
            "--rock-zone-y", str(y0), str(y1), "--rock-size", str(size0), str(size1),
            "--rock-min-spacing", str(spacing), "--rock-centerline-clear", str(clear),
            "--rock-spawn-clear", "12", "--rock-seed", str(seed)]


def gen_held_command(path: Path, duration, throttle):
    """Constant forward-command CSV: models a connection-loss / frozen state
    where the operator's last command keeps being applied and can no longer be
    corrected (no steering feedback to the path)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["time", "steering_op", "throttle_op", "braking_op"])
        t = 0.0
        while t <= duration + 2.0:
            w.writerow([f"{t:.3f}", "0.0", f"{throttle:.3f}", "0.0"])
            t += 0.1


# ---- scenario registry ------------------------------------------------------
# Each entry encodes one documented failure mode. ``cite`` names the source its
# operator-failure parameters are taken from, ``latency`` selects whether the
# mode is driven through the 5G environment, and ``obstacles`` supplies either a
# static rock field or a traffic preset.
def scenarios(scratch: Path):
    freeze = scratch / "frozen_command.csv"; gen_held_command(freeze, 20, 0.60)
    return [
        dict(key="1_pio_oscillation",
             title="Latency-induced steering oscillation (PIO)",
             cite="VSD 2025; Orosz IV 2024",
             terrain="dirt", time=18, latency=True,
             driver=["--path", "straight", "--synthetic-operator",
                     "--synthetic-operator-speed", "6",
                     "--synthetic-operator-klat", "1.15",
                     "--synthetic-operator-kpsi", "2.7"],
             # flanking-rock corridor (rocks kept >=2.5 m off centre): oscillation
             # drifts toward the walls.
             obstacles=rocks(6, 16, 50, -4.5, 4.5, 0.7, 1.0, 6, 2.5, 5)),
        dict(key="2_delayed_hazard",
             title="Delayed braking / late hazard response",
             cite="VSD 2025; 5G eval 2025",
             terrain="dirt", time=14, latency=True,
             driver=["--path", "straight", "--synthetic-operator",
                     "--synthetic-operator-speed", "6"],
             obstacles=rocks(1, 22, 22, 0, 0, 1.2, 1.2, 6, 0.0, 7)),
        dict(key="3_missed_obstacle",
             title="Missed obstacle (narrow FOV / low awareness)",
             cite="Bogdoll 2022; Orosz 2024",
             # path-tracking operator that never perceives/avoids the obstacle
             # (drives its intended straight line into it); no latency -- pure
             # unawareness. Path-tracking so it returns to course after the dodge.
             terrain="dirt", time=14, latency=False,
             driver=["--path", "straight", "--synthetic-operator",
                     "--synthetic-operator-speed", "6"],
             obstacles=rocks(1, 20, 20, 0, 0, 1.3, 1.3, 6, 0.0, 8)),
        dict(key="4_peripheral_hazard",
             title="Late-revealed peripheral hazard",
             cite="Bogdoll 2022",
             terrain="dirt", time=14, latency=True,
             driver=["--path", "straight", "--synthetic-operator",
                     "--synthetic-operator-speed", "6"],
             # The rock lies in the path but offset to one side, so avoidance
             # requires a definite directional dodge. The 1.0 m offset keeps the
             # hazard clear of the corridor edge, where a marginal conflict would
             # make the filter engage and disengage repeatedly and the recorded
             # clearance would reflect that chatter rather than the avoidance.
             obstacles=rocks(1, 21, 21, 1.0, 1.0, 1.1, 1.1, 6, 0.0, 9)),
        dict(key="5_connection_freeze",
             title="Connection loss / frozen command",
             cite="Brettin 2025",
             terrain="dirt", time=14, latency=True,
             driver=["--path", "straight", "--replay-cmds", str(freeze)],
             obstacles=rocks(1, 22, 22, 0, 0, 1.2, 1.2, 6, 0.0, 7)),
        dict(key="6_overspeed",
             title="Overspeed for the latency/conditions (low grip)",
             cite="5G eval 2025; Orosz 2024",
             # path-tracking operator commanding a speed too high for the low-grip
             # clay under 5G delay -> short reaction window into the hazard.
             terrain="clay", time=14, latency=True,
             driver=["--path", "straight", "--synthetic-operator",
                     "--synthetic-operator-speed", "9"],
             obstacles=rocks(1, 26, 26, 0, 0, 1.3, 1.3, 6, 0.0, 7)),
        # --- modes involving other traffic -------------------------------
        # The first six modes place a static hazard. A hazard that is itself
        # closing on the ego, as every documented teleoperation risk involving
        # other road users does, leaves the filter less time than its own
        # reaction budget would suggest, and a rock cannot reproduce that.
        # Traffic is driven by the same manager the convoy counterfactual uses,
        # so the two studies share one traffic model.
        dict(key="7_lead_brake_delay",
             title="Delayed reaction to lead-vehicle braking",
             cite="VSD 2025; Orosz IV 2024",
             terrain="dirt", time=16, latency=True,
             driver=["--path", "straight", "--synthetic-operator",
                     "--synthetic-operator-speed", "6"],
             obstacles=["--convoy", "lead_brake", "--traffic-detail", "primitives"]),
        dict(key="8_late_cut_in",
             title="Late cut-in by an adjacent vehicle",
             cite="Bogdoll 2022; VSD 2025",
             terrain="dirt", time=16, latency=True,
             driver=["--path", "straight", "--synthetic-operator",
                     "--synthetic-operator-speed", "6"],
             obstacles=["--convoy", "cut_in", "--traffic-detail", "primitives"]),
        dict(key="9_freeze_into_stalled",
             title="Connection loss approaching a stalled vehicle",
             cite="Brettin 2025",
             terrain="dirt", time=16, latency=True,
             driver=["--path", "straight", "--replay-cmds", str(freeze)],
             obstacles=["--convoy", "stalled", "--traffic-detail", "primitives"]),
        dict(key="10_oncoming_pio",
             title="Oncoming vehicle during latency-induced oscillation",
             cite="VSD 2025; Orosz IV 2024",
             terrain="dirt", time=16, latency=True,
             driver=["--path", "straight", "--synthetic-operator",
                     "--synthetic-operator-speed", "6",
                     "--synthetic-operator-klat", "1.15",
                     "--synthetic-operator-kpsi", "2.7"],
             obstacles=["--convoy", "oncoming", "--traffic-detail", "primitives"]),
    ]


def build_cmd(sc, flavor, port, sim_diag: Path):
    cmd = [sys.executable, "-u", str(SIM), "--transport", "ros",
           "--terrain", sc["terrain"], "--time", str(sc["time"]),
           "--vis-mode", "none", "--sim-port", str(port),
           "--mesh-resolution", "0.08", "--no-noise",
           "--speed", "6", *sc["driver"], *sc["obstacles"],
           "--sim-diag-csv", str(sim_diag)]
    if sc["latency"]:
        # Staggered replay phases: scenario i enters the trace at 30*i s modulo
        # its 300 s period. Each run is short, so a battery that always started
        # at the head of the trace would sample only its nominal opening stretch
        # and never encounter a command-channel spike or an outage. Staggering
        # makes the battery collectively span the profile's regimes, while the
        # arms of one scenario share a phase and therefore stay paired.
        cmd += ["--latency-profile-json", LAT5G,
                "--latency-phase-s", str((30 * sc["index"]) % 300)]
    if flavor != "none":
        # Every filter receives the same true-soil terrain belief and otherwise
        # runs at its own defaults, with no per-filter tuning, so a difference in
        # failure-mode outcome is attributable to the filter rather than to an
        # asymmetric configuration.
        cmd += ["--safety-filter", "--safety-flavor", flavor,
                "--shield-init-terrain-belief", sc["terrain"]]
    return cmd


def metrics(sim_diag: Path):
    out = dict(status="ok", collided=0, min_clear=np.nan, mean_dsteer=np.nan,
               mean_dthrottle=np.nan, max_abs_y=np.nan, steer_reversals=0,
               progress_x=np.nan, final_v=np.nan)
    if not sim_diag.exists():
        out["status"] = "no_trace"; return out
    try:
        d = pd.read_csv(sim_diag)
    except Exception as e:  # noqa: BLE001
        out["status"] = f"err:{e}"; return out
    if len(d) == 0:
        out["status"] = "empty"; return out
    num = lambda c, dv=0.0: pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series([dv] * len(d))
    out["collided"] = int((num("collisions").fillna(0) > 0).any())
    clr = num("nearest_clearance_m")
    out["min_clear"] = float(clr.min()) if clr.notna().any() else np.nan
    out["mean_dsteer"] = float((num("steering") - num("steering_op")).abs().mean())
    out["mean_dthrottle"] = float((num("throttle") - num("throttle_op")).abs().mean())
    y = num("y"); out["max_abs_y"] = float(y.abs().max())
    st = num("steering").to_numpy()
    out["steer_reversals"] = int((np.diff(np.sign(st)) != 0).sum())  # oscillation proxy
    out["progress_x"] = float(num("x").iloc[-1])
    out["final_v"] = float(num("speed").iloc[-1])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filters", nargs="+", default=list(FILTERS), choices=list(FILTERS))
    ap.add_argument("--only", nargs="+", default=None, help="run only these scenario keys")
    ap.add_argument("--render", action="store_true", help="render top-down GIFs")
    ap.add_argument("--base-port", type=int, default=15900)
    ap.add_argument("--timeout", type=float, default=320)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # One top-level ROS benchmark at a time: concurrent sweeps can collide on
    # a DDS domain and silently mix runs. The handle is kept on ``args`` so
    # the exclusive flock is held until this process exits.
    from paper_provenance import acquire_paper_ros_lease
    args._ros_lease = acquire_paper_ros_lease("teleop_failure_modes")

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "benchmarking" / "results" / f"teleop_failure_modes_{ts}"
    raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    write_manifest(out, args,
                   "Teleoperation failure-mode battery under the 5G profile.")
    scs = scenarios(out)
    for _i, _sc in enumerate(scs):
        _sc["index"] = _i
    if args.only:
        scs = [s for s in scs if s["key"] in args.only]

    rows = []
    port = args.base_port
    metrics_csv = out / "results.csv"
    for sc in scs:
        for flavor in args.filters:
            run_dir = raw / f"{sc['key']}__{flavor}"; run_dir.mkdir(parents=True, exist_ok=True)
            sim_diag = run_dir / "sim_diag.csv"
            cmd = build_cmd(sc, flavor, port, sim_diag); port += 2
            print(f"\n=== {sc['key']} | filter={flavor} ===")
            # The exit status must reach the results row: a simulator that
            # crashed or timed out can still leave a partial sim_diag.csv
            # whose metrics parse cleanly, and a row silently labelled "ok"
            # would report a truncated run as a completed scenario.
            rc: object = None
            try:
                rc = subprocess.run(cmd, cwd=str(ROOT), timeout=args.timeout,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
                print("  timeout")
            m = metrics(sim_diag)
            if rc == "timeout":
                m["status"] = "timeout"
            elif isinstance(rc, int) and rc != 0 and m["status"] == "ok":
                m["status"] = f"exit_{rc}"
            if m["status"] != "ok":
                print(f"  WARNING: run status = {m['status']}")
            m.update(scenario=sc["key"], title=sc["title"], cite=sc["cite"], filter=flavor)
            rows.append(m)
            pd.DataFrame(rows).to_csv(metrics_csv, index=False)
            print(f"  collided={m['collided']} min_clear={m['min_clear']:.2f} "
                  f"|dsteer|={m['mean_dsteer']:.3f} maxY={m['max_abs_y']:.2f} "
                  f"reversals={m['steer_reversals']} progress={m['progress_x']:.1f}")
            if args.render and flavor in ("none", "mpsf"):
                try:
                    sys.path.insert(0, str(Path(__file__).resolve().parent))
                    from render_topdown import render
                    render(run_dir, out=run_dir / "topdown.gif",
                           title=f"{sc['title']} [{flavor}]")
                except Exception as e:  # noqa: BLE001
                    print(f"  render failed: {e}")

    _matrix(rows, out)
    print(f"\nDone: {out}")
    return 0


def _matrix(rows, out: Path):
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"]
    if ok.empty:
        print("no ok runs"); return
    print("\n" + "=" * 92)
    print(" TELEOP FAILURE-MODE x FILTER  (collided / min_clear m / |dsteer|)")
    print("=" * 92)
    scen = ok["scenario"].unique()
    hdr = f"{'scenario':22s}"
    for fl in FILTERS:
        hdr += f" | {fl:^16s}"
    print(hdr); print("-" * 92)
    for s in scen:
        line = f"{s:22s}"
        for fl in FILTERS:
            r = ok[(ok["scenario"] == s) & (ok["filter"] == fl)]
            if len(r):
                r = r.iloc[0]
                line += f" | {('HIT' if r['collided'] else 'ok '):3s} {r['min_clear']:5.2f} {r['mean_dsteer']:4.2f}"
            else:
                line += f" | {'--':^16s}"
        print(line)
    print("=" * 92)
    # prevented tally: baseline none collided, filter did not
    prev = {}
    for fl in FILTERS:
        if fl == "none":
            continue
        cnt = 0; base = 0
        for s in scen:
            n = ok[(ok["scenario"] == s) & (ok["filter"] == "none")]
            f = ok[(ok["scenario"] == s) & (ok["filter"] == fl)]
            if len(n) and len(f):
                base += int(n.iloc[0]["collided"])
                if n.iloc[0]["collided"] and not f.iloc[0]["collided"]:
                    cnt += 1
        prev[fl] = (cnt, base)
    print("collisions prevented vs unfiltered baseline:")
    for fl, (c, b) in prev.items():
        print(f"  {fl:10s}: {c}/{b}")
    df.to_csv(out / "results.csv", index=False)


if __name__ == "__main__":
    raise SystemExit(main())
