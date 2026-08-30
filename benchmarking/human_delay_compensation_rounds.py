#!/usr/bin/env python3
"""Human-in-the-loop rounds at a physical wheel station under transport delay.

This script runs the single-operator demonstration reported at the end of
Sec. 4.3 of the manuscript, in which an operator completes a preregistered set
of trials at the wheel station with arms sharing seeds and trace phases, and
each filter's contact and goal-completion outcomes are recorded under live human
intent. It also records the operator command traces that
``convoy_counterfactual_eval.py`` replays deterministically, which is where the
causal filter attribution is made: live driving cannot reproduce identical
intent, so the live rounds establish that the composition works under a real
operator while the replay establishes what the filter caused.

An operator drives the HMMWV through a hazard field while the simulator-side
safety filter screens the delayed commands. Each round delays both the operator
command path and the driver's point-of-view camera feed, modelling the uplink
and the downlink of a teleoperation link respectively; the camera delay is
``--camera-delay-scale`` times the command delay, symmetric by default. Delaying
the camera as well as the command matters because the operator closes the loop
on what the video shows, so a command-only delay would understate the difficulty
of the task.

With ``--latency-profile-json``, every round instead runs under a seeded,
time-varying profile in which the ``control`` and ``manual`` channels carry the
command uplink and the ``camera`` channel the asymmetric video downlink; this
replaces the fixed-delay sweep with one profile condition per cell.
``--live-hud`` places an instrument overlay (``simulation/teleop/hil_hud.py``) on
each round's ROS 2 telemetry stream, showing the commanded and applied steering
and throttle so that a filter takeover is visible to the operator as it happens.

Rounds run one at a time. The script writes raw simulator diagnostics per round
and summarizes tracking, speed, contact, clearance, and intervention metrics for
every (filter, delay) cell, together with the per-pair table written to
``paired_live_results.csv``.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_NN_MODEL,
    LAUNCHER,
    LOGS_DIR,
    PATH_ROCK_ZONES,
    PROJECT_ROOT,
    TERRAINS,
    ensure_runtime_env,
    parse_collision_csv,
    parse_log_summary,
    parse_shield_csv,
    save_summary_markdown,
    timestamped_result_dir,
    write_manifest,
)

SIM_DIR = PROJECT_ROOT / "simulation"
sys.path.insert(0, str(SIM_DIR))
import flatpath  # noqa: E402,F401
from reference_path import ReferencePath, generate_path_waypoints  # noqa: E402
from traffic import CONVOY_DESCRIPTIONS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-id", default="hil_single_operator_pilot",
                   help="Anonymized study/protocol identifier written to every "
                        "plan and result row.")
    p.add_argument("--operator-id", default="op01",
                   help="Anonymized operator identifier (letters, numbers, '.', "
                        "'_' or '-' only; do not use a participant name).")
    p.add_argument("--input-device-model", default="unspecified",
                   help="Detected/manual controller model recorded as hardware "
                        "provenance (for example Logitech_G923_Racing_Wheel).")
    p.add_argument("--filters", nargs="+", default=["none", "dob_cbf"],
                   choices=["none", "dob_cbf", "mpsf"])
    p.add_argument("--delays", nargs="+", type=float, default=[0.0, 0.15, 0.30],
                   help="Operator command-path (uplink) delays in seconds.")
    p.add_argument("--camera-delay-scale", type=float, default=1.0,
                   help="Camera (downlink) delay as a multiple of the command "
                        "delay. A value of 1.0 is a symmetric link; values above "
                        "1.0 model the heavier video downlink that a real "
                        "cellular link exhibits. Applies to the fixed --delays "
                        "sweep; under a latency profile the asymmetry comes from "
                        "the profile's own camera channel.")
    p.add_argument("--latency-profile-json",
                   default="data/latency_profiles/5g_nhits_geforce.json",
                   help="Run every round under a seeded, time-varying latency "
                        "profile in place of the fixed --delays sweep, with the "
                        "control and manual channels carrying the command uplink "
                        "and the camera channel the asymmetric video downlink. "
                        "It replaces the constant uplink, downlink, and teleop "
                        "delays and leaves one delay condition per cell. The "
                        "default is the measurement-grounded 5G profile the "
                        "manuscript describes; pass an empty value to use the "
                        "fixed --delays sweep instead.")
    p.add_argument("--live-hud", action="store_true",
                   help="Place the instrument overlay "
                        "(simulation/teleop/hil_hud.py) on each round's ROS 2 "
                        "telemetry stream. It draws the commanded steering angle "
                        "against the applied one, and a throttle bar, so that a "
                        "filter takeover is visible to the operator as it occurs.")
    p.add_argument("--hud-corner", choices=["br", "bl", "tr", "tl"], default="br",
                   help="Screen corner to dock the live HUD overlay into.")
    p.add_argument("--hud-wheel-lock", type=float, default=450.0,
                   help="HUD wheel rotation at full steer; set to half the G29's "
                        "lock-to-lock range to match the physical wheel (default 450).")
    # Render settings for interactive driving. The defaults trade the autonomous
    # studies' fidelity for real-time execution, because an operator cannot
    # drive a session that runs slower than real time.
    p.add_argument("--cam-width", type=int, default=1280,
                   help="Driver POV render width (px), GPU-upscaled to the display. "
                        "The delayed POV dedupes frames by render TimeStamp so the "
                        "readback runs ~30/s (not per physics step), which holds RT "
                        "~1.0x up to ~1600x1000 on single-vehicle scenes. Default 1280 "
                        "(16:10). Multi-vehicle scenes (convoy/platoon) run ~0.7x from "
                        "the 5-vehicle soil physics, not render.")
    p.add_argument("--cam-height", type=int, default=800,
                   help="Driver POV render height (px). 800 = 16:10 with width 1280.")
    p.add_argument("--cam-fov", type=float, default=1.05,
                   help="Driver POV camera horizontal FOV (rad, ~1.05=60deg).")
    p.add_argument("--cam-rate", type=float, default=30.0,
                   help="Driver POV camera render rate (Hz).")
    p.add_argument("--cam-fullscreen", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Display the driver POV fullscreen (renders at cam W x H, "
                        "scaled to the screen). Use --no-cam-fullscreen for a window.")
    p.add_argument("--delayed-pov", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Show the driver POV through a software frame-delay buffer so "
                        "the operator SEES the camera-channel latency (Chrono's SetLag "
                        "does not delay the display). On by default for live rounds; "
                        "--no-delayed-pov reverts to the real-time view.")
    p.add_argument("--pov-no-flip", action="store_true",
                   help="Disable the delayed POV's default vertical flip (it is flipped "
                        "upright by default; use this if your POV shows upside down).")
    p.add_argument("--convoy", nargs="+", default=[""],
                   help="Convoy scenario(s) the operator must avoid, swept as "
                        "separate rounds (lead_brake/cut_in/stalled/swerver/convoy/"
                        "platoon/oncoming/double_cut/stop_and_go/jam/overtake/"
                        "gauntlet/rear_approach). Empty = open course (rocks only).")
    p.add_argument("--traffic-detail", choices=["auto", "mesh", "primitives"],
                   default="mesh", help="Traffic render detail (mesh|auto|primitives).")
    p.add_argument("--mesh-resolution", type=float, default=0.12,
                   help="SCM grid spacing (m). The default of 0.12 sustains real "
                        "time for a driven round: the terrain triangle count "
                        "dominates the camera render cost, and 1080p at 30 Hz "
                        "runs in real time at 0.12 but at roughly 0.55x at 0.08. "
                        "An operator cannot drive below real time, so this "
                        "setting is what makes live rounds possible; the "
                        "autonomous studies use 0.08 for force fidelity.")
    p.add_argument("--terrains", nargs="+", default=["clay", "sand"], choices=list(TERRAINS))
    p.add_argument("--paths", nargs="+", default=["straight"],
                   help="Course shape. 'straight' (default) is the forward "
                        "corridor for the human drive-and-avoid task; the weaving "
                        "paths (sinusoidal/lane_change) are for autonomous tracking.")
    p.add_argument("--speeds", nargs="+", type=float, default=[4.0])
    p.add_argument("--bumpiness", nargs="+", type=int, default=[0, 4])
    p.add_argument("--rounds", type=int, default=1,
                   help="Repeated human rounds per condition.")
    p.add_argument("--base-seed", type=int, default=910)
    p.add_argument("--order", choices=["fixed", "randomized-blocks"], default="fixed",
                   help="Trial order. 'randomized-blocks' puts every experimental "
                        "condition once in each repetition block, then applies a "
                        "reproducible shuffle; use this for live study collection.")
    p.add_argument("--order-seed", type=int, default=20260719,
                   help="Seed for the reproducible randomized-block trial order.")
    p.add_argument("--practice-rounds", type=int, default=0,
                   help="Familiarization rounds run before the study and stored "
                        "under practice/ (never included in results.csv). They use "
                        "no filter and no latency profile.")
    p.add_argument("--time", type=float, default=25.0)
    p.add_argument("--lead-in", type=float, default=5.0)
    p.add_argument("--goal-distance", type=float, default=50.0,
                   help="Forward distance (m) the ego must cover for the round to "
                        "count as reaching the goal. Guards against the trivial "
                        "'sit still -> 0 collisions' result: a round is a clean "
                        "success only if it is collision-free AND reaches the goal.")
    p.add_argument("--rocks", type=int, default=5)
    p.add_argument("--rock-min-spacing", type=float, default=6.0,
                   help="Min spacing (m) between rocks -> steerable boulder field "
                        "(wide gaps; worst-case clear gap well over the HMMWV width).")
    p.add_argument("--rock-centerline-clear", type=float, default=3.0,
                   help="Half-width (m) where rock density is thinned for the lead's line.")
    p.add_argument("--rock-spawn-clear", type=float, default=8.0,
                   help="Rock-free radius (m) around the spawn.")
    p.add_argument("--rock-size", type=float, nargs=2, default=[0.5, 1.4],
                   help="Rock diameter range (m). Field rocks are smaller/denser.")
    p.add_argument("--manual-mode", choices=["g29", "wasd"], default="g29")
    p.add_argument("--vis-mode", choices=["irrlicht", "sensor", "both", "none"], default="irrlicht",
                   help="Driver view. 'irrlicht' (default) rasterizes the fixed "
                        "driver POV -- faster than the ray-traced 'sensor' camera on "
                        "a large deformable terrain, so it allows more traffic at "
                        "real-time. 'sensor' adds modelled camera (downlink) latency.")
    p.add_argument("--shield-horizon", type=int, default=12)
    p.add_argument("--safety-buffer", type=float, default=0.25)
    p.add_argument("--cbf-profile",
                   choices=["conservative", "balanced", "permissive", "custom"],
                   default="conservative",
                   help="DOB-CBF intrusiveness preset for driving comparisons. "
                        "conservative = max safety / most intrusive; balanced; "
                        "permissive = least intrusive; custom = use the explicit "
                        "--cbf-alpha / --cbf-forward-bias overrides. (Non-DOB-CBF "
                        "filter types plug in here as they are added.)")
    p.add_argument("--cbf-alpha", type=float, default=None,
                   help="Override CBF class-K gain (higher = acts later = less intrusive).")
    p.add_argument("--cbf-forward-bias", type=float, default=None,
                   help="Override CBF barrier forward shift, m (lower = less intrusive).")
    p.add_argument("--auto-start", action="store_true",
                   help="Do not wait for Enter before each round.")
    p.add_argument("--dry-run", action="store_true",
                   help="Only write manifest and command plan; do not launch Chrono.")
    p.add_argument("--timeout", type=float, default=360.0)
    p.add_argument("--base-port", type=int, default=10400)
    p.add_argument("--quick", action="store_true",
                   help="Single short WASD-compatible smoke round.")
    return p.parse_args()


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class RoundCondition:
    """One live study condition before ports, paths, and commands are assigned."""

    filter_name: str
    delay: float
    terrain: str
    path: str
    speed: float
    bumpiness: int
    seed: int
    convoy: str
    repetition: int
    block: int


def validate_args(args: argparse.Namespace) -> None:
    for name in ("study_id", "operator_id"):
        value = str(getattr(args, name, ""))
        if not value or not _SAFE_ID.fullmatch(value):
            raise SystemExit(
                f"--{name.replace('_', '-')} must contain only letters, numbers, "
                "'.', '_' or '-' (got {value!r})"
            )
    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1")
    if args.practice_rounds < 0:
        raise SystemExit("--practice-rounds cannot be negative")
    if len(set(args.filters)) != len(args.filters):
        raise SystemExit("--filters contains duplicates")
    if args.order == "randomized-blocks" and len(args.filters) < 2:
        raise SystemExit("--order randomized-blocks requires at least two filters")


def build_round_conditions(args: argparse.Namespace) -> list[RoundCondition]:
    """Create a balanced, reproducible live-study schedule.

    In randomized-block mode, each repetition is a complete block containing
    every filter/scenario/plant condition exactly once.  The paired filters in
    a block share the same simulation seed, while the block shuffle prevents
    filter state from being aliased with practice, learning, or fatigue.
    """

    def cells_for_rep(rep: int) -> list[RoundCondition]:
        cells: list[RoundCondition] = []
        for convoy in args.convoy:
            for filter_name in args.filters:
                for delay in args.delays:
                    for terrain in args.terrains:
                        for path in args.paths:
                            for speed in args.speeds:
                                for bump in args.bumpiness:
                                    cells.append(RoundCondition(
                                        filter_name=filter_name,
                                        delay=delay,
                                        terrain=terrain,
                                        path=path,
                                        speed=speed,
                                        bumpiness=bump,
                                        seed=args.base_seed + rep,
                                        convoy=convoy,
                                        repetition=rep,
                                        block=rep + 1,
                                    ))
        return cells

    def pair_key(cell: RoundCondition) -> tuple:
        return (
            cell.delay, cell.terrain, cell.path, cell.speed, cell.bumpiness,
            cell.seed, cell.convoy, cell.repetition,
        )

    if args.order == "randomized-blocks":
        rng = random.Random(args.order_seed)
        out: list[RoundCondition] = []
        for rep in range(args.rounds):
            block = cells_for_rep(rep)
            # With the canonical two-filter study, balance which filter is seen
            # first for each paired cell across repetitions.  A pure shuffle can
            # accidentally put the same filter first for one scenario in all
            # five blocks, turning filter into an order effect.  Rejection also
            # prevents the identical scenario/seed pair from appearing back to
            # back and avoids runs of three identical filter states.
            if len(args.filters) == 2:
                keys = list(dict.fromkeys(pair_key(cell) for cell in block))
                preferred_first = {
                    key: args.filters[(rep + pair_index) % 2]
                    for pair_index, key in enumerate(keys)
                }
                for _ in range(10_000):
                    rng.shuffle(block)
                    positions = {
                        key: [i for i, cell in enumerate(block) if pair_key(cell) == key]
                        for key in keys
                    }
                    precedence_ok = all(
                        len(pos) == 2
                        and block[min(pos)].filter_name == preferred_first[key]
                        for key, pos in positions.items()
                    )
                    adjacent_pair = any(
                        pair_key(block[i]) == pair_key(block[i + 1])
                        for i in range(len(block) - 1)
                    )
                    filters = [cell.filter_name for cell in out[-2:] + block]
                    three_same = any(
                        filters[i] == filters[i + 1] == filters[i + 2]
                        for i in range(len(filters) - 2)
                    )
                    if precedence_ok and not adjacent_pair and not three_same:
                        break
                else:  # pragma: no cover - indicates an impossible custom matrix
                    raise RuntimeError(
                        "could not construct a balanced randomized block; "
                        "use --order fixed for this custom condition matrix"
                    )
            else:
                rng.shuffle(block)
            out.extend(block)
        return out

    # Deterministic nested ordering for callers outside the live study, where
    # counterbalancing has no meaning because no operator sees the sequence.
    out = []
    for convoy in args.convoy:
        for filter_name in args.filters:
            for delay in args.delays:
                for terrain in args.terrains:
                    for path in args.paths:
                        for speed in args.speeds:
                            for bump in args.bumpiness:
                                for rep in range(args.rounds):
                                    out.append(RoundCondition(
                                        filter_name=filter_name,
                                        delay=delay,
                                        terrain=terrain,
                                        path=path,
                                        speed=speed,
                                        bumpiness=bump,
                                        seed=args.base_seed + rep,
                                        convoy=convoy,
                                        repetition=rep,
                                        block=rep + 1,
                                    ))
    return out


def command_for_round(args: argparse.Namespace, run_dir: Path, idx: int, filter_name: str,
                      delay: float, terrain: str, path: str, speed: float,
                      bump: int, seed: int, convoy: str = "",
                      latency_profile_json: str | None = None,
                      latency_phase_s: float = 0.0) -> list[str]:
    sim_port = args.base_port + 2 * idx
    ctrl_port = sim_port + 1
    camera_delay = delay * args.camera_delay_scale
    cmd = [
        sys.executable, "-u", str(LAUNCHER),
        "--terrain", terrain,
        "--path", path,
        "--speed", str(speed),
        "--time", str(args.time),
        "--lead-in", str(args.lead_in),
        "--bumpiness", str(bump),
        "--rocks", str(args.rocks),
        "--rock-seed", str(seed),
        "--sim-seed", str(seed),
        "--sim-port", str(sim_port),
        "--ctrl-port", str(ctrl_port),
        "--transport", "ros",
        "--vis-mode", args.vis_mode,
        "--manual-honor-time",
        "--sim-diag-csv", str(run_dir / "sim_diag.csv"),
        "--nn-model", DEFAULT_NN_MODEL,
        # Torque-command plant, matching the configuration run.py applies to
        # every autonomous study through HIL_SIM_EXTRA. A manual round without
        # it would drive a different plant and would not be comparable to any
        # published arm. The remaining flags of that configuration act on the
        # NMPC and have no counterpart under manual control.
        "--simple-powertrain",
        "--cam-width", str(args.cam_width),
        "--cam-height", str(args.cam_height),
        "--cam-fov", str(args.cam_fov),
        "--cam-rate", str(args.cam_rate),
    ]
    if args.cam_fullscreen:
        cmd.append("--cam-fullscreen")
    if getattr(args, "delayed_pov", False):
        cmd.append("--delayed-pov")
        if getattr(args, "pov_no_flip", False):
            cmd.append("--pov-no-flip")
    if convoy:
        cmd += ["--convoy", convoy, "--traffic-detail", args.traffic_detail]
    if args.goal_distance > 0:
        cmd += ["--goal-distance", str(args.goal_distance)]
    if args.mesh_resolution is not None:
        cmd += ["--mesh-resolution", str(args.mesh_resolution)]
    profile_json = (args.latency_profile_json
                    if latency_profile_json is None else latency_profile_json)
    if profile_json:
        # The profile drives every channel: control and manual carry the command
        # uplink, camera the asymmetric video downlink. It replaces the constant
        # delays. Each block enters the trace at a staggered phase, so the
        # session as a whole samples the environment's outages rather than only
        # its nominal opening stretch, while the paired filter-off and
        # filter-on cells of a block share a phase and stay comparable.
        if latency_phase_s:
            cmd += ["--latency-phase-s", str(latency_phase_s)]
        cmd += ["--latency-profile-json", profile_json,
                "--latency-profile-log", str(run_dir / "latency_profile.csv")]
    else:
        cmd += ["--manual-input-delay", str(delay),
                "--camera-input-delay", str(camera_delay)]
    cmd.append("--wasd" if args.manual_mode == "wasd" else "--manual")
    if args.rocks > 0:
        min_spacing = args.rock_min_spacing
        if convoy:
            # With traffic in the lane, keep the centerline thinned: the cars are
            # the lane hazard and the driver swerves off-centre into the rocks.
            zone = PATH_ROCK_ZONES.get(path, PATH_ROCK_ZONES["sinusoidal"])
            centerline_clear = args.rock_centerline_clear
        else:
            # Without traffic the rocks are the only hazard, so they must lie in
            # the driving corridor. The wide default field clears the centerline
            # and extends to 98 m, which leaves a straight bypass to a goal near
            # 40 m and presents the operator with an effectively empty course.
            # The field is therefore narrowed to the start-to-goal corridor with
            # no centerline clearance and tighter spacing, which places roughly
            # four rocks in the corridor and produces a staggered weave. Spacing
            # remains at or above the minimum, so a traversable gap always
            # exists and the task stays solvable.
            gx = args.goal_distance if args.goal_distance > 0 else 45.0
            zone = {"x": (8.0, gx + 4.0), "y": (-5.0, 5.0)}
            centerline_clear = 0.0
            min_spacing = min(min_spacing, 4.5)
        cmd += [
            "--rock-zone-x", str(zone["x"][0]), str(zone["x"][1]),
            "--rock-zone-y", str(zone["y"][0]), str(zone["y"][1]),
            "--rock-size", str(args.rock_size[0]), str(args.rock_size[1]),
            "--rock-min-spacing", str(min_spacing),
            "--rock-centerline-clear", str(centerline_clear),
            "--rock-spawn-clear", str(args.rock_spawn_clear),
        ]
    if filter_name != "none":
        # DOB-CBF intrusiveness presets, as (class-K alpha, forward bias in m).
        # A higher alpha and a lower forward bias make the filter act later and
        # closer to the hazard, so it intervenes less at the cost of a tighter
        # safety margin. The presets name points on that trade-off, which the
        # convoy studies characterize quantitatively.
        _cbf_profiles = {
            "conservative": (5.0, 3.0),
            "balanced": (15.0, 1.0),
            "permissive": (30.0, 1.0),
        }
        _alpha, _fbias = _cbf_profiles.get(args.cbf_profile, (5.0, 3.0))
        if args.cbf_alpha is not None:
            _alpha = args.cbf_alpha
        if args.cbf_forward_bias is not None:
            _fbias = args.cbf_forward_bias
        cmd += [
            "--safety-filter",
            "--safety-flavor", filter_name,
            "--safety-buffer", str(args.safety_buffer),
            "--shield-horizon", str(args.shield_horizon),
            "--cbf-alpha", str(_alpha),
            "--cbf-forward-bias", str(_fbias),
        ]
        if not profile_json:
            # In profile mode the sim samples the control channel and feeds
            # the sim-side filter; a fixed --teleop-delay would override it.
            cmd += ["--teleop-delay", str(delay)]
    return cmd


def parse_sim_diag(path: Path, ref_path_name: str, speed: float, lead_in: float,
                   metric_start: float = 3.0) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    u = pd.to_numeric(df["speed"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(t) & (t >= metric_start)
    if not mask.any():
        mask = np.isfinite(t)
    xp, yp = generate_path_waypoints(ref_path_name, lead_in=lead_in)
    ref = ReferencePath(xp, yp, v_target=speed)
    cte = []
    for xi, yi in zip(x, y):
        if math.isfinite(xi) and math.isfinite(yi):
            cte.append(ref.closest_point_on_path(float(xi), float(yi))["e_lat"])
        else:
            cte.append(math.nan)
    cte_arr = np.asarray(cte, dtype=float)
    cte_m = cte_arr[mask & np.isfinite(cte_arr)]
    u_m = u[mask & np.isfinite(u)]
    progress = math.nan
    good_xy = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good_xy) >= 2:
        progress = float(np.sum(np.hypot(np.diff(x[good_xy]), np.diff(y[good_xy]))))
    clearance = pd.to_numeric(df.get("nearest_clearance_m", pd.Series(dtype=float)), errors="coerce")
    clearance_m = clearance[mask] if len(clearance) == len(df) else clearance
    return {
        "n_samples": int(len(df)),
        "rms_cte_m": float(np.sqrt(np.mean(cte_m ** 2))) if len(cte_m) else math.nan,
        "max_abs_cte_m": float(np.max(np.abs(cte_m))) if len(cte_m) else math.nan,
        "mean_abs_cte_m": float(np.mean(np.abs(cte_m))) if len(cte_m) else math.nan,
        "mean_speed_mps": float(np.mean(u_m)) if len(u_m) else math.nan,
        "speed_ratio": float(np.mean(u_m) / speed) if len(u_m) and speed > 1e-6 else math.nan,
        "progress_m": progress,
        "final_x_m": float(x[good_xy][-1]) if np.count_nonzero(good_xy) else math.nan,
        "final_y_m": float(y[good_xy][-1]) if np.count_nonzero(good_xy) else math.nan,
        "min_clearance_m": float(np.nanmin(clearance_m)) if len(clearance_m) and np.isfinite(clearance_m).any() else math.nan,
    }


def _maybe_launch_hud(args: argparse.Namespace, idx: int, run_dir: Path):
    """Launch the live HMI overlay on this round's ROS 2 stream (or return None).

    The HUD only subscribes, so it cannot perturb the sim/controller loops; it
    CONNECTs (late binder is fine) and starts updating once the sim binds.
    """
    if not args.live_hud:
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    sim_port = args.base_port + 2 * idx
    ctrl_port = sim_port + 1
    hud_log = (run_dir / "hud.log").open("w")
    # The launcher derives ROS_DOMAIN_ID from the simulator port modulo 101, so
    # the overlay must join that same domain. On any other domain it subscribes
    # successfully but receives nothing, and the operator drives a round with a
    # blank instrument display and no indication that anything is wrong.
    hud_env = dict(os.environ)
    hud_env["ROS_DOMAIN_ID"] = str(sim_port % 101)
    proc = subprocess.Popen(
        [sys.executable, str(SIM_DIR / "teleop" / "hil_hud.py"),
         "--sim-port", str(sim_port), "--ctrl-port", str(ctrl_port),
         "--transport", "ros",
         "--corner", args.hud_corner, "--wheel-lock-deg", str(args.hud_wheel_lock)],
        cwd=str(PROJECT_ROOT), stdout=hud_log, stderr=subprocess.STDOUT,
        env=hud_env,
    )
    proc._hud_log = hud_log  # keep the handle so we can close it on teardown
    return proc


def _stop_hud(proc) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
    log = getattr(proc, "_hud_log", None)
    if log is not None:
        log.close()


def guard_instant_death(rc: int, wall_s: float, run_dir: Path, phase: str) -> None:
    """Halt the session when a round exits before the operator could have driven.

    A round that terminates within seconds presented the operator with nothing,
    yet it still produces a result row. A configuration fault that kills every
    launch, such as a graphics-library mismatch that aborts the renderer at
    startup, would therefore consume an entire session of trials without any
    visible signal. This guard prints the tail of the round's log and requires
    an explicit decision before further rounds are attempted.
    """
    if rc == 0 or wall_s >= 8.0:
        return
    print(f"\n  *** {phase} ROUND DIED IN {wall_s:.1f}s (exit {rc}) -- the sim "
          "never reached the operator. Last log lines:")
    log = run_dir / "run.log"
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines()[-8:]:
            print(f"    | {line}")
    ans = input("  Continue with the remaining rounds anyway? [y/N] ").strip().lower()
    if ans != "y":
        raise SystemExit(
            "Aborting the session: fix the failure above and rerun. "
            "Completed rounds are preserved in this results directory.")


def run_round(cmd: list[str], run_dir: Path, timeout: float) -> tuple[int, float, str]:
    ensure_runtime_env()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    # Route the sim-side collision logger and the safety-filter/shield loggers
    # into this run's dir (same mechanism the parallel sweeps use), so each
    # round's cbf/mppi/nmpc shield CSV + collision log land here -- otherwise
    # they fall back to a shared global dir and the intrusiveness metrics
    # (mean_abs_dsteer/dthrottle, intervention rate) are lost.
    env = dict(os.environ)
    env["HIL_RUN_LOG_DIR"] = str(run_dir)
    t0 = time.time()
    with log_path.open("w") as f:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env=env,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
            f.write(f"\nTIMEOUT after {timeout:.1f}s\n")
    return rc, time.time() - t0, log_path.read_text(errors="replace")


def collect_global_logs(run_dir: Path, created_after: float) -> tuple[str, str]:
    """Resolve the per-run collision + shield log paths.

    With HIL_RUN_LOG_DIR set in run_round, the sim/safety loggers write these
    straight into run_dir; fall back to the shared global LOGS_DIR for any that
    an older path still drops there.
    """
    collision_csv = ""
    shield_csv = ""
    for name in ("collision_log.csv", "cbf_filter_log.csv", "mppi_shield_log.csv", "nmpc_shield_log.csv"):
        local = run_dir / name
        if local.exists():
            src = local
        else:
            src = LOGS_DIR / name
            if not (src.exists() and src.stat().st_mtime >= created_after - 2.0):
                continue
            shutil.copy2(src, run_dir / name)
            src = run_dir / name
        if name == "collision_log.csv":
            collision_csv = str(src)
        elif not shield_csv:
            shield_csv = str(src)
    return collision_csv, shield_csv


def clear_global_logs() -> None:
    """Remove stale shared logs before a sequential live round starts."""
    for name in ("collision_log.csv", "mppi_shield_log.csv",
                 "nmpc_shield_log.csv", "cbf_filter_log.csv"):
        path = LOGS_DIR / name
        if path.exists():
            path.unlink()


def plot_figures(results_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(results_csv)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return
    for col in ("rms_cte_m", "speed_ratio", "collisions",
                "min_clearance_m", "intervention_rate_pct"):
        if col not in ok.columns:
            ok[col] = math.nan
    fig_dir = out_dir / "figures"
    for col in ("mean_abs_dsteer", "mean_abs_dthrottle"):
        if col not in ok.columns:
            ok[col] = math.nan
    if "contact_run" not in ok.columns:
        ok["contact_run"] = (
            pd.to_numeric(ok["collisions"], errors="coerce").fillna(0) > 0
        ).astype(int)

    # Under a latency profile the session has one delay condition rather than a
    # sweep, so the filter-off and filter-on outcomes are plotted directly and
    # every trial point is retained. A line plot against delay would collapse to
    # a single x-coordinate and would conceal the within-operator variability
    # that repeating each condition across blocks exists to expose.
    profile_mode = (
        "latency_profile" in ok.columns
        and ok["latency_profile"].fillna("").astype(str).str.len().gt(0).any()
    )
    if profile_mode:
        for col in ("clean_success", "reached_goal", "progress_m"):
            if col not in ok.columns:
                ok[col] = math.nan
        order = [name for name in ("none", "dob_cbf")
                 if name in set(ok["filter"])]
        metrics = [
            ("contact_run", "Contact run (0/1)", "Native-contact safety"),
            ("min_clearance_m", "Minimum clearance (m)", "Safety margin"),
            ("clean_success", "Clean success (0/1)", "Safety + completion"),
            ("intervention_rate_pct", "Intervention rate (%)", "How often"),
            ("mean_abs_dsteer", "Mean |Δ steer|", "Steering intrusiveness"),
            ("mean_abs_dthrottle", "Mean |Δ throttle|", "Throttle intrusiveness"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.1))
        colors = {"none": "#6b7280", "dob_cbf": "#2563eb"}
        labels = {"none": "Filter off", "dob_cbf": "DOB-CBF"}
        for ax, (metric, ylabel, title) in zip(axes.flat, metrics):
            for xpos, filter_name in enumerate(order):
                values = pd.to_numeric(
                    ok.loc[ok["filter"] == filter_name, metric], errors="coerce"
                ).dropna().to_numpy(dtype=float)
                if not len(values):
                    continue
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ax.bar(xpos, mean, width=0.58, color=colors[filter_name],
                       alpha=0.72, yerr=std, capsize=3, zorder=1)
                jitter = np.linspace(-0.16, 0.16, len(values)) if len(values) > 1 else [0.0]
                ax.scatter(xpos + np.asarray(jitter), values, s=18,
                           facecolor="white", edgecolor=colors[filter_name],
                           linewidth=0.8, alpha=0.9, zorder=2)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([labels[name] for name in order])
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=9.5)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(
            "Single-operator live pilot under seeded statistical latency\n"
            "bars: mean ± within-operator SD; points: individual trials",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "human_live_pilot_summary.png", dpi=220)
        plt.close(fig)
        return

    summary = ok.groupby(["filter", "delay_s"], sort=False).agg(
        rms_cte=("rms_cte_m", "mean"),
        collisions=("collisions", "mean"),
        clearance=("min_clearance_m", "mean"),
        speed_ratio=("speed_ratio", "mean"),
        intervention=("intervention_rate_pct", "mean"),
        dsteer=("mean_abs_dsteer", "mean"),
        dthrottle=("mean_abs_dthrottle", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for filter_name, sub in summary.groupby("filter", sort=False):
        axes[0, 0].plot(sub["delay_s"], sub["collisions"], marker="o", label=filter_name)
        axes[0, 1].plot(sub["delay_s"], sub["clearance"], marker="o", label=filter_name)
        axes[0, 2].plot(sub["delay_s"], sub["intervention"], marker="o", label=filter_name)
        axes[1, 0].plot(sub["delay_s"], sub["dsteer"], marker="o", label=filter_name)
        axes[1, 1].plot(sub["delay_s"], sub["dthrottle"], marker="o", label=filter_name)
        axes[1, 2].plot(sub["delay_s"], sub["speed_ratio"], marker="o", label=filter_name)
    labels = [
        ("Unique obstacles hit (lower better)", "Safety"),
        ("Minimum clearance (m, higher better)", "Safety margin"),
        ("Intervention rate (%)", "Intrusiveness: how often"),
        ("Mean |Δ steer| (filtered − operator)", "Intrusiveness: how much"),
        ("Mean |Δ throttle| (filtered − operator)", "Intrusiveness: how much"),
        ("Speed retention (achieved / target)", "Task progress"),
    ]
    for ax, (label, title) in zip(axes.flat, labels):
        ax.set_xlabel("Operator command delay (s)")
        ax.set_ylabel(label)
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Human-in-the-loop delay compensation rounds: safety × intrusiveness")
    fig.tight_layout()
    fig.savefig(fig_dir / "human_delay_compensation_summary.png", dpi=220)
    plt.close(fig)

    pivot = ok.pivot_table(index="delay_s", columns="filter", values="collisions", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.45 * len(pivot.columns) + 4, 3.6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if math.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("Filter")
    ax.set_ylabel("Delay (s)")
    ax.set_title("Mean unique obstacles hit")
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(fig_dir / "human_delay_collision_heatmap.png", dpi=220)
    plt.close(fig)


def brief_round(args: argparse.Namespace, i: int, total: int, filter_name: str,
                delay: float, terrain: str, path: str, speed: float, bump: int,
                convoy: str = "", *, phase: str = "STUDY", block: int = 0,
                repetition: int = 0) -> None:
    """Print an operator-facing briefing: scenario, goal, filter, and latency."""
    bar = "=" * 64
    print(f"\n{bar}")
    if phase == "PRACTICE":
        print(f"  PRACTICE {i + 1} of {total} -- excluded from study results")
    elif block:
        print(f"  STUDY ROUND {i + 1} of {total}  |  BLOCK {block}  |  "
              f"REPETITION {repetition + 1}")
    else:
        print(f"  STUDY ROUND {i + 1} of {total}")
    print("-" * 64)
    # --- scenario ---
    extras = f", bumpiness {bump}" if bump else ""
    print(f"  SCENARIO : {terrain} terrain, straight forward course, "
          f"{args.time:.0f}s run{extras}.")
    if convoy:
        desc = CONVOY_DESCRIPTIONS.get(convoy, convoy)
        print(f"             Convoy lead: {desc} (it waits for you to move, then "
              f"picks a line through the field).")
    if args.rocks > 0:
        print(f"             {args.rocks}-rock boulder field spanning the full "
              f"width -- thread a route, you can't go around.")
    elif not convoy:
        print(f"             Open course, no obstacles.")
    # --- goal ---
    print(f"  GOAL     : drive FORWARD and reach the far end (~{args.goal_distance:.0f} m "
          f"ahead) within")
    print(f"             the {args.time:.0f}s run WITHOUT hitting any vehicle or rock. "
          f"You must")
    print(f"             keep moving (a round that stops short does NOT count, even")
    print(f"             with no collision). Steady pace ~{speed:g} m/s; weave around")
    print(f"             hazards freely -- there is no line to follow.")
    # --- safety filter ---
    if filter_name == "none":
        print(f"  FILTER   : NONE -- your commands go straight to the vehicle "
              f"(no safety net).")
    else:
        print(f"  FILTER   : {filter_name.upper()} -- screens your commands and may "
              f"override")
        print(f"             steering/throttle to avoid a collision.")
    # --- latency ---
    if args.latency_profile_json:
        import os
        print(f"  LATENCY  : seeded time-varying statistical link "
              f"({os.path.basename(args.latency_profile_json)}).")
        print(f"             Command + camera delay fluctuate, with bursts/outages "
              f"up to ~0.45 s.")
    elif delay > 0:
        cam = delay * args.camera_delay_scale
        print(f"  LATENCY  : CONSTANT -- {delay * 1000:.0f} ms on your commands "
              f"(uplink),")
        print(f"             {cam * 1000:.0f} ms on the camera feed (downlink). "
              f"Inputs and view will lag.")
    else:
        print(f"  LATENCY  : none (0 ms) -- real-time control and view.")
    print(bar)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.filters = ["none"]
        args.delays = [0.0]
        args.terrains = ["clay"]
        args.paths = ["sinusoidal"]
        args.speeds = [4.0]
        args.bumpiness = [0]
        args.rounds = 1
        args.time = min(args.time, 8.0)
        args.manual_mode = "wasd"

    validate_args(args)

    prof_tag = ""
    if args.latency_profile_json:
        args.latency_profile_json = str(
            Path(args.latency_profile_json).expanduser().resolve())
        prof_tag = Path(args.latency_profile_json).stem
        if args.delays != [0.0]:
            print(f"[latency] profile '{prof_tag}' overrides the --delays "
                  "sweep; collapsing to one profile condition per cell.")
        args.delays = [0.0]

    out_dir = timestamped_result_dir("human_delay_compensation_rounds")
    write_manifest(out_dir, args, "Human-in-the-loop manual delay compensation rounds.")
    print(f"Output: {out_dir}")

    planned: list[dict] = []
    for idx, condition in enumerate(build_round_conditions(args)):
        cell = prof_tag if prof_tag else f"delay{condition.delay:.2f}"
        cv = condition.convoy if condition.convoy else "open"
        pair_id = (
            f"b{condition.block:02d}_{cv}_{cell}_{condition.terrain}_"
            f"{condition.path}_v{condition.speed:g}_b{condition.bumpiness}_"
            f"seed{condition.seed}"
        )
        run_dir = out_dir / "raw" / (
            f"{idx:04d}_{condition.filter_name}_{cv}_{cell}_{condition.terrain}_"
            f"{condition.path}_v{condition.speed:g}_b{condition.bumpiness}_"
            f"r{condition.repetition}"
        )
        cmd = command_for_round(
            args, run_dir, idx, condition.filter_name, condition.delay,
            condition.terrain, condition.path, condition.speed,
            condition.bumpiness, condition.seed, condition.convoy,
            latency_phase_s=60.0 * ((condition.block - 1) % 5),
        )
        planned.append({
            "idx": idx,
            "phase": "study",
            "study_id": args.study_id,
            "operator_id": args.operator_id,
            "input_device_model": args.input_device_model,
            "block": condition.block,
            "repetition": condition.repetition,
            "pair_id": pair_id,
            "filter": condition.filter_name,
            "delay_s": condition.delay,
            "convoy": condition.convoy,
            "terrain": condition.terrain,
            "path": condition.path,
            "speed_mps": condition.speed,
            "bumpiness": condition.bumpiness,
            "seed": condition.seed,
            "run_dir": run_dir,
            "cmd": cmd,
        })

    plan_rows = [
        {**{k: v for k, v in row.items() if k not in ("run_dir", "cmd")},
         "run_dir": str(row["run_dir"]), "command": " ".join(row["cmd"])}
        for row in planned
    ]
    pd.DataFrame(plan_rows).to_csv(out_dir / "round_plan.csv", index=False)

    # Familiarization rounds are written outside raw/, because trace-directory
    # replay and every downstream analysis discover study rounds under raw/
    # alone; keeping practice separate prevents it from entering any result.
    # Practice uses the same scenarios with no filter and no latency stressor.
    practice_args = copy.copy(args)
    practice_args.latency_profile_json = ""
    practice_dir = out_dir / "practice"
    practice_dir.mkdir()
    practice_planned: list[dict] = []
    for idx in range(args.practice_rounds):
        convoy = args.convoy[idx % len(args.convoy)]
        terrain = args.terrains[0]
        path = args.paths[0]
        speed = args.speeds[0]
        bump = args.bumpiness[0]
        seed = args.base_seed + 10000 + idx
        cv = convoy if convoy else "open"
        run_dir = practice_dir / f"{idx:04d}_none_{cv}_no_latency_seed{seed}"
        cmd = command_for_round(
            practice_args, run_dir, idx, "none", 0.0, terrain, path,
            speed, bump, seed, convoy, latency_profile_json="",
        )
        practice_planned.append({
            "idx": idx, "phase": "practice", "study_id": args.study_id,
            "operator_id": args.operator_id,
            "input_device_model": args.input_device_model,
            "filter": "none", "delay_s": 0.0,
            "convoy": convoy, "terrain": terrain, "path": path,
            "speed_mps": speed, "bumpiness": bump, "seed": seed,
            "run_dir": run_dir, "cmd": cmd,
        })
    practice_plan_rows = [
        {**{k: v for k, v in row.items() if k not in ("run_dir", "cmd")},
         "run_dir": str(row["run_dir"]), "command": " ".join(row["cmd"])}
        for row in practice_planned
    ]
    pd.DataFrame(practice_plan_rows).to_csv(out_dir / "practice_plan.csv", index=False)
    if args.dry_run:
        print(f"Dry run wrote study plan: {out_dir / 'round_plan.csv'}")
        print(f"Dry run wrote practice plan: {out_dir / 'practice_plan.csv'}")
        return

    practice_rows: list[dict] = []
    if practice_planned:
        print("\nFAMILIARIZATION: these rounds are logged separately and excluded "
              "from every study summary.")
    for item in practice_planned:
        clear_global_logs()
        brief_round(
            practice_args, item["idx"], len(practice_planned), item["filter"],
            item["delay_s"], item["terrain"], item["path"], item["speed_mps"],
            item["bumpiness"], item["convoy"], phase="PRACTICE",
        )
        print(f"  (practice output -> {item['run_dir']})")
        if not args.auto_start:
            input("\n  Press Enter when you're ready for this practice round...")
        hud_proc = _maybe_launch_hud(practice_args, item["idx"], item["run_dir"])
        created_after = time.time()
        try:
            rc, wall_s, text = run_round(item["cmd"], item["run_dir"], args.timeout)
        finally:
            _stop_hud(hud_proc)
        guard_instant_death(rc, wall_s, item["run_dir"], "PRACTICE")
        collision_csv, shield_csv = collect_global_logs(item["run_dir"], created_after)
        practice_row = {
            **{k: v for k, v in item.items() if k not in ("run_dir", "cmd")},
            "excluded_from_study": 1,
            "run_dir": str(item["run_dir"]),
            "rc": rc,
            "wall_s": wall_s,
            "status": "ok" if rc == 0 else f"exit_{rc}",
            "sim_diag_csv": str(item["run_dir"] / "sim_diag.csv"),
            "collision_csv": collision_csv,
            "shield_csv": shield_csv,
        }
        practice_row.update(parse_log_summary(text))
        practice_row.update(parse_collision_csv(
            Path(collision_csv) if collision_csv else None))
        practice_row.update(parse_sim_diag(
            item["run_dir"] / "sim_diag.csv", item["path"],
            item["speed_mps"], args.lead_in))
        practice_rows.append(practice_row)
        pd.DataFrame(practice_rows).to_csv(
            out_dir / "practice_results_excluded.csv", index=False)

    if practice_planned:
        print("\nFamiliarization complete. The recorded study begins next. "
              "Take a short break before continuing.")

    rows: list[dict] = []
    total = len(planned)
    previous_block = 0
    for item in planned:
        i = item["idx"]
        filter_name = item["filter"]
        delay = item["delay_s"]
        terrain = item["terrain"]
        path = item["path"]
        speed = item["speed_mps"]
        bump = item["bumpiness"]
        seed = item["seed"]
        convoy = item["convoy"]
        run_dir = item["run_dir"]
        cmd = item["cmd"]
        if args.order == "randomized-blocks" and item["block"] != previous_block:
            if previous_block:
                print("\nBLOCK COMPLETE: take a short rest before starting the next block.")
            print(f"\nBEGINNING RANDOMIZED BLOCK {item['block']} OF {args.rounds}")
            previous_block = item["block"]
        clear_global_logs()
        brief_round(
            args, i, total, filter_name, delay, terrain, path, speed, bump, convoy,
            phase="STUDY",
            block=item["block"] if args.order == "randomized-blocks" else 0,
            repetition=item["repetition"],
        )
        print(f"  (raw output -> {run_dir})")
        if not args.auto_start:
            input("\n  Press Enter when you're ready to drive this round...")
        hud_proc = _maybe_launch_hud(args, i, run_dir)
        created_after = time.time()
        try:
            rc, wall_s, text = run_round(cmd, run_dir, args.timeout)
        finally:
            _stop_hud(hud_proc)
        guard_instant_death(rc, wall_s, run_dir, "STUDY")
        collision_csv, shield_csv = collect_global_logs(run_dir, created_after)
        row = {
            "experiment": "human_delay_compensation_rounds",
            "phase": "study",
            "study_id": args.study_id,
            "operator_id": args.operator_id,
            "input_device_model": args.input_device_model,
            "block": item["block"],
            "repetition": item["repetition"],
            "pair_id": item["pair_id"],
            "order_position": i,
            "order_seed": args.order_seed,
            "filter": filter_name,
            "variant": f"{filter_name}_delay{delay:.2f}",
            "delay_s": delay,
            "camera_delay_s": delay * args.camera_delay_scale,
            "latency_profile": prof_tag,
            "convoy": convoy,
            "terrain": terrain,
            "path": path,
            "speed_mps": speed,
            "bumpiness": bump,
            "seed": seed,
            "run_dir": str(run_dir),
            "rc": rc,
            "wall_s": wall_s,
            "status": "ok" if rc == 0 else f"exit_{rc}",
            "sim_diag_csv": str(run_dir / "sim_diag.csv"),
            "collision_csv": collision_csv,
            "shield_csv": shield_csv,
        }
        row.update(parse_log_summary(text))
        row.update(parse_collision_csv(Path(collision_csv) if collision_csv else None))
        # The continuous clearance trace takes precedence over the event log's
        # sparse proximity rows, which need not contain the run minimum.
        # Collision truth continues to come from the body-contact log.
        row.update(parse_sim_diag(run_dir / "sim_diag.csv", path, speed, args.lead_in))
        row.update(parse_shield_csv(Path(shield_csv) if shield_csv else None))
        # A round counts as a clean success only if it both avoids collisions
        # and reaches the goal distance. Requiring both is what keeps the metric
        # meaningful: a vehicle that never moves records no collisions, and a
        # collision-only criterion would score that as a perfect outcome.
        _prog = row.get("progress_m", math.nan)
        row["goal_distance_m"] = args.goal_distance
        _final_x = row.get("final_x_m", math.nan)
        row["goal_reached_log"] = int("GOAL REACHED:" in text)
        row["reached_goal"] = int(
            args.goal_distance <= 0
            or row["goal_reached_log"]
            or (math.isfinite(_final_x) and _final_x >= args.goal_distance)
        )
        row["clean_success"] = int(row["reached_goal"] and row.get("collisions", 0) == 0)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
        print(f"    {row['status']}: collisions={row.get('collisions', 0)} "
              f"progress={_prog:.0f}/{args.goal_distance:.0f}m "
              f"{'REACHED' if row['reached_goal'] else 'DID-NOT-REACH'}"
              f"{' [clean success]' if row['clean_success'] else ''}")

    results_csv = out_dir / "results.csv"
    results_df = pd.DataFrame(rows)
    results_df.to_csv(results_csv, index=False)
    # A metric column exists only if at least one run produced it: the
    # intervention rate and the per-step command corrections, for instance, are
    # written by the filter and are absent from a session that ran the
    # unfiltered arm alone. Backfilling the missing columns with NaN lets the
    # aggregation below run over any subset of filters without special cases.
    for col in ("rms_cte_m", "speed_ratio", "collisions",
                "min_clearance_m", "intervention_rate_pct",
                "mean_abs_dsteer", "mean_abs_dthrottle", "near_misses",
                "reached_goal", "clean_success", "progress_m"):
        if col not in results_df.columns:
            results_df[col] = math.nan
    results_df["contact_run"] = (
        pd.to_numeric(results_df["collisions"], errors="coerce").fillna(0) > 0
    ).astype(int)
    results_df.to_csv(results_csv, index=False)
    summary = results_df.groupby(["filter", "delay_s"], sort=False).agg(
        n_runs=("status", "count"),
        n_ok=("status", lambda s: int((s == "ok").sum())),
        contact_runs=("contact_run", "sum"),
        contact_rate=("contact_run", "mean"),
        clean_success_rate=("clean_success", "mean"),
        goal_completion_rate=("reached_goal", "mean"),
        rms_cte_m_mean=("rms_cte_m", "mean"),
        rms_cte_m_std=("rms_cte_m", "std"),
        speed_ratio_mean=("speed_ratio", "mean"),
        speed_ratio_std=("speed_ratio", "std"),
        collisions_mean=("collisions", "mean"),
        collisions_std=("collisions", "std"),
        near_misses_mean=("near_misses", "mean"),
        min_clearance_m_mean=("min_clearance_m", "mean"),
        min_clearance_m_std=("min_clearance_m", "std"),
        intervention_rate_pct_mean=("intervention_rate_pct", "mean"),
        intervention_rate_pct_std=("intervention_rate_pct", "std"),
        mean_abs_dsteer_mean=("mean_abs_dsteer", "mean"),
        mean_abs_dthrottle_mean=("mean_abs_dthrottle", "mean"),
    ).reset_index()
    summary.insert(0, "input_device_model", args.input_device_model)
    summary.insert(0, "operator_id", args.operator_id)
    summary.insert(0, "study_id", args.study_id)
    summary.to_csv(out_dir / "summary_by_filter_delay.csv", index=False)

    by_scenario = results_df.groupby(["filter", "convoy"], sort=False).agg(
        n_runs=("status", "count"),
        n_ok=("status", lambda s: int((s == "ok").sum())),
        contact_runs=("contact_run", "sum"),
        contact_rate=("contact_run", "mean"),
        min_clearance_m_median=("min_clearance_m", "median"),
        clean_success_rate=("clean_success", "mean"),
        goal_completion_rate=("reached_goal", "mean"),
        progress_m_mean=("progress_m", "mean"),
        intervention_rate_pct_mean=("intervention_rate_pct", "mean"),
        mean_abs_dsteer_mean=("mean_abs_dsteer", "mean"),
        mean_abs_dthrottle_mean=("mean_abs_dthrottle", "mean"),
    ).reset_index()
    by_scenario.insert(0, "input_device_model", args.input_device_model)
    by_scenario.insert(0, "operator_id", args.operator_id)
    by_scenario.insert(0, "study_id", args.study_id)
    by_scenario.to_csv(out_dir / "summary_by_filter_scenario.csv", index=False)

    # Each pair is the same operator, scenario, latency trace, plant seed, and
    # repetition under filter-off vs DOB-CBF.  The live commands are naturally
    # not identical; this paired table measures the complete operator+filter
    # loop, while convoy_counterfactual_eval.py remains the identical-intent
    # causal replay analysis.
    pair_metrics = [
        "contact_run", "collisions", "min_clearance_m", "clean_success",
        "reached_goal", "progress_m", "mean_abs_dsteer", "mean_abs_dthrottle",
    ]
    paired = results_df.pivot_table(
        index=["study_id", "operator_id", "pair_id", "block", "repetition",
               "convoy", "seed"],
        columns="filter", values=pair_metrics, aggfunc="first",
    )
    paired.columns = [f"{metric}_{filter_name}" for metric, filter_name in paired.columns]
    paired = paired.reset_index()
    if {"contact_run_none", "contact_run_dob_cbf"}.issubset(paired.columns):
        paired["contact_prevented_live"] = (
            paired["contact_run_none"] - paired["contact_run_dob_cbf"])
    if {"min_clearance_m_none", "min_clearance_m_dob_cbf"}.issubset(paired.columns):
        paired["clearance_gain_m_live"] = (
            paired["min_clearance_m_dob_cbf"] - paired["min_clearance_m_none"])
    if {"clean_success_none", "clean_success_dob_cbf"}.issubset(paired.columns):
        paired["clean_success_gain_live"] = (
            paired["clean_success_dob_cbf"] - paired["clean_success_none"])
    paired.to_csv(out_dir / "paired_live_results.csv", index=False)
    save_summary_markdown(
        out_dir,
        "Human Delay Compensation Rounds",
        summary,
        [
            (f"Study design: single-operator pilot `{args.study_id}`; operator "
             f"identifier `{args.operator_id}`; {args.rounds} randomized complete "
             "blocks. Do not interpret repetitions as independent participants."),
            "Noise policy: sensor noise enabled in every run.",
            (f"Delay policy: seeded time-varying statistical latency profile `{prof_tag}` on "
             "every round -- control/manual channels = command uplink, camera "
             "channel = asymmetric video downlink (supersedes fixed delays).")
            if prof_tag else
            ("Delay policy: each round delays both the operator command path "
             "(`--manual-input-delay`, plus `--teleop-delay` so the predictive "
             "filter horizon is delay-aware) and the driver POV camera feed "
             "(`--camera-input-delay`)."),
            f"Camera delay = {args.camera_delay_scale:g} x command delay "
            "(--camera-delay-scale; 1.0 = symmetric link)."
            if not prof_tag else
            "Live HMI overlay (simulation/teleop/hil_hud.py) attached per round."
            if args.live_hud else "Constant-delay sweep mode.",
        ],
    )
    plot_figures(results_csv, out_dir)
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
