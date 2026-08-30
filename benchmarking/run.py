#!/usr/bin/env python3
"""Single orchestrator for the manuscript's simulation campaign.

This script runs, in one pass, every closed-loop experiment whose numbers appear
in ``my_paper/acmd_fullpaper.tex``, and on a complete paper-tier suite it
regenerates the manuscript figures and tables from the result directories that
same suite produced. Each command's description names the manuscript item it
supports, and the suite manifest records that mapping alongside the exact
command line, so every published value can be traced back to one invocation.

Every learned tire-force query in the campaign is answered by a network
supervised solely by the controlled single-tire Chrono SCM test stand; no
learned component of the control or estimation stack is trained on vehicle
trajectories.

Three tiers are available. ``smoke`` exercises each experiment end to end on a
minimal matrix and checks that it is wired correctly; ``pilot`` runs a reduced
matrix for development; ``paper`` runs the published matrix and is the only tier
permitted to republish manuscript artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from common import bounded_ros_workers, default_ros_workers
except ModuleNotFoundError:  # imported as a package by the tests
    from benchmarking.common import bounded_ros_workers, default_ros_workers

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

RESULT_PREFIX = {
    "tire_models": "mpc_tire_model_sweep",
    "tire_models_calibrated": "mpc_tire_model_sweep_calibrated",
    "tire_estimator": "tire_model_with_estimator_ablation",
    "speed_profile": "speed_profile_ablation",
    "grit_speed": "grit_adaptive_speed_matrix",
    "teleop_battery": "teleop_failure_modes",
    "safety": "safety_filter_sweep_planner_aware",
    "live_grit_integration": "safety_filter_sweep_live_grit_mpsf_paper",
    "convoy_cf": "convoy_counterfactual_eval",
    "latency_awareness": "latency_awareness_ablation",
}

# A command's freshly generated result directory is the publication source for
# its manuscript values, except where the publisher owns a separately locked
# selector of its own (the hash-verified terrain-estimator evidence matrix).
# Entries here override that default for a named command.
AUTHORITATIVE_PUBLISH_SOURCES: dict[str, Path] = {}


@dataclass(frozen=True)
class Command:
    name: str
    argv: list[str]
    estimated_runs: int
    description: str
    tracks_result_directory: bool = True


def py(script: str, *arguments: str) -> list[str]:
    return [sys.executable, "-u", str(HERE / script), *map(str, arguments)]


# --- Plant and longitudinal-channel configuration of the deployed stack ------
# The deployed controller commands drive torque directly and obtains its
# longitudinal force balance from the surrogate-drag feedforward of
# Sec. 3.3 rather than from a reactive throttle disturbance observer. The
# tire-model comparison is the single exception: it leaves the disturbance
# observer active on every arm, because the feedforward evaluates the neural
# surrogate and is therefore unavailable to the analytical baselines; enabling it
# for one arm alone would confound the tire-model contrast with a
# longitudinal-channel advantage. The terrain-estimator evidence and the
# teleoperation battery declare their own configuration and accept no override.
# ``HIL_SIM_EXTRA`` is read by ``common.launch_and_collect`` and appended to
# every launch, so one setting here reconfigures a whole command's matrix.
_TIRE_CMP_EXTRA = "--simple-powertrain"
_DEPLOYED_EXTRA = "--simple-powertrain --ff-drag-surrogate --dob-ki 0 --dob-max 0"
# The integration study is the deployed stack with live estimation and a delayed
# channel added, so it extends the deployed flags rather than replacing them.
# Replacing them would restore the full nonlinear powertrain and the throttle
# disturbance observer, and its contact counts would then not be comparable with
# the safety matrix they are read against.
_LIVE_GRIT_EXTRA = (
    _DEPLOYED_EXTRA
    + " --terrain-estimator --terrain-estimator-backend grit"
    + " --teleop-delay 0.30 --terrain-id-probe"
)
_SIM_EXTRA_BY_COMMAND: dict[str, str | None] = {
    "tire_models": _TIRE_CMP_EXTRA,     # disturbance observer active on every arm
    "tire_models_calibrated": _TIRE_CMP_EXTRA,  # same plant as the main tire matrix
    "terrain_estimator": None,          # hash-verified evidence matrix
    "teleop_battery": None,             # the battery declares its own 5G profile
    "live_grit_integration": _LIVE_GRIT_EXTRA,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["smoke", "pilot", "paper"], default="pilot")
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--workers", type=bounded_ros_workers,
                        default=default_ros_workers())
    parser.add_argument("--timeout", type=float, default=500.0)
    parser.add_argument("--base-port", type=int, default=20000)
    parser.add_argument("--port-stride", type=int, default=2500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def matrix(tier: str) -> dict[str, object]:
    if tier == "pilot":
        return {
            "terrains": ["clay", "dirt", "sand"],
            "paths": ["sinusoidal", "lane_change", "right_left"],
            "speeds": ["5", "7"],
            "bumps": ["0", "4"],
            "seeds": 2,
            "time": "12",
            "estimator_n": 20,
            "transition_pairs": 10,
        }
    return {
        "terrains": ["clay", "dirt", "sand"],
        "paths": ["sinusoidal", "lane_change", "right_left"],
        "speeds": ["5", "7", "9"],
        "bumps": ["0", "4", "8"],
        "seeds": 5,
        "time": "15",
        "estimator_n": 100,
        "transition_pairs": 40,
    }


def common_args(values: dict[str, object]) -> list[str]:
    return [
        "--terrains", *values["terrains"],
        "--paths", *values["paths"],
        "--speeds", *values["speeds"],
        "--bumpiness", *values["bumps"],
        "--seeds", str(values["seeds"]),
        "--time", str(values["time"]),
    ]


def product_count(*items: object) -> int:
    result = 1
    for item in items:
        result *= item if isinstance(item, int) else len(item)
    return result


def build_commands(args: argparse.Namespace) -> list[Command]:
    names = [
        "tire_models", "tire_models_calibrated", "tire_estimator", "terrain_estimator",
        "speed_profile", "grit_speed", "safety", "live_grit_integration",
        "convoy_cf", "latency_awareness", "teleop_battery",
    ]
    ports = {name: args.base_port + index * args.port_stride
             for index, name in enumerate(names)}
    if max(ports.values()) + args.port_stride > 65535:
        raise SystemExit("Port plan exceeds 65535; lower --base-port or --port-stride")

    if args.tier == "smoke":
        commands = [
            Command("tire_models", py("mpc_tire_model_sweep.py", "--quick",
                    "--base-port", ports["tire_models"]), 3, "Tire-model wiring"),
            Command("tire_estimator", py("tire_model_with_estimator_ablation.py",
                    "--quick", "--variants", "nn_static", "nn_estimator",
                    "nn_parent_estimator", "nn_fixed_fallback",
                    "--estimator-backend", "grit",
                    "--base-port", ports["tire_estimator"]), 4,
                    "Joint/parent/fallback terrain conditioning"),
            Command(
                "terrain_estimator",
                [sys.executable, "-m", "unittest",
                 "tests.benchmarking.test_develop_joint_estimator",
                 "tests.benchmarking.test_score_joint_estimator",
                 "tests.benchmarking.test_active_estimator_diagnostics",
                 "tests.benchmarking.test_metric_extraction"],
                1, "Joint n/phi estimator and locked-evidence contracts",
                tracks_result_directory=False,
            ),
            Command(
                "safety_and_policy_tests",
                # These suites are pytest-style, which ``unittest`` collects
                # as zero tests; they must run here, in the environment that
                # has casadi and acados, or they run nowhere. Plugin autoload
                # is disabled because the ROS overlay registers pytest
                # plugins with unmet dependencies.
                ["env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
                 sys.executable, "-m", "pytest", "-q",
                 "tests/simulation/test_mpsf_safety.py",
                 "tests/simulation/test_controller_terrain_policy.py"],
                1, "MPSF safety contract and controller snapshot policy",
                tracks_result_directory=False,
            ),
            Command("speed_profile", py("speed_profile_ablation.py", "--terrains", "clay",
                    "--paths", "sinusoidal", "--speeds", "5", "--seeds", "1",
                    "--time", "10", "--base-port", ports["speed_profile"]), 2,
                    "Static vs terrain-aware speed profile"),
            Command("grit_speed", py("grit_adaptive_speed_matrix.py", "--terrains", "clay",
                    "--paths", "sinusoidal", "--bumpiness", "0", "--seeds", "1",
                    "--workers", "1", "--base-port", ports["grit_speed"]), 4,
                    "GRIT adaptive-speed wiring"),
            Command("safety", py("safety_filter_sweep.py", "--quick", "--flavors", "none",
                    "dob_cbf", "--blind-and-aware", "--output-suffix", "planner_aware",
                    "--base-port", ports["safety"]), 3, "Native-contact safety wiring"),
            Command("convoy_cf", py("convoy_counterfactual_eval.py", "--reckless-throttle",
                    "0.6", "--convoy", "stalled", "--filters", "none", "dob_cbf",
                    "--delays", "0.0", "--time", "10", "--base-port", ports["convoy_cf"]),
                    2, "Paired counterfactual wiring"),
            Command("latency_awareness", py("latency_awareness_ablation.py", "--convoy", "stalled",
                    "--delays", "0.15", "--reckless-throttle", "0.6", "--time", "10",
                    "--base-port", ports["latency_awareness"]), 3,
                    "Blind/aware delay wiring"),
        ]
    else:
        values = matrix(args.tier)
        common = common_args(values)
        n_base = product_count(values["terrains"], values["paths"], values["speeds"],
                               values["bumps"], int(values["seeds"]))
        # The terrain estimator requires no ground datum, so roughness is held
        # at zero here: the conditioning matrix is flat, which lets all four
        # arms share one reference design and isolates the terrain-parameter
        # source as the only difference between them.
        tire_estimator_args = [
            "--terrains", *values["terrains"],
            "--paths", *values["paths"],
            "--speeds", *values["speeds"],
            "--bumpiness", "0",
            "--seeds", str(values["seeds"]),
            "--time", "20", "--metric-start", "8",
            "--estimator-backend", "grit",
        ]
        n_tire_estimator = product_count(
            values["terrains"], values["paths"], values["speeds"],
            int(values["seeds"]),
        )
        terrain_estimator_command = Command(
            "terrain_estimator",
            py("joint_n_phi_evidence.py"),
            72,
            "Table 2, Fig. 2: hash-verified terrain-estimator confirmation",
            tracks_result_directory=False,
        )
        commands = [
            # Table 1 runs one at a time. It reports solver cost alongside
            # tracking error, and both are contention-sensitive: under eight
            # workers the mean solve time of every arm roughly doubles and the
            # neural arm, the most expensive to solve, loses tracking accuracy
            # that the analytical arms do not. Serial execution is what makes
            # the solve column meaningful and the arms comparable.
            Command("tire_models", py("mpc_tire_model_sweep.py", "--models", "pacejka",
                    "tmeasy", "tire_force_static", *common, "--workers", "1",
                    "--base-port", ports["tire_models"]),
                    3 * n_base, "Table 1: matched-terrain tire-model benchmark"),
            Command("tire_estimator", py("tire_model_with_estimator_ablation.py", "--variants",
                    "nn_static", "nn_estimator", "nn_parent_estimator",
                    "nn_fixed_fallback", *tire_estimator_args,
                    "--base-port", ports["tire_estimator"]),
                    4 * n_tire_estimator,
                    "Sec. 3.2: terrain-conditioning robustness, four sources"),
            terrain_estimator_command,
            Command("speed_profile", py("speed_profile_ablation.py", "--paths",
                    "double_lane_change", "sinusoidal", "--base-port", ports["speed_profile"]),
                    48, "Sec. 3.3: terrain-aware speed-reference component study"),
            Command("grit_speed", py("grit_adaptive_speed_matrix.py", "--base-port",
                    ports["grit_speed"]), 540,
                    "Fig. 3: GRIT terrain-adaptive speed matrix"),
            Command("tire_models_calibrated", py("mpc_tire_model_sweep.py",
                    "--models", "pacejka_oracle", "pacejka_rigfit",
                    "--output-prefix", "mpc_tire_model_sweep_calibrated",
                    "--workers", "1",
                    "--base-port", ports["tire_models_calibrated"]), 810,
                    "Table 1 rows 2-3: calibrated-Pacejka arms, collected serially"),
            Command("safety", py("safety_filter_sweep.py", "--flavors", "none", "dob_cbf",
                    "--blind-and-aware", "--output-suffix", "planner_aware", *common,
                    "--base-port", ports["safety"]), 4 * n_base,
                    "Table 3 matrix rows: planner-blind and planner-aware safety"),
            Command("live_grit_integration", py("safety_filter_sweep.py",
                    "--flavors", "none", "mpsf", "--blind-and-aware",
                    "--terrains", "clay", "dirt", "sand", "--paths", "sinusoidal",
                    "--speeds", "7", "--bumpiness", "0", "--seeds", "5",
                    "--time", "24", "--lead-in", "6", "--rocks", "5",
                    "--output-suffix", "live_grit_mpsf_paper",
                    "--base-port", ports["live_grit_integration"]), 60,
                    "Sec. 4.2: online estimation under filtering and delay"),
            Command("convoy_cf", py("convoy_counterfactual_eval.py", "--reckless-throttle", "0.4",
                    "0.6", "0.8", "--convoy", "double_cut", "jam", "oncoming", "--filters",
                    "none", "dob_cbf", "--delays", "0.0", "0.1", "0.2", "0.3", "0.4",
                    "--base-port", ports["convoy_cf"]), 90,
                    "Table 3 convoy rows: paired convoy counterfactual"),
            # The primary endpoint is the three moving-approach scenarios;
            # lead_brake and rear_approach are the preregistered negative
            # controls. The script's default scenario set contains none of the
            # three, so it must be named explicitly or the paired endpoint has
            # no cells to score.
            Command("latency_awareness", py("latency_awareness_ablation.py",
                    "--convoy", "oncoming", "jam", "double_cut",
                    "lead_brake", "rear_approach",
                    "--base-port", ports["latency_awareness"]), 90,
                    "Sec. 4.2: paired delay-awareness ablation"),
            Command("teleop_battery", py("teleop_failure_modes.py", "--base-port",
                    ports["teleop_battery"]), 30,
                    "Table 4, Fig. 4: teleoperation failure-mode battery"),
        ]

    if args.only:
        wanted = set(args.only)
        unknown = wanted - {command.name for command in commands}
        if unknown:
            raise SystemExit(f"Unknown --only values: {', '.join(sorted(unknown))}")
        commands = [command for command in commands if command.name in wanted]
    return commands


def add_parallel_flags(command: Command, args: argparse.Namespace) -> list[str]:
    argv = list(command.argv)
    no_workers = {
        "terrain_estimator",
        # The battery sequences its own scenarios and takes no worker count.
        "teleop_battery",
        # pytest takes neither flag.
        "safety_and_policy_tests",
    }
    no_timeout = {
        "terrain_estimator",
        "safety_and_policy_tests",
    }
    if command.name not in no_workers and "--workers" not in argv:
        argv += ["--workers", str(args.workers)]
    if command.name not in no_timeout and "--timeout" not in argv:
        argv += ["--timeout", str(args.timeout)]
    return argv


def main() -> int:
    args = parse_args()
    os.environ.setdefault("ACADOS_SOURCE_DIR", str(Path.home() / "Documents/sbel/acados"))
    commands = build_commands(args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = RESULTS / f"acmd_{args.tier}_{stamp}"
    rows = []
    for command in commands:
        argv = add_parallel_flags(command, args)
        rows.append([command.name, command.estimated_runs, command.description, " ".join(argv)])
    if not args.dry_run:
        suite_dir.mkdir(parents=True, exist_ok=True)
        with (suite_dir / "suite_manifest.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["name", "estimated_runs", "description", "command"])
            writer.writerows(rows)

    total = sum(command.estimated_runs for command in commands)
    print(f"ACMD rig suite: tier={args.tier} commands={len(commands)} estimated_runs={total}")
    print(
        "Manifest: "
        + (str(suite_dir / "suite_manifest.csv") if not args.dry_run else "not written (dry run)")
    )
    failures = []
    source_map: dict[str, str] = {}
    replication_map: dict[str, str] = {}
    source_map_path = suite_dir / "publish_source_map.json"
    for index, command in enumerate(commands, start=1):
        argv = add_parallel_flags(command, args)
        print(f"\n[{index}/{len(commands)}] {command.name}\n  {' '.join(argv)}", flush=True)
        if args.dry_run:
            continue
        prefix = RESULT_PREFIX.get(command.name)
        before = (
            {path.resolve() for path in RESULTS.glob(f"{prefix}_*") if path.is_dir()}
            if command.tracks_result_directory and prefix is not None
            else set()
        )
        _extra = _SIM_EXTRA_BY_COMMAND.get(command.name, _DEPLOYED_EXTRA)
        _env = dict(os.environ)
        if _extra is not None:
            _env["HIL_SIM_EXTRA"] = _extra
        elif "HIL_SIM_EXTRA" in _env:
            del _env["HIL_SIM_EXTRA"]
        completed = subprocess.run(argv, cwd=ROOT, env=_env)
        if completed.returncode:
            failures.append(command.name)
            if not args.continue_on_error:
                break
            continue
        if not command.tracks_result_directory:
            continue
        if prefix is None:
            failures.append(f"{command.name}:missing_result_prefix")
            if not args.continue_on_error:
                break
            continue
        after = {path.resolve() for path in RESULTS.glob(f"{prefix}_*") if path.is_dir()}
        created = sorted(after - before)
        if len(created) != 1:
            failures.append(f"{command.name}:result_provenance")
            print(
                f"ERROR: expected exactly one new {prefix}_* directory, got "
                f"{[str(path) for path in created]}"
            )
            if not args.continue_on_error:
                break
            continue
        replication_map[prefix] = str(created[0])
        selected = AUTHORITATIVE_PUBLISH_SOURCES.get(command.name, created[0]).resolve()
        if (
            not selected.is_dir()
            or selected.parent != RESULTS.resolve()
            or not selected.name.startswith(prefix + "_")
        ):
            failures.append(f"{command.name}:authoritative_source")
            print(f"ERROR: invalid authoritative source for {prefix}: {selected}")
            if not args.continue_on_error:
                break
            continue
        source_map[prefix] = str(selected)
        source_map_path.write_text(json.dumps(source_map, indent=2) + "\n")
        (suite_dir / "replication_result_map.json").write_text(
            json.dumps(replication_map, indent=2) + "\n"
        )

    publish_requested = (
        not args.dry_run
        and not failures
        and not args.no_figures
        and args.tier == "paper"
        and not args.only
    )
    if publish_requested:
        figure_env = dict(os.environ)
        figure_env["ACMD_PUBLISH_SOURCE_MAP"] = str(source_map_path.resolve())
        completed = subprocess.run(
            py("make_paper_figures.py"), cwd=ROOT, env=figure_env
        )
        if completed.returncode:
            failures.append("make_paper_figures")
    elif not args.dry_run and not failures and not args.no_figures:
        print(
            "Skipping paper publication for a non-paper or partial suite; "
            "this prevents mixing the new run with stale generations."
        )
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
