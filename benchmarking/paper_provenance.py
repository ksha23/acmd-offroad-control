"""Provenance contract shared by the benchmarks that produce paper values.

Three properties must hold before a benchmark row is admissible evidence, and
this module establishes each one: that no other ROS benchmark was running
concurrently, that the row's simulation was launched with the configuration it
was asked for, and that the source tree it ran from was fully committed.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, TextIO

import fcntl


ROOT = Path(__file__).resolve().parents[1]

DOWNSTREAM_SOURCE_FILES = (
    "benchmarking/paper_provenance.py",
    "benchmarking/common.py",
    "benchmarking/active_estimator_diagnostics.py",
    "benchmarking/tire_model_with_estimator_ablation.py",
    "simulation/control/acados_mpc_controller_node.py",
    "simulation/runtime/launch_decoupled.py",
    "simulation/runtime/chrono_sim_node.py",
    "simulation/estimators/scalar_parent_terrain_estimator.py",
    "simulation/estimators/grit_terrain_estimator.py",
    "simulation/estimators/terrain_parameterization.py",
    "simulation/shared/param_consistency.py",
    "simulation/tire_models/four_wheel_projection.py",
    "simulation/tire_models/nn_tire_model.py",
    "simulation/tire_models/tire_input_features.py",
)

_ROS_LEASE_PATH = Path("/tmp/offroad_control_acmd_ros.lock")
_PATH_SPEED_PATTERN = re.compile(
    r"^\s*Path:\s*([^,]+),\s*v_target:\s*([-+0-9.eE]+)\s*m/s\s*$",
    flags=re.MULTILINE,
)


def acquire_paper_ros_lease(study: str) -> TextIO:
    """Fail closed if another paper benchmark already owns the ROS graph.

    The launcher maps ports onto a finite range of DDS domain identifiers, so
    two sweep programs can collide on a domain even when their port labels
    differ, and the resulting cross-talk would silently mix runs.  An
    exclusive lease makes that collision impossible rather than detectable
    after the fact.  Keeping the returned handle alive holds the lease.
    """

    handle = _ROS_LEASE_PATH.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(
            "another ACMD ROS benchmark is active: " + owner
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} study={study}\n")
    handle.flush()
    return handle


def launch_identity_contract(
    run_dir: Path,
    *,
    expected_path: str,
    expected_speed_mps: float,
    expected_seed: int,
    expected_sim_port: int,
    expected_ctrl_port: int,
) -> dict[str, Any]:
    """Parse the child log and bind a result row to its requested launch.

    A mismatch indicates DDS cross-talk or task misrouting and invalidates the
    row before any performance metric is considered.
    """

    log_path = Path(run_dir) / "run.log"
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""

    path_speed = _PATH_SPEED_PATTERN.findall(text)
    seeds = re.findall(
        r"^\s*Simulation seed:\s*([-+0-9]+)", text, flags=re.MULTILINE
    )
    domains = re.findall(
        r"^\[launch\] ROS_DOMAIN_ID=([0-9]+)", text, flags=re.MULTILINE
    )
    state_subscriptions = re.findall(
        r"Subscribing to state from localhost:([0-9]+)", text
    )
    control_publications = re.findall(
        r"Publishing controls on port ([0-9]+)", text
    )
    state_publications = re.findall(
        r"Publishing state on port ([0-9]+)", text
    )
    control_subscriptions = re.findall(
        r"Subscribing to controls from localhost:([0-9]+)", text
    )

    observed_path = path_speed[0][0].strip() if len(path_speed) == 1 else ""
    observed_speed = (
        float(path_speed[0][1]) if len(path_speed) == 1 else math.nan
    )
    observed_seed = int(seeds[0]) if len(seeds) == 1 else -1
    observed_domain = int(domains[0]) if len(domains) == 1 else -1
    observed_sim_ports = {
        *(int(value) for value in state_subscriptions),
        *(int(value) for value in state_publications),
    }
    observed_ctrl_ports = {
        *(int(value) for value in control_publications),
        *(int(value) for value in control_subscriptions),
    }
    matches = bool(
        observed_path == str(expected_path)
        and math.isclose(
            observed_speed, float(expected_speed_mps), rel_tol=0.0, abs_tol=1e-9
        )
        and observed_seed == int(expected_seed)
        and observed_domain == int(expected_sim_port) % 101
        and observed_sim_ports == {int(expected_sim_port)}
        and observed_ctrl_ports == {int(expected_ctrl_port)}
    )
    return {
        "launch_identity_match": matches,
        "observed_path": observed_path,
        "observed_speed_mps": observed_speed,
        "observed_sim_seed": observed_seed,
        "observed_ros_domain_id": observed_domain,
        "observed_sim_ports": sorted(observed_sim_ports),
        "observed_ctrl_ports": sorted(observed_ctrl_ports),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git provenance query failed")
    return result.stdout.strip()


def downstream_repository_provenance() -> dict[str, Any]:
    """Record the exact committed sources a run executed.

    The record states whether the run is eligible as paper evidence, which
    requires a clean worktree and every contract file committed.  Ineligible
    runs are annotated rather than refused, so development runs proceed while
    remaining permanently distinguishable from publishable ones.
    """


    missing = [relative for relative in DOWNSTREAM_SOURCE_FILES
               if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing downstream paper sources: {missing}")
    tracked_status = _git("status", "--porcelain", "--untracked-files=no")
    uncommitted = []
    for relative in DOWNSTREAM_SOURCE_FILES:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        if result.returncode:
            uncommitted.append(relative)
    eligible = not tracked_status and not uncommitted
    return {
        "code_git_head": _git("rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(tracked_status),
        "uncommitted_source_files": sorted(uncommitted),
        "paper_evidence_eligible": bool(eligible),
        "source_sha256": {
            relative: _sha256(ROOT / relative)
            for relative in DOWNSTREAM_SOURCE_FILES
        },
    }
