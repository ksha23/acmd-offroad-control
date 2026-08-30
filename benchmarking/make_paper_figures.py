#!/usr/bin/env python3
"""Regenerate and gate the four figures the manuscript includes.

This program owns the figure contract. It reads the set of figures
``my_paper/acmd_fullpaper.tex`` actually includes and requires it to equal the
set this pipeline produces, so a figure can neither appear in the manuscript
without a generator nor be generated without being used. It then runs the
publishers and figure makers in dependency order and records a manifest
naming, for every figure, its digest, its generator, and the digest of each
input the generator read.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FIGURES = ROOT / "my_paper" / "paper_figures"
TEX = ROOT / "my_paper" / "acmd_fullpaper.tex"

EXPECTED = {
    "sys_arch.png",
    "terrain_profile_validation.png",
    "grit_adaptive_speed.png",
    "teleop_scenarios.png",
}

GENERATORS = [
    "publish_paper_figures.py",
    "make_fig_sys_arch.py",
    "make_fig_terrain_profile.py",
    "make_fig_grit_adaptive_speed.py",
    "publish_teleop_scenario_traces.py",
    "make_fig_teleop_scenarios.py",
]

FIGURE_RECIPES = {
    "sys_arch.png": ("make_fig_sys_arch.py", []),
    "terrain_profile_validation.png": (
        "make_fig_terrain_profile.py", [
            "terrain_joint_scored_runs.csv",
            "terrain_joint_summary.csv",
            "terrain_joint_paired_bootstrap.csv",
            "terrain_joint_decision.json",
            "terrain_joint_evidence.json",
        ]
    ),
    "grit_adaptive_speed.png": (
        "make_fig_grit_adaptive_speed.py",
        ["grit_adaptive_speed_matrix_summary.csv",
         "grit_adaptive_speed_trace_grit.csv",
         "grit_adaptive_speed_trace_conservative.csv",
         "grit_adaptive_speed_trace_aggressive.csv"],
    ),
    "teleop_scenarios.png": (
        "make_fig_teleop_scenarios.py", ["teleop_scenario_traces.csv"],
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_figure_manifest() -> None:
    records = []
    for figure_name in sorted(EXPECTED):
        generator_name, input_names = FIGURE_RECIPES[figure_name]
        figure_path = FIGURES / figure_name
        generator_path = HERE / generator_name
        input_paths = [FIGURES / name for name in input_names]
        missing = [
            str(path) for path in [figure_path, generator_path, *input_paths]
            if not path.is_file()
        ]
        if missing:
            raise RuntimeError(
                "Cannot record figure provenance; missing: " + ", ".join(missing)
            )
        records.append({
            "figure": figure_name,
            "sha256": digest(figure_path),
            "generator": str(generator_path.relative_to(ROOT)),
            "generator_sha256": digest(generator_path),
            "inputs": [
                {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
                for path in input_paths
            ],
        })
    (FIGURES / "figure_manifest.json").write_text(
        json.dumps({"schema_version": 1, "figures": records}, indent=2) + "\n"
    )


def run(script: str) -> None:
    completed = subprocess.run([sys.executable, "-u", str(HERE / script)], cwd=ROOT)
    if completed.returncode:
        raise RuntimeError(f"{script} failed with exit code {completed.returncode}")


def tex_figures() -> set[str]:
    names = set()
    for raw in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", TEX.read_text()):
        path = Path(raw)
        names.add(path.name if path.suffix else f"{path.name}.png")
    return names


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-published",
        action="store_true",
        help="Regenerate the figures from the committed paper_figures CSVs "
             "only, skipping the publishers (which validate against the raw "
             "result generations restored per DATA.md). For clean clones.",
    )
    args = parser.parse_args()
    publishers = {"publish_paper_figures.py", "publish_teleop_scenario_traces.py"}

    FIGURES.mkdir(parents=True, exist_ok=True)
    included = tex_figures()
    if included != EXPECTED:
        raise RuntimeError(
            f"Figure contract mismatch: TeX-only={sorted(included - EXPECTED)}, "
            f"pipeline-only={sorted(EXPECTED - included)}"
        )
    for script in GENERATORS:
        if args.from_published and script in publishers:
            print(f"[skip] {script} (--from-published)")
            continue
        run(script)
        print(f"[ok] {script}")
    if not args.from_published:
        # Re-run the numeric publisher to confirm every bound source still
        # passes its fail-closed gate. Generated figures are not copied from
        # or written into evidence directories, so this pass cannot overwrite
        # them.
        run("publish_paper_figures.py")
        print("[ok] publish manifest refreshed after figure generation")
    missing = sorted(name for name in EXPECTED if not (FIGURES / name).is_file())
    if missing:
        raise RuntimeError(f"Missing generated figures: {missing}")
    write_figure_manifest()
    print("[ok] final figure hashes and recipes recorded")
    print(f"All {len(EXPECTED)} ACMD figures regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
