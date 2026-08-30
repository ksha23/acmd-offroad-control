#!/usr/bin/env python3
"""Publish the clearance traces behind the teleoperation failure-mode figure.

The raw scenario-battery output is too large to track, so the three panels the
manuscript shows are extracted here into a committed long-format CSV, and the
figure is drawn from that file. A reader with the repository alone can
therefore regenerate the figure, and the accompanying manifest records which
run and which file each trace came from. Each row carries the ego pose and the
geometry of the nearest hazard at that step, so the hazard is drawn where the
simulator placed it rather than inferred from the ego path.

Collision truth is native Chrono ego-body--obstacle-body contact; the published
``collisions`` column carries that count, and the run is rejected if it declares
any other collision source. Signed clearance is the geometric diagnostic and
travels beside it, never in place of it.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarking" / "results"
DEST = ROOT / "my_paper" / "paper_figures"

PANELS = ["3_missed_obstacle", "4_peripheral_hazard", "9_freeze_into_stalled"]
ARMS = ["none", "dob_cbf", "mpsf"]
KEEP = ["time", "x", "y", "collisions", "nearest_clearance_m",
        "hazard_x", "hazard_y", "hazard_r"]
NATIVE = "chrono_body_contact"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="", help="explicit battery results dir; "
                    "default: the run dirs recorded in the committed manifest "
                    "(falls back to newest teleop_failure_modes_* only when no "
                    "manifest exists)")
    args = ap.parse_args()
    if args.dir:
        runs = [Path(args.dir).resolve()]
        if not (runs[0] / "results.csv").is_file():
            raise SystemExit(f"--dir has no results.csv: {runs[0]}")
    else:
        # Republishing must be idempotent, so a rerun without --dir
        # re-extracts the selection the manifest records rather than the newest
        # matching run, which may be a probe or a different battery variant.
        manifest_path = DEST / "teleop_scenario_traces_manifest.json"
        runs = []
        if manifest_path.is_file():
            recorded = json.loads(manifest_path.read_text()).get("sources", [])
            seen: list[Path] = []
            for entry in recorded:
                run_dir = (ROOT / entry["path"]).parents[2]
                if run_dir not in seen:
                    seen.append(run_dir)
            if all(r.is_dir() for r in seen):
                runs = seen
            else:
                raise SystemExit(
                    "manifest-recorded battery dirs are missing on disk; pass "
                    "--dir explicitly to re-publish from a different run")
        if not runs:
            runs = sorted(RESULTS.glob("teleop_failure_modes_*"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise SystemExit("no teleop_failure_modes results to publish")

    frames, sources = [], []
    for scen in PANELS:
        for arm in ARMS:
            # The panels are distributed across battery runs, so resolve each
            # scenario/arm pair against the newest run that holds its trace.
            src = next((r / "raw" / f"{scen}__{arm}" / "sim_diag.csv" for r in runs
                        if (r / "raw" / f"{scen}__{arm}" / "sim_diag.csv").is_file()), None)
            if src is None:
                raise SystemExit(f"missing trace for {scen}__{arm}")
            d = pd.read_csv(src)
            if any(c not in d.columns for c in KEEP):
                raise SystemExit(f"{src} lacks {KEEP}")
            observed = set(str(v) for v in d.get("collision_source", pd.Series(dtype=str))
                           .dropna().unique()) - {"", "none"}
            if observed - {NATIVE}:
                raise SystemExit(f"{src} declares non-native collision source: {sorted(observed)}")
            frames.append(d[KEEP].assign(scenario=scen, arm=arm))
            sources.append({"scenario": scen, "arm": arm,
                            "path": str(src.relative_to(ROOT)), "sha256": digest(src),
                            "rows": int(len(d))})

    out = pd.concat(frames, ignore_index=True)[["scenario", "arm", *KEEP]]
    DEST.mkdir(parents=True, exist_ok=True)
    out.to_csv(DEST / "teleop_scenario_traces.csv", index=False)
    (DEST / "teleop_scenario_traces_manifest.json").write_text(json.dumps({
        "schema_version": 1, "study": "teleop_failure_modes",
        "panels": PANELS, "arms": ARMS, "collision_source": NATIVE,
        "sources": sources,
    }, indent=2) + "\n")
    print(f"published {len(out)} rows from {len(sources)} traces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
