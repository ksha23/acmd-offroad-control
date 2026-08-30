#!/usr/bin/env python3
"""Audit every benchmark result set for runs that end stopped.

A run that comes to a halt and stays there scores well on any harm metric --
no contact, no near miss, a large minimum clearance -- while making no
progress along its reference path. Where the halt is the safety filter
refusing a commanded throttle, the harm numbers are the arithmetic of a
parked vehicle rather than evidence of avoidance. This script finds those
runs across the tracked result directories, classifies why each stopped, and
reports progress alongside the harm metrics so the two are read together.

A run counts as stopped when the median speed over its last quarter falls
below --stop-speed. It counts as stopping *early* when it last exceeded
1 m/s before --early-fraction of its elapsed time, which separates a vehicle
halted mid-course from one that simply finished and came to rest.

Classification uses the commanded and applied controls in `sim_diag.csv`:

  filter_refuses   the command source asks for throttle and the applied
                   throttle is near zero -- a filter or shield is holding the
                   vehicle
  no_command       the command source itself asks for no throttle
  no_response      throttle reaches the plant and the vehicle stays put

Usage:
    python benchmarking/audit_stalled_runs.py                  # all results
    python benchmarking/audit_stalled_runs.py --pattern 'safety_filter*'
    python benchmarking/audit_stalled_runs.py --csv out.csv    # per-run rows
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import math
import statistics
import sys
from pathlib import Path
from typing import Iterator, Optional

RESULTS = Path(__file__).resolve().parent / "results"

# Column names differ across generations of the diagnostic writer; the first
# name present in a row is the one used.
SPEED_KEYS = ("speed", "u_meas", "v")
TIME_KEYS = ("time", "sim_time")
APPLIED_THROTTLE_KEYS = ("throttle",)
COMMANDED_THROTTLE_KEYS = ("throttle_op", "throttle_cmd", "throttle")


def pick(rows, keys) -> Optional[str]:
    """First of ``keys`` that carries a value somewhere in ``rows``.

    Deciding from the first row alone would skip a channel that is blank at
    ``t = 0`` and populated later -- a command column before the first command
    arrives, for instance -- and silently downgrade the run to unclassified.
    """
    sample = rows[:200]
    for key in keys:
        if any(row.get(key) not in ("", None) for row in sample):
            return key
    return None


def median_of(rows, key, default=float("nan")) -> float:
    values = []
    for row in rows:
        try:
            values.append(abs(float(row[key])))
        except (TypeError, ValueError):
            continue
    return statistics.median(values) if values else default


def find_diag(run_dir: Path) -> Optional[Path]:
    """Locate a run's sim_diag.csv, which sits either in the run directory or
    one level down in the launcher's timestamped subdirectory."""
    direct = run_dir / "sim_diag.csv"
    if direct.exists():
        return direct
    for child in sorted(run_dir.iterdir()):
        if child.is_dir():
            candidate = child / "sim_diag.csv"
            if candidate.exists():
                return candidate
    return None


def iter_runs(result_dir: Path) -> Iterator[tuple[str, Path]]:
    raw = result_dir / "raw"
    if not raw.is_dir():
        return
    for run_dir in sorted(raw.iterdir()):
        if not run_dir.is_dir():
            continue
        diag = find_diag(run_dir)
        if diag is not None:
            yield run_dir.name, diag


def audit_run(diag: Path, stop_speed: float, early_fraction: float
              ) -> Optional[dict]:
    with diag.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 50:
        return None

    speed_key = pick(rows, SPEED_KEYS)
    time_key = pick(rows, TIME_KEYS)
    if speed_key is None or time_key is None:
        return None

    try:
        times = [float(r[time_key]) for r in rows]
        speeds = [abs(float(r[speed_key])) for r in rows]
    except (TypeError, ValueError):
        return None

    elapsed = times[-1] - times[0]
    if elapsed <= 0:
        return None

    tail = rows[int(len(rows) * 0.75):]
    tail_speed = median_of(tail, speed_key)
    stopped = tail_speed < stop_speed

    moving = [t for t, s in zip(times, speeds) if s > 1.0]
    last_moving = max(moving) if moving else times[0]
    stopped_early = (last_moving - times[0]) < early_fraction * elapsed

    distance = 0.0
    if "x" in rows[0] and "y" in rows[0]:
        try:
            for a, b in zip(rows, rows[1:]):
                distance += math.hypot(float(b["x"]) - float(a["x"]),
                                       float(b["y"]) - float(a["y"]))
        except (TypeError, ValueError):
            distance = float("nan")
    else:
        distance = float("nan")

    reason = ""
    if stopped:
        applied_key = pick(rows, APPLIED_THROTTLE_KEYS)
        commanded_key = pick(rows, COMMANDED_THROTTLE_KEYS)
        if applied_key and commanded_key and commanded_key != applied_key:
            applied = median_of(tail, applied_key, 0.0)
            commanded = median_of(tail, commanded_key, 0.0)
            if commanded <= 0.3:
                reason = "no_command"
            elif applied < 0.1:
                reason = "filter_refuses"
            else:
                reason = "no_response"
        else:
            reason = "unclassified"

    return dict(elapsed=elapsed, tail_speed=tail_speed, stopped=stopped,
                stopped_early=stopped_early, distance=distance, reason=reason)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--pattern", default="*",
                        help="glob over result-directory names")
    parser.add_argument("--stop-speed", type=float, default=0.30,
                        help="m/s; tail median below this counts as stopped")
    parser.add_argument("--early-fraction", type=float, default=0.70,
                        help="fraction of elapsed time; last motion before "
                             "this counts as stopping early")
    parser.add_argument("--min-runs", type=int, default=4,
                        help="skip result sets with fewer readable runs")
    parser.add_argument("--only-flagged", action="store_true",
                        help="report only sets where some run stops early")
    parser.add_argument("--csv", type=Path,
                        help="write one row per run to this path")
    args = parser.parse_args(argv)

    if not args.results_dir.is_dir():
        print(f"no results directory at {args.results_dir}", file=sys.stderr)
        return 1

    per_run = []
    summaries = []
    for result_dir in sorted(args.results_dir.iterdir()):
        if not result_dir.is_dir():
            continue
        if not fnmatch.fnmatch(result_dir.name, args.pattern):
            continue
        runs = []
        for run_name, diag in iter_runs(result_dir):
            verdict = audit_run(diag, args.stop_speed, args.early_fraction)
            if verdict is None:
                continue
            runs.append(verdict)
            per_run.append(dict(result_set=result_dir.name, run=run_name,
                                **verdict))
        if len(runs) < args.min_runs:
            continue
        early = sum(1 for r in runs if r["stopped_early"])
        ended_stopped = sum(1 for r in runs if r["stopped"])
        if args.only_flagged and early == 0:
            continue
        reasons = {}
        for r in runs:
            if r["stopped_early"] and r["reason"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        distances = [r["distance"] for r in runs if not math.isnan(r["distance"])]
        summaries.append(dict(
            name=result_dir.name, n=len(runs), early=early,
            ended=ended_stopped,
            distance=statistics.median(distances) if distances else float("nan"),
            reasons=reasons))

    summaries.sort(key=lambda s: s["early"] / max(s["n"], 1), reverse=True)
    print(f"{'result set':56s} {'runs':>5s} {'stop early':>11s} {'ended':>10s} "
          f"{'dist_med':>9s}  why")
    for s in summaries:
        pct = 100.0 * s["early"] / max(s["n"], 1)
        why = ", ".join(f"{k}={v}" for k, v in
                        sorted(s["reasons"].items(), key=lambda kv: -kv[1]))
        dist = "n/a" if math.isnan(s["distance"]) else f"{s['distance']:.1f}"
        epct = 100.0 * s["ended"] / max(s["n"], 1)
        print(f"{s['name'][:56]:56s} {s['n']:5d} {s['early']:5d} ({pct:3.0f}%) "
              f"{s['ended']:4d} ({epct:3.0f}%) {dist:>9s}  {why}")

    if args.csv:
        fieldnames = ["result_set", "run", "elapsed", "tail_speed", "stopped",
                      "stopped_early", "distance", "reason"]
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_run)
        print(f"\nper-run rows -> {args.csv} ({len(per_run)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
