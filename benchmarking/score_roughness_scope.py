#!/usr/bin/env python3
"""Confirm the terrain estimator's scope on rough ground.

The manuscript's estimator accuracy is measured on smooth terrain. This
module establishes how far that result carries onto rough ground, by
replaying the estimator over a rough-terrain trace matrix and testing it
against absolute limits derived from the smooth-terrain evidence.

The test is single-arm non-inferiority: there is no comparator, and every
endpoint is a fixed cap or floor rather than a relative comparison, so the
rough-terrain result stands or falls on its own. The caps are frozen before
any rough-terrain data exists, and the digest of the protocol document is
recorded in the decision file this module writes.

The six endpoints, all of which must pass: mean signed ``n`` error over the
soft band stays below its cap, so the estimator does not become optimistic on
soft soil; ``n`` MAE and ``phi`` MAE stay below theirs; the Spearman
correlation between true and estimated ``n`` stays above its floor; the
publication rate stays above its minimum; and the soft band holds enough
traces for the first endpoint to be meaningful.

Traces on which the estimator never accepts a snapshot are scored at the
labelled control fallback (n=0.50, phi=13 deg) rather than dropped, because
that is the value control would have used. Any failure whose recorded reason
is something other than the absence of an accepted snapshot aborts scoring,
so a genuine defect cannot be absorbed into the fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SOFT_BAND_N_MAX = 0.58
MIN_SOFT_TRACES = 25
SOFT_SIGNED_CAP = 0.08
N_MAE_CAP = 0.134
PHI_MAE_CAP_DEG = 2.23
N_SPEARMAN_FLOOR = 0.80
MIN_PUBLISH_RATE = 0.90
FALLBACK_N = 0.50
FALLBACK_PHI_DEG = 13.0
NO_ACCEPT_TOKEN = "no accepted snapshot"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs/JOINT_N_PHI_B4_SCOPE_PREREGISTRATION.md",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    truth = pd.read_csv(
        args.collection_dir / "truth.csv", float_precision="round_trip"
    )
    est = pd.read_csv(args.replay_dir / "estimates.csv")
    failed = est[est["status"] != "ok"]
    bad_reason = failed[
        ~failed["failure"].astype(str).str.contains(NO_ACCEPT_TOKEN)
    ]
    if len(bad_reason):
        raise SystemExit(
            "non-declarable failures present: "
            + ", ".join(bad_reason["trace_id"].astype(str))
        )
    est = est.copy()
    mask = est["status"] != "ok"
    est.loc[mask, "final_est_n"] = FALLBACK_N
    est.loc[mask, "final_est_phi_deg"] = FALLBACK_PHI_DEG
    est.loc[mask, "final_snapshot_was_published"] = False
    m = truth.merge(
        est[
            [
                "trace_id",
                "final_est_n",
                "final_est_phi_deg",
                "final_snapshot_was_published",
            ]
        ],
        on="trace_id",
        validate="one_to_one",
    )
    if len(m) != len(truth):
        raise SystemExit("replay does not cover the truth matrix")

    n_err = m["final_est_n"] - m["n_true"]
    phi_err = (m["final_est_phi_deg"] - m["phi_true_deg"]).abs()
    soft = m["n_true"] < SOFT_BAND_N_MAX
    soft_count = int(soft.sum())
    published = (
        m["final_snapshot_was_published"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "1.0"})
    )

    soft_signed = float(n_err[soft].mean()) if soft_count else float("nan")
    n_mae = float(n_err.abs().mean())
    phi_mae = float(phi_err.mean())
    rho = float(stats.spearmanr(m["n_true"], m["final_est_n"]).statistic)
    publish_rate = float(published.mean())

    endpoints = {
        "1_soft_signed": {
            "value": soft_signed,
            "cap": SOFT_SIGNED_CAP,
            "pass": bool(soft_count and soft_signed <= SOFT_SIGNED_CAP),
        },
        "2_n_mae": {
            "value": n_mae,
            "cap": N_MAE_CAP,
            "pass": bool(n_mae <= N_MAE_CAP),
        },
        "3_phi_mae_deg": {
            "value": phi_mae,
            "cap": PHI_MAE_CAP_DEG,
            "pass": bool(phi_mae <= PHI_MAE_CAP_DEG),
        },
        "4_n_spearman": {
            "value": rho,
            "floor": N_SPEARMAN_FLOOR,
            "pass": bool(rho >= N_SPEARMAN_FLOOR),
        },
        "5_publish_rate": {
            "value": publish_rate,
            "minimum": MIN_PUBLISH_RATE,
            "pass": bool(publish_rate >= MIN_PUBLISH_RATE),
        },
        "6_validity": {
            "soft_trace_count": soft_count,
            "minimum": MIN_SOFT_TRACES,
            "pass": bool(soft_count >= MIN_SOFT_TRACES),
        },
    }
    accept = bool(all(entry["pass"] for entry in endpoints.values()))
    decision = {
        "preregistration_sha256": _sha256_file(args.preregistration),
        "truth_sha256": _sha256_file(args.collection_dir / "truth.csv"),
        "trace_count": int(len(m)),
        "never_accepting_cells": sorted(
            est.loc[mask, "trace_id"].astype(str)
        ),
        "endpoints": endpoints,
        "accept": accept,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"SCOPE_DECISION accept={accept} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
