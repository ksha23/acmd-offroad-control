#!/usr/bin/env python3
"""Acceptance gate for a candidate joint terrain-estimator configuration.

A candidate configuration may replace the published one only by passing every
endpoint below on a fresh seed. The estimator accuracy the manuscript reports
is therefore the accuracy of a configuration selected against criteria fixed
before its data existed, not one chosen after inspecting the outcome. This
module is frozen together with those criteria, and the digest of the
protocol document is recorded in the decision file it writes.

Both arms replay the identical sensor traces; plant truth is joined here, once
inference is complete. A trace on which an estimator never accepts a snapshot
is scored at the labelled control fallback (n=0.50, phi=13 deg) and counted as
unpublished, because that is the value control would have used.

The seven endpoints, all of which must pass:

1. Soft-band reduction. Over soils with ``n`` below the soft-band edge, the
   candidate's mean absolute ``n`` error is lower than the published arm's,
   and a one-sided Wilcoxon signed-rank test rejects at alpha 0.05.
2. Wide-band monotonicity. The candidate's Spearman correlation between true
   and estimated ``n`` is at least the published arm's, so a soft-band gain
   may not be bought by losing rank ordering elsewhere.
3. Sign-aware interior bias. The candidate's mean signed ``n`` error on
   interior soils lies within an asymmetric band. The band is wider on the
   soft side because underestimating ``n`` yields a conservative grip
   assumption, whereas overestimating it does not.
4. Interior accuracy. The candidate's interior ``n`` MAE does not exceed the
   published arm's by more than a fixed margin.
5. Friction-angle guard. The candidate's ``phi`` MAE does not exceed the
   published arm's by more than a fixed ratio, so an improvement in ``n`` may
   not be traded against the other estimated coordinate.
6. Publication rate. Both arms publish on at least the minimum fraction of
   traces, which keeps the comparison between two estimators that are
   actually available to control.
7. Validity. The soft band contains enough traces for endpoint 1 to be
   meaningful.
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
INTERIOR_N_MIN = 0.58
INTERIOR_N_MAX = 0.95
MIN_SOFT_TRACES = 25
WILCOXON_ALPHA = 0.05
FIRM_INTERIOR_CAP = 0.05
SOFT_INTERIOR_CAP = -0.15
INTERIOR_MAE_MARGIN = 0.05
PHI_MAE_RATIO_LIMIT = 1.05
MIN_PUBLISH_RATE = 0.95
FALLBACK_N = 0.50
FALLBACK_PHI_DEG = 13.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_arm(directory: Path, truth: pd.DataFrame, label: str) -> pd.DataFrame:
    estimates = pd.read_csv(directory / "estimates.csv")
    estimates = estimates.copy()
    failed_mask = estimates["status"] != "ok"
    estimates.loc[failed_mask, "final_est_n"] = FALLBACK_N
    estimates.loc[failed_mask, "final_est_phi_deg"] = FALLBACK_PHI_DEG
    estimates.loc[failed_mask, "final_snapshot_was_published"] = False
    merged = truth.merge(
        estimates[
            [
                "trace_id",
                "final_est_n",
                "final_est_phi_deg",
                "final_snapshot_was_published",
                "status",
            ]
        ],
        on="trace_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(truth):
        raise ValueError(
            f"{label}: replay covers {len(merged)} of {len(truth)} truth traces"
        )
    merged["failed_cell"] = merged["status"] != "ok"
    merged["n_err"] = merged["final_est_n"] - merged["n_true"]
    merged["phi_abs_err"] = (
        merged["final_est_phi_deg"] - merged["phi_true_deg"]
    ).abs()
    merged["published"] = (
        merged["final_snapshot_was_published"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "1.0"})
    )
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--published-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs/JOINT_N_PHI_ATTEMPT4_PREREGISTRATION.md",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    truth = pd.read_csv(
        args.collection_dir / "truth.csv", float_precision="round_trip"
    )
    candidate = _load_arm(args.candidate_dir, truth, "candidate")
    published = _load_arm(args.published_dir, truth, "published")

    soft_mask = truth["n_true"] < SOFT_BAND_N_MAX
    interior_mask = (truth["n_true"] >= INTERIOR_N_MIN) & (
        truth["n_true"] <= INTERIOR_N_MAX
    )
    soft_count = int(soft_mask.sum())
    validity_pass = soft_count >= MIN_SOFT_TRACES

    # 1 soft-band paired reduction
    soft_pub = published.loc[soft_mask.values, "n_err"].abs().to_numpy()
    soft_cand = candidate.loc[soft_mask.values, "n_err"].abs().to_numpy()
    wilcoxon = stats.wilcoxon(
        soft_pub, soft_cand, alternative="greater", zero_method="zsplit"
    )
    wilcoxon_p = float(wilcoxon.pvalue)
    e1 = bool(np.mean(soft_cand) < np.mean(soft_pub) and wilcoxon_p < WILCOXON_ALPHA)

    # 2 wide-band Spearman
    pub_ok = published[~published["failed_cell"]]
    rho_candidate = float(
        stats.spearmanr(candidate["n_true"], candidate["final_est_n"]).statistic
    )
    rho_published = float(
        stats.spearmanr(pub_ok["n_true"], pub_ok["final_est_n"]).statistic
    )
    e2 = bool(rho_candidate >= rho_published)

    # 3 sign-aware interior floors
    interior_bias = float(candidate.loc[interior_mask.values, "n_err"].mean())
    e3 = bool(SOFT_INTERIOR_CAP <= interior_bias <= FIRM_INTERIOR_CAP)

    # 4 interior MAE bound
    interior_mae_candidate = float(
        candidate.loc[interior_mask.values, "n_err"].abs().mean()
    )
    pub_ok_interior = pub_ok[
        (pub_ok["n_true"] >= INTERIOR_N_MIN) & (pub_ok["n_true"] <= INTERIOR_N_MAX)
    ]
    interior_mae_published = float(pub_ok_interior["n_err"].abs().mean())
    e4 = bool(
        interior_mae_candidate <= interior_mae_published + INTERIOR_MAE_MARGIN
    )

    # 5 phi MAE guard
    phi_mae_candidate = float(candidate["phi_abs_err"].mean())
    phi_mae_published = float(pub_ok["phi_abs_err"].mean())
    e5 = bool(phi_mae_candidate <= phi_mae_published * PHI_MAE_RATIO_LIMIT)

    # 6 publish rates
    publish_rate_candidate = float(candidate["published"].mean())
    publish_rate_published = float(published["published"].mean())
    e6 = bool(
        publish_rate_candidate >= MIN_PUBLISH_RATE
        and publish_rate_published >= MIN_PUBLISH_RATE
    )

    accept = bool(validity_pass and e1 and e2 and e3 and e4 and e5 and e6)

    decision = {
        "preregistration_sha256": _sha256_file(args.preregistration),
        "truth_sha256": _sha256_file(args.collection_dir / "truth.csv"),
        "trace_count": int(len(truth)),
        "soft_trace_count": soft_count,
        "candidate_failed_cells": sorted(
            candidate.loc[candidate["failed_cell"], "trace_id"]
        ),
        "published_failed_cells": sorted(
            published.loc[published["failed_cell"], "trace_id"]
        ),
        "endpoints": {
            "1_soft_band_reduction": {
                "soft_abs_err_published": float(np.mean(soft_pub)),
                "soft_abs_err_candidate": float(np.mean(soft_cand)),
                "soft_signed_err_published": float(
                    published.loc[soft_mask.values, "n_err"].mean()
                ),
                "soft_signed_err_candidate": float(
                    candidate.loc[soft_mask.values, "n_err"].mean()
                ),
                "wilcoxon_p_one_sided": wilcoxon_p,
                "pass": e1,
            },
            "2_wide_band_spearman": {
                "rho_candidate": rho_candidate,
                "rho_published_ok_cells": rho_published,
                "pass": e2,
            },
            "3_sign_aware_interior": {
                "interior_mean_signed_err": interior_bias,
                "firm_cap": FIRM_INTERIOR_CAP,
                "soft_cap": SOFT_INTERIOR_CAP,
                "pass": e3,
            },
            "4_interior_mae_bound": {
                "interior_mae_candidate": interior_mae_candidate,
                "interior_mae_published": interior_mae_published,
                "margin": INTERIOR_MAE_MARGIN,
                "pass": e4,
            },
            "5_phi_mae": {
                "phi_mae_candidate_deg": phi_mae_candidate,
                "phi_mae_published_deg": phi_mae_published,
                "limit_ratio": PHI_MAE_RATIO_LIMIT,
                "pass": e5,
            },
            "6_publish_rate": {
                "publish_rate_candidate": publish_rate_candidate,
                "publish_rate_published": publish_rate_published,
                "minimum": MIN_PUBLISH_RATE,
                "pass": e6,
            },
            "7_validity_gate": {
                "soft_trace_count": soft_count,
                "minimum": MIN_SOFT_TRACES,
                "pass": validity_pass,
            },
        },
        "accept": accept,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"ATTEMPT4_DECISION accept={accept} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
