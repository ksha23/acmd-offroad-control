#!/usr/bin/env python3
"""Score the delay-awareness ablation behind the manuscript's safety-filter claim.

The ablation asks whether telling the safety filter the one-way command delay
improves the clearance it achieves. Its design is five convoy scenarios by two
command delays by three throttle intentions by three filter arms, and this
scorer reduces that matrix to the endpoints below.

The protocol is fixed in advance of the contrast, and this module states it in
full so the analysis cannot be chosen after seeing the data.

Primary endpoint: paired minimum clearance, ``dob_aware`` minus ``dob_blind``,
over the dynamic scenarios alone, by two-sided Wilcoxon signed-rank test at
alpha 0.05. Superiority additionally requires a median improvement of at least
0.25 m, so that a difference too small to matter operationally is reported as
detectable and practically negligible rather than as a benefit. Direction is
not assumed: a significant result favouring the blind arm is an overcorrection
finding and is reported as one.

The stationary scenarios and ``rear_approach`` are negative controls, chosen
because the delay compensation has no authority in either. They may not be
pooled into the primary endpoint, and an effect appearing in them indicates a
confound rather than a benefit.

Usage:
  python benchmarking/score_latency_ablation.py <result_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

# The dynamic set is defined by the mechanism's own property: closest approach
# occurs at nonzero speed, so the speed-proportional standoff term the delay
# compensation adds has authority in these scenarios and vanishes elsewhere.
DYNAMIC = ("oncoming", "jam", "double_cut")
SUPERIORITY_MEDIAN_M = 0.25
ALPHA = 0.05


def paired(frame: pd.DataFrame, scenarios: tuple[str, ...]) -> pd.DataFrame:
    sub = frame[frame.convoy.isin(scenarios)]
    aware = sub[sub.variant == "dob_aware"].set_index("cell")
    blind = sub[sub.variant == "dob_blind"].set_index("cell")
    both = aware.join(blind, lsuffix="_aware", rsuffix="_blind", how="inner")
    return both


def report(frame: pd.DataFrame, label: str, primary: bool) -> None:
    if frame.empty:
        print(f"\n## {label}: no paired cells")
        return
    delta = frame.min_clearance_m_aware - frame.min_clearance_m_blind
    print(f"\n## {label}  (n = {len(frame)} paired cells)")
    print(f"   clearance  aware {frame.min_clearance_m_aware.mean():+.3f} m   "
          f"blind {frame.min_clearance_m_blind.mean():+.3f} m")
    print(f"   paired delta (aware - blind): mean {delta.mean():+.3f} m, "
          f"median {delta.median():+.3f} m, aware better in "
          f"{int((delta > 0).sum())}/{len(delta)}")
    if delta.abs().sum() == 0:
        print("   Wilcoxon: undefined, every pair identical")
        return
    p = wilcoxon(delta).pvalue
    print(f"   Wilcoxon signed-rank: p = {p:.4f}")
    if not primary:
        verdict = "effect present (CONFOUND)" if p < ALPHA else "null, as expected"
        print(f"   control verdict: {verdict}")
        return
    if p >= ALPHA:
        print(f"   VERDICT: null. Delay awareness does not change closest "
              f"approach at alpha = {ALPHA}. The filter is delay-robust across "
              f"the modelled channel without being told the delay.")
    elif delta.median() >= SUPERIORITY_MEDIAN_M:
        print(f"   VERDICT: superiority. Delay awareness improves closest "
              f"approach by {delta.median():+.3f} m median, above the "
              f"{SUPERIORITY_MEDIAN_M} m threshold fixed in advance.")
    elif delta.median() <= -SUPERIORITY_MEDIAN_M:
        print(f"   VERDICT: blind superior by {-delta.median():.3f} m median. "
              f"Forward prediction over a delayed channel overcorrects.")
    else:
        print(f"   VERDICT: detectable but negligible. Median {delta.median():+.3f} m "
              f"is inside the {SUPERIORITY_MEDIAN_M} m practical threshold.")


def main() -> None:
    directory = Path(sys.argv[1])
    frame = pd.read_csv(directory / "results.csv")
    bad = frame[frame.status != "ok"]
    if len(bad):
        print(f"WARNING: {len(bad)} non-ok rows; the preregistration requires "
              f"re-running failed cells at the same design.")
    frame = frame[frame.status == "ok"]

    print(f"=== delay-awareness ablation: {directory.name} ===")
    print(f"cells per arm: {frame.groupby('variant').size().to_dict()}")

    print("\n--- secondary: collisions and mobility by arm ---")
    for label, scen in (("dynamic", DYNAMIC),
                        ("controls", tuple(s for s in frame.convoy.unique()
                                           if s not in DYNAMIC))):
        sub = frame[frame.convoy.isin(scen)]
        if sub.empty:
            continue
        agg = sub.groupby("variant").agg(
            collisions=("collided", "sum"), cells=("collided", "size"),
            clearance=("min_clearance_m", "mean"),
            progress=("progress_x_m", "mean"),
            dsteer=("mean_abs_dsteer", "mean"))
        print(f"\n  [{label}: {', '.join(sorted(set(scen)))}]")
        print(agg.round(3).to_string())

    report(paired(frame, DYNAMIC), "PRIMARY ENDPOINT (dynamic scenarios)", True)
    controls = tuple(s for s in frame.convoy.unique() if s not in DYNAMIC)
    for scenario in controls:
        report(paired(frame, (scenario,)), f"control: {scenario}", False)


if __name__ == "__main__":
    main()
