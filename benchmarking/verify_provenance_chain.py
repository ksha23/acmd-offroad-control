#!/usr/bin/env python3
"""Verify that every trained artifact traces to a hash-matched training corpus.

A hash identifies a corpus but not the program that produced it, so this
verifier classifies each corpus by how far its provenance reaches. A DERIVED
corpus names the collector that produced it, by commit and binary hash, and is
therefore regenerable from source. A PRIMARY corpus is an archived input that
downstream work takes as given, because the exact program that produced it is
not recoverable and a rebuilt collector yields measurably different data.

Both categories support the reproducibility claim that matters, namely that
every published number can be regenerated from a hash-identified input.
Declaring the distinction explicitly is what keeps the weaker case visible:
each PRIMARY corpus must carry a written reason, so the exemption cannot be
granted silently or inherited by a new corpus.

The verifier checks three properties:

* every model's recorded `training_csv_sha256` resolves to a file on disk
  whose hash still matches, so no artifact silently depends on data that has
  changed since it was trained;
* every corpus is classified PRIMARY or DERIVED, and each DERIVED corpus
  carries a manifest naming its collector commit and binary;
* nothing is treated as DERIVED without that manifest to support it.

Usage:
  python benchmarking/verify_provenance_chain.py [--json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Corpora whose collector source is not in version control. Each entry must say
# why, so the exemption cannot be granted silently or inherited by accident.
PRIMARY_CORPORA = {
    "data/tire_rig/rate_v2_100k.csv": (
        "Collected 2026-07-21 18:02, about two hours before the collector was "
        "first committed (591e389, 20:20). The working state read slip from the "
        "rig rather than the tire; the vendored Chrono exposes no such accessor, "
        "so the committed predecessor (f8b057e) does not compile and the source "
        "is absent from the reflog and every dangling object. Rebuilt collectors "
        "produce data that trains measurably flatter surrogates, so this file is "
        "an archived input, not a regenerable one."
    ),
    "data/tire_rig_static/train.csv": (
        "Collected 2026-05-20 by the pre-repository SCM_Final single-tire rig, "
        "before this repository's collector-manifest convention existed; no "
        "collector commit or binary hash was recorded. Recovered 2026-08-03 "
        "from the original SCM_Final tree with an exact hash match to the "
        "checkpoint's recorded training_csv_sha256, and tracked in-repo since. "
        "Trains the frozen scalar-parent checkpoint tire_force_static_parent "
        "(the live scalar-arm force model and UKF references; the Table 2 "
        "parent replay itself evaluates tire_force_rate, matched to GRIT) "
        "and anchors the global analytical-baseline friction calibration "
        "PACEJKA_MU = 0.42 (benchmarking/calibrate_analytical_tires.py); "
        "nn_training/train_scalar_parent.sh reproduces the training end to end."
    ),
}

# Checkpoints exempt from recording a training_csv_sha256, each with the reason
# stated so the exemption is declared rather than silently accepted. This map is
# empty: every checkpoint in nn_models records the hash of its training corpus.
DECLARED_LEGACY_MODELS: dict[str, str] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    findings: list[dict] = []
    failures = 0

    # Hash every candidate corpus once. Doing it per model is quadratic and the
    # corpora are tens of megabytes each.
    by_hash: dict[str, Path] = {}
    for path in sorted(ROOT.rglob("*.csv")):
        if path.stat().st_size > 1_000_000 and "results" not in path.parts:
            by_hash.setdefault(sha256(path), path)

    for metrics_path in sorted((ROOT / "nn_models").glob("*/test_metrics.json")):
        model = metrics_path.parent.name
        payload = json.loads(metrics_path.read_text())
        recorded = payload.get("architecture", {}).get("training_csv_sha256")
        if not recorded:
            if model in DECLARED_LEGACY_MODELS:
                findings.append({"model": model, "status": "LEGACY",
                                 "detail": DECLARED_LEGACY_MODELS[model]})
            else:
                findings.append({"model": model, "status": "NO_HASH",
                                 "detail": "checkpoint records no training_csv_sha256"})
                failures += 1
            continue

        matches = [by_hash[recorded]] if recorded in by_hash else []
        if not matches:
            if model in DECLARED_LEGACY_MODELS:
                findings.append({"model": model, "status": "LEGACY",
                                 "detail": DECLARED_LEGACY_MODELS[model]})
            else:
                findings.append({"model": model, "status": "INPUT_MISSING",
                                 "detail": f"no file on disk hashes to {recorded[:16]}"})
                failures += 1
            continue

        corpus = matches[0].relative_to(ROOT).as_posix()
        manifest = matches[0].parent / "MANIFEST.json"
        if corpus in PRIMARY_CORPORA:
            findings.append({"model": model, "corpus": corpus, "status": "PRIMARY",
                             "detail": PRIMARY_CORPORA[corpus]})
        elif manifest.exists():
            man = json.loads(manifest.read_text())
            findings.append({"model": model, "corpus": corpus, "status": "DERIVED",
                             "detail": f"collector {man.get('repo_head','?')[:12]}, "
                                       f"binary {man.get('binary_sha256','?')[:16]}"})
        else:
            findings.append({"model": model, "corpus": corpus, "status": "UNDECLARED",
                             "detail": "no MANIFEST.json and not declared PRIMARY; "
                                       "the collector that produced it is unidentified"})
            failures += 1

    if args.json:
        print(json.dumps(findings, indent=2))
    else:
        width = max((len(f["model"]) for f in findings), default=5) + 2
        for f in findings:
            print(f"{f['status']:<12}{f['model']:<{width}}{f.get('corpus','-')}")
            if f["status"] in ("PRIMARY", "LEGACY", "UNDECLARED", "INPUT_MISSING", "NO_HASH"):
                print(f"{'':<12}  {f['detail'][:150]}")
        print(f"\n{len(findings)} artifacts, {failures} unresolved")
        print("PRIMARY and LEGACY are accepted, declared boundaries; "
              "UNDECLARED is not.")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
