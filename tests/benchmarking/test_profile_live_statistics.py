"""Fail-closed checks for the canonical profile-live statistics artifact.

The source generations are read from the committed publish manifest (the
same selection `data_sync/data_snapshot.list` backs up), so this test runs
on a clean restore of the documented backup set and never drifts from the
published selection. It guards the committed canonical artifact by
recomputing it from those sources and asserting byte-equality.
"""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from benchmarking.profile_live_statistics import (
    build_profile_live_statistics,
    canonical_json,
    source_input_hashes,
    validate_profile_live_statistics,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarking" / "results"
CANONICAL = ROOT / "my_paper" / "paper_figures" / "profile_live_statistics.json"
PUBLISH_MANIFEST = ROOT / "my_paper" / "paper_figures" / "publish_manifest.json"


def _published_source(prefix: str) -> Path:
    for entry in json.loads(PUBLISH_MANIFEST.read_text()):
        if entry.get("prefix") == prefix and entry.get("source_dir"):
            return ROOT / entry["source_dir"]
    raise RuntimeError(f"publish manifest records no source for {prefix!r}")


CONDITIONING = _published_source("tire_model_with_estimator_ablation")


class ProfileLiveStatisticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = build_profile_live_statistics(CONDITIONING, root=ROOT)

    def test_recomputation_matches_committed_canonical_artifact(self):
        # The published artifact must be exactly reproducible from the pinned
        # fresh sources -- this is the reproducibility contract for every paper
        # value drawn from profile_live_statistics.json.
        committed = json.loads(CANONICAL.read_text())
        self.assertEqual(canonical_json(self.payload), canonical_json(committed))

        # Confirm the recomputation produced the joint-estimator schema, so a
        # payload built under a different schema cannot pass this test by
        # comparing equal against an equally stale committed artifact.
        conditioning = self.payload["studies"]["conditioning"]
        self.assertEqual(
            set(conditioning["comparisons"]),
            {
                "joint_vs_fixed_fallback",
                "joint_vs_historical_scalar_parent",
                "joint_vs_matched_terrain_oracle",
            },
        )
        self.assertEqual(set(self.payload["studies"]), {"conditioning"})

    def test_sources_are_exactly_hashed_and_canonical_json_is_stable(self):
        hashes = source_input_hashes(self.payload)
        self.assertEqual(len(hashes), 3)
        expected_results = CONDITIONING / "results.csv"
        relative = str(expected_results.relative_to(ROOT))
        self.assertEqual(
            hashes[relative], hashlib.sha256(expected_results.read_bytes()).hexdigest()
        )
        self.assertEqual(canonical_json(self.payload), canonical_json(self.payload))

    def test_exact_recomputation_accepts_artifact_and_rejects_tampering(self):
        validate_profile_live_statistics(self.payload, CONDITIONING, root=ROOT)
        tampered = copy.deepcopy(self.payload)
        tampered["studies"]["conditioning"]["comparisons"][
            "joint_vs_fixed_fallback"
        ]["treatment_wins"] += 1
        with self.assertRaisesRegex(RuntimeError, "do not match"):
            validate_profile_live_statistics(tampered, CONDITIONING, root=ROOT)


if __name__ == "__main__":
    unittest.main()
