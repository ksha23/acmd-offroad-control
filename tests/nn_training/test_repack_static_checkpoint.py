"""Identity and fail-closed contracts for the checkpoint provenance repack.

Attaching provenance to a checkpoint must not change the model it describes,
so these tests establish that the operation is deterministic, preserves every
state tensor, and produces an identical hash from identical inputs. They also
establish that conflicting or unverifiable provenance is refused rather than
overwritten, which is what makes the resulting record trustworthy.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from nn_training.repack_static_checkpoint import (
    PROVENANCE,
    repack_checkpoint,
    sha256_file,
    state_dict_sha256,
)


class StaticCheckpointRepackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.model_dir = cls.root / "nn_models" / "tire_force_static_parent"
        cls.committed_checkpoint = torch.load(
            cls.model_dir / "best_terrain_nn.pt",
            map_location="cpu",
            weights_only=True,
        )

    def _write_legacy_source(self, directory: Path) -> Path:
        legacy = {
            key: value
            for key, value in self.committed_checkpoint.items()
            if key not in PROVENANCE
        }
        source = directory / "legacy.pt"
        torch.save(legacy, source)
        return source

    def test_repack_is_deterministic_and_preserves_all_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = self._write_legacy_source(directory)
            scaler = directory / "scalers.pkl"
            shutil.copy2(self.model_dir / "scalers.pkl", scaler)
            scaler_sha_before = sha256_file(scaler)

            outputs = [directory / "first.pt", directory / "second.pt"]
            manifests = [directory / "first.json", directory / "second.json"]
            proofs = [
                repack_checkpoint(source, scaler, output, manifest)
                for output, manifest in zip(outputs, manifests, strict=True)
            ]

            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            self.assertEqual(
                outputs[0].read_bytes(),
                (self.model_dir / "best_terrain_nn.pt").read_bytes(),
            )
            self.assertEqual(proofs[0], proofs[1])
            self.assertEqual(
                json.loads(manifests[0].read_text()),
                json.loads(manifests[1].read_text()),
            )
            self.assertEqual(sha256_file(scaler), scaler_sha_before)
            self.assertEqual(proofs[0]["max_abs_prediction_difference"], 0.0)
            self.assertIs(proofs[0]["tensor_values_identical"], True)
            self.assertIs(proofs[0]["scaler_file_unchanged"], True)

            repacked = torch.load(
                outputs[0], map_location="cpu", weights_only=True
            )
            for key, expected in PROVENANCE.items():
                self.assertEqual(repacked[key], expected)
            self.assertEqual(
                state_dict_sha256(repacked["model_state_dict"]),
                state_dict_sha256(
                    self.committed_checkpoint["model_state_dict"]
                ),
            )
            for name, before in self.committed_checkpoint[
                "model_state_dict"
            ].items():
                self.assertTrue(
                    torch.equal(before, repacked["model_state_dict"][name]),
                    name,
                )

    def test_in_place_rerun_validates_without_rewriting_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "best_terrain_nn.pt"
            scaler = directory / "scalers.pkl"
            manifest = directory / "repack_manifest.json"
            for source, destination in (
                (self.model_dir / "best_terrain_nn.pt", checkpoint),
                (self.model_dir / "scalers.pkl", scaler),
                (self.model_dir / "repack_manifest.json", manifest),
            ):
                shutil.copy2(source, destination)
            checkpoint_before = checkpoint.read_bytes()
            manifest_before = manifest.read_bytes()

            proof = repack_checkpoint(
                checkpoint, scaler, checkpoint, manifest
            )

            self.assertEqual(checkpoint.read_bytes(), checkpoint_before)
            self.assertEqual(manifest.read_bytes(), manifest_before)
            self.assertEqual(
                proof["repacked_checkpoint_sha256"], sha256_file(checkpoint)
            )

    def test_conflicting_or_unbound_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scaler = directory / "scalers.pkl"
            shutil.copy2(self.model_dir / "scalers.pkl", scaler)
            conflicting = dict(self.committed_checkpoint)
            conflicting["training_source"] = "unbound_source"
            source = directory / "conflicting.pt"
            torch.save(conflicting, source)
            with self.assertRaisesRegex(ValueError, "conflicting training_source"):
                repack_checkpoint(
                    source,
                    scaler,
                    directory / "output.pt",
                    directory / "manifest.json",
                )


if __name__ == "__main__":
    unittest.main()
