"""Contracts for the embedded tire-force network the NMPC evaluates.

The controller evaluates the force network in batches through CasADi, so
these tests establish that the batched path is numerically identical to the
scalar one at any batch width and that malformed feature rows are rejected.
They also establish that every checkpoint the loader accepts carries verified
single-tire rig provenance, and that a checkpoint missing it fails closed.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from simulation.estimators.terrain_parameterization import terrain_params_for_n
from simulation.tire_models.nn_tire_model import NNTireModel, load_nn_tire_model


class NNTireModelBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.model = load_nn_tire_model(
            cls.root / "nn_models" / "tire_force_static_parent",
            terrain_params_for_n(0.7),
        )

    def test_feature_row_batch_matches_scalar_casadi_path(self) -> None:
        rows = []
        expected = []
        for n_value, alpha in ((0.52, -0.12), (0.7, 0.03), (1.05, 0.18)):
            terrain = terrain_params_for_n(n_value)
            rows.append([
                0.05, alpha, 5.0, 5800.0, 0.1,
                terrain["Kphi"], terrain["Kc"], terrain["n"], terrain["c"],
                self.model.phi_feature_value(terrain["phi"]), terrain["k"],
            ])
            expected.append(self.model.predict_numeric(
                alpha=alpha,
                Fz=5800.0,
                u=5.0,
                kappa=0.05,
                n_terrain=n_value,
                steering_rate=0.1,
                terrain_params=terrain,
            ))
        actual = self.model.predict_feature_rows(np.asarray(rows))
        np.testing.assert_allclose(actual, np.asarray(expected), atol=1.0e-9)

    def test_casadi_predict_batch_matches_scalar_path(self) -> None:
        # The NMPC embeds ``predict_batch`` (the CasADi Function built by
        # _build_batch) into its dynamics; nothing else exercised it
        # numerically, and a feature-row swap inside _build_batch survived
        # mutation testing against the rest of this file. Evaluate the
        # compiled Function directly and pin it to the scalar path across
        # operating points that differ in every feature, so any row/ordering
        # mutation shifts at least one output.
        model = self.model
        count = model._BATCH
        terrain = terrain_params_for_n(0.62)
        alphas = np.linspace(-0.2, 0.25, count)
        loads = np.linspace(4200.0, 7000.0, count)
        us = np.linspace(2.0, 9.0, count)
        kappas = np.linspace(-0.3, 0.35, count)
        n_ts = np.linspace(0.5, 1.05, count)
        srs = np.linspace(-0.2, 0.2, count)
        phi_feat = model.phi_feature_value(terrain["phi"])
        fx_b, fy_b = model.predict_batch(
            alphas, loads, us, kappas, n_ts, srs,
            terrain["Kphi"], terrain["Kc"], terrain["c"], phi_feat,
            terrain["k"])
        fx_b = np.asarray(fx_b).ravel()
        fy_b = np.asarray(fy_b).ravel()
        for i in range(count):
            fx_s, fy_s = model.predict_numeric(
                alpha=float(alphas[i]), Fz=float(loads[i]), u=float(us[i]),
                kappa=float(kappas[i]), n_terrain=float(n_ts[i]),
                steering_rate=float(srs[i]), terrain_params=terrain)
            self.assertAlmostEqual(fx_b[i], fx_s, places=6)
            self.assertAlmostEqual(fy_b[i], fy_s, places=6)

    def test_feature_contract_rejects_wrong_width_and_nonfinite_rows(self) -> None:
        with self.assertRaises(ValueError):
            self.model.predict_feature_rows(np.ones((2, 10)))
        bad = np.ones((2, 11))
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.model.predict_feature_rows(bad)

    def test_static_checkpoint_has_embedded_rig_provenance(self) -> None:
        checkpoint = torch.load(
            self.model.model_dir / "best_terrain_nn.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.assertEqual(checkpoint["checkpoint_format"], "tire_force_static_mlp")
        self.assertEqual(
            checkpoint["training_source"], "chrono_scm_single_tire_rig"
        )
        self.assertEqual(
            checkpoint["training_csv_sha256"],
            "926938c0f44e6c3914a4e1e99f05d304ca710a8690188ada1c014aa7fad5923c",
        )
        self.assertEqual(checkpoint["seed"], 42)

    def test_active_rate_checkpoint_passes_provenance_contract(self) -> None:
        model = NNTireModel(
            self.root / "nn_models" / "tire_force_rate",
            terrain_params_for_n(0.7),
        )
        self.assertEqual(model.model_format, "tire_force_rate_mlp")
        self.assertEqual(model.input_dim, 14)

    def test_loader_fails_closed_on_missing_or_invalid_provenance(self) -> None:
        source = self.model.model_dir
        base = torch.load(
            source / "best_terrain_nn.pt", map_location="cpu", weights_only=True
        )
        cases = (
            (
                "missing_format",
                lambda checkpoint: checkpoint.pop("checkpoint_format"),
                "missing required rig checkpoint metadata",
            ),
            (
                "wrong_format",
                lambda checkpoint: checkpoint.__setitem__(
                    "checkpoint_format", "tire_force_rate_mlp"
                ),
                "format does not match",
            ),
            (
                "wrong_source",
                lambda checkpoint: checkpoint.__setitem__(
                    "training_source", "whole_vehicle_trace"
                ),
                "not provenance-bound",
            ),
            (
                "invalid_hash",
                lambda checkpoint: checkpoint.__setitem__(
                    "training_csv_sha256", "ABC123"
                ),
                "64 lowercase hex digits",
            ),
            (
                "invalid_seed",
                lambda checkpoint: checkpoint.__setitem__("seed", True),
                "nonnegative integer",
            ),
            (
                "invalid_output",
                lambda checkpoint: checkpoint.__setitem__("output_size", 3),
                "two tire forces",
            ),
        )
        terrain = terrain_params_for_n(0.7)
        for name, mutate, pattern in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                model_dir = Path(temporary) / f"invalid_{name}"
                model_dir.mkdir()
                shutil.copy2(source / "scalers.pkl", model_dir / "scalers.pkl")
                checkpoint = dict(base)
                mutate(checkpoint)
                torch.save(checkpoint, model_dir / "best_terrain_nn.pt")
                with self.assertRaisesRegex(ValueError, pattern):
                    NNTireModel(model_dir, terrain)

    def test_batch_width_has_no_dead_rows(self) -> None:
        """Every batch row the solver builds must be consumed by it.

        The NMPC evaluates the force MLP as one batch per right-hand-side call
        and reads the result by index. A batch wider than the highest index
        read is dead work, because the network and its sensitivities are
        evaluated for rows nothing consumes, and that cost falls directly on
        the reported solve time. This test pins batch width to the indices the
        solver actually references, so the two cannot drift apart.
        """
        source = (Path(__file__).resolve().parents[2]
                  / "simulation" / "control" / "acados_mpc_solver.py").read_text()
        indices = [int(i) for i in re.findall(r"F[xy]s_all\[(\d+)\]", source)]
        self.assertTrue(indices, "found no batch consumers to check")
        self.assertEqual(
            max(indices) + 1, NNTireModel._BATCH,
            f"solver consumes rows 0..{max(indices)} but the batch is "
            f"{NNTireModel._BATCH} wide; widen the consumers or narrow _BATCH",
        )


if __name__ == "__main__":
    unittest.main()
