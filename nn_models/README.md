# Tire-rig neural checkpoints

Every checkpoint in this directory is supervised by the controlled single-tire
Chrono SCM rig alone: no vehicle run, vehicle trace, or plant terrain-truth
channel enters any learned component of the control and estimation stack.

| Directory | Role |
| --- | --- |
| `tire_force_static/` | Deployed static (steering-rate-only) tire-force map at the commanded operating point: NMPC internal model, DOB-CBF, MPSF, speed profile, and force adaptation. |
| `tire_force_rate/` | Force surrogate the GRIT joint estimator (backend identifier `grit`) evaluates over the independent (n, phi) grid. |
| `tire_force_static_parent/` | Scalar-parent comparison model (backend identifier `scalar_parent`, Table 2). |

Each directory holds the weights (`best_terrain_nn.pt`), the input and output
scalers (`scalers.pkl`), the recorded scores (`test_metrics.json`), and a
`TRAINING_METADATA.md` giving the architecture, the input list, the training
and holdout CSVs with their SHA-256 hashes, and the seed.

`nn_training/train_deployed_surrogates.sh` rebuilds both deployed checkpoints
end to end. It runs `nn_training/train_variant.py` on
`data/tire_rig_commanded` — a DERIVED corpus collected by the tracked
`data_collection/build_rig_cmd/` binaries, whose sources are
`data_collection/collect_static_data.cpp` and
`data_collection/collect_rate_data.cpp` — at the commanded operating point with
the recorded seed, and scores against the tracked independent holdout.
Re-running it reproduces `tire_force_static`'s published metrics exactly and
`tire_force_rate`'s to within training nondeterminism in the fourth decimal.

`nn_training/train_scalar_parent.sh`, together with
`nn_training/repack_static_checkpoint.py`, rebuilds `tire_force_static_parent`
from the tracked PRIMARY corpus `data/tire_rig_static/train.csv`.

Every checkpoint records its `training_csv_sha256`, which
`benchmarking/verify_provenance_chain.py` resolves against the corpora on disk.
No checkpoint receives realized plant soil values, vehicle-run labels, or a
Chrono terrain-truth channel at runtime.
