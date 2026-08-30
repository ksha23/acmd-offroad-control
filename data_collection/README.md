# Tire-rig and validation data collection

The neural training path consists of two Chrono SCM `ChTireTestRig` collectors:

| File | Purpose |
| --- | --- |
| `collect_static_data.cpp` | Uniform-LHS single-tire force samples without finite-difference rate inputs. |
| `collect_rate_data.cpp` | Uniform-LHS single-tire force samples with slip, speed, and steering-rate inputs. |

Both write exact per-wheel SCM `(Fx, Fy, My, sinkage)` labels together with the
operating point and all six Bekker–Mohr parameters. `My` is the spin-axis
contact moment at the spindle, and `sinkage` is wheel penetration relative to
the initially undeformed rig surface. Mesh spacing is pinned to the deployed
`0.08` m through `--mesh-spacing` rather than sampled over `0.08`–`0.12`, so the
surrogate is trained on the grid the runtime uses.

`run_dallas_scm.py` is an evaluation collector rather than a neural-model
trainer. It records HMMWV state and IMU traces used to validate the four-wheel
projection of the tire-rig UKF. No collector here creates supervision for a
vehicle-level neural model.

## Building the collectors

`build_collectors.sh` compiles both directly against the Chrono build, without
a CMake tree.

```bash
data_collection/build_collectors.sh          # writes data_collection/build_rig_cmd/
```

It sets the `CHRONO_DATA_DIR` and `CHRONO_VEHICLE_DATA_DIR` definitions the
sources require. Override `CHRONO_FORK` or `EIGEN_INCLUDE` if either lives
elsewhere, and pass an output directory as the first argument to build
somewhere other than `build_rig_cmd/`.

`data_collection/build_rig_cmd/` holds the binaries that collected
`data/tire_rig_commanded`; `data/tire_rig_commanded/MANIFEST.json` records the
collector commit, the binary SHA-256, the build flags, and the per-split seeds.
Because any collector edit changes the data, each corpus is collected end to
end by a single binary.

## Training on a corpus

Train through `nn_training/train_variant.py`, or through the wrappers that
reproduce the published checkpoints:
`nn_training/train_deployed_surrogates.sh` for the two deployed surrogates, and
`nn_training/train_scalar_parent.sh` with
`nn_training/repack_static_checkpoint.py` for the scalar-parent comparison
model.

Rig corpora produced by different collector builds are not interchangeable.
Matched-operating-point comparisons between builds differ by more than the mean
force magnitude itself, so training on one collection and evaluating against
another measures the gap between collectors rather than generalisation. Train
and score each surrogate within a single corpus and its own held-out split.

Large CSV and NPZ inputs beyond the tracked corpora are restored with
`data_sync/data_sync.sh pull`, as documented in `DATA.md`.
