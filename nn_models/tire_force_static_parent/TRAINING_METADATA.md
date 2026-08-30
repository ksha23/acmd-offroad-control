# Controlled-rig instantaneous force map

This 946-parameter MLP, with hidden widths 32 and 16, is the instantaneous
controlled-rig force map used by the force-grid and force-UKF comparisons and
by the sinkage--dynamics estimator. Its 11 inputs are slip ratio, slip angle,
speed, vertical load, steering rate, and the six Bekker--Mohr soil parameters;
its outputs are longitudinal and lateral tire force. Here, *static* means that
the model has no finite-difference history inputs. Steering rate remains an
explicit measured rig coordinate.

The checkpoint is supervised only by the controlled single-tire Chrono SCM
test rig in `data/tire_rig_static/train.csv`, collected by
`data_collection/collect_static_data.cpp`. No whole-vehicle trace or
whole-vehicle force label is used. The collector uses a uniform
Latin-hypercube design over tire operating conditions and soil parameters.
The training CSV SHA-256 is
`926938c0f44e6c3914a4e1e99f05d304ca710a8690188ada1c014aa7fad5923c`.

Reproduce the fit with:

```bash
nn_training/train_scalar_parent.sh
```

The wrapper invokes `nn_training/train_variant.py` in `static` mode with
hidden widths 32 and 16, 300 epochs, learning rate 0.01, patience 50, batch
size 256, and seed 42. The stored held-out metrics are $R^2=0.9872$ for
$F_x$, $R^2=0.9865$ for $F_y$, RMSE 204.6 N for $F_x$, and RMSE 148.3 N for
$F_y$.

The original learned tensors and scalers predated the embedded provenance
schema. `nn_training/repack_static_checkpoint.py` performs the deterministic,
metadata-only migration to `tire_force_static_mlp`. The committed
`repack_manifest.json` binds the legacy and repacked checkpoint hashes, the
unchanged scaler and tensor hashes, and an exact before/after prediction probe.
No parameter was retrained or numerically changed by that migration.
