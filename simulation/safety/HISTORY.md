# Safety filter history

`README.md` describes the filters as they are. This note
records what they replaced, and why. Every earlier version is in the git
history.

## As published (tag `acmd2026-published`)

The ACMD 2026 paper's filters, at `git show acmd2026-published:simulation/safety/README.md`.

* `dob_cbf`: a pointwise, second-order barrier QP on an ellipse around the
  vehicle (weights along and across the heading, a forward shift of the centre)
  with a disturbance observer. Under a teleop delay it inflated each obstacle by
  the distance covered over the round trip and predicted forward over the queue
  of past commands. Its gains, weights, forward shift and obstacle buffer were
  hand-set.
* `mpsf`: a kinematic-bicycle OCP over 20 x 0.1 s with braking and cornering
  authority from the rig surrogate, one footprint circle (1.0 m radius plus a
  1.6 m forward extent), a per-stage speed funnel tied to the horizon length,
  and the initial state advanced over the command round trip.

## The reformulation (branch `filter-fixes`, 2026-11)

Each step was a flag beside the published filters until the last one replaced
them.

1. **Terminal set** (`79733d5`, `37c181d`, `bbd6433`, `6504026`). The MPSF's
   safety depended on its horizon through the speed funnel. A plan ending in a
   braking set, a check of every executed command by straight braking, a
   friction ellipse and the plant's pedal rate made it horizon-independent.
2. **Textbook filters** (`4a97d42`, `a84f9ad`, `b8345a9`, `3700a78`). A DOB-CBF
   with a distance barrier, braking-distance and sensing-edge barriers, and an
   observer bound in every row, and an MPSF on the NMPC's surrogate dynamic
   model with an error tube, set-based traffic prediction and no delay advance.
   A plan checked from an advanced state and then run from the real one is not
   the plan that was checked, and in the failure-mode battery that let the
   vehicle stop against a rock.
3. **Grade preview** (`4e604c6`, `3406212`, `f4e7c48`). Gravity along the
   heading from an IMU, an oracle map or a lidar-like map.
4. **No hand-set numbers** (`c18b386`). Every number from the plant's spec, the
   rig surrogate or bounds identified on separate plant manoeuvres: the plant's
   powertrain, drawbar pull with the soil's motion resistance, the pedal as a
   state at its rate limit, and identified stop, drift, tube and observer
   bounds with a held-out half.
5. **The body rectangle** (`4ce9bde`, `9098205`, `1db8233`). The footprint and
   other vehicles as the HMMWV's collision rectangle instead of discs, the tube
   growing with the open-loop error, and bounds identified up to 11 m/s.
6. **Discrete-time condition** (`2c81788`). The DOB-CBF's condition as the
   discrete-time one with two pedal secants, and the backup's stop integrated
   while the pedal is still on the throttle (a standing vehicle could be
   throttled into a car ahead).
7. **Replacement**. The filters of step 6 became the only ones. Removed with the
   published and intermediate filters: the delay projection and the MPSF's
   delay advance, the command predictor, the `--safety-delay-*` options, the
   per-call `cbf_filter_log.csv` (sim_diag.csv carries the operator's and the
   filtered commands), the hedged and grip-scaled soil authority, and every
   tuning flag of the published filters. The delay-awareness ablation
   (`benchmarking/latency_awareness_ablation.py`) tested the delay projection,
   so it went too. Its frozen results and scorer stay as evidence for the
   published paper. The filters' clock is always simulation time.

The results behind each step (late detection, the failure-mode battery, the
static matrix, the delay sweep and the sensitivity to the free design
parameters) are in the journal workspace, `experiments/filter_fixes/` and
`experiments/derived/`.
