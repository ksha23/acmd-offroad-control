# Environment setup

Reproducing this project needs three things: a Python environment (via conda),
a **from-source PyChrono build** (the deformable-terrain plant), and — to run
the NMPC controller — a **from-source acados build**.

Both source builds are pinned as **git submodules** under `third_party/`, at
the exact upstream commits this project is built against:

```bash
git clone <this-repo> && cd offroad-control
git submodule update --init --recursive third_party/acados   # acados + its deps
git submodule update --init            third_party/chrono    # ~1 GB Chrono clone
```

- `third_party/chrono` → `github.com/projectchrono/chrono` @ `81d8f2491`, on
  `main`. That commit carries the CUDA 13 / GCC 15 / Thrust 3.x build
  compatibility this toolchain needs, together with the Sensor and SCM
  functionality the project uses.
- `third_party/acados` → `github.com/acados/acados` @ `8d6cd69ff`, with its
  blasfeo and hpipm submodules (hence `--recursive`).

> **Validated configuration:** the `scm-terrain` Python 3.12 environment
> against the pinned Chrono commit (clean full Chrono build; CUDA 13.1,
> SWIG 4.2, Vehicle, Sensor with OptiX, Irrlicht, and VSG enabled):
> - **Plant:** `chrono_setup` builds HMMWV_Full with SCM deformable terrain and
>   steps under load.
> - **Closed-loop acados NMPC:** headless `launch_decoupled` completes with an
>   RMS lateral CTE of about 0.024 m on a sand lane change.
> - **GPU sensor camera:** a ray-traced driver-POV frame renders on the GPU via
>   OptiX.
>
> `simulation/runtime/chrono_setup.py` resolves the Chrono data-path helpers
> (`SetVehicleDataPath`, `GetVehicleDataFile`) back-compatibly, so both the
> pinned commit and older Chrono trees work.

The submodules pin *what* to build, not the build itself; build each one with
the steps below.

## 1. Python environment (conda)

```bash
conda env create -f environment.yml      # env "scm-terrain", pinned deps
conda activate scm-terrain
```

This installs the pip/conda stack (numpy, scipy, pandas, matplotlib,
scikit-learn, casadi, pyzmq, pygame, torch, and optional msgpack/PySDL2). It
does **not** install PyChrono or acados — those are built from source below.

## 2. PyChrono (required) — build from source

The code imports `pychrono.core`, `pychrono.vehicle`, `pychrono.sensor`, and
`pychrono.irrlicht`, so Chrono must be built **with those modules enabled**. No
pip or conda package ships all of them — in particular the **Sensor** module,
used for the ray-traced driver-POV camera, is absent from the conda-forge
`pychrono` package — hence the source build.

The source is the pinned submodule `third_party/chrono`. Build it with CMake:

| CMake flag | Enables | Extra prerequisites |
|---|---|---|
| `-DCH_ENABLE_MODULE_PYTHON=ON`   | SWIG Python bindings (`pychrono`) | SWIG, a matching Python |
| `-DCH_ENABLE_MODULE_VEHICLE=ON`  | HMMWV and **SCM** deformable terrain | Chrono data dir |
| `-DCH_ENABLE_MODULE_IRRLICHT=ON` | chase-cam visualization | Irrlicht |
| `-DCH_ENABLE_MODULE_SENSOR=ON`   | ray-traced driver-POV camera | **CUDA + NVIDIA OptiX SDK** + GLFW/GLEW |
| `-DCH_ENABLE_MODULE_ROS=ON`      | the Chrono::ROS state and `/clock` publishers the paper runs require | ROS 2 Jazzy packages |

Build Chrono against the **same Python** as the conda environment (this project
uses Python 3.12; the pinned Chrono commit's SWIG bindings link
`libpython3.12`). Then, in the activated environment, expose the bindings and
the Chrono data directory:

```bash
cmake -S third_party/chrono -B third_party/chrono/build -DCH_ENABLE_MODULE_PYTHON=ON \
  -DCH_ENABLE_MODULE_VEHICLE=ON -DCH_ENABLE_MODULE_IRRLICHT=ON \
  -DCH_ENABLE_MODULE_SENSOR=ON -DCH_ENABLE_MODULE_ROS=ON
cmake --build third_party/chrono/build -j
export PYTHONPATH="$PWD/third_party/chrono/build/bin:$PYTHONPATH"
export CHRONO_DATA_DIR="$PWD/third_party/chrono/data/"     # HMMWV meshes etc.
python -c "import pychrono.vehicle, pychrono.sensor, pychrono.irrlicht; print('pychrono OK')"
```

Persist these by dropping them in
`$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh` so they are set on activation.

> Headless and no-GPU note: `pychrono.sensor` needs CUDA and OptiX at build time
> and a GPU at run time. The autonomous sweeps run `--vis-mode none` (no camera)
> and need Vehicle alone; the driver-POV and human-in-the-loop features need
> Sensor. `./collect_hil.sh` launches the demonstration station (wheel, delayed
> POV, live HUD, preflight checks).

## 3. acados (required to run the NMPC controller)

The tire and estimator benchmarks and the paper-figure regeneration do **not**
need acados. Running the acados NMPC controller does.

```bash
# build the pinned submodule third_party/acados (cmake + make; see its docs)
pip install -e third_party/acados/interfaces/acados_template
export ACADOS_SOURCE_DIR="$PWD/third_party/acados"
export LD_LIBRARY_PATH="$PWD/third_party/acados/lib:$LD_LIBRARY_PATH"
python -c "import acados_template; print('acados_template OK')"
```

`ACADOS_SOURCE_DIR` must be set before any acados import: the controller
preloads `libacados.so` from it via `ctypes`.

## 4. ROS 2 and the Chrono::ROS paper transport

The simulation-to-controller link defaults to direct ROS 2 `rclpy`/DDS topics
(`--transport ros`). In the same default paper configuration, Chrono's
`ChROSPythonManager` exposes the chassis through `/clock` and
`~/chrono/vehicle/state/{pose,twist,accel}`. These are distinct paths: the
application-specific controller packets do not pass through a Chrono::ROS
handler. Missing `pychrono.ros` is a configuration error for a paper run, and
`--no-chrono-ros` is a development-only option. A self-contained ZeroMQ
transport is available outside the paper path as `--transport zmq`; run
`python simulation/runtime/launch_decoupled.py --help` for the full interface.

ROS 2 Jazzy (Python 3.12, matching `scm-terrain`) is required for the default
transport, along with the prebuilt `chrono_ros_interfaces` workspace and a
PyChrono build carrying the ROS module. Source them before the PyChrono
environment so the selected from-source bindings win on `LD_LIBRARY_PATH`:

```bash
source /opt/ros/jazzy/setup.bash
source ~/packages/chrono_ros_ws/install/setup.bash    # chrono_ros_interfaces
# ... then the PyChrono/acados exports from sections 2 and 3 ...
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib:$LD_LIBRARY_PATH"
python -c "import rclpy, pychrono.ros; print('ros stack OK')"
```

Parallel sweeps set a unique `ROS_DOMAIN_ID` per worker to isolate their DDS
graphs. The ROS and ZeroMQ transports run at closed-loop parity.

## 5. Verify

```bash
conda activate scm-terrain
python -c "import numpy, scipy, pandas, matplotlib, sklearn, casadi, zmq, pygame, torch; print('py stack OK')"
python -c "import pychrono.vehicle, pychrono.sensor; print('pychrono OK')"
python -c "import acados_template; print('acados OK')"   # if you built acados
python benchmarking/verify_provenance_chain.py           # corpora and checkpoints
```

The unit and contract tests under `tests/` run with the standard library
runner, for example
`python -m unittest tests.benchmarking.test_score_joint_estimator`; `pytest.ini`
configures `pytest` for the same tree if that runner is installed.

Restore the large result generations with `data_sync/data_sync.sh pull <tag>`,
as described in `DATA.md`.
