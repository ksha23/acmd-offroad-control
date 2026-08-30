#!/usr/bin/env bash
# Build the two Chrono SCM single-tire rig collectors.
#
# The collectors are two self-contained translation units, so they are compiled
# and linked directly against an existing Chrono build rather than through a
# CMake project. Keeping the build to one readable command means the binary
# that collects a corpus can be rebuilt, hashed, and recorded in that corpus's
# manifest without reproducing a build system.
set -euo pipefail
CHRONO="${CHRONO_FORK:-$HOME/Documents/sbel/chrono_fork}"
BUILD="$CHRONO/build"
EIGEN="${EIGEN_INCLUDE:-/usr/include/eigen3}"
OUT="${1:-$(dirname "$0")/build_rig_cmd}"
mkdir -p "$OUT"
for src in collect_static_data collect_rate_data; do
  g++ -O2 -fopenmp -std=c++17 \
    -I"$CHRONO/chrono/src" -I"$BUILD" -I"$BUILD/chrono" -I"$EIGEN" \
    -I/usr/include/irrlicht -I"$CHRONO/chrono/src/chrono_thirdparty" \
    -DCHRONO_DATA_DIR="\"$CHRONO/chrono/data/\"" \
    -DCHRONO_VEHICLE_DATA_DIR="\"$CHRONO/chrono/data/vehicle/\"" \
    "$(dirname "$0")/$src.cpp" -o "$OUT/$src" \
    -L"$BUILD/lib" -lChrono_core -lChrono_vehicle -lChronoModels_vehicle \
    -lChrono_irrlicht -Wl,-rpath,"$BUILD/lib"
  echo "built $OUT/$src"
done
