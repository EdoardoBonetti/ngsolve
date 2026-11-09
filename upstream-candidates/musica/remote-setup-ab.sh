#!/bin/bash
# A/B variant of the musica-run remote-setup.sh (CPU only, UMFPACK on), used by ab.sh.
# Set up the managed dir on MUSICA: module hook, venv and cmake configure of the
# synced NGSolve sources in ROOT/src/ngsolve. musica-run.sh runs it via
# ssh "bash -s -- ROOT" when ROOT/build/ngsolve has no Makefile, then builds.
#
# Python, numpy (SciPy-bundle), OpenCASCADE and NGSolve are linked through the
# EESSI rpath wrappers, no LD_LIBRARY_PATH needed. venv/bin/activate sources
# venv/bin/ngs_modules.sh, which loads the module set in ssh commands and jobs.
set -eo pipefail
ROOT=$1
VENV=$ROOT/venv SOURCES=$ROOT/src/ngsolve BUILD_DIR=$ROOT/build/ngsolve

MODULES="buildenv/default-foss-2025a Python/3.13.1-GCCcore-14.2.0 SciPy-bundle/2025.06-gfbf-2025a occt/7.9.1-GCCcore-14.2.0 CMake/3.31.3-GCCcore-14.2.0"

[ -f "$SOURCES/CMakeLists.txt" ] || { echo "no ngsolve sources in $SOURCES"; exit 1; }
rm -rf "$VENV" "$BUILD_DIR"
mkdir -p "$VENV/bin" "$BUILD_DIR"

cat > "$VENV/bin/ngs_modules.sh" <<HOOK
# Loads the EESSI/ASC modules matching this NGSolve build. Sourced by
# bin/activate before the venv is put on PATH, so the venv python wins.
export EESSI_SITE_SOFTWARE_PREFIX=\${EESSI_SITE_SOFTWARE_PREFIX:-/cvmfs/software.asc.ac.at}
export EESSI_ACCELERATOR_TARGET_OVERRIDE=\${EESSI_ACCELERATOR_TARGET_OVERRIDE:-accel/nvidia/cc90}
export LMOD_MODULERC=\${LMOD_MODULERC:-/cvmfs/software.eessi.io/host_injections/modulerc.lua}
export EESSI_MODULE_STICKY=true
# site workaround, module load fails silently without it
export LMOD_CACHED_LOADS=no

if ! type module >/dev/null 2>&1; then
    . /cvmfs/software.eessi.io/versions/2025.06/init/lmod/sh >/dev/null
fi
module purge >/dev/null 2>&1 || true
module load $MODULES
HOOK
. "$VENV/bin/ngs_modules.sh"
for v in EBROOTPYTHON EBROOTOCCT EBROOTGCCCORE; do
    [ -n "${!v}" ] || { echo "$v not set, module load failed"; exit 1; }
done
module list

# system site packages give numpy/scipy from SciPy-bundle (rpath linked)
python3 -m venv --system-site-packages "$VENV"
sed -i "1i . \"$VENV/bin/ngs_modules.sh\"" "$VENV/bin/activate"
. "$VENV/bin/activate"

# FindPython3 without the Interpreter component (first call in netgen) stops
# at the empty venv include dir, so pass headers and library explicitly
PY_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_LIBRARY=$(python3 -c "import sysconfig as s; print(s.get_config_var('LIBDIR') + '/' + s.get_config_var('LDLIBRARY'))")
[ -f "$PY_INCLUDE_DIR/Python.h" ] || { echo "Python.h not found in $PY_INCLUDE_DIR"; exit 1; }
[ -f "$PY_LIBRARY" ] || { echo "$PY_LIBRARY not found"; exit 1; }

cd "$BUILD_DIR"
cmake "$SOURCES" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_SUPERBUILD=ON \
  -DCMAKE_INSTALL_PREFIX="$VENV" \
  -DPython3_EXECUTABLE="$VENV/bin/python3" \
  -DPython3_INCLUDE_DIR="$PY_INCLUDE_DIR" \
  -DPython3_LIBRARY="$PY_LIBRARY" \
  -DUSE_OCC=ON \
  -DOpenCascade_DIR="$EBROOTOCCT/lib64/cmake/opencascade" \
  -DUSE_CUDA=OFF \
  -DUSE_GUI=OFF \
  -DUSE_UMFPACK=ON -DBUILD_UMFPACK=ON \
  -DBUILD_STUB_FILES=OFF
