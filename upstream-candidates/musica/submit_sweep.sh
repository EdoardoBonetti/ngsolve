#!/bin/bash
# submit_sweep: submit musica/sweep_ab.py to one MUSICA CPU node (48 cores, 192 GB) against the A/B builds.
# Writes sweep_meta.json (commit of these scripts) and copies the benchmarks next to the driver.
# Needs both builds done once: ./musica/ab.sh upstream musica/smoke_ab.py ...; same for fork.
# Run from anywhere:  upstream-candidates/musica/submit_sweep.sh [--time 04:00:00]
set -eo pipefail
cd "$(dirname "$0")/.."
c=$(git rev-parse --short HEAD); git diff --quiet -- . || c="$c-dirty"
printf '{"script_commit": "%s", "date": "%s"}\n' "$c" "$(date '+%Y-%m-%d %H:%M:%S')" > musica/sweep_meta.json
w=(); for f in common.py bench_fieldlines.py bench_simd_edgeface.py bench_hoprol.py bench_complex_precond.py \
             bench_refine.py bench_gplhr.py musica/sweep_meta.json; do w+=(--with "$f"); done
exec musica/ab.sh fork musica/sweep_ab.py --gpus 0 --cpus 48 --time "${1:-04:00:00}" "${w[@]}"
