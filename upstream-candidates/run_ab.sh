#!/bin/bash
# run_ab: every fork-vs-upstream experiment on both builds, one job at a time.
# Each run writes results/<name>_<variant>.json (cost, machine, command, commit).
# Needs ~/Applications/ngsuite/install-dev (fork) and upstream-ref/install (pristine master).
# Run:  ./run_ab.sh 2>&1 | tee results/run_ab.log
set -u
cd "$(dirname "$0")"
for V in upstream fork; do
  (
    source env.sh $V
    echo "=== $V: $(python3 -c 'import ngsolve; print(ngsolve.__version__)' 2>/dev/null)"
    python3 tests_ab.py
    python3 bench_simd_edgeface.py
    python3 bench_fieldlines.py
    python3 bench_refine.py
    python3 bench_equivkeys.py --keys 0
    python3 bench_equivkeys.py --keys 1
    python3 bench_complex_precond.py
    python3 bench_hoprol.py --flag hoprolongation
    [ $V = fork ] && python3 bench_hoprol.py --flag localhoprolongation
    for h in 0.12 0.08 0.06 0.045; do
      python3 bench_gplhr.py --method arnoldi --maxh $h
      [ $V = fork ] && python3 bench_gplhr.py --method gplhr --maxh $h
    done
  )
done
echo "=== done"
