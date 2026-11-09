#!/bin/bash
# submit_sweep: run musica/sweep_ab.py on MUSICA as one Slurm job per chunk (dev QOS: 48 cores, 192 GB,
# 10 min each; Slurm runs at most two at a time), wait for all of them, and fetch the run dir back.
# The regular CPU queue had multi-day waits on 2026-11-09, hence the 10-minute chunks (CHUNKS in sweep_ab.py).
# Needs both A/B builds done once (./musica/ab.sh upstream|fork musica/smoke_ab.py --gpus 0 --cpus 4).
# Run:  upstream-candidates/musica/submit_sweep.sh [chunk ...]      (default: every chunk)
# Then: copy <run dir>/results/*.json to results/musica/ and run plot_sweep.py.
set -eo pipefail
cd "$(dirname "$0")/.."
HOST=musica SLURM=/opt/slurm/rocky-9.6/25-11-3-1/bin
chunks=("$@")
[ ${#chunks[@]} -gt 0 ] || chunks=($(SWEEP_LIST=1 python3 musica/sweep_ab.py))
c=$(git rev-parse --short HEAD); git diff --quiet -- . || c="$c-dirty"
run=$(date +%Y%m%d-%H%M%S)_sweep_ab; ldir=$HOME/musica_runs/ab-fork/$run
mkdir -p "$ldir"
cp common.py bench_fieldlines.py bench_simd_edgeface.py bench_hoprol.py bench_complex_precond.py bench_refine.py \
   bench_gplhr.py musica/sweep_ab.py "$ldir/"
printf '{"script_commit": "%s", "date": "%s"}\n' "$c" "$(date '+%Y-%m-%d %H:%M:%S')" > "$ldir/sweep_meta.json"
for k in "${chunks[@]}"; do cat > "$ldir/job_$k.sh" <<JOB
#!/bin/bash
#SBATCH --job-name ab_$k
#SBATCH -p zen4_0768
#SBATCH --qos dev_zen4_0768
#SBATCH --cpus-per-task=48 --mem=192G
#SBATCH --threads-per-core=1
#SBATCH --time=00:10:00
#SBATCH --output=job_$k.out
. ~/musica-ab-fork/venv/bin/activate 2>/dev/null
export NGS_NUM_THREADS=48 SWEEP_CHUNK=$k SWEEP_BUDGET_S=540
echo "start \$(date)"
python3 sweep_ab.py
echo "end \$(date)"
JOB
done
d=$(ssh $HOST 'bash -lc "echo \$DATA"' 2>/dev/null | tail -1); rdir=${d:?no DATA dir}/musica-run/$run
rsync -az --mkpath "$ldir/" "$HOST:$rdir/"
ids=$(ssh $HOST "cd $rdir && for k in ${chunks[*]}; do $SLURM/sbatch --parsable job_\$k.sh; done" | tr '\n' ' ')
ids=${ids% }
echo "== $run: jobs $ids"
while [ -n "$(ssh $HOST "$SLURM/squeue -h -j ${ids// /,}" 2>/dev/null)" ]; do sleep 30; done
ssh $HOST "$SLURM/sacct -X -n -j ${ids// /,} -o JobName%14,State,Elapsed"
rsync -az "$HOST:$rdir/" "$ldir/"
echo "== $ldir"; cat "$ldir"/job_*.out | grep -E 'rc=|Error|Traceback|CANCEL|TIME' || true
