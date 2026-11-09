"""sweep_ab: small / medium / large A/B sweep of every benchmark, pristine upstream master against the fork.

Purpose: show for each proposal whether the fork beats upstream at three problem sizes, on one MUSICA CPU node,
with both builds compiled the same way (Release, CPU only, UMFPACK; musica/remote-setup-ab.sh). Runs alternate
upstream and fork per benchmark and size, so both see the same node load; each run is its own process
(peak memory per run).
Run (from upstream-candidates/):  musica/submit_sweep.sh        (one job per chunk, dev QOS, <10 min each)
Chunks: SWEEP_CHUNK=<name> runs only that chunk (see CHUNKS); unset runs everything.
Mac dry run with tiny sizes:      SWEEP_LOCAL=1 python3 musica/sweep_ab.py   (uses env.sh builds)
Output: results/<benchmark>_<size>_<variant>.json (cost, machine, command, commit) + logs/ + sweep_summary_<chunk>.json.
Correct: every run exits 0 except upstream's real-BDDC-on-complex run (fails by design); the fork's numbers
(matrix norms, eigenvalues, traced lengths, errors) match upstream's.
"""
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = bool(os.environ.get("SWEEP_LOCAL"))
BENCH = HERE.parent if LOCAL else HERE                # on MUSICA the benchmarks are copied next to this script
T = int(os.environ.get("NGS_NUM_THREADS", "10" if LOCAL else "48"))
PY = {v: os.path.expanduser(f"~/musica-ab-{v}/venv") for v in ("upstream", "fork")}

if LOCAL:   # tiny sizes, only to check that every run starts and writes its JSON
    S = {"fieldlines": {"S": "--maxh 0.2 --lines 50 --reps 1"},
         "simd_edgeface": {"S": "--maxh 0.3 --reps 1"},
         "hoprol": {"S": "--maxdof 2e4"},
         "complex_precond": {"S": "--maxh 0.3"},
         "refine": {"S": "--maxh 0.3 --levels 2"},
         "gplhr": {"S": "--maxh 0.3 --levels 1"}}
else:
    S = {"fieldlines": {"S": "--maxh 0.1 --lines 500", "M": "--maxh 0.05 --lines 2000 --reps 2",
                        "L": "--maxh 0.025 --lines 10000 --reps 1"},
         "simd_edgeface": {"S": "--maxh 0.16", "M": "--maxh 0.08", "L": "--maxh 0.04"},
         "hoprol": {"S": "--maxdof 1e5", "M": "--maxdof 1e6", "L": "--maxdof 1e7"},
         "complex_precond": {"S": "--maxh 0.16", "M": "--maxh 0.08", "L": "--maxh 0.04"},
         "refine": {"S": "--maxh 0.2 --levels 3", "M": "--maxh 0.1 --levels 3", "L": "--maxh 0.1 --levels 4"},
         # 5k (XS, where Arnoldi should still win), 20k, 83k, 287k dofs (L: Arnoldi + UMFPACK ~5 min, ~25 GB)
         "gplhr": {"XS": "--maxh 0.25 --levels 1", "S": "--maxh 0.3 --levels 2", "M": "--maxh 0.35 --levels 3",
                   "L": "--maxh 0.25 --levels 3"}}

# jobs of at most 10 minutes (MUSICA dev QOS): chunk -> [(benchmark, size, variants or None for all)]
CHUNKS = {"fl_SM": [("fieldlines", "S", None), ("fieldlines", "M", None)],
          "fl_L": [("fieldlines", "L", None)],
          "simd": [("simd_edgeface", z, None) for z in "SML"],
          "hop_SM": [("hoprol", "S", None), ("hoprol", "M", None)],
          "hop_L": [("hoprol", "L", None)],
          "cpx_SM": [("complex_precond", "S", None), ("complex_precond", "M", None)],
          "cpx_L": [("complex_precond", "L", None)],
          "refine": [("refine", z, None) for z in "SML"],
          "eig_XSSM": [("gplhr", z, None) for z in ("XS", "S", "M")],
          "eig_L_fork": [("gplhr", "L", ("fork",))],
          "eig_L_up": [("gplhr", "L", ("upstream",))]}

# per benchmark: (variant, script, extra args, tag suffix); upstream first, then the fork
RUNS = {
    "fieldlines": [("upstream", "bench_fieldlines.py", "", ""), ("fork", "bench_fieldlines.py", "", ""),
                   ("fork", "bench_fieldlines.py", "--threads 1", "_t1")],
    "simd_edgeface": [("upstream", "bench_simd_edgeface.py", "", ""), ("fork", "bench_simd_edgeface.py", "", "")],
    "hoprol": [("upstream", "bench_hoprol.py", "--flag hoprolongation", ""),
               ("fork", "bench_hoprol.py", "--flag localhoprolongation", ""),
               ("fork", "bench_hoprol.py", "--flag hoprolongation", "")],
    "complex_precond": [("upstream", "bench_complex_precond.py", "--pre real", ""),
                        ("upstream", "bench_complex_precond.py", "--pre complex", ""),
                        ("upstream", "bench_complex_precond.py", "--pre system", ""),
                        ("fork", "bench_complex_precond.py", "--pre real", "")],
    "refine": [("upstream", "bench_refine.py", "", ""), ("fork", "bench_refine.py", "", "")],
    "gplhr": [("upstream", "bench_gplhr.py", "--method arnoldi --krylov 20 --num 4", ""),
              ("fork", "bench_gplhr.py", "--method gplhr --pre multigrid --num 4", "")],
}


def env_for(variant, tag):
    e = dict(os.environ, NGS_VARIANT=variant, NGS_TAG=tag, NGS_MACHINE="mac" if LOCAL else "MUSICA",
             OMP_NUM_THREADS=str(T), OPENBLAS_NUM_THREADS=str(T), FLEXIBLAS_NUM_THREADS=str(T))
    meta = HERE / "sweep_meta.json"
    if meta.exists():
        e["NGS_SCRIPT_COMMIT"] = json.loads(meta.read_text())["script_commit"]
    if not LOCAL:
        e["VIRTUAL_ENV"] = PY[variant]
        e["PATH"] = f"{PY[variant]}/bin:" + ":".join(p for p in e["PATH"].split(":") if "musica-ab-" not in p)
    return e


def command(variant, script, args):
    if LOCAL:
        return ["bash", "-c", f'source env.sh {variant} && exec python3 {script} {args}']
    return [f"{PY[variant]}/bin/python3", script, *args.split()]


if os.environ.get("SWEEP_LIST"):                       # for submit_sweep.sh
    print(" ".join(CHUNKS)); sys.exit()
CHUNK = os.environ.get("SWEEP_CHUNK")
todo = [(b, z, None) for b, sz in S.items() for z in sz] if LOCAL or not CHUNK else CHUNKS[CHUNK]
deadline = time.perf_counter() + float(os.environ.get("SWEEP_BUDGET_S", "1e9"))   # stop runs before Slurm does
summary = []
(BENCH / "logs").mkdir(exist_ok=True)
for bench, size, only in todo:
        sargs = S[bench][size]
        for variant, script, extra, suffix in RUNS[bench]:
            if only and variant not in only:
                continue
            tag = ("dry" if LOCAL else "") + size + suffix
            args = f"{sargs} {extra} --threads {T}" if "--threads" not in extra else f"{sargs} {extra}"
            log = BENCH / "logs" / f"{bench}_{tag}_{variant}_{extra.replace('--', '').replace(' ', '')}.log"
            t = time.perf_counter()
            try:
                r = subprocess.run(command(variant, script, args), cwd=BENCH, env=env_for(variant, tag),
                                   stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                   timeout=max(5.0, min(5400.0, deadline - time.perf_counter())))
                rc = r.returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            dt = time.perf_counter() - t
            summary.append({"bench": bench, "size": tag, "variant": variant, "script": script, "args": args,
                            "returncode": rc, "wall_s": dt, "log": log.name})
            print(f"{bench:16s} {tag:5s} {variant:9s} {extra:40s} rc={rc} {dt:8.1f}s", flush=True)
            (BENCH / f"sweep_summary_{CHUNK or 'all'}.json").write_text(json.dumps(summary, indent=1))
print("done")
