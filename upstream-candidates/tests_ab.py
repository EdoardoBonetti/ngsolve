"""tests_ab: run the fork's regression tests against the active NGSolve build (env.sh fork|upstream).

Purpose: the proof for each bug fix is a test that fails (or crashes) on upstream master and passes
on the fork. Each test file runs in its own pytest process, so a segfault is recorded, not fatal.
Run:  source env.sh fork|upstream; python3 tests_ab.py
Correct: every test passes on the fork; the per-test outcome on upstream is the evidence.
"""
import re, subprocess, sys, time
from pathlib import Path
from common import save

ROOT = Path(__file__).resolve().parent.parent
NGS = ROOT / "tests" / "pytest"
NG = ROOT / "external_dependencies" / "netgen" / "tests" / "pytest"
FILES = [(NGS, f) for f in [
    "test_edgefacetangential_simd.py", "test_tangential_simd_bbnd.py", "test_other_bnd_assembled.py",
    "test_inverse_temporary.py", "test_irspace_set.py", "test_equivalence_keys.py", "test_periodic.py",
    "test_device_krylov.py", "test_l2prolongation_refined.py", "test_localhoprolongation.py",
    "test_hcurlcurl_prolongation.py", "test_hcurlcurl_prolongation_trig.py", "test_refine1d.py",
    "test_solvers.py", "test_tendexlines.py"]] + [(NG, f) for f in ["test_split_alfeld.py", "test_refine1d.py"]]

t0 = time.perf_counter()
out = {}
for d, f in FILES:
    key = ("netgen/" if d == NG else "") + f
    t = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", f],
                       cwd=d, capture_output=True, text=True, timeout=1800)
    res = {}
    for line in p.stdout.splitlines():
        m = re.match(r"(PASSED|FAILED|ERROR|SKIPPED)\s+(\S+)", line)
        if m:
            res[m.group(2).split("::", 1)[-1]] = m.group(1)
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr.strip()[-200:]
    out[key] = {"returncode": p.returncode, "crashed": p.returncode < 0, "tests": res,
                "summary": tail[-200:], "seconds": time.perf_counter() - t}
    print(f"{key:45s} rc={p.returncode:4d}  {tail[-90:]}")
save("tests", out, t0, None)
