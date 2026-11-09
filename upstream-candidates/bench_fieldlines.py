"""bench_fieldlines: time to trace field lines of a 3D vector field (the engine of ngsolve.webgui.FieldLines).

Purpose: the fork's netgen FieldLineCalc tests the element of the previous step before the search
tree (the hint was dead code upstream) and traces the lines in parallel.
Field: B = (-(y-0.5), x-0.5, 0.3) + 0.2 (sin(2 pi z), 0, 0) on the unit cube, H(curl)-free CF.
Calls cf._BuildFieldLines (what webgui.FieldLines calls) with fixed start points, since
FieldLines itself draws random start points.
Run:  source env.sh fork|upstream; python3 bench_fieldlines.py [--maxh 0.05] [--lines 2000]
Correct: the same number of lines and points in both variants (to the tracing tolerance).
"""
import argparse, time
import numpy as np
from common import save, best_of
from ngsolve import *
from netgen.csg import unit_cube

p = argparse.ArgumentParser()
p.add_argument("--maxh", type=float, default=0.05)
p.add_argument("--lines", type=int, default=2000)
p.add_argument("--reps", type=int, default=3)
p.add_argument("--threads", type=int, default=10)
a_ = p.parse_args()
t0 = time.perf_counter()
SetNumThreads(a_.threads)
ngsglobals.msg_level = 0
mesh = Mesh(unit_cube.GenerateMesh(maxh=a_.maxh))
B = CF((-(y - 0.5) + 0.2 * sin(2 * pi * z), x - 0.5, 0.3))
rng = np.random.default_rng(0)
pts = [tuple(p) for p in 0.05 + 0.9 * rng.random((a_.lines, 3))]
res = {}
def run():
    with TaskManager():
        # mesh, start points, number, length, max points per line, thickness, tolerance, direction, randomized
        res["d"] = B._BuildFieldLines(mesh, pts, len(pts), 0.5, 500, 0.0015, 0.0005, 0, False)
tbest, ts = best_of(run, a_.reps)
d = res["d"]
npts = {k: len(v) for k, v in d.items() if hasattr(v, "__len__") and not isinstance(v, str)}
save("fieldlines", {"ne": mesh.ne, "lines": a_.lines, "trace_s_best": tbest, "trace_s_all": ts,
                    "sizes": npts},
     t0, a_.threads)
