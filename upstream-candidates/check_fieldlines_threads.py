"""check_fieldlines_threads: field-line output and time at a given thread count (argv[1]).

Purpose: show that the fork traces the same total line length as upstream and that its output does not
depend on the thread count. Run: source env.sh fork|upstream; python3 check_fieldlines_threads.py 1|10
Correct: identical segment count and length at 1 and 10 threads; total length equal across builds to ~1e-4."""
import sys, numpy as np, time
from ngsolve import *
from netgen.csg import unit_cube
ngsglobals.msg_level = 0
SetNumThreads(int(sys.argv[1]))
mesh = Mesh(unit_cube.GenerateMesh(maxh=0.05))
B = CF((-(y - 0.5) + 0.2 * sin(2 * pi * z), x - 0.5, 0.3))
rng = np.random.default_rng(0)
pts = [tuple(p) for p in 0.05 + 0.9 * rng.random((2000, 3))]
with TaskManager():
    t = time.perf_counter()
    d = B._BuildFieldLines(mesh, pts, len(pts), 0.5, 500, 0.0015, 0.0005, 0, False)
    dt = time.perf_counter() - t
ps = np.array(d["pstart"]).reshape(-1, 3); pe = np.array(d["pend"]).reshape(-1, 3)
inside = lambda P: ((P > -1e-9) & (P < 1 + 1e-9)).all(axis=1).mean()
seglen = np.linalg.norm(pe - ps, axis=1)
print(f"threads={sys.argv[1]} time={dt:.2f}s segments={len(ps)} total length={seglen.sum():.3f} "
      f"inside={inside(pe):.4f} max seg={seglen.max():.4f}")
