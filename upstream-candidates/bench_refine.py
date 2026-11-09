"""bench_refine: wall time of mesh.Refine() (netgen bisection + topology update) in 3D.

Purpose: the fork's netgen skips the second AnisotropicClusters::Update per Refine when the
topology has not changed (bit-identical result). Uniform and adaptive (30 % random marks) refinement.
Run:  source env.sh fork|upstream; python3 bench_refine.py [--maxh 0.1] [--levels 3]
Correct: identical element counts per level in both variants.
"""
import argparse, time
import numpy as np
from common import save
from ngsolve import *
from netgen.csg import unit_cube

p = argparse.ArgumentParser()
p.add_argument("--maxh", type=float, default=0.1)
p.add_argument("--levels", type=int, default=3)
p.add_argument("--threads", type=int, default=10)
a_ = p.parse_args()
t0 = time.perf_counter()
SetNumThreads(a_.threads)
ngsglobals.msg_level = 0
out = {}
for mode in ["uniform", "adaptive"]:
    rng = np.random.default_rng(0)
    mesh = Mesh(unit_cube.GenerateMesh(maxh=a_.maxh))
    lev = []
    with TaskManager():
        for l in range(a_.levels + (2 if mode == "adaptive" else 0)):
            if mode == "adaptive":
                marks = rng.random(mesh.ne) < 0.3
                for el in mesh.Elements():
                    mesh.SetRefinementFlag(el, bool(marks[el.nr]))
            t = time.perf_counter(); mesh.Refine(); dt = time.perf_counter() - t
            lev.append({"ne": mesh.ne, "nv": mesh.nv, "refine_s": dt})
    out[mode] = lev
save("refine", out, t0, a_.threads)
