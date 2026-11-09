"""bench_hoprol: cost of transferring a high-order H1 GridFunction under adaptive refinement.

Purpose: upstream offers hoprolongation=True (global H1<->L2 conversion per level); the fork adds
localhoprolongation=True (exact, cost proportional to the refined elements, no per-level operators).
Setup: L-shaped domain, H1(order); each level marks the half of the elements closest to the
re-entrant corner (deterministic, same meshes in both variants) until ndof > --maxdof;
mesh.Refine, fes.Update and gfu.Update are timed per level.
Run:  source env.sh fork|upstream; python3 bench_hoprol.py --flag hoprolongation|localhoprolongation
Correct: the transferred function equals the interpolated polynomial to round-off (err column).
"""
import argparse, time
import numpy as np
from common import save
from ngsolve import *
from netgen.occ import MoveTo, OCCGeometry

p = argparse.ArgumentParser()
p.add_argument("--flag", default="hoprolongation", choices=["hoprolongation", "localhoprolongation"])
p.add_argument("--order", type=int, default=4)
p.add_argument("--maxdof", type=float, default=1.5e6)
p.add_argument("--threads", type=int, default=10)
a_ = p.parse_args()
t0 = time.perf_counter()
SetNumThreads(a_.threads)
ngsglobals.msg_level = 0
shape = MoveTo(-1, -1).Rectangle(2, 2).Face() - MoveTo(0, -1).Rectangle(1, 1).Face()
mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.05))
fes = H1(mesh, order=a_.order, **{a_.flag: True})
gfu = GridFunction(fes)
f = x**a_.order - 2 * x * y**(a_.order - 1) + y
gfu.Set(f)
lev = []
with TaskManager():
    while fes.ndof < a_.maxdof:
        pts = np.array([v.point for v in mesh.vertices])
        dist = np.array([np.linalg.norm(pts[[v.nr for v in el.vertices]].mean(axis=0)) for el in mesh.Elements()])
        cut = np.median(dist)
        for el in mesh.Elements():
            mesh.SetRefinementFlag(el, bool(dist[el.nr] <= cut))
        t = time.perf_counter(); mesh.Refine(); t1 = time.perf_counter()
        fes.Update(); t2 = time.perf_counter()
        gfu.Update(); t3 = time.perf_counter()
        err = Integrate((gfu - f)**2 * dx, mesh)**0.5
        lev.append({"ndof": fes.ndof, "refine_s": t1 - t, "fes_update_s": t2 - t1,
                    "gf_update_s": t3 - t2, "err": err})
        print(lev[-1])
save(f"hoprol_{a_.flag}", {"order": a_.order, "flag": a_.flag, "levels": lev,
                            "total_update_s": sum(d["fes_update_s"] + d["gf_update_s"] for d in lev)},
     t0, a_.threads)
