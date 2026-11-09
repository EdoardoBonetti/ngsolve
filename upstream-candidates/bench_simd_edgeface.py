"""bench_simd_edgeface: assembly time of an edge (BBND) form with specialcf.EdgeFaceTangentialVectors.

Purpose: upstream has no SIMD Evaluate for EdgeFaceTangentialVectors, so the integrator throws
ExceptionNOSIMD once and then assembles in scalar mode for good; the fork evaluates it in SIMD.
Form: a(u,v) = sum_E int_E (E^T u) . (E^T v) ds, u, v in VectorH1(order) on the unit cube.
Run:  source env.sh fork|upstream; python3 bench_simd_edgeface.py [--maxh 0.08] [--order 3]
Correct: identical matrix norm in both variants; simd_evaluate True in the fork, False upstream.
"""
import argparse, time
from common import save, best_of
from ngsolve import *
from netgen.csg import unit_cube

p = argparse.ArgumentParser()
p.add_argument("--maxh", type=float, default=0.08)
p.add_argument("--order", type=int, default=3)
p.add_argument("--reps", type=int, default=3)
p.add_argument("--threads", type=int, default=10)
a_ = p.parse_args()
t0 = time.perf_counter()
SetNumThreads(a_.threads)
ngsglobals.msg_level = 0
with TaskManager():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=a_.maxh))
    fes = VectorH1(mesh, order=a_.order)
    u, v = fes.TnT()
    E = specialcf.EdgeFaceTangentialVectors(3)
    a = BilinearForm(fes)
    a += InnerProduct(E.trans * u, E.trans * v) * dx(element_vb=BBND)
    a.Assemble()                                  # warm-up (upstream: first SIMD attempt fails here)
    tbest, ts = best_of(a.Assemble, a_.reps)
    norm = a.mat.AsVector().Norm()
save("simd_edgeface", {"ne": mesh.ne, "nedges": mesh.nedge, "ndof": fes.ndof, "order": a_.order,
                       "assemble_s_best": tbest, "assemble_s_all": ts,
                       "simd_evaluate": a.integrators[0].simd_evaluate, "mat_norm": norm},
     t0, a_.threads)
