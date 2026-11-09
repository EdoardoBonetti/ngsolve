"""bench_complex_precond: GMRES on a complex system with a real BDDC, against the complex BDDCs upstream needs.

Purpose: upstream applies a real operator to a complex vector through x.FV<double>(), which reads the n
complex entries as 2n interleaved doubles: a real preconditioner on a complex system raises (or silently
returns garbage). The fork applies real operators to Re and Im separately, so one real BDDC (real
factorizations, half the memory) serves the complex system. Upstream's two workarounds are measured too.
Problem: int grad u . grad v + i*omega u v dx = int v dx on the unit cube, u = 0 on the boundary, H1(order) complex.
Preconditioners (--pre):
  real     BDDC of the real form int grad u . grad v + u v dx on the real space (fork only),
  complex  the same form assembled on the complex space (complex BDDC; upstream workaround 1),
  system   BDDC of the complex system matrix itself (upstream workaround 2).
Run:  source env.sh fork|upstream; python3 bench_complex_precond.py [--maxh 0.08] [--order 3] [--pre real]
Correct: every variant that runs reaches the tolerance; the true residual matches the reported one;
'real' and 'complex' need the same number of iterations (same operator, different arithmetic).
"""
import argparse, time
from common import save
from ngsolve import *
from ngsolve.krylovspace import GMRESSolver
from netgen.occ import unit_cube

p = argparse.ArgumentParser()
p.add_argument("--maxh", type=float, default=0.08)
p.add_argument("--order", type=int, default=3)
p.add_argument("--omega", type=float, default=1.0)
p.add_argument("--pre", default="real", choices=["real", "complex", "system"])
p.add_argument("--maxiter", type=int, default=200)
p.add_argument("--threads", type=int, default=10)
a_ = p.parse_args()
t0 = time.perf_counter()
SetNumThreads(a_.threads)
ngsglobals.msg_level = 0
with TaskManager():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=a_.maxh))
    fesc = H1(mesh, order=a_.order, complex=True, dirichlet=".*")
    u, v = fesc.TnT()
    f = LinearForm(v * dx).Assemble()
    # setup = assembly of the system + construction of the preconditioner (what a user pays before GMRES)
    ts = time.perf_counter()
    a = BilinearForm(grad(u) * grad(v) * dx + 1j * a_.omega * u * v * dx)
    if a_.pre == "system":
        pre = Preconditioner(a, "bddc")
        a.Assemble()
    else:
        a.Assemble()
        if a_.pre == "real":
            fesr = H1(mesh, order=a_.order, dirichlet=".*")
            ur, vr = fesr.TnT()
            ap = BilinearForm(grad(ur) * grad(vr) * dx + ur * vr * dx)
        else:
            ap = BilinearForm(grad(u) * grad(v) * dx + u * v * dx)
        pre = Preconditioner(ap, "bddc")
        ap.Assemble()
    tsetup = time.perf_counter() - ts
    its, res = [], []
    solver = GMRESSolver(mat=a.mat, pre=pre, tol=1e-10, maxiter=a_.maxiter,
                         callback=lambda it, r: (its.append(it), res.append(r)))
    gfu = GridFunction(fesc)
    t = time.perf_counter()
    error = None
    try:
        gfu.vec.data = solver * f.vec
    except Exception as e:
        error = str(e)[:300]
    tsolve = time.perf_counter() - t
    r = f.vec.CreateVector(); r.data = f.vec - a.mat * gfu.vec
    rf = r.CreateVector(); rf.data = Projector(fesc.FreeDofs(), True) * r
    ff = r.CreateVector(); ff.data = Projector(fesc.FreeDofs(), True) * f.vec
save(f"complex_precond_{a_.pre}", {"ndof": fesc.ndof, "order": a_.order, "pre": a_.pre, "iterations": len(its),
                                   "first_res": res[0] if res else None, "last_res": res[-1] if res else None,
                                   "true_rel_res": rf.Norm() / ff.Norm(), "setup_s": tsetup, "solve_s": tsolve,
                                   "total_s": tsetup + tsolve, "error": error}, t0, a_.threads)
