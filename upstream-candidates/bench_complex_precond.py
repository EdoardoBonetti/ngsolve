"""bench_complex_precond: GMRES on a complex system, preconditioned by a real BDDC.

Purpose: upstream applies a real operator to a complex vector through x.FV<double>(), which reads
the n complex entries as 2n interleaved doubles: the preconditioner silently returns a wrong
result and GMRES stagnates. The fork applies real operators to Re and Im separately.
Problem: int grad u . grad v + i*omega u v dx = int v dx on the unit cube, u = 0 on the boundary,
H1(order) complex; preconditioner: BDDC of the real form int grad u . grad v + u v dx.
Run:  source env.sh fork|upstream; python3 bench_complex_precond.py [--maxh 0.08] [--order 3]
Correct: the fork converges in a few tens of iterations; the true residual matches the reported one.
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
p.add_argument("--maxiter", type=int, default=200)
p.add_argument("--threads", type=int, default=10)
a_ = p.parse_args()
t0 = time.perf_counter()
SetNumThreads(a_.threads)
ngsglobals.msg_level = 0
with TaskManager():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=a_.maxh))
    fesc = H1(mesh, order=a_.order, complex=True, dirichlet=".*")
    fesr = H1(mesh, order=a_.order, dirichlet=".*")
    u, v = fesc.TnT()
    a = BilinearForm(grad(u) * grad(v) * dx + 1j * a_.omega * u * v * dx).Assemble()
    f = LinearForm(v * dx).Assemble()
    ur, vr = fesr.TnT()
    ar = BilinearForm(grad(ur) * grad(vr) * dx + ur * vr * dx)
    pre = Preconditioner(ar, "bddc")
    ar.Assemble()
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
    fd = fesc.FreeDofs()
    rn = sum(abs(r[i])**2 for i in range(len(r)) if fd[i])**0.5 if fesc.ndof < 2e5 else None
save("complex_precond", {"ndof": fesc.ndof, "order": a_.order, "iterations": len(its),
                         "first_res": res[0] if res else None, "last_res": res[-1] if res else None,
                         "true_rel_res": rn / f.vec.Norm() if rn is not None else None,
                         "solve_s": tsolve, "error": error}, t0, a_.threads)
