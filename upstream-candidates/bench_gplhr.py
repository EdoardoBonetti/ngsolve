"""bench_gplhr: a few eigenvalues of a non-Hermitian 3D pencil near a target, without factorizing.

Purpose: upstream offers ArnoldiSolver with shift-and-invert, i.e. a sparse direct factorization of
A - sigma M; the fork adds solvers.GPLHR (Vecharynski-Yang-Xue 2016), which needs only products
with A, M and a preconditioner (BDDC). Compared: wall time, peak memory, eigenvalue agreement.
Pencil: -Laplace u + b . grad u = lam u on the unit cube, b = (8, 6, 4), u = 0 on the boundary,
H1(order) complex; num eigenvalues closest to sigma.
Run:  source env.sh fork|upstream; python3 bench_gplhr.py --method arnoldi|gplhr [--maxh 0.08]
Exact spectrum of the continuous problem: |b|^2/4 + pi^2 (k^2 + l^2 + m^2) = 58.61, 88.22 (x3), 117.83 (x3), ...
Correct: both methods find the same eigenvalues; num=4 takes the triple 88.22 whole. Arnoldi returns Ritz
values without a convergence test: give it a larger Krylov space (--krylov 20). Run each method in its
own process (peak memory is per process).
"""
import argparse, time
from common import save
from ngsolve import *
from netgen.occ import unit_cube

p = argparse.ArgumentParser()
p.add_argument("--method", default="arnoldi", choices=["arnoldi", "gplhr"])
p.add_argument("--maxh", type=float, default=0.08)
p.add_argument("--order", type=int, default=2)
p.add_argument("--num", type=int, default=5)
p.add_argument("--sigma", type=float, default=0.0)
p.add_argument("--krylov", type=int, default=0, help="Arnoldi: number of vectors (default num)")
p.add_argument("--threads", type=int, default=10)
a_ = p.parse_args()
t0 = time.perf_counter()
SetNumThreads(a_.threads)
ngsglobals.msg_level = 0
with TaskManager():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=a_.maxh))
    fes = H1(mesh, order=a_.order, complex=True, dirichlet=".*")
    u, v = fes.TnT()
    b = CF((8, 6, 4))
    a = BilinearForm(grad(u) * grad(v) * dx + (b * grad(u)) * v * dx).Assemble()
    m = BilinearForm(u * v * dx).Assemble()
    t = time.perf_counter()
    if a_.method == "arnoldi":
        nk = a_.krylov or a_.num
        gfu = GridFunction(fes, multidim=nk)
        lams = ArnoldiSolver(a.mat, m.mat, fes.FreeDofs(), list(gfu.vecs), shift=a_.sigma, inverse="umfpack")
        lams = list(lams)
        extra = {"inverse": "umfpack", "krylov": nk}
    else:
        from ngsolve.solvers import GPLHR
        lap = BilinearForm(grad(u) * grad(v) * dx)
        pre = Preconditioner(lap, "bddc")
        lap.Assemble()
        lams, vecs = GPLHR(a.mat, m.mat, pre, num=a_.num, sigma=a_.sigma, m=1, maxit=300, tol=1e-8,
                           freedofs=fes.FreeDofs(), printrates=False)
        lams = list(lams)
        extra = {"pre": "bddc(-Laplace)"}
    tsolve = time.perf_counter() - t
lams = sorted(lams, key=lambda l: abs(l - a_.sigma))[:a_.num]
tag = f"_k{a_.krylov}" if a_.method == "arnoldi" and a_.krylov else ""
save(f"gplhr_{a_.method}{tag}_n{a_.num}_h{a_.maxh}", {"method": a_.method, "ndof": fes.ndof, "order": a_.order,
     "num": a_.num, "sigma": a_.sigma, "solve_s": tsolve,
     "eigenvalues": [[l.real, l.imag] for l in lams], **extra}, t0, a_.threads)
