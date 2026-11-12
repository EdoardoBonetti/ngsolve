"""bench_lazy_energy: build, Apply and first linearization of a realcompiled Variation energy, uses_diff off and on.

Purpose: with symbolic_integrator_uses_diff on, SymbolicEnergy differentiates the energy density once (the gradient
blocks dcf, used by Apply) and, on master, also twice in the constructor (the Hessian blocks ddcf, used only by the
linearization); a realcompiled energy then compiles its Hessian although a matrix-free Apply never uses it. The fork
builds the Hessian on first use (branch lazy-energy-hessian). Energy (as in tests/pytest/test_lazy_trial_jacobian.py):
H1**N on the unit square, order p, E(u) = sum_i u_i^4/4 + u_i u_(i+1)^2/2 + (1 + u_i^2) (d_x u_i)^2/2 + (d_y u_i)^2/2,
Compile(realcompile=True, wait=True); trial proxies u and Grad(u), Hessian blocks (u, u), (Grad u, u), (Grad u, Grad u).
Reports the seconds to compile the energy, to build a nonassemble form with uses_diff on, the first Apply, and the
first and second AssembleLinearization of an assembled form with uses_diff off and on (where the fork compiles the
Hessian). Each run is a fresh process: CompileCode caches libraries only in memory, and ngscxx uses no ccache here.
Run:  source env.sh fork|upstream; python3 bench_lazy_energy.py --N 8 [--maxh 0.3] [--order 2] [--threads 8]
Correct: Apply, Energy and the linearized matrix with uses_diff on equal uses_diff off to 1e-12 (rel_diff).
"""
import argparse, time
from common import save
from netgen.geom2d import unit_square
from ngsolve import *

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=8)
p.add_argument("--maxh", type=float, default=0.3)
p.add_argument("--order", type=int, default=2)
p.add_argument("--threads", type=int, default=8)
a_ = p.parse_args()
SetNumThreads(a_.threads)
t0 = time.perf_counter()
ngsglobals.msg_level = 0
N = a_.N
mesh = Mesh(unit_square.GenerateMesh(maxh=a_.maxh))
fes = H1(mesh, order=a_.order) ** N
u = fes.TrialFunction()
gu = Grad(u)
E = sum(u[i] ** 4 / 4 + u[i] * u[(i + 1) % N] ** 2 / 2 + (1 + u[i] ** 2) * gu[i, 0] ** 2 / 2 + gu[i, 1] ** 2 / 2
        for i in range(N))
state = GridFunction(fes)
state.Set(CF(tuple(0.5 * sin((i + 1) * x) * cos(y + i) for i in range(N))))
x_ = state.vec
res = {"N": N, "order": a_.order, "maxh": a_.maxh, "unknowns": fes.ndof}

t = time.perf_counter()
E = E.Compile(realcompile=True, wait=True)
res["energy_compile_s"] = time.perf_counter() - t


def energy_form(nonassemble=False):
    a = BilinearForm(fes, nonassemble=nonassemble)
    a += Variation(E * dx)
    return a


def rel_diff(a, b):
    return (a - b).Norm() / b.Norm()


ys, mats, energies, forms = {}, {}, {}, {}
for diff in (False, True):
    key = "on" if diff else "off"
    ngsglobals.symbolic_integrator_uses_diff = diff
    t = time.perf_counter()
    a = energy_form(nonassemble=True)
    res[f"build_nonassemble_{key}_s"] = time.perf_counter() - t
    ys[diff] = x_.CreateVector()
    t = time.perf_counter()
    a.Apply(x_, ys[diff])
    res[f"first_apply_{key}_s"] = time.perf_counter() - t
    energies[diff] = a.Energy(x_)
    b = energy_form()
    with TaskManager():
        t = time.perf_counter()
        b.AssembleLinearization(x_)
        res[f"first_linearization_{key}_s"] = time.perf_counter() - t
        t = time.perf_counter()
        b.AssembleLinearization(x_)
        res[f"second_linearization_{key}_s"] = time.perf_counter() - t
    forms[diff] = b  # AsVector() is a view of the matrix: keep the form alive and copy the values
    mats[diff] = b.mat.AsVector().CreateVector()
    mats[diff].data = b.mat.AsVector()

res["rel_diff_apply"] = rel_diff(ys[True], ys[False])
res["rel_diff_energy"] = abs(energies[True] - energies[False]) / abs(energies[False])
res["rel_diff_linearization"] = rel_diff(mats[True], mats[False])
save(f"lazy_energy_N{N}_p{a_.order}", res, t0, a_.threads)
