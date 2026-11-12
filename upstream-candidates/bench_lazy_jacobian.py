"""bench_lazy_jacobian: build and Apply of a realcompiled matrix-free form, symbolic_integrator_uses_diff off and on.

Purpose: with uses_diff on, the symbolic integrators differentiate the integrand by the test functions (dcf_dtest,
so Apply evaluates the integrand once) and, on master, also by the trial functions in the constructor; a realcompiled
integrand then compiles its trial Jacobian (the largest derivative) although a matrix-free Apply never uses it. The
fork builds the Jacobian on first use (branch lazy-trial-jacobian). Form: L2**N on the unit square, order p,
F(u) . v dx with F_i(u) = u_i u_(i+1) + u_i^3 / 3, plus B (u - u.Other()) . v on the inner facets, both
Compile(realcompile=True, wait=True); nonassemble. Reports per variant the seconds to build the form (the derivative
compiles) and the min and median Apply time over --rounds rounds of --reps calls, the two variants interleaved after
one warm-up call each (single short runs were off by up to 1.6x); with --lin 1 also the first AssembleLinearization
of an assembled form (where the fork compiles the Jacobian).
Run:  source env.sh fork|upstream; python3 bench_lazy_jacobian.py --N 24 --maxh 0.03 --order 2 [--lin 1] [--threads 8]
Correct: Apply with uses_diff on equals uses_diff off to 1e-12 (rel_diff).
"""
import argparse, time
from common import save
from netgen.geom2d import unit_square
from ngsolve import *

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=24)
p.add_argument("--maxh", type=float, default=0.03)
p.add_argument("--order", type=int, default=2)
p.add_argument("--reps", type=int, default=30)
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--lin", type=int, default=0)
p.add_argument("--threads", type=int, default=8)
a_ = p.parse_args()
SetNumThreads(a_.threads)
t0 = time.perf_counter()
ngsglobals.msg_level = 0
N = a_.N
mesh = Mesh(unit_square.GenerateMesh(maxh=a_.maxh))
fes = L2(mesh, order=a_.order, dgjumps=True) ** N
u, v = fes.TnT()
flux = CF(tuple(u[i] * u[(i + 1) % N] + u[i] ** 3 / 3 for i in range(N)))
B = CF(tuple(1.0 if j == i else (0.5 if j == (i + 1) % N else 0.0) for i in range(N) for j in range(N)), dims=(N, N))
t = time.perf_counter()
vol = InnerProduct(flux, v).Compile(realcompile=True, wait=True)
jump = InnerProduct(B * (u - u.Other()), v).Compile(realcompile=True, wait=True)
res = {"N": N, "order": a_.order, "maxh": a_.maxh, "ne": mesh.ne, "ndof": fes.ndof,
       "integrand_compile_s": time.perf_counter() - t}
state = GridFunction(fes)
state.Set(CF(tuple(0.5 * sin((i + 1) * x) * cos(y + i) for i in range(N))))
w = state.vec


def form(nonassemble):
    a = BilinearForm(fes, nonassemble=nonassemble)
    a += vol * dx
    a += jump * dx(skeleton=True)
    return a


forms, ys, rows, times = {}, {}, {}, {False: [], True: []}
with TaskManager():
    for flag in (False, True):
        ngsglobals.symbolic_integrator_uses_diff = flag
        t = time.perf_counter()
        forms[flag] = form(nonassemble=True)
        rows[flag] = {"build_s": time.perf_counter() - t}
        ys[flag] = w.CreateVector()
        forms[flag].Apply(w, ys[flag])       # warm-up
    for _ in range(a_.rounds):
        for flag in (False, True):
            for _ in range(a_.reps):
                t = time.perf_counter(); forms[flag].Apply(w, ys[flag]); times[flag].append(time.perf_counter() - t)
    for flag in (False, True):
        ts = sorted(times[flag])
        rows[flag].update(apply_ms_min=1e3 * ts[0], apply_ms_median=1e3 * ts[len(ts) // 2], apply_calls=len(ts))
        if a_.lin:
            ngsglobals.symbolic_integrator_uses_diff = flag
            al = form(nonassemble=False)
            t = time.perf_counter(); al.AssembleLinearization(w); rows[flag]["first_linearization_s"] = time.perf_counter() - t
        res[f"uses_diff_{int(flag)}"] = rows[flag]
res["rel_diff"] = (ys[True] - ys[False]).Norm() / ys[False].Norm()
save(f"lazy_jacobian_N{N}_p{a_.order}", res, t0, a_.threads)
