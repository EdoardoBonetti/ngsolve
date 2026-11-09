"""bench_equivkeys: compile time of a deep symbolic derivative with and without equivalence keys.

Purpose: with ngsglobals.code_uses_equivalence_keys = True, upstream gives a scalar zero and a
one-component vector zero the same key, so the generated C++ references an undeclared variable and
compilation fails; the fork fixes the key, and merging common subexpressions shrinks the code.
Expression: the trace of the Hessian of psi^5 plus |grad psi|^2 plus sum_ij (d_i d_j psi)^2, with
psi = 1 + exp(-r^2/4) / (2 (r + 0.3)) and r = |(x, y, z)| (an XCTS-like manufactured source term;
r as a norm is what produces the one-component zero).
Run:  source env.sh fork|upstream; python3 bench_equivkeys.py --keys 0|1
Correct: compiled and interpreted integrals agree to 1e-10 when the compile succeeds.
"""
import argparse, time, traceback
from common import save
from ngsolve import *
from netgen.csg import unit_cube

p = argparse.ArgumentParser()
p.add_argument("--keys", type=int, default=1)
a_ = p.parse_args()
t0 = time.perf_counter()
ngsglobals.msg_level = 0
ngsglobals.code_uses_equivalence_keys = bool(a_.keys)
X = CF((x, y, z))
r = InnerProduct(X, X) ** 0.5
psi = 1 + exp(-r * r / 4) / (2 * (r + 0.3))
cs = (x, y, z)
expr = sum((psi**5).Diff(c).Diff(c) for c in cs) + sum(psi.Diff(c) * psi.Diff(c) for c in cs) \
       + sum(psi.Diff(c).Diff(d)**2 for c in cs for d in cs)
mesh = Mesh(unit_cube.GenerateMesh(maxh=0.3))
res = {"keys": a_.keys}
t = time.perf_counter()
try:
    ec = expr.Compile(realcompile=True, maxderiv=0, wait=True)
    res["compile_s"] = time.perf_counter() - t
    I1, I2 = Integrate(ec, mesh, order=6), Integrate(expr, mesh, order=6)
    res.update(ok=True, rel_diff=abs(I1 - I2) / abs(I2))
except Exception as e:
    res.update(ok=False, compile_s=time.perf_counter() - t, error=str(e).strip().splitlines()[-1][:300] if str(e).strip() else repr(e))
save(f"equivkeys_k{a_.keys}", res, t0, 1)
