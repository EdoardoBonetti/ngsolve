"""check_upstream_bugs: real operators on complex vectors, and the equivalence-key compile failure.

Run: source env.sh fork|upstream; python3 check_upstream_bugs.py
Correct (fork): SparseMatrix error ~1e-14 (also upstream), Jacobi no exception, test expression compiles."""
from ngsolve import *
from netgen.occ import unit_cube
ngsglobals.msg_level = 0
mesh = Mesh(unit_cube.GenerateMesh(maxh=0.3))
fr = H1(mesh, order=2); fc = H1(mesh, order=2, complex=True)
u, v = fr.TnT()
a = BilinearForm(grad(u)*grad(v)*dx + u*v*dx).Assemble()
f = x + y*z
gc = GridFunction(fc); gc.Set((1+2j)*f)
gr = GridFunction(fr); gr.Set(f)
ref = a.mat.CreateColVector(); ref.data = a.mat * gr.vec
for name, op in [("SparseMatrix", a.mat), ("Jacobi", a.mat.CreateSmoother(fr.FreeDofs()))]:
    yv = gc.vec.CreateVector()
    try:
        yv.data = op * gc.vec
        if name == "SparseMatrix":
            err = max(abs(yv[i] - (1+2j)*ref[i]) for i in range(len(ref))) / max(abs(ref[i]) for i in range(len(ref)))
            print(f"{name} * complex vector: rel. error vs exact = {err:.3e}")
        else:
            print(f"{name} * complex vector: no exception, |y| = {yv.Norm():.4e}")
    except Exception as e:
        print(f"{name} * complex vector raises: {str(e)[:110]}")
X = CF((x, y, z)); r = InnerProduct(X, X) ** 0.5
g = exp(-r * r / 8); lap = sum(g.Diff(c).Diff(c) for c in (x, y, z))
ngsglobals.code_uses_equivalence_keys = True
try:
    lap.Compile(realcompile=True, maxderiv=0, wait=True); print("equivalence keys, test expression: compiles")
except Exception as e:
    print("equivalence keys, test expression: FAILS:", str(e).strip().splitlines()[-1][:150])
