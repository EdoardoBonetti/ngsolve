"""smoke_ab: check an A/B build on MUSICA (version, UMFPACK, a small solve).

Purpose: after ab.sh built ~/musica-ab-<variant>, confirm that it imports, reports the expected
commit, and has UMFPACK (needed by the ArnoldiSolver baseline).
Run:  ./ab.sh fork|upstream smoke_ab.py --gpus 0 --cpus 4 --time 00:05:00
Correct: prints the version (fork: ...-gddcd4c270, upstream: ...-g6e701b568) and "umfpack ok".
"""
import ngsolve, netgen
from ngsolve import *
from netgen.occ import unit_square

print("ngsolve", ngsolve.__version__, ngsolve.__file__)
print("netgen", getattr(netgen, "__version__", None))
mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
fes = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()
a = BilinearForm(grad(u) * grad(v) * dx + u.Deriv()[0] * v * dx).Assemble()
f = LinearForm(v * dx).Assemble()
gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="umfpack") * f.vec
r = f.vec.CreateVector(); r.data = f.vec - a.mat * gfu.vec
rf = r.CreateVector(); rf.data = Projector(fes.FreeDofs(), True) * r
print("umfpack ok, residual", rf.Norm())
