"""
Device Krylov solvers (DeviceCGSolver, DeviceTFQMRSolver) on the
registered gpu device - the host reference device if there is no gpu.

The exact-convergence tests use an L2 order-0 mass matrix on a uniform
mesh with h = 1/4: its block-Jacobi inverse is exact, CG and TFQMR
converge in one step with residual exactly 0, and without the SafeDiv
guards the next step before the residual check computed 0/0 = NaN.
"""

import pytest
from math import isnan
from netgen.geom2d import unit_square
from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh
from ngsolve.krylovspace import CGSolver, DeviceCGSolver, DeviceTFQMRSolver

try:
    import ngsolve.gpu as gpu
    has_device = gpu.GetGPUDevice().has_float64
except Exception:
    has_device = False

pytestmark = pytest.mark.skipif(not has_device, reason="no device with fp64")


def mass_l2_0(n):
    mesh = MakeStructured2DMesh(quads=True, nx=n, ny=n)
    fes = L2(mesh, order=0)
    u, v = fes.TnT()
    mat = BilinearForm(u*v*dx).Assemble().mat
    pre = mat.CreateBlockSmoother([[i] for i in range(mat.height)])
    return mat, pre


def solve(inv, dmat, f):
    # rhs and solution as device vectors, the result back on the host
    rhs = dmat.CreateColVector()
    rhs.data = f
    sol = dmat.CreateColVector()
    sol.data = inv * rhs
    x = f.CreateVector()
    x.data = sol
    return x


def test_safediv():
    mat, pre = mass_l2_0(2)
    dvec = mat.CreateDeviceMatrix().CreateColVector()
    a, b, c = [dvec.CreateScalar() for i in range(3)]
    a.Set(3)
    b.Set(2)
    c.data = a.SafeDiv(b)
    assert c.Get() == 1.5
    c.data = (a*a).SafeDiv(b)
    assert c.Get() == 4.5
    b.Set(0)
    c.data = a.SafeDiv(b)
    assert c.Get() == 0
    c.data = (a-a).SafeDiv(b*a)
    assert c.Get() == 0


@pytest.mark.parametrize("solver", [DeviceCGSolver, DeviceTFQMRSolver])
@pytest.mark.parametrize("record", [False, True])
def test_exact_convergence(solver, record):
    n = 4
    mat, pre = mass_l2_0(n)
    dmat = mat.CreateDeviceMatrix()
    f = mat.CreateColVector()
    f[:] = 1
    inv = solver(dmat, pre.CreateDeviceMatrix(), tol=1e-12, check=10, record=record,
                 printrates=False)
    x = solve(inv, dmat, f)
    assert not any(isnan(xi) for xi in x)
    assert max(abs(xi - n*n) for xi in x) < 1e-12 * n*n
    assert inv.iterations < inv.maxiter     # converged, not stopped by maxiter


@pytest.mark.parametrize("record", [False, True])
def test_devicecg_matches_cg(record):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
    fes = H1(mesh, order=2, dirichlet=".*")
    u, v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    f = LinearForm(x*y*v*dx).Assemble()
    jac = a.mat.CreateSmoother(fes.FreeDofs())

    cg = CGSolver(a.mat, jac, tol=1e-10, maxiter=500, printrates=False)
    x_host = f.vec.CreateVector()
    x_host.data = cg * f.vec

    dmat = a.mat.CreateDeviceMatrix()
    dcg = DeviceCGSolver(dmat, jac.CreateDeviceMatrix(), tol=1e-10, maxiter=500, check=1,
                         record=record, printrates=False)
    x_dev = solve(dcg, dmat, f.vec)

    assert dcg.iterations == cg.iterations
    diff = x_dev.CreateVector()
    diff.data = x_dev - x_host
    assert Norm(diff) < 1e-8 * Norm(x_host)
