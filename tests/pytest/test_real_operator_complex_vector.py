"""test_real_operator_complex_vector: a real operator applied to a complex vector acts on Re and Im separately.

Before the fix (commit 26266ce2a) the kernels took x.FV<double>(), which views n complex entries as 2n
interleaved doubles: sparse inverses and preconditioners raised or returned wrong numbers, and GMRES on a
complex system with a real BDDC stagnated.
Correct: op * (xr + i xi) == op * xr + i op * xi to rounding, for every operator below.
"""
import pytest
from ngsolve import *
from netgen.occ import unit_cube

ngsglobals.msg_level = 0


@pytest.fixture(scope="module")
def spaces():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.35))
    fesr = H1(mesh, order=3, dirichlet="left|right")
    fesc = H1(mesh, order=3, complex=True, dirichlet="left|right")
    return fesr, fesc


def real_operator(kind, fesr):
    u, v = fesr.TnT()
    a = BilinearForm(grad(u) * grad(v) * dx + u * v * dx, symmetric=kind != "umfpack")
    pre = Preconditioner(a, "bddc") if kind == "bddc" else None
    a.Assemble()
    fd = fesr.FreeDofs()
    if kind == "matrix":
        return a.mat
    if kind == "bddc":
        return pre
    if kind == "jacobi":
        return a.mat.CreateSmoother(fd)
    if kind == "blockjacobi":
        blocks = [[d for d in fesr.GetDofNrs(el) if d >= 0 and fd[d]] for el in fesr.mesh.Elements()]
        return a.mat.CreateBlockSmoother(blocks)
    try:
        return a.mat.Inverse(fd, inverse=kind)
    except Exception as e:
        pytest.skip(f"{kind} not in this build: {e}")


@pytest.mark.parametrize("kind", ["matrix", "sparsecholesky", "umfpack", "bddc", "jacobi", "blockjacobi"])
def test_real_operator_on_complex_vector(spaces, kind):
    fesr, fesc = spaces
    op = real_operator(kind, fesr)
    xc = GridFunction(fesc).vec.CreateVector()
    xc.SetRandom()
    xi_ = xc.CreateVector(); xi_.SetRandom()
    xc.data += 1j * xi_                                   # genuinely complex entries
    xr, xi = GridFunction(fesr).vec.CreateVector(), GridFunction(fesr).vec.CreateVector()
    xr.FV().NumPy()[:] = xc.FV().NumPy().real
    xi.FV().NumPy()[:] = xc.FV().NumPy().imag
    yr, yi = xr.CreateVector(), xi.CreateVector()
    yr.data = op * xr
    yi.data = op * xi
    yc = xc.CreateVector()
    yc.data = op * xc
    ref = yr.FV().NumPy() + 1j * yi.FV().NumPy()
    err = abs(yc.FV().NumPy() - ref).max()
    assert err <= 1e-10 * max(abs(ref).max(), 1.0), f"{kind}: max deviation {err:.3e}"
