# EdgeFaceTangentialVectors has a SIMD Evaluate: assembling with it does not
# switch the integrator to scalar mode, and gives the scalar result
from ngsolve import *
import pytest


def assemble(mesh, simd):
    fes = VectorH1(mesh, order=2)
    u, v = fes.TnT()
    E = specialcf.EdgeFaceTangentialVectors(3)
    a = BilinearForm(fes)
    a += InnerProduct(E.trans * u, E.trans * v) * dx(element_vb=BBND)
    a.integrators[0].simd_evaluate = simd
    a.Assemble()
    return a


def test_edgefacetangential_simd():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.4))
    a = assemble(mesh, True)
    assert a.integrators[0].simd_evaluate   # no fallback to scalar evaluation
    b = assemble(mesh, False)
    da = a.mat.ToDense().NumPy()
    db = b.mat.ToDense().NumPy()
    assert abs(da - db).max() < 1e-12 * abs(db).max()


def test_edgefacetangential_orthonormal():
    # t1, t2 are unit vectors orthogonal to the edge, and det(t, t1, t2) > 0
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.4))
    E = specialcf.EdgeFaceTangentialVectors(3)
    t1, t2 = E[:, 0], E[:, 1]
    t = specialcf.tangential(3)
    err = Integrate(((t1 * t1 - 1)**2 + (t2 * t2 - 1)**2 + (t1 * t)**2 + (t2 * t)**2) * dx(element_vb=BBND), mesh)
    assert err < 1e-20
