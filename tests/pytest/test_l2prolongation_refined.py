# L2-type spaces created on an already refined mesh must still use the exact
# high-order prolongation: a polynomial of degree <= order stays exact after
# further refinements. Before the fix, L2 / SurfaceL2 fell back to a
# prolongation keeping only the lowest dof per element, and
# VectorL2(hoprolongation=True) in 3D threw "vi0 not found".
import pytest
from ngsolve import *
from netgen.geom2d import unit_square
from netgen.csg import unit_cube

order = 3


def _mesh(dim):
    if dim == 2:
        return Mesh(unit_square.GenerateMesh(maxh=0.3))
    return Mesh(unit_cube.GenerateMesh(maxh=0.5))


def _space(name, mesh):
    dim = mesh.dim
    if name == "L2":
        return L2(mesh, order=order), x**3 - 2*x*y*y + (z*z*y if dim == 3 else y**2), VOL
    if name == "VectorL2":
        f = CF((x**3 - y, x*y*y, x*y*z + z**3)[:dim])
        return VectorL2(mesh, order=order, hoprolongation=True), f, VOL
    if name == "SurfaceL2":
        return SurfaceL2(mesh, order=order), x**3 - 2*x*y*z + y*y, BND


@pytest.mark.parametrize("nref0", [0, 1, 2])
@pytest.mark.parametrize("name,dim", [("L2", 2), ("L2", 3), ("VectorL2", 2),
                                      ("VectorL2", 3), ("SurfaceL2", 3)])
def test_prolongation_after_refine(name, dim, nref0):
    mesh = _mesh(dim)
    for _ in range(nref0):
        mesh.Refine()
    fes, f, vb = _space(name, mesh)
    gf = GridFunction(fes)
    if vb == BND:
        gf.Set(f, definedon=mesh.Boundaries(".*"))
    else:
        gf.Set(f)

    def err():
        return sqrt(Integrate(InnerProduct(gf - f, gf - f), mesh, vb))

    assert err() < 1e-10
    for _ in range(2):
        mesh.Refine()
        assert err() < 1e-10
