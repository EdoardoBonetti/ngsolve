# HCurlCurl (Regge) on tetrahedra has an exact prolongation: the spaces on a refined mesh are
# nested, so a GridFunction (any coarse-space function, here random dofs) keeps its values
# under mesh.Refine(), and the prolongation matrix is available for multigrid. Also for a
# space created on an already refined mesh.
import pytest
import numpy as np
from ngsolve import *
from netgen.csg import unit_cube


def interior_points(mesh, lam=(0.7, 0.13, 0.1, 0.07)):
    # One point per element at fixed barycentric coordinates, away from the element faces and from
    # the cuts of the next refinement: the function jumps across faces, and a point within the
    # search tolerance of a face could be located in the neighbour on one level only.
    return [np.dot(lam, [mesh[v].point for v in el.vertices]) for el in mesh.Elements(VOL)]


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("nref0", [0, 1])
def test_hcurlcurl_prolongation(order, nref0):
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.5))
    for _ in range(nref0):
        mesh.Refine()
    fes = HCurlCurl(mesh, order=order)
    assert fes.Prolongation() is not None
    gf = GridFunction(fes, autoupdate=True)
    gf.vec.SetRandom()
    pts = interior_points(mesh)
    before = np.array([gf(mesh(*p)) for p in pts])
    nc = fes.ndof
    mesh.Refine()
    after = np.array([gf(mesh(*p)) for p in pts])
    assert np.abs(after - before).max() < 1e-9 * (1 + np.abs(before).max())
    P = fes.Prolongation().CreateMatrix(nref0 + 1)
    assert (P.height, P.width) == (fes.ndof, nc)
