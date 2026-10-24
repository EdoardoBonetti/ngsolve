# HCurlCurl (Regge) on triangles has an exact prolongation: mesh.Refine() keeps the spaces
# nested, so a GridFunction (any coarse-space function, here random dofs) keeps its values,
# and the prolongation matrix is available for multigrid. Also for a space created on an
# already refined mesh.
import pytest
import numpy as np
from ngsolve import *
from netgen.geom2d import unit_square


def interior_points(mesh, lam=(0.61, 0.27, 0.12)):
    # One point per element at fixed barycentric coordinates, away from the element edges and from
    # the cuts of the next refinement: the function jumps across edges, and a point within the
    # search tolerance of an edge could be located in the neighbour on one level only.
    return [np.dot(lam, [mesh[v].point for v in el.vertices]) for el in mesh.Elements(VOL)]


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("nref0", [0, 1])
def test_hcurlcurl_prolongation_trig(order, nref0):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
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


@pytest.mark.parametrize("order", [0, 1, 2])
def test_hcurlcurl_prolongation_trig_l2(order):
    # a smooth coarse function keeps its L2 norm under refinement
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    fes = HCurlCurl(mesh, order=order)
    gc = GridFunction(fes)
    gc.Set(CF((1 + x * y, x - y * y, x - y * y, 2 + x)).Reshape((2, 2)), dual=True)
    gf = GridFunction(fes, autoupdate=True)
    gf.vec.data = gc.vec
    norm_before = Integrate(InnerProduct(gf, gf), mesh)
    mesh.Refine()
    assert abs(Integrate(InnerProduct(gf, gf), mesh) - norm_before) < 1e-10 * norm_before
