# HCurlCurl (Regge) on tetrahedra has an exact prolongation: the spaces on a refined mesh are
# nested, so a GridFunction keeps its values under mesh.Refine(), and the prolongation matrix
# is available for multigrid. Also for a space created on an already refined mesh.
import pytest
import numpy as np
from ngsolve import *
from netgen.csg import unit_cube


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("nref0", [0, 1])
def test_hcurlcurl_prolongation(order, nref0):
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.5))
    for _ in range(nref0):
        mesh.Refine()
    fes = HCurlCurl(mesh, order=order)
    assert fes.Prolongation() is not None
    gf = GridFunction(fes, autoupdate=True)
    gf.Set(CF((1+x*y, x*z-y, z, x*z-y, 2+y*y, y*z, z, y*z, 3+x)).Reshape((3, 3)), dual=True)
    pts = np.random.default_rng(1).uniform(0.05, 0.95, size=(50, 3))
    before = np.array([gf(mesh(*p)) for p in pts])
    nc = fes.ndof
    mesh.Refine()
    after = np.array([gf(mesh(*p)) for p in pts])
    assert np.abs(after - before).max() < 1e-9
    P = fes.Prolongation().CreateMatrix(nref0 + 1)
    assert (P.height, P.width) == (fes.ndof, nc)
