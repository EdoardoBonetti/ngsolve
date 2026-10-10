# H1(..., localhoprolongation=True): the local high-order prolongation must
# reproduce the global one (hoprolongation=True) after adaptive refinement:
# both are the exact embedding of the coarse function into the fine space.
import numpy as np
import pytest
from ngsolve import Mesh, H1, GridFunction, TaskManager, x, y, z, Integrate, dx
from netgen.geom2d import unit_square
from netgen.csg import unit_cube


def _run(dim, order, nref, seed=0, onlyonce=False):
    if dim == 2:
        meshes = [Mesh(unit_square.GenerateMesh(maxh=0.2)) for _ in range(2)]
    else:
        meshes = [Mesh(unit_cube.GenerateMesh(maxh=0.4)) for _ in range(2)]
    spaces = [H1(meshes[0], order=order, hoprolongation=True),
              H1(meshes[1], order=order, localhoprolongation=True)]
    gfs = [GridFunction(fes) for fes in spaces]
    rng = np.random.default_rng(seed)
    vals = rng.standard_normal(spaces[0].ndof)
    for g in gfs:
        g.vec.FV().NumPy()[:] = vals
    errs = []
    for lev in range(nref):
        flags = rng.random(meshes[0].ne) < 0.3
        for m in meshes:
            for el in m.Elements():
                m.SetRefinementFlag(el, bool(flags[el.nr]))
            m.Refine(onlyonce=onlyonce)
        assert spaces[0].ndof == spaces[1].ndof
        a, b = (g.vec.FV().NumPy() for g in gfs)
        errs.append(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))
        # also the function itself: same field on the fine mesh
        diff = Integrate((gfs[0] - gfs[1]) ** 2 * dx, meshes[0]) ** 0.5
        norm = Integrate(gfs[0] ** 2 * dx, meshes[0]) ** 0.5
        errs[-1] = max(errs[-1], diff / norm)
    return errs


@pytest.mark.parametrize("dim,order", [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 3)])
def test_local_equals_global(dim, order):
    with TaskManager():
        errs = _run(dim, order, nref=4 if dim == 2 else 3)
    assert max(errs) < 1e-10, errs


@pytest.mark.parametrize("onlyonce", [True, False])
def test_local_equals_global_onlyonce(onlyonce):
    errs = _run(2, 4, nref=4, seed=3, onlyonce=onlyonce)
    assert max(errs) < 1e-10, errs


def test_exact_for_polynomials():
    # a degree-p polynomial is reproduced exactly after refinement
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.25))
    fes = H1(mesh, order=3, localhoprolongation=True)
    g = GridFunction(fes)
    f = x**3 - 2*x*y**2 + y
    g.Set(f)
    rng = np.random.default_rng(1)
    for _ in range(3):
        for el in mesh.Elements():
            mesh.SetRefinementFlag(el, bool(rng.random() < 0.4))
        mesh.Refine()
        assert Integrate((g - f) ** 2 * dx, mesh) ** 0.5 < 1e-11
