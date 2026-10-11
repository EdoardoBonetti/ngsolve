# u.Other(bnd=g) in dx(element_boundary=True): on boundary facets the assembled
# matrix uses the part of g that is linear in the trial functions
from ngsolve import *
from ngsolve.meshes import Make1DMesh
import pytest


def facet_matrix(fes, g):
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += u.Other(bnd=g(u)) * v * dx(element_boundary=True)
    a.Assemble()
    return a.mat.ToDense().NumPy()


def test_other_bnd_1d():
    mesh = Make1DMesh(2)
    fes = L2(mesh, order=0, dgjumps=True)
    assert facet_matrix(fes, lambda u: mesh.BoundaryCF({"left": 5 * u, "right": 7 * u})).tolist() == [[5, 1], [1, 7]]
    assert facet_matrix(fes, lambda u: 5 * u).tolist() == [[5, 1], [1, 5]]
    # a constant g is a right hand side, the matrix gets no boundary term
    assert facet_matrix(fes, lambda u: CF(3)).tolist() == [[0, 1], [1, 0]]


def test_other_bnd_2d():
    # u.Other(bnd=-2*u) == u.Other() on interior facets, -2*u on boundary facets
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = L2(mesh, order=2, dgjumps=True)
    u, v = fes.TnT()
    n = specialcf.normal(2)
    a = BilinearForm(fes)
    a += u.Other(bnd=-2 * u) * (grad(v) * n) * dx(element_boundary=True)
    a.Assemble()
    b = BilinearForm(fes)
    b += u.Other() * (grad(v) * n) * dx(element_boundary=True)
    b += -2 * u * (grad(v) * n) * ds(skeleton=True)
    b.Assemble()
    da = a.mat.ToDense().NumPy()
    db = b.mat.ToDense().NumPy()
    assert abs(da - db).max() < 1e-12 * abs(db).max()
