"""1D adaptive refinement through NGSolve: SetRefinementFlag + Refine + GridFunction.Update."""
import pytest
from math import sqrt
from ngsolve import *
from ngsolve.meshes import Make1DMesh


def flag_and_refine(mesh, pred):
    for el in mesh.Elements(VOL):
        mesh.SetRefinementFlag(el, pred(el))
    mesh.Refine()


def test_element_count_and_parents():
    mesh = Make1DMesh(8)
    flag_and_refine(mesh, lambda el: el.nr < 3)
    assert mesh.ne == 11
    assert mesh.levels == 2
    # appended children know their parent; original elements have no parent
    # (GetParentElement returns an invalid ElementId, whose .nr is out of range)
    parents = [mesh.GetParentElement(ElementId(VOL, i)).nr for i in range(mesh.ne)]
    assert all(p >= mesh.ne for p in parents[:8])
    assert sorted(parents[8:]) == [0, 1, 2]


def test_refine_all_doubles():
    mesh = Make1DMesh(5)
    flag_and_refine(mesh, lambda el: True)
    assert mesh.ne == 10
    flag_and_refine(mesh, lambda el: True)
    assert mesh.ne == 20
    assert mesh.levels == 3


@pytest.mark.parametrize("order", [0, 1, 2, 3, 5])
def test_l2_update_is_exact(order):
    # a piecewise polynomial of degree <= order lies in the space before and
    # after refinement, so Update() must reproduce it to round-off
    mesh = Make1DMesh(6)
    fes = L2(mesh, order=order, dgjumps=True)
    gfu = GridFunction(fes)
    f = sum(c * x**k for k, c in enumerate([0.3, -1.2, 2.0, 0.7, -0.4, 0.9][:order + 1]))
    gfu.Set(f)
    flag_and_refine(mesh, lambda el: el.nr % 2 == 0)
    fes.Update()
    gfu.Update()
    err = sqrt(Integrate((gfu - f)**2, mesh))
    assert err < 1e-12


def test_l2_update_two_levels():
    mesh = Make1DMesh(4)
    fes = L2(mesh, order=3, dgjumps=True)
    gfu = GridFunction(fes)
    f = 1 + x - 3 * x**2 + 0.5 * x**3
    gfu.Set(f)
    for k in range(3):
        flag_and_refine(mesh, lambda el: (el.nr + k) % 3 == 0)
        fes.Update()
        gfu.Update()
    assert sqrt(Integrate((gfu - f)**2, mesh)) < 1e-12


def test_h1_order1_update_is_exact():
    # (only order 1: NGSolve's high-order H1 prolongation transfers the
    # linear part exactly but not the higher modes, in 1D as in 2D/3D)
    mesh = Make1DMesh(6)
    fes = H1(mesh, order=1)
    gfu = GridFunction(fes)
    gfu.Set(0.3 - 1.2 * x)
    flag_and_refine(mesh, lambda el: el.nr in (1, 4))
    fes.Update()
    gfu.Update()
    assert sqrt(Integrate((gfu - (0.3 - 1.2 * x))**2, mesh)) < 1e-12


def test_solve_on_refined_mesh():
    # a Laplace solve after non-uniform refinement must still converge
    mesh = Make1DMesh(10)
    flag_and_refine(mesh, lambda el: el.nr < 5)
    fes = H1(mesh, order=3, dirichlet="left|right")
    u, v = fes.TnT()
    a = BilinearForm(grad(u) * grad(v) * dx).Assemble()
    from math import pi
    f = LinearForm(pi**2 * sin(pi * x) * v * dx).Assemble()
    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec
    assert sqrt(Integrate((gfu - sin(pi * x))**2, mesh)) < 1e-5
