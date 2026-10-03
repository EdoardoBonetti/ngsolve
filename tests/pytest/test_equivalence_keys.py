"""Code generation with ngsglobals.code_uses_equivalence_keys = True
(common subexpressions merged by equivalence key)."""
import pytest
from ngsolve import *
from netgen.csg import unit_cube


@pytest.fixture
def cse():
    old = ngsglobals.code_uses_equivalence_keys
    ngsglobals.code_uses_equivalence_keys = True
    yield
    ngsglobals.code_uses_equivalence_keys = old


def test_scalar_and_one_component_zero_are_distinct(cse):
    """A scalar zero and a one-component vector zero used to get the same
    equivalence key, so generated code referenced an undeclared variable
    (e.g. the second derivative of exp(-r^2) with r = |(x,y,z)|)."""
    X = CF((x, y, z))
    r = InnerProduct(X, X) ** 0.5
    g = exp(-r * r / 8)
    lap = sum(g.Diff(c).Diff(c) for c in (x, y, z))
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.4))
    lapc = lap.Compile(realcompile=True, maxderiv=0, wait=True)
    assert abs(Integrate(lapc, mesh) - Integrate(lap, mesh)) < 1e-12
