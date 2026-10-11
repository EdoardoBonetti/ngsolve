# GridFunction(IntegrationRuleSpace).Set agrees with Interpolate also when the
# element rule has more points than one SIMD block (3D, order >= 6)
from ngsolve import *
from ngsolve.comp import IntegrationRuleSpace
from ngsolve.meshes import MakeStructured3DMesh
import pytest


@pytest.mark.parametrize("hexes", [True, False])
@pytest.mark.parametrize("order", [5, 6, 8])
def test_irspace_set(hexes, order):
    mesh = MakeStructured3DMesh(hexes, nx=2, ny=1, nz=1)
    fes = IntegrationRuleSpace(mesh, order=order)**3
    gs = GridFunction(fes)
    gs.Set(CF((x, y, z)))
    gi = GridFunction(fes)
    gi.Interpolate(CF((x, y, z)))
    a, b = gs.vec.FV().NumPy(), gi.vec.FV().NumPy()
    assert abs(b).max() <= 1
    assert abs(a - b).max() < 1e-10
    # Apply of the space operator: int x . x = 1 on the unit cube
    irs = fes.components[0].GetIntegrationRules()
    assert abs(Integrate(InnerProduct(gs, gs) * dx(intrules=irs), mesh) - 1) < 1e-10
