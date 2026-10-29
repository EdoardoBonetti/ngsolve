"""SIMD bilinear forms on volume edges (element_vb=BBND) with the tangential vector.

The SIMD ComputeNormalsAndMeasure for volume edges set the measure but not the tangential vector, so
specialcf.tangential evaluated to 0 in the SIMD element-matrix path: such forms assembled to zero (scalar
evaluation, LinearForms and Integrate were correct). EdgeFaceTangentialVectors gained a SIMD Evaluate, so forms
using it no longer fell back to scalar evaluation and hit the same bug.
"""
import numpy as np
import pytest
from ngsolve import *
from netgen.occ import unit_cube


@pytest.mark.parametrize("which", ["tt", "ef", "shift"])
def test_tangential_bbnd_simd_matches_scalar(which):
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.6))
    fes = HCurlCurl(mesh, order=1)
    u, v = fes.TnT()
    t = specialcf.tangential(3)
    ef = specialcf.EdgeFaceTangentialVectors(3)
    form = {"tt": InnerProduct(t, u * t) * InnerProduct(t, v * t),
            "ef": InnerProduct(ef[:, 0], u * t) * InnerProduct(t, v * t),
            "shift": (t[0] + 2) * InnerProduct(t, u * t) * InnerProduct(ef[:, 1], v * t)}[which]
    g = GridFunction(fes)
    g.vec.FV().NumPy()[:] = np.random.default_rng(1).standard_normal(fes.ndof)
    vals = []
    for simd in (True, False):
        a = BilinearForm(fes)
        a += SymbolicBFI(form, VOL, element_vb=BBND, simd_evaluate=simd)
        a.Assemble()
        w = a.mat.CreateColVector()
        w.data = a.mat * g.vec
        vals.append(InnerProduct(g.vec, w))
    assert abs(vals[0]) > 1e-8
    assert abs(vals[0] - vals[1]) <= 1e-10 * abs(vals[1])
