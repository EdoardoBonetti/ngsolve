"""test_hddf_edge_shapes: the edge bubbles of HDivDivFacetSpace on a tetrahedron.

On BBND integration points the edge shapes of HDivDivFacetHOElement<ET_TET> are
    phi_{e,i} = lam_a lam_b P_i(lam_a - lam_b) t (x) t,   t = x_b - x_a,
for the edge e = (a, b) of the reference tetrahedron.  On the reference tetrahedron
itself the mapping is the identity, so the BBND mass matrix has the diagonal entries
    int_e |phi_{e,i}|^2 ds = |t|^4 |e| int_0^1 (s(1-s) P_i(2s-1))^2 ds,
which is |e|^5/30 for i = 0 and |e|^5/210 for i = 1.  Before the fix, lam_2 was read
from ip(3) (the integration weight), so the three edges at vertex 2 were wrong.
"""
import math
import pytest
from ngsolve import *
from netgen.meshing import Mesh as NGMesh, MeshPoint, Pnt, Element3D, Element2D, FaceDescriptor


def reference_tet():
    m = NGMesh(dim=3)
    pts = [m.Add(MeshPoint(Pnt(*p))) for p in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]]
    m.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    m.SetMaterial(1, "dom")
    m.Add(Element3D(1, pts))
    for f in [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]:
        m.Add(Element2D(1, [pts[i] for i in f]))
    return Mesh(m)


@pytest.mark.parametrize("order", [1, 2])
def test_hddf_edge_bubble_mass(order):
    mesh = reference_tet()
    ei = ElementId(VOL, 0)
    el = mesh[ei]
    # the identity map is needed for the closed-form values below
    assert [mesh[v].point for v in el.vertices] == [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]

    fes = HDivDivFacetSpace(mesh, order=order)
    u, v = fes.TnT()
    bfi = SymbolicBFI(InnerProduct(u, v), element_vb=BBND, bonus_intorder=4)
    fel = fes.GetFE(ei)
    M = bfi.CalcElementMatrix(fel, mesh.GetTrafo(ei))

    first_edge = fel.ndof - 6 * order          # element-local: face shapes, then edge shapes
    moments = [1 / 30, 1 / 210]
    for enr, e in enumerate(el.edges):
        a, b = [mesh[vv].point for vv in mesh[e].vertices]
        L = math.dist(a, b)
        for i in range(order):
            k = first_edge + enr * order + i
            assert M[k, k] == pytest.approx(L**5 * moments[i], rel=1e-12), (enr, i)
