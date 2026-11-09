import pytest

from netgen.csg import *
from netgen.geom2d import SplineGeometry
from ngsolve import *
from numpy import pi


def test_periodicH1():
    # incoming wave direction
    d=[1./sqrt(3),1./sqrt(3),1/sqrt(3)]
    
    # wave number
    k = 10
    Lx = 2*pi/(k*d[0])
    Ly = 2*pi/(k*d[1])
    Lz = 2*pi/(k*d[2])
    print(Lx)

    geo = CSGeometry()

    left = Plane(Pnt(0,0,0),Vec(-1,0,0))
    right = Plane(Pnt(Lx,0,0),Vec(1,0,0))
    back = Plane(Pnt(0,0,0),Vec(0,-1,0))
    front = Plane(Pnt(0,Ly,0),Vec(0,1,0))
    bot = Plane(Pnt(0,0,0),Vec(0,0,-1))
    top = Plane(Pnt(0,0,Lz),Vec(0,0,1))

    
    cube = left * right * top * bot * back * front
    
    geo.Add(cube)
    geo.PeriodicSurfaces(left,right)
    geo.PeriodicSurfaces(back,front)
    geo.PeriodicSurfaces(bot,top)

    mesh = Mesh(geo.GenerateMesh(maxh=min(Lx,Ly,Lz)/10))
    
    fes = Periodic(H1(mesh,order=2, complex=True))
    
    u,v = fes.TrialFunction(), fes.TestFunction()
    
    a = BilinearForm(fes)
    a += SymbolicBFI(grad(u)*grad(v) + u*v)

    c = Preconditioner(a,"direct")

    f = LinearForm(fes)
    f += SymbolicLFI((k*k+1) * exp(1J * k * (d[0] * x + d[1] * y + d[2] * z)) * v)

    u = GridFunction(fes)

    u_exact = exp(1J*k*(d[0] * x + d[1] * y + d[2] * z))

    with TaskManager():
        a.Assemble()
        f.Assemble()
        ainv = CGSolver(a.mat,c.mat)
        u.vec.data = ainv * f.vec
        error = sqrt(Integrate(Conj(u-u_exact)*(u-u_exact),mesh).real)
        assert error < 1e-2


def test_periodic_set_order():
    geometry = SplineGeometry()
    points = [geometry.AppendPoint(*point)
              for point in ((0, 0), (1, 0), (1, 1), (0, 1))]
    geometry.Append(["line", points[0], points[1]], bc="outer")
    master = geometry.Append(["line", points[1], points[2]], bc="periodic")
    geometry.Append(["line", points[2], points[3]], bc="outer")
    geometry.Append(["line", points[0], points[3]], leftdomain=0,
                    rightdomain=1, copy=master, bc="periodic")

    mesh = Mesh(geometry.GenerateMesh(maxh=0.2))
    base_space = TangentialFacetFESpace(
        mesh, order=0, order_policy=ORDER_POLICY.VARIABLE)
    fes = Periodic(base_space)

    periodic_edges = mesh.GetPeriodicNodePairs(EDGE)
    assert periodic_edges
    (master_edge, minion_edge), _ = periodic_edges[0]

    fes.SetOrder(NodeId(EDGE, master_edge), 1)
    assert base_space.GetOrder(NodeId(EDGE, master_edge)) == 1
    assert base_space.GetOrder(NodeId(EDGE, minion_edge)) == 1

    fes.SetOrder(NodeId(EDGE, minion_edge), 2)
    assert base_space.GetOrder(NodeId(EDGE, master_edge)) == 2
    assert base_space.GetOrder(NodeId(EDGE, minion_edge)) == 2


def test_quasiperiodic():
    k=1.
    Lx=1.
    Ly=2.
    Lz=1.5
    d=[1./sqrt(3),1./sqrt(3),1./sqrt(3)]

    geo = CSGeometry()

    left = Plane(Pnt(0,0,0),Vec(-1,0,0))
    right = Plane(Pnt(Lx,0,0),Vec(1,0,0))
    bot = Plane(Pnt(0,0,0),Vec(0,0,-1))
    top = Plane(Pnt(0,0,Lz),Vec(0,0,1))
    back = Plane(Pnt(0,0,0),Vec(0,-1,0))
    front = Plane(Pnt(0,Ly,0),Vec(0,1,0))

    cube = left * right * top * bot * back * front

    geo.Add(cube)
    geo.PeriodicSurfaces(left,right)
    geo.PeriodicSurfaces(back,front)
    geo.PeriodicSurfaces(bot,top)

    mesh = Mesh(geo.GenerateMesh(maxh=0.3))
    fx = exp(1J*k*d[0]*Lx)
    fy = exp(1J*k*d[1]*Ly)
    fz = exp(1J*k*d[2]*Lz)
    factors = [fx,fy,fz]
    fes = Periodic(H1(mesh,order=5,complex=True),phase=factors)
    u,v = fes.TnT()

    a = BilinearForm(fes)
    a += SymbolicBFI(grad(u) * grad(v) + u * v)
    f = LinearForm(fes)
    f += SymbolicLFI((k**2+1)*exp(1J*k*(d[0]*x+d[1]*y+d[2]*z)) * v)

    u = GridFunction(fes,name="u")

    with TaskManager():
        a.Assemble()
        f.Assemble()
        u.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

    u_true = exp(1J * k * (d[0] * x + d[1] * y + d[2] * z))
    assert sqrt(Integrate(Conj(u_true-u)*(u_true-u),mesh).real) < 1e-8


# ---------------------------------------------------------------------------
# Periodic meshes whose points are numbered at random.
#
# Facet integrals (dx(skeleton=True), element_boundary with .Other()) map the
# integration points of a facet into both neighbouring elements through their
# vertex numbers. At a periodic facet pair the two elements see two different
# (identified) facets; their vertex numbers are in the same order only if every
# periodic translation preserves the order of the point numbers, as it does for
# lexicographically numbered meshes. The tests below renumber the points of a
# structured periodic mesh at random and check that nothing changes.

import numpy as np
from netgen.meshing import Mesh as NGMesh, MeshPoint, Element2D, Element3D, Element1D, \
    FaceDescriptor, Pnt
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh


def _renumbered(ngmesh, perm):
    """copy of a periodic netgen mesh, its point i moved to position perm[i]"""
    dim = ngmesh.dim
    m = NGMesh(dim=dim)
    pts = [p.p for p in ngmesh.Points()]
    order = np.argsort(perm)
    new = [None] * len(pts)
    for k in order:
        new[k] = m.Add(MeshPoint(Pnt(*pts[k])))
    remap = lambda el: [new[v.nr - 1] for v in el.vertices]
    if dim == 3:
        for i in range(ngmesh.GetNFaceDescriptors()):
            fd = ngmesh.FaceDescriptor(i + 1)
            m.Add(FaceDescriptor(surfnr=i + 1, domin=fd.domin, domout=fd.domout, bc=fd.bc))
        for el in ngmesh.Elements3D():
            m.Add(Element3D(el.index, remap(el)))
        for el in ngmesh.Elements2D():
            m.Add(Element2D(el.index, remap(el)))
    else:
        m.SetMaterial(1, "dom")
        for el in ngmesh.Elements2D():
            m.Add(Element2D(el.index, remap(el)))
        for el in ngmesh.Elements1D():
            m.Add(Element1D(remap(el), index=el.index))
    for a, b, nr in ngmesh.GetIdentifications():
        m.AddPointIdentification(new[a.nr - 1], new[b.nr - 1], identnr=nr, type=2)
    return Mesh(m)


def _periodic_meshes(dim, n=2, nrandom=2, seed=0):
    if dim == 3:
        mesh = MakeStructured3DMesh(hexes=False, nx=n, periodic_x=True, periodic_y=True, periodic_z=True)
    else:
        mesh = MakeStructured2DMesh(quads=False, nx=n, ny=n, periodic_x=True, periodic_y=True)
    rng = np.random.default_rng(seed)
    npts = len(mesh.ngmesh.Points())
    return [mesh] + [_renumbered(mesh.ngmesh, rng.permutation(npts)) for _ in range(nrandom)]


def _field(dim):
    if dim == 3:
        return sin(2 * pi * (x + 2 * y + 0.3)) * cos(2 * pi * (y - z + 0.1)) + sin(2 * pi * (z + x))
    return sin(2 * pi * (x + 2 * y + 0.3)) * cos(2 * pi * (y + 0.1)) + sin(2 * pi * x)


def _jumps(mesh, fes, gf):
    """sum over facets of |[[gf]]|^2, by every assembly path for facet integrals"""
    u, v = fes.TnT()
    jump = lambda w: w - w.Other()
    res = {}
    a = BilinearForm(fes)                                       # facet-wise, assembled
    a += jump(u) * jump(v) * dx(skeleton=True)
    a.Assemble()
    res["facetwise"] = InnerProduct(gf.vec, a.mat * gf.vec)
    a_lin = BilinearForm(fes)                                   # linearization
    a_lin += jump(u) * jump(v) * dx(skeleton=True)
    a_lin.AssembleLinearization(gf.vec)
    res["linearized"] = InnerProduct(gf.vec, a_lin.mat * gf.vec)
    a_na = BilinearForm(fes, nonassemble=True)                  # facet-wise, matrix-free apply
    a_na += jump(u) * jump(v) * dx(skeleton=True)
    tmp = gf.vec.CreateVector()
    a_na.Apply(gf.vec, tmp)
    res["facetwise apply"] = InnerProduct(gf.vec, tmp)
    # element-wise (each element integrates over its whole boundary: each facet twice)
    a_el = BilinearForm(fes)
    a_el += 0.5 * jump(u) * jump(v) * dx(element_boundary=True)
    a_el.Assemble()
    res["elementwise"] = InnerProduct(gf.vec, a_el.mat * gf.vec)
    a_elna = BilinearForm(fes, nonassemble=True)
    a_elna += 0.5 * jump(u) * jump(v) * dx(element_boundary=True)
    a_elna.Apply(gf.vec, tmp)
    res["elementwise apply"] = InnerProduct(gf.vec, tmp)
    f = LinearForm(fes)                                         # facet-wise linear form
    f += jump(gf) * jump(v) * dx(skeleton=True)
    f.Assemble()
    res["linearform"] = InnerProduct(gf.vec, f.vec)
    return res


@pytest.mark.parametrize("dim", [2, 3])
def test_periodic_skeleton_random_numbering(dim):
    # a continuous periodic function has no jumps, also across the periodic facets
    for mesh in _periodic_meshes(dim):
        fes = Periodic(H1(mesh, order=3, dgjumps=True))
        gf = GridFunction(fes)
        gf.Set(_field(dim))
        for path, val in _jumps(mesh, fes, gf).items():
            assert abs(val) < 1e-10, (path, val)


@pytest.mark.parametrize("dim", [2, 3])
def test_periodic_dg_jumps_random_numbering(dim):
    # jumps of the element-wise L2 projection: the same number on every numbering
    ref = None
    for mesh in _periodic_meshes(dim):
        fes = L2(mesh, order=2, dgjumps=True)
        gf = GridFunction(fes)
        gf.Set(_field(dim))
        res = _jumps(mesh, fes, gf)
        vals = np.array(list(res.values()))
        assert np.allclose(vals, vals[0], rtol=1e-10), res
        if ref is None:
            ref = vals[0]
        assert abs(vals[0] - ref) < 1e-10 * ref, (vals[0], ref)


def test_periodic_spaces_random_numbering():
    # higher-order edge/face dofs of identified nodes are matched with their orientation:
    # the L2 projection of a smooth periodic field gives the same error on every numbering
    nv = specialcf.normal(3)
    P = Id(3) - OuterProduct(nv, nv)
    f1 = _field(3)
    f3 = CF((f1, f1 * f1, sin(2 * pi * (z - x))))
    cases = [
        (lambda m: H1(m, order=4), f1, dx, lambda w: w),
        (lambda m: HCurl(m, order=3), f3, dx, lambda w: w),
        (lambda m: HDiv(m, order=3), f3, dx, lambda w: w),
        (lambda m: HCurlCurl(m, order=2), OuterProduct(f3, f3), dx, lambda w: w),
        (lambda m: HDivDiv(m, order=2), OuterProduct(f3, f3), dx, lambda w: w),
        (lambda m: FacetFESpace(m, order=3), f1, dx(element_boundary=True), lambda w: w),
        # only the tangential trace of a TangentialFacetFESpace function is single-valued
        (lambda m: TangentialFacetFESpace(m, order=2), f3, dx(element_boundary=True), lambda w: P * w),
    ]
    meshes = _periodic_meshes(3)
    for space, f, dX, tr in cases:
        errs = []
        for mesh in meshes:
            fes = Periodic(space(mesh))
            u, v = fes.TnT()
            a = BilinearForm(InnerProduct(tr(u), tr(v)) * dX).Assemble()
            l = LinearForm(InnerProduct(tr(f), tr(v)) * dX).Assemble()
            gf = GridFunction(fes)
            gf.vec.data = a.mat.Inverse(fes.FreeDofs()) * l.vec
            errs.append(sqrt(Integrate(InnerProduct(tr(gf - f), tr(gf - f)) * dX, mesh)))
        assert max(errs) - min(errs) < 1e-8 * max(errs), (fes.type, errs)


_dual_set_meshes = {}

def _dual_set_mesh(dim):
    """unit-ish square / box, every pair of opposite sides identified periodically"""
    if dim not in _dual_set_meshes:
        import netgen.occ as occ
        per = occ.IdentificationType.PERIODIC
        if dim == 2:
            shape = occ.Rectangle(1, 1.2).Face()
            sides, maxh = shape.edges, 0.3
            dirs = (occ.X, occ.Y)
        else:
            shape = occ.Box((0, 0, 0), (1, 1.2, 1.5))
            sides, maxh = shape.faces, 0.4
            dirs = (occ.X, occ.Y, occ.Z)
        for d, name in zip(dirs, ("px", "py", "pz")):
            sides.Min(d).Identify(sides.Max(d), name, per)
        mesh = Mesh(occ.OCCGeometry(shape, dim=dim).GenerateMesh(maxh=maxh))
        # no element may contain a vertex together with its periodic image
        fes = Periodic(H1(mesh, order=1))
        for el in fes.Elements(VOL):
            assert len(set(el.dofs)) == len(el.dofs)
        _dual_set_meshes[dim] = mesh
    return _dual_set_meshes[dim]


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("space", [H1, HCurl, HDiv, HCurlCurl, HDivDiv])
def test_periodic_dual_set(space, dim):
    # The dual interpolant Set(cf, dual=True) is element-local, and for a periodic cf the
    # identified dofs get the same value from both sides: Periodic(V) gives the interpolant of V.
    # Spaces caching the inverse duality matrix per vertex-order class must take the class from
    # the vertex numbers PeriodicFESpace::GetFE sets, not from the mesh numbering.
    mesh = _dual_set_mesh(dim)
    s = [sin(2*pi*x), cos(2*pi*y/1.2), sin(2*pi*z/1.5)][:dim]
    if space is H1:
        cf = s[0] * s[1] * (s[2] if dim == 3 else 1) + 0.3
    elif space in (HCurl, HDiv):
        cf = CF(tuple(s[i] * s[(i+1) % dim] + 0.1*(i+1) for i in range(dim)))
    else:
        m = CF(tuple(s[i] * s[j] + 0.2*(i == j) + 0.1*j for i in range(dim) for j in range(dim)), dims=(dim, dim))
        cf = 0.5 * (m + m.trans)
    for order in range(1 if space is H1 else 0, 4):
        gper = GridFunction(Periodic(space(mesh, order=order)))
        gper.Set(cf, dual=True)
        g = GridFunction(space(mesh, order=order))
        g.Set(cf, dual=True)
        err = sqrt(Integrate(InnerProduct(gper - g, gper - g), mesh))
        assert err < 1e-10, f"{space.__name__} order {order}: |periodic - plain| = {err}"
