"""Fingerprints of the major NGSolve/Netgen features, for comparing two builds.

Each feature is a function returning a dict {key: numpy array} of results that
must not change between builds (meshes, topology, dof counts, matrices,
solutions, iteration counts, eigenvalues, adaptive-loop histories).  The
wall time of each feature is recorded too.

    python fingerprints.py OUT.npz          record with the build on PYTHONPATH
    python fingerprints.py --list           list the features
    NGS_FP_SERIAL=1 python fingerprints.py OUT.npz   deterministic (serial) recording

Compare two recordings with compare.py, or run test_crossversion.py with
NGS_CROSSVERSION_REF=OUT.npz.
"""
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
from ngsolve import *
from netgen.geom2d import unit_square
from netgen.csg import unit_cube

ngsglobals.msg_level = 0

# NGS_FP_SERIAL=1: run without TaskManager. Serial runs are deterministic, so two
# builds with the same numerics must then agree bit for bit (compare.py --exact).
TM = nullcontext if os.environ.get("NGS_FP_SERIAL") == "1" else TaskManager

FEATURES = {}
# Features whose results vary at round-off level between two runs of the *same*
# build even in serial mode (the sparse direct factorizations use threaded BLAS).
# They are compared with a tolerance in --exact mode as well.
NOT_REPRODUCIBLE = {"direct_solvers", "iterative_solvers", "eigenvalues", "adaptive_loop_zz"}


def feature(f):
    FEATURES[f.__name__] = f
    return f


# ---------------------------------------------------------------- helpers
def mesh_arrays(mesh, pre):
    ng = mesh.ngmesh
    out = {pre + "points": np.array([p.p for p in ng.Points()]),
           pre + "vol": np.array([[v.nr for v in el.vertices] for el in mesh.Elements(VOL)]),
           pre + "volindex": np.array([el.index for el in mesh.Elements(VOL)]),
           pre + "bnd": np.array([[v.nr for v in el.vertices] for el in mesh.Elements(BND)]),
           pre + "bndindex": np.array([el.index for el in mesh.Elements(BND)]),
           pre + "edges": np.array([[v.nr for v in e.vertices] for e in mesh.edges])}
    if mesh.dim == 3:
        out[pre + "faces"] = np.array([sorted(v.nr for v in f.vertices) for f in mesh.faces])
    return out


def csr(mat, pre):
    r, c, v = mat.COO()
    r, c, v = np.array(r), np.array(c), np.array(v)
    order = np.lexsort((c, r))
    return {pre + "rows": r[order], pre + "cols": c[order], pre + "vals": v[order]}


def mark_near(mesh, point, frac):
    """Mark the fraction frac of elements closest to point (deterministic)."""
    P = np.array([v.point for v in mesh.vertices])
    centers = np.array([P[[v.nr for v in el.vertices]].mean(axis=0) for el in mesh.Elements(VOL)])
    d = np.linalg.norm(centers - np.array(point[:mesh.dim]), axis=1)
    cut = np.sort(d)[int(frac * len(d))]
    for el in mesh.Elements(VOL):
        mesh.SetRefinementFlag(el, bool(d[el.nr] <= cut))


def lshape(maxh):
    from netgen.occ import WorkPlane, OCCGeometry
    wp = (WorkPlane().MoveTo(-1, -1).LineTo(0, -1).LineTo(0, 0)
          .LineTo(1, 0).LineTo(1, 1).LineTo(-1, 1).Close())
    return Mesh(OCCGeometry(wp.Face(), dim=2).GenerateMesh(maxh=maxh))


# ---------------------------------------------------------------- meshing
@feature
def mesh_generation():
    from netgen.occ import Box, Sphere, Pnt, OCCGeometry
    out = {}
    out.update(mesh_arrays(Mesh(unit_square.GenerateMesh(maxh=0.05)), "sq/"))
    out.update(mesh_arrays(Mesh(unit_cube.GenerateMesh(maxh=0.2)), "cube/"))
    out.update(mesh_arrays(lshape(0.1), "lshape/"))
    shape = Box(Pnt(0, 0, 0), Pnt(1, 1, 1)) - Sphere(Pnt(1, 1, 1), 0.5)
    out.update(mesh_arrays(Mesh(OCCGeometry(shape).GenerateMesh(maxh=0.3)), "occ3d/"))
    return out


@feature
def refine_uniform():
    out = {}
    m2 = Mesh(unit_square.GenerateMesh(maxh=0.2))
    m3 = Mesh(unit_cube.GenerateMesh(maxh=0.4))
    for lev in range(3):
        m2.Refine(); m3.Refine()
        out.update(mesh_arrays(m2, f"2d/L{lev}/"))
        out.update(mesh_arrays(m3, f"3d/L{lev}/"))
    return out


@feature
def refine_adaptive():
    out = {}
    for name, mesh, nref in [("2d", Mesh(unit_square.GenerateMesh(maxh=0.2)), 6),
                             ("3d", Mesh(unit_cube.GenerateMesh(maxh=0.4)), 4),
                             ("occ", lshape(0.3), 6)]:
        for onlyonce in (False, True):
            m = Mesh(mesh.ngmesh.Copy())
            for lev in range(nref):
                mark_near(m, (0.0, 0.0, 0.0) if name == "occ" else (0.3, 0.4, 0.5), 0.2)
                m.Refine(onlyonce=onlyonce)
            pre = f"{name}/once{int(onlyonce)}/"
            out.update(mesh_arrays(m, pre))
            out[pre + "vparents"] = np.array([m.GetParentVertices(v) for v in range(m.nv)])
            out[pre + "eparents"] = np.array([m.GetParentElement(el).nr for el in m.Elements(VOL)])
    return out


@feature
def curved_geometry():
    from netgen.occ import Circle, OCCGeometry, Sphere, Pnt
    out = {}
    disk = Mesh(OCCGeometry(Circle((0, 0), 1).Face(), dim=2).GenerateMesh(maxh=0.3))
    ball = Mesh(OCCGeometry(Sphere(Pnt(0, 0, 0), 1)).GenerateMesh(maxh=0.5))
    for lev in range(2):
        for name, m in (("disk", disk), ("ball", ball)):
            m.Curve(3)
            out[f"{name}/L{lev}/measure"] = np.array([Integrate(1, m), Integrate(1, m, BND)])
            m.Refine()
    return out


# ---------------------------------------------------------------- spaces
@feature
def fespace_ndofs():
    m2 = Mesh(unit_square.GenerateMesh(maxh=0.2))
    m3 = Mesh(unit_cube.GenerateMesh(maxh=0.4))
    spaces = {"H1": H1, "L2": L2, "HDiv": HDiv, "HCurl": HCurl, "VectorH1": VectorH1,
              "Facet": FacetFESpace, "HDivDiv": HDivDiv, "HCurlCurl": HCurlCurl}
    out = {}
    for mname, m in (("2d", m2), ("3d", m3)):
        for sname, S in spaces.items():
            nd = []
            for p in range(1, 5):
                fes = S(m, order=p)
                nd.append([fes.ndof, sum(fes.FreeDofs()), sum(fes.FreeDofs(True))])
            out[f"{mname}/{sname}"] = np.array(nd)
        fes = H1(m, order=3, dirichlet=".*") * HDiv(m, order=2) * NumberSpace(m)
        out[f"{mname}/compound"] = np.array([fes.ndof, sum(fes.FreeDofs()), sum(fes.FreeDofs(True))])
    return out


# ---------------------------------------------------------------- assembly
@feature
def assemble_matrices():
    m2 = Mesh(unit_square.GenerateMesh(maxh=0.1))
    m3 = Mesh(unit_cube.GenerateMesh(maxh=0.3))
    out = {}
    with TM():
        fes = H1(m2, order=3, dirichlet="left|bottom")
        u, v = fes.TnT()
        a = BilinearForm((Grad(u) * Grad(v) + (1 + x * y) * u * v) * dx + u * v * ds("right")).Assemble()
        out.update(csr(a.mat, "h1/"))
        f = LinearForm(sin(3 * x) * v * dx + y * v * ds("top")).Assemble()
        out["h1/f"] = f.vec.FV().NumPy().copy()

        fes = HCurl(m3, order=2)
        u, v = fes.TnT()
        a = BilinearForm(curl(u) * curl(v) * dx + u * v * dx).Assemble()
        out.update(csr(a.mat, "hcurl/"))

        V, Q = HDiv(m2, order=2), L2(m2, order=1)
        X = V * Q
        (s, p), (t, q) = X.TnT()
        a = BilinearForm((s * t + div(s) * q + div(t) * p) * dx).Assemble()
        out.update(csr(a.mat, "mixed/"))

        fes = L2(m2, order=2, dgjumps=True)
        u, v = fes.TnT()
        n, h = specialcf.normal(2), specialcf.mesh_size
        jump = lambda w: w - w.Other()
        avg = lambda w: 0.5 * (grad(w) + grad(w.Other())) * n
        a = BilinearForm(grad(u) * grad(v) * dx
                         + (10 * 4 / h * jump(u) * jump(v) - avg(u) * jump(v) - avg(v) * jump(u)) * dx(skeleton=True)
                         + (10 * 4 / h * u * v - grad(u) * n * v - grad(v) * n * u) * ds(skeleton=True)).Assemble()
        out.update(csr(a.mat, "dg/"))

        fes = H1(m2, order=4, dirichlet=".*")
        u, v = fes.TnT()
        a = BilinearForm(Grad(u) * Grad(v) * dx, condense=True).Assemble()
        out.update(csr(a.mat, "condensed/"))
        g = GridFunction(fes)
        g.Set(x * (1 - x) * y)
        w = g.vec.CreateVector()
        w.data = a.harmonic_extension * g.vec
        out["condensed/hext"] = w.FV().NumPy().copy()
        w.data = a.inner_solve * g.vec
        out["condensed/isolve"] = w.FV().NumPy().copy()
    return out


# ---------------------------------------------------------------- solvers
def poisson(mesh, p, **kw):
    fes = H1(mesh, order=p, dirichlet=".*", **kw)
    u, v = fes.TnT()
    a = BilinearForm(Grad(u) * Grad(v) * dx)
    f = LinearForm(10 * v * dx)
    return fes, a, f


@feature
def direct_solvers():
    m = Mesh(unit_square.GenerateMesh(maxh=0.05))
    out = {}
    with TM():
        fes, a, f = poisson(m, 3)
        a.Assemble(); f.Assemble()
        g = GridFunction(fes)
        for inv in ("umfpack", "sparsecholesky", "pardiso"):
            try:
                g.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse=inv) * f.vec
            except Exception:
                continue
            out[inv] = g.vec.FV().NumPy().copy()
    return out


@feature
def iterative_solvers():
    out = {}
    with TM():
        m = Mesh(unit_square.GenerateMesh(maxh=0.05))
        fes, a, f = poisson(m, 2)
        for pre in ("local", "bddc", "h1amg"):
            c = Preconditioner(a, pre)
            a.Assemble(); f.Assemble()
            g = GridFunction(fes)
            sol = solvers.CGSolver(a.mat, c.mat, tol=1e-10, maxiter=500)
            g.vec.data = sol * f.vec
            out[f"{pre}/its"] = np.array([sol.iterations])
            out[f"{pre}/u"] = g.vec.FV().NumPy().copy()
        g = GridFunction(fes)
        a.Assemble()
        sol = solvers.GMRes(a.mat, f.vec, pre=Projector(fes.FreeDofs(), True), x=g.vec,
                            tol=1e-10, maxsteps=2000, printrates=False)
        out["gmres/u"] = g.vec.FV().NumPy().copy()
    return out


@feature
def multigrid_refinement():
    """Multigrid preconditioner over refined levels, for low- and high-order
    prolongations (touches the prolongation code)."""
    out = {}
    with TM():
        for p, kw in ((1, {}), (3, {}), (3, {"hoprolongation": True})):
            m = Mesh(unit_square.GenerateMesh(maxh=0.3))
            fes, a, f = poisson(m, p, **kw)
            c = Preconditioner(a, "multigrid", coarsetype="direct",
                               smoother="block" if p > 1 else "point")
            a.Assemble()
            its = []
            for lev in range(4):
                mark_near(m, (0.2, 0.3, 0), 0.3)
                m.Refine()
                a.Assemble(); f.Assemble()
                g = GridFunction(fes)
                sol = solvers.CGSolver(a.mat, c.mat, tol=1e-10, maxiter=300)
                g.vec.data = sol * f.vec
                its.append(sol.iterations)
            key = f"p{p}{'ho' if kw else ''}"
            out[key + "/its"] = np.array(its)
            out[key + "/u"] = g.vec.FV().NumPy().copy()
    return out


@feature
def prolongation_autoupdate():
    out = {}
    with TM():
        for dim in (2, 3):
            for p in (1, 2, 4) if dim == 2 else (1, 2, 3):
                for ho in (False, True):
                    m = Mesh(unit_square.GenerateMesh(maxh=0.3)) if dim == 2 else Mesh(unit_cube.GenerateMesh(maxh=0.5))
                    fes = H1(m, order=p, hoprolongation=ho) if ho else H1(m, order=p)
                    g = GridFunction(fes, autoupdate=True)
                    g.Set(sin(2 * x) * cos(y + 0.3 * z))
                    for lev in range(3):
                        mark_near(m, (0.2, 0.3, 0.4), 0.3)
                        m.Refine()
                    out[f"{dim}d/p{p}/ho{int(ho)}"] = g.vec.FV().NumPy().copy()
    return out


@feature
def interpolation_integration():
    out = {}
    with TM():
        m = Mesh(unit_square.GenerateMesh(maxh=0.1))
        cf = sin(5 * x) * exp(y)
        for S, p in ((H1, 5), (L2, 3), (HCurl, 2), (HDiv, 2)):
            fes = S(m, order=p) if S in (H1, L2) else S(m, order=p)
            g = GridFunction(fes)
            g.Set(cf if S in (H1, L2) else CF((cf, x * y)))
            out[f"set/{S.__name__}"] = g.vec.FV().NumPy().copy()
        g = GridFunction(H1(m, order=3))
        g.Interpolate(cf)
        out["interpolate"] = g.vec.FV().NumPy().copy()
        out["integrals"] = np.array([Integrate(cf, m), Integrate(cf * cf, m, order=12),
                                     Integrate(cf, m, BND), Integrate((g - cf) ** 2, m)])
        mip = m(0.3, 0.7)
        out["pointeval"] = np.array([g(mip), cf(mip)])
        vals = Integrate(cf, m, element_wise=True)
        out["elementwise"] = vals.NumPy().copy()
    return out


@feature
def eigenvalues():
    m = Mesh(unit_square.GenerateMesh(maxh=0.1))
    with TM():
        fes = H1(m, order=3, dirichlet=".*")
        u, v = fes.TnT()
        a = BilinearForm(Grad(u) * Grad(v) * dx).Assemble()
        b = BilinearForm(u * v * dx).Assemble()
        pre = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")
        lams, _ = solvers.PINVIT(a.mat, b.mat, pre, num=6, maxit=40, printrates=False)
    return {"pinvit": np.array(lams)}


@feature
def nonlinear_newton():
    m = Mesh(unit_square.GenerateMesh(maxh=0.1))
    with TM():
        fes = H1(m, order=3, dirichlet=".*")
        u, v = fes.TnT()
        a = BilinearForm((Grad(u) * Grad(v) + u ** 3 * v - 10 * v) * dx)
        g = GridFunction(fes)
        solvers.Newton(a, g, printing=False, maxerr=1e-12)
    return {"u": g.vec.FV().NumPy().copy()}


@feature
def adaptive_loop_zz():
    """Standard ZZ-estimator AFEM loop on the L-shape (refine, autoupdate,
    assemble, solve, estimate, mark)."""
    m = lshape(0.5)
    out = {}
    with TM():
        fes = H1(m, order=3, dirichlet=".*", autoupdate=True)
        u, v = fes.TnT()
        a = BilinearForm(Grad(u) * Grad(v) * dx)
        f = LinearForm(1 * v * dx)
        g = GridFunction(fes, autoupdate=True)
        space_flux = HDiv(m, order=2, autoupdate=True)
        gf_flux = GridFunction(space_flux, autoupdate=True)
        hist = []
        for lev in range(8):
            a.Assemble(); f.Assemble()
            g.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
            gf_flux.Set(grad(g))
            err = Integrate((grad(g) - gf_flux) ** 2, m, VOL, element_wise=True).NumPy()
            hist.append([fes.ndof, np.sqrt(err.sum()), Integrate(g, m)])
            mx = err.max()
            for el in m.Elements():
                m.SetRefinementFlag(el, bool(err[el.nr] > 0.25 * mx))
            m.Refine()
    out["hist"] = np.array(hist)
    out["final_u"] = g.vec.FV().NumPy().copy()
    return out


@feature
def time_stepping_heat():
    m = Mesh(unit_square.GenerateMesh(maxh=0.05))
    with TM():
        fes = H1(m, order=2, dirichlet=".*")
        u, v = fes.TnT()
        dt = 1e-3
        mstar = BilinearForm(u * v * dx + dt * Grad(u) * Grad(v) * dx).Assemble()
        mass = BilinearForm(u * v * dx).Assemble()
        inv = mstar.mat.Inverse(fes.FreeDofs())
        g = GridFunction(fes)
        g.Set(exp(-50 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)), dual=False)
        g.vec.data = Projector(fes.FreeDofs(), True) * g.vec
        r = g.vec.CreateVector()
        for _ in range(50):
            r.data = mass.mat * g.vec
            g.vec.data = inv * r
    return {"u": g.vec.FV().NumPy().copy()}


@feature
def jit_compile():
    m = Mesh(unit_square.GenerateMesh(maxh=0.1))
    cf = (sin(x) * y + x ** 2).Compile(realcompile=True, wait=True)
    return {"int": np.array([Integrate(cf, m)])}


@feature
def pickling_and_io():
    import pickle, tempfile, os
    m = Mesh(unit_square.GenerateMesh(maxh=0.2))
    m.Refine()
    fes = H1(m, order=3)
    g = GridFunction(fes)
    g.Set(x * y * y)
    g2 = pickle.loads(pickle.dumps(g))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.vol.gz")
        m.ngmesh.Save(p)
        m3 = Mesh(p)
    return {"gf": g2.vec.FV().NumPy().copy(), "reloaded": np.array([m3.ne, m3.nv, m3.nedge])}


def record(path, names=None):
    out, times = {}, {}
    import ngsolve
    for name, f in FEATURES.items():
        if names and name not in names:
            continue
        t0 = time.perf_counter()
        try:
            res = f()
            status = "ok"
        except Exception as e:
            res, status = {}, f"error: {type(e).__name__}: {e}"
        times[name] = time.perf_counter() - t0
        for k, v in res.items():
            out[f"{name}::{k}"] = np.asarray(v)
        out[f"{name}::__status"] = np.array(status)
        out[f"{name}::__time"] = np.array(times[name])
        out[f"{name}::__reproducible"] = np.array(name not in NOT_REPRODUCIBLE)
        print(f"{name:28s} {times[name]:8.3f}s  {status}", flush=True)
    out["__version"] = np.array(ngsolve.__version__)
    out["__path"] = np.array(ngsolve.__file__)
    out["__serial"] = np.array(os.environ.get("NGS_FP_SERIAL") == "1")
    np.savez_compressed(path, **out)


if __name__ == "__main__":
    if sys.argv[1] == "--list":
        print("\n".join(FEATURES))
    else:
        record(sys.argv[1], sys.argv[2:])
