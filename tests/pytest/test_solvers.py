from netgen.geom2d import unit_square
from ngsolve import *
import pytest
from ngsolve.krylovspace import *

def test_arnoldi():
    SetHeapSize (10*1000*1000)
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.25))
    fes1 = L2(mesh,order=8,complex=True,dirichlet="top|bottom|left|right")
    fes2 = H1(mesh,order=10,complex=True,dirichlet="top|bottom|left|right")
    fes = FESpace([fes1,fes2])

    u1,u2 = fes.TrialFunction()
    v1,v2 = fes.TestFunction()

    a = BilinearForm(fes)
    a += SymbolicBFI(grad(u2)*grad(v2)+u1*v1)

    m = BilinearForm(fes)
    m += SymbolicBFI(u1*v2+u2*v1)

    u = GridFunction(fes,multidim=40)

    a.Assemble()
    m.Assemble()

    lam = ArnoldiSolver(a.mat,m.mat,fes.FreeDofs(),list(u.vecs),1)
    print("ev = ", lam)
    Draw(u.components[0])
    Draw(u.components[1])

    evec = GridFunction(fes2)

    def laplace(gf):
        hesse = gf.Operator("hesse")
        return hesse[0]+hesse[3]

    for i in range(5):
        evec.vec.data = u.components[1].vecs[i]
        error = Integrate(Norm(laplace(evec) + lam[i].real*lam[i].real*evec),mesh)
        print("error[",i,"] = ",error)
        assert error < 1e-7

    Draw(laplace(evec),mesh,"laplace")

def test_newton_with_dirichlet():
    mesh = Mesh (unit_square.GenerateMesh(maxh=0.3))
    V = H1(mesh, order=3, dirichlet=[1,2,3,4])
    u,v = V.TnT()
    a = BilinearForm(V)
    a += (grad(u) * grad(v) + 3*u**3*v- 1 * v)*dx
    gfu = GridFunction(V)
    dirichlet = GridFunction(V)
    dirichlet.Set(0)
    newton = solvers.Newton(a, gfu, dirichletvalues=dirichlet.vec)


def test_krylovspace_solvers():
    solvers = [CGSolver, GMRESSolver, MinResSolver, TFQMRSolver] # , QMRSolver]
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = H1(mesh, order=4, dirichlet=".*")
    u,v = fes.TnT()
    f = LinearForm(32 * (y*(1-y)+x*(1-x)) * v * dx).Assemble()
    a = BilinearForm(grad(u)*grad(v)*dx)
    c = Preconditioner(a, type="bddc")
    a.Assemble()
    u = GridFunction(fes)
    exact = 16*x*(1-x)*y*(1-y)
    for solver in solvers:
        inv = solver(mat=a.mat, pre=c)
        u.vec.data = inv * f.vec
        error = sqrt(Integrate((u-exact)*(u-exact), mesh))
        print(solver.name, ": iterations = ", inv.iterations)
        print("Error = ", error)
        assert inv.iterations < 40
        # p4 should be exact
        assert error < 1e-12


def _convection_diffusion_pencil(b, sigma, maxh=0.15):
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes = H1(mesh, order=2, complex=True, dirichlet=".*")
    u, v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx + (b*grad(u))*v*dx).Assemble()
    m = BilinearForm(u*v*dx).Assemble()
    ashift = BilinearForm(grad(u)*grad(v)*dx + (b*grad(u))*v*dx - sigma*u*v*dx).Assemble()
    lap = BilinearForm(grad(u)*grad(v)*dx)
    pre = Preconditioner(lap, "bddc")
    lap.Assemble()
    return fes, a, m, ashift, pre


def _check_eigenpairs(fes, a, m, lams, vecs, sigma, num, tol):
    import numpy as np
    import scipy.linalg
    fd = np.array(fes.FreeDofs(), dtype=bool)
    A = a.mat.ToDense().NumPy()[np.ix_(fd, fd)]
    M = m.mat.ToDense().NumPy()[np.ix_(fd, fd)]
    ref = scipy.linalg.eigvals(A, M)
    ref = ref[np.argsort(np.abs(ref - sigma))][:num]
    for lam in lams:
        assert min(abs(lam - ref)) < tol * abs(lam)
    for j in range(num):
        x = vecs[j].FV().NumPy()[fd]
        assert np.linalg.norm(A @ x - lams[j] * (M @ x)) < 1e2 * tol * abs(lams[j]) * np.linalg.norm(M @ x)


def test_gplhr_convection_diffusion():
    # -Laplace u + b.grad u = lam u: real spectrum, strongly non-normal operator
    sigma = 0
    fes, a, m, ashift, pre = _convection_diffusion_pencil(CF((8, 6)), sigma)
    lams, vecs = solvers.GPLHR(a.mat, m.mat, pre, num=3, sigma=sigma, m=1, maxit=100, tol=1e-10,
                               freedofs=fes.FreeDofs(), printrates=False)
    _check_eigenpairs(fes, a, m, lams, vecs, sigma, 3, 1e-8)
    # LOBPCG-style search direction and no Krylov blocks
    # (num=3 keeps the nearly double eigenvalue lam_2 ~ lam_3 together)
    lams, vecs = solvers.GPLHR(a.mat, m.mat, pre, num=3, sigma=sigma, m=0, thick=False, maxit=300, tol=1e-10,
                               freedofs=fes.FreeDofs(), printrates=False)
    _check_eigenpairs(fes, a, m, lams, vecs, sigma, 3, 1e-8)


def test_gplhr_interior_complex_target():
    # rotating flow: complex eigenvalues; complex target inside the spectrum,
    # preconditioner = a few TFQMR steps on (A - sigma M) with BDDC of the Laplacian
    sigma = 50+30j
    fes, a, m, ashift, pre = _convection_diffusion_pencil(30*CF((-(y-0.5), x-0.5)), sigma)
    inner = TFQMRSolver(ashift.mat, pre=pre, maxiter=20, tol=1e-12)
    lams, vecs = solvers.GPLHR(a.mat, m.mat, inner, num=3, sigma=sigma, m=1, maxit=100, tol=1e-10,
                               freedofs=fes.FreeDofs(), printrates=False)
    _check_eigenpairs(fes, a, m, lams, vecs, sigma, 3, 1e-8)


def test_gplhr_needs_complex_vectors():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = H1(mesh, order=1)
    u, v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    m = BilinearForm(u*v*dx).Assemble()
    with pytest.raises(ValueError):
        solvers.GPLHR(a.mat, m.mat, IdentityMatrix(fes.ndof), printrates=False)



if __name__ == "__main__":
    # test_arnoldi()
    test_krylovspace_solvers()
