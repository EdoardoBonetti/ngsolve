# (temporary sparse matrix).Inverse() keeps the matrix alive: sparse
# factorizations hold only a weak_ptr to it and need it for Height/Width,
# e.g. in a product operator inv @ B
from ngsolve import *
import gc
import pytest


@pytest.mark.parametrize("inverse", [None, "pardiso", "umfpack", "sparsecholesky"])
def test_inverse_of_temporary(inverse):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = H1(mesh, order=2, dirichlet=".*") * L2(mesh, order=1)
    u, v = fes.TnT()
    m = BilinearForm(u[0] * v[0] * dx + u[1] * v[1] * dx).Assemble()
    l = BilinearForm(grad(u[0]) * grad(v[0]) * dx + u[1] * v[1] * dx).Assemble()
    kwargs = {} if inverse is None else {"inverse": inverse}
    try:
        inv = (m.mat + 0.1 * l.mat).CreateSparseMatrix().Inverse(fes.FreeDofs(), **kwargs)
    except Exception as e:
        pytest.skip(f"inverse {inverse} not available: {e}")
    gc.collect()
    w = m.mat.CreateColVector()
    w.SetRandom()
    r1 = w.CreateVector()
    r1.data = inv @ l.mat * w
    r2 = w.CreateVector()
    r2.data = inv * (l.mat * w).Evaluate()
    assert (inv.height, inv.width) == (fes.ndof, fes.ndof)
    assert Norm(r1 - r2) < 1e-12 * Norm(r2)
