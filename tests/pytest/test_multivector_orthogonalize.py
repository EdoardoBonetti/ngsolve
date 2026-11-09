"""MultiVector.Orthogonalize: orthonormal columns and a correct R factor (V_old = V_new R, R upper triangular).

Without ipmat, Orthogonalize recurses on two halves of the vectors. It used to apply the coupling block of the
two halves transposed, which is invisible as long as the first half holds a single vector (n = 2, 3) and gives
off-diagonal Gram entries of 1e-2 to 1e-1 from n = 4 on; the returned R factor had that block below the diagonal.
"""
import numpy as np
import pytest
from netgen.geom2d import unit_square
from ngsolve import BilinearForm, GridFunction, H1, Mesh, MultiVector, dx

NS = list(range(1, 17))


@pytest.fixture(scope="module")
def spaces():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
    out = {}
    for cplx in (False, True):
        fes = H1(mesh, order=2, complex=cplx)
        u, v = fes.TnT()
        mass = BilinearForm(u * v * dx).Assemble().mat
        out[cplx] = (fes, mass)
    return out


def columns(mv, n):
    return np.column_stack([np.array(mv[i].FV(), copy=True) for i in range(n)])


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("cplx", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("with_ipmat", [False, True], ids=["euclidean", "ipmat"])
def test_orthogonalize(spaces, n, cplx, with_ipmat):
    fes, mass = spaces[cplx]
    rng = np.random.default_rng(1000 * n + 10 * cplx + with_ipmat)
    gf = GridFunction(fes)
    mv = MultiVector(gf.vec, n)
    for i in range(n):
        x = rng.standard_normal(fes.ndof)
        if cplx:
            x = x + 1j * rng.standard_normal(fes.ndof)
        mv[i].FV().NumPy()[:] = x
    V_old = columns(mv, n)

    R = np.array(mv.Orthogonalize(mass if with_ipmat else None))
    V_new = columns(mv, n)

    if with_ipmat:  # Gram matrix in the ipmat inner product: V^H M V
        MV = MultiVector(gf.vec, n)
        for i in range(n):
            MV[i].data = mass * mv[i]
        G = V_new.conj().T @ columns(MV, n)
    else:
        G = V_new.conj().T @ V_new
    gram_err = np.abs(G - np.eye(n)).max()
    assert gram_err < 1e-12, f"n={n}: max |G - I| = {gram_err:.2e}"

    assert R.shape == (n, n)
    assert np.abs(np.tril(R, -1)).max(initial=0) == 0, "R factor is not upper triangular"
    r_err = np.abs(V_old - V_new @ R).max() / np.abs(V_old).max()
    assert r_err < 1e-12, f"n={n}: max |V_old - V_new R| / max |V_old| = {r_err:.2e}"
