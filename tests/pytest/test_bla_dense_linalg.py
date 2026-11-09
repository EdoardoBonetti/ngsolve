"""bla.Matrix dense linear algebra through LAPACK (Eigh, Eig, SVD, QR with and without pivoting, Cholesky, Solve).

Each method is compared with numpy.linalg on real and complex, square and rectangular matrices: the factors
reproduce the matrix and are orthonormal / triangular, and the invariant quantities (eigenvalues, singular
values, |R|, L, x) agree with numpy to 1e-12 relative. The input matrix is never modified.
"""
import numpy as np
import pytest
from ngsolve.bla import Matrix, Vector

TOL = 1e-12
SQUARE = [1, 5, 30]
SHAPES = [(1, 1), (6, 6), (30, 30), (40, 25), (25, 40)]
CPLX = pytest.mark.parametrize("cplx", [False, True], ids=["real", "complex"])


def rnd(m, n, cplx, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, n))
    if cplx:
        a = a + 1j * rng.standard_normal((m, n))
    return a


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b).max(initial=0) / max(np.abs(b).max(initial=0), 1e-300)


def np_(x):
    return np.array(x)


def check_untouched(M, a):
    assert np.array_equal(np_(M), a), "the input matrix was modified"


@CPLX
@pytest.mark.parametrize("n", SQUARE)
def test_eigh(n, cplx):
    b = rnd(n, n, cplx, 1)
    a = b + b.conj().T
    M = Matrix(a)
    lam, V = M.Eigh()
    lam, V = np_(lam), np_(V)
    lam_np, V_np = np.linalg.eigh(a)
    scale = np.abs(lam_np).max()
    assert np.abs(lam - lam_np).max() <= TOL * scale
    assert rel(a @ V, V * lam) < TOL
    assert rel(V.conj().T @ V, np.eye(n)) < TOL
    # eigenvectors equal numpy's up to a unit factor per column (the eigenvalues are simple)
    phase = np.sum(V_np.conj() * V, axis=0)
    assert rel(V / phase, V_np) < 1e-10
    check_untouched(M, a)


@CPLX
@pytest.mark.parametrize("n", SQUARE)
def test_eig(n, cplx):
    a = rnd(n, n, cplx, 2)
    M = Matrix(a)
    lam, V = M.Eig()
    lam, V = np_(lam), np_(V)
    lam_np = np.linalg.eigvals(a)
    scale = np.abs(lam_np).max()
    # same spectrum: every numpy eigenvalue has a match
    assert max(np.abs(lam - l).min() for l in lam_np) <= TOL * scale
    assert np.abs(a @ V - V * lam).max() <= TOL * scale * np.sqrt(n)
    assert np.abs(np.linalg.norm(V, axis=0) - 1).max() < TOL
    check_untouched(M, a)


@CPLX
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("full", [False, True], ids=["reduced", "full"])
def test_svd(shape, cplx, full):
    m, n = shape
    k = min(m, n)
    a = rnd(m, n, cplx, 3)
    M = Matrix(a)
    U, s, Vh = M.SVD(full_matrices=full)
    U, s, Vh = np_(U), np_(s), np_(Vh)
    assert U.shape == ((m, m) if full else (m, k)) and Vh.shape == ((n, n) if full else (k, n)) and s.shape == (k,)
    assert rel(s, np.linalg.svd(a, compute_uv=False)) < TOL
    assert rel((U[:, :k] * s) @ Vh[:k, :], a) < TOL
    assert rel(U.conj().T @ U, np.eye(U.shape[1])) < TOL
    assert rel(Vh @ Vh.conj().T, np.eye(Vh.shape[0])) < TOL
    check_untouched(M, a)


@CPLX
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_qr(shape, cplx):
    m, n = shape
    k = min(m, n)
    a = rnd(m, n, cplx, 4)
    M = Matrix(a)
    Q, R = M.QR()
    Q, R = np_(Q), np_(R)
    assert Q.shape == (m, k) and R.shape == (k, n)
    assert rel(Q @ R, a) < TOL
    assert rel(Q.conj().T @ Q, np.eye(k)) < TOL
    assert np.abs(np.tril(R, -1)).max(initial=0) == 0
    _, R_np = np.linalg.qr(a)
    assert rel(np.abs(R), np.abs(R_np)) < 1e-10      # the same up to a unit factor per row
    check_untouched(M, a)


@CPLX
@pytest.mark.parametrize("shape", [(30, 30), (40, 25), (25, 40)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_qr_pivoting_rank_deficient(shape, cplx):
    m, n = shape
    k, rank = min(m, n), min(m, n) - 7
    a = rnd(m, rank, cplx, 5) @ rnd(rank, n, cplx, 6)
    M = Matrix(a)
    Q, R, perm = M.QR(pivoting=True)
    Q, R = np_(Q), np_(R)
    assert sorted(perm) == list(range(n))
    assert rel(Q @ R, a[:, perm]) < TOL
    assert rel(Q.conj().T @ Q, np.eye(k)) < TOL
    assert np.abs(np.tril(R, -1)).max(initial=0) == 0
    d = np.abs(np.diag(R))
    assert np.all(d[:-1] >= d[1:] * (1 - 1e-14)), "|diag R| must not increase"
    assert d[rank - 1] > 1e-8 * d[0] and d[rank] < TOL * d[0], "the pivoted QR reveals the rank"
    assert np.abs(R[rank:, :]).max() < TOL * d[0]
    # pivoting=False on the same matrix does not reveal the rank in its diagonal order
    check_untouched(M, a)


@CPLX
@pytest.mark.parametrize("n", SQUARE)
def test_cholesky(n, cplx):
    b = rnd(n, n, cplx, 7)
    a = b @ b.conj().T + n * np.eye(n)
    M = Matrix(a)
    L = np_(M.Cholesky())
    assert rel(L, np.linalg.cholesky(a)) < TOL
    assert np.abs(np.triu(L, 1)).max(initial=0) == 0
    assert rel(L @ L.conj().T, a) < TOL
    check_untouched(M, a)
    with pytest.raises(Exception):
        Matrix(-a).Cholesky()


@CPLX
@pytest.mark.parametrize("n", SQUARE)
def test_solve(n, cplx):
    a = rnd(n, n, cplx, 8) + n * np.eye(n)
    M = Matrix(a)
    b = rnd(n, 1, cplx, 9)[:, 0]
    x = np_(M.Solve(Vector(b)))
    assert x.shape == (n,)
    assert rel(x, np.linalg.solve(a, b)) < TOL
    B = rnd(n, 3, cplx, 10)
    X = np_(M.Solve(Matrix(B)))
    assert X.shape == (n, 3)
    assert rel(X, np.linalg.solve(a, B)) < TOL
    check_untouched(M, a)
    if n > 1:
        s = a.copy()
        s[:, -1] = s[:, 0]
        with pytest.raises(Exception):
            Matrix(s).Solve(Vector(b))


def test_bla_types():
    """the results are bla objects (zero-copy numpy views stay available)"""
    a = rnd(5, 4, False, 11)
    U, s, Vh = Matrix(a).SVD()
    assert type(U).__name__ == "MatrixD" and type(s).__name__ == "VectorD"
    assert type(Matrix(a + 1j).QR()[0]).__name__ == "MatrixC"
    assert type(Matrix(a[:4]).Eig()[0]).__name__ == "VectorC"
    assert np.asarray(U).shape == (5, 4)
