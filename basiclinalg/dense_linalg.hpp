#ifndef FILE_DENSE_LINALG
#define FILE_DENSE_LINALG

/*
  Dense factorizations and eigenvalue problems of general (row-major) matrices through LAPACK.

  The input matrix is not modified (it is copied to column-major work storage). The outputs are
  allocated by the caller with the sizes given below; k = min(m, n) for an m x n matrix A.
  Conventions follow numpy.linalg:

    DenseEigh      A = V diag(lam) V^H for real symmetric / complex Hermitian A (lower triangle used),
                   lam ascending (n), V (n x n) with the eigenvectors as columns              [?syevd / ?heevd]
    DenseEig       A V = V diag(lam), right eigenvectors as columns, unit 2-norm (n, n x n)    [?geev]
    DenseSVD       A = U diag(s) Vh, s descending (k); full: U m x m, Vh n x n, else U m x k, Vh k x n
                                                                                              [?gesdd]
    DenseQR        A = Q R, Q m x k with orthonormal columns, R k x n upper triangular;
                   with a permutation array of size n (pivoting): A(:, perm) = Q R and
                   |R(0,0)| >= |R(1,1)| >= ... (rank revealing)                     [?geqrf / ?geqp3, ?orgqr / ?ungqr]
    DenseCholesky  A = L L^H, L lower triangular (n x n), A symmetric / Hermitian positive definite [?potrf]
    DenseSolve     A X = B, LU with partial pivoting; B and X are n x nrhs                 [?gesv]

  Errors (singular matrix, not positive definite, no convergence, no LAPACK) throw ngcore::Exception.
*/

namespace ngbla
{
  NGS_DLL_HEADER void DenseEigh (FlatMatrix<double> a, FlatVector<double> lam, FlatMatrix<double> evecs);
  NGS_DLL_HEADER void DenseEigh (FlatMatrix<Complex> a, FlatVector<double> lam, FlatMatrix<Complex> evecs);

  NGS_DLL_HEADER void DenseEig (FlatMatrix<double> a, FlatVector<Complex> lam, FlatMatrix<Complex> evecs);
  NGS_DLL_HEADER void DenseEig (FlatMatrix<Complex> a, FlatVector<Complex> lam, FlatMatrix<Complex> evecs);

  NGS_DLL_HEADER void DenseSVD (FlatMatrix<double> a, FlatMatrix<double> u, FlatVector<double> s,
                                FlatMatrix<double> vh, bool full_matrices);
  NGS_DLL_HEADER void DenseSVD (FlatMatrix<Complex> a, FlatMatrix<Complex> u, FlatVector<double> s,
                                FlatMatrix<Complex> vh, bool full_matrices);

  // perm.Size() == 0: no pivoting; perm.Size() == a.Width(): column pivoting, perm is 0-based
  NGS_DLL_HEADER void DenseQR (FlatMatrix<double> a, FlatMatrix<double> q, FlatMatrix<double> r,
                               FlatArray<int> perm);
  NGS_DLL_HEADER void DenseQR (FlatMatrix<Complex> a, FlatMatrix<Complex> q, FlatMatrix<Complex> r,
                               FlatArray<int> perm);

  NGS_DLL_HEADER void DenseCholesky (FlatMatrix<double> a, FlatMatrix<double> l);
  NGS_DLL_HEADER void DenseCholesky (FlatMatrix<Complex> a, FlatMatrix<Complex> l);

  // x = A^{-1} b; b and x may be the same matrix
  NGS_DLL_HEADER void DenseSolve (FlatMatrix<double> a, FlatMatrix<double> b, FlatMatrix<double> x);
  NGS_DLL_HEADER void DenseSolve (FlatMatrix<Complex> a, FlatMatrix<Complex> b, FlatMatrix<Complex> x);
}

#endif
