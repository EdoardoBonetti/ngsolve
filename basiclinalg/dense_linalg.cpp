/*
  Dense factorizations and eigenvalue problems through LAPACK, see dense_linalg.hpp.

  Every routine copies its (row-major) input to column-major work storage, queries LAPACK for the
  optimal workspace (lwork = -1), runs it, and copies the results to the caller's row-major outputs.
*/

#include <bla.hpp>
#include "dense_linalg.hpp"

namespace ngbla
{
#ifdef LAPACK

  namespace
  {
    inline integer WorkSize (double w) { return max (integer(1), integer(w)); }
    inline integer WorkSize (Complex w) { return max (integer(1), integer(w.real())); }

    void CheckInfo (integer info, const string & routine, const string & failure)
    {
      if (info < 0)
        throw Exception (routine + ": LAPACK argument " + ToString(-info) + " has an illegal value");
      if (info > 0)
        throw Exception (routine + ": " + failure + " (LAPACK info = " + ToString(info) + ")");
    }

    template <typename T> using ColMat = Matrix<T, ColMajor>;
  }


  template <typename T>
  void T_DenseEigh (FlatMatrix<T> a, FlatVector<double> lam, FlatMatrix<T> evecs)
  {
    static Timer t("DenseEigh"); RegionTimer reg(t);
    integer n = a.Height();
    if (a.Width() != size_t(n) || lam.Size() != size_t(n) || evecs.Height() != size_t(n) || evecs.Width() != size_t(n))
      throw Exception ("DenseEigh: needs a square matrix, and n eigenvalues and n x n eigenvectors");
    if (n == 0) return;

    ColMat<T> A(n, n);
    A = a;
    char jobz = 'V', uplo = 'L';
    integer lda = n, info = 0, lwork = -1, liwork = -1, iwork0 = 0;
    T work0 = 0;
    if constexpr (is_same<T, double>::value)
      {
        dsyevd_ (&jobz, &uplo, &n, A.Data(), &lda, lam.Data(), &work0, &lwork, &iwork0, &liwork, &info);
        lwork = WorkSize(work0); liwork = max (integer(1), iwork0);
        Array<double> work(lwork);
        Array<integer> iwork(liwork);
        dsyevd_ (&jobz, &uplo, &n, A.Data(), &lda, lam.Data(), work.Data(), &lwork, iwork.Data(), &liwork, &info);
      }
    else
      {
        integer lrwork = -1;
        double rwork0 = 0;
        zheevd_ (&jobz, &uplo, &n, A.Data(), &lda, lam.Data(), &work0, &lwork, &rwork0, &lrwork,
                 &iwork0, &liwork, &info);
        lwork = WorkSize(work0); lrwork = WorkSize(rwork0); liwork = max (integer(1), iwork0);
        Array<Complex> work(lwork);
        Array<double> rwork(lrwork);
        Array<integer> iwork(liwork);
        zheevd_ (&jobz, &uplo, &n, A.Data(), &lda, lam.Data(), work.Data(), &lwork, rwork.Data(), &lrwork,
                 iwork.Data(), &liwork, &info);
      }
    CheckInfo (info, "DenseEigh", "the eigenvalue algorithm did not converge");
    evecs = A;
  }


  template <typename T>
  void T_DenseEig (FlatMatrix<T> a, FlatVector<Complex> lam, FlatMatrix<Complex> evecs)
  {
    static Timer t("DenseEig"); RegionTimer reg(t);
    integer n = a.Height();
    if (a.Width() != size_t(n) || lam.Size() != size_t(n) || evecs.Height() != size_t(n) || evecs.Width() != size_t(n))
      throw Exception ("DenseEig: needs a square matrix, and n eigenvalues and n x n eigenvectors");
    if (n == 0) return;

    ColMat<T> A(n, n), VR(n, n);
    A = a;
    char jobvl = 'N', jobvr = 'V';
    integer lda = n, ldvl = 1, ldvr = n, info = 0, lwork = -1;
    T work0 = 0, vl0 = 0;
    if constexpr (is_same<T, double>::value)
      {
        Vector<double> wr(n), wi(n);
        dgeev_ (&jobvl, &jobvr, &n, A.Data(), &lda, wr.Data(), wi.Data(), &vl0, &ldvl, VR.Data(), &ldvr,
                &work0, &lwork, &info);
        lwork = WorkSize(work0);
        Array<double> work(lwork);
        dgeev_ (&jobvl, &jobvr, &n, A.Data(), &lda, wr.Data(), wi.Data(), &vl0, &ldvl, VR.Data(), &ldvr,
                work.Data(), &lwork, &info);
        CheckInfo (info, "DenseEig", "the QR algorithm did not converge");
        // a complex pair (wr +- i wi) shares the columns j, j+1 of VR: v_j = VR_j + i VR_{j+1}, v_{j+1} = conj(v_j)
        for (integer j = 0; j < n; j++)
          {
            if (wi(j) == 0 || j+1 == n)
              {
                lam(j) = wr(j);
                for (integer i = 0; i < n; i++)
                  evecs(i,j) = VR(i,j);
              }
            else
              {
                lam(j) = Complex(wr(j), wi(j));
                lam(j+1) = Complex(wr(j+1), wi(j+1));
                for (integer i = 0; i < n; i++)
                  {
                    evecs(i,j) = Complex(VR(i,j), VR(i,j+1));
                    evecs(i,j+1) = Complex(VR(i,j), -VR(i,j+1));
                  }
                j++;
              }
          }
      }
    else
      {
        Array<double> rwork(2*n);
        zgeev_ (&jobvl, &jobvr, &n, A.Data(), &lda, lam.Data(), &vl0, &ldvl, VR.Data(), &ldvr,
                &work0, &lwork, rwork.Data(), &info);
        lwork = WorkSize(work0);
        Array<Complex> work(lwork);
        zgeev_ (&jobvl, &jobvr, &n, A.Data(), &lda, lam.Data(), &vl0, &ldvl, VR.Data(), &ldvr,
                work.Data(), &lwork, rwork.Data(), &info);
        CheckInfo (info, "DenseEig", "the QR algorithm did not converge");
        evecs = VR;
      }
  }


  template <typename T>
  void T_DenseSVD (FlatMatrix<T> a, FlatMatrix<T> u, FlatVector<double> s, FlatMatrix<T> vh, bool full)
  {
    static Timer t("DenseSVD"); RegionTimer reg(t);
    integer m = a.Height(), n = a.Width(), k = min(m, n);
    integer ucols = full ? m : k, vhrows = full ? n : k;
    if (u.Height() != size_t(m) || u.Width() != size_t(ucols) || s.Size() != size_t(k)
        || vh.Height() != size_t(vhrows) || vh.Width() != size_t(n))
      throw Exception ("DenseSVD: output sizes do not match (U m x " + string(full ? "m" : "k")
                       + ", s k, Vh " + string(full ? "n" : "k") + " x n, k = min(m,n))");
    if (k == 0)
      {
        u = Identity(ucols); vh = Identity(vhrows);
        return;
      }

    ColMat<T> A(m, n), U(m, ucols), VT(vhrows, n);
    A = a;
    char jobz = full ? 'A' : 'S';
    integer lda = m, ldu = m, ldvt = vhrows, info = 0, lwork = -1;
    Array<integer> iwork(8*k);
    T work0 = 0;
    if constexpr (is_same<T, double>::value)
      {
        dgesdd_ (&jobz, &m, &n, A.Data(), &lda, s.Data(), U.Data(), &ldu, VT.Data(), &ldvt,
                 &work0, &lwork, iwork.Data(), &info);
        lwork = WorkSize(work0);
        Array<double> work(lwork);
        dgesdd_ (&jobz, &m, &n, A.Data(), &lda, s.Data(), U.Data(), &ldu, VT.Data(), &ldvt,
                 work.Data(), &lwork, iwork.Data(), &info);
      }
    else
      {
        integer lrwork = k * max (5*k+7, 2*max(m,n)+2*k+1);
        Array<double> rwork(lrwork);
        zgesdd_ (&jobz, &m, &n, A.Data(), &lda, s.Data(), U.Data(), &ldu, VT.Data(), &ldvt,
                 &work0, &lwork, rwork.Data(), iwork.Data(), &info);
        lwork = WorkSize(work0);
        Array<Complex> work(lwork);
        zgesdd_ (&jobz, &m, &n, A.Data(), &lda, s.Data(), U.Data(), &ldu, VT.Data(), &ldvt,
                 work.Data(), &lwork, rwork.Data(), iwork.Data(), &info);
      }
    CheckInfo (info, "DenseSVD", "the SVD did not converge");
    u = U;
    vh = VT;
  }


  template <typename T>
  void T_DenseQR (FlatMatrix<T> a, FlatMatrix<T> q, FlatMatrix<T> r, FlatArray<int> perm)
  {
    static Timer t("DenseQR"); RegionTimer reg(t);
    integer m = a.Height(), n = a.Width(), k = min(m, n);
    bool pivoting = perm.Size() > 0;
    if (q.Height() != size_t(m) || q.Width() != size_t(k) || r.Height() != size_t(k) || r.Width() != size_t(n)
        || (pivoting && perm.Size() != size_t(n)))
      throw Exception ("DenseQR: output sizes do not match (Q m x k, R k x n, perm n or 0, k = min(m,n))");
    for (size_t i = 0; i < perm.Size(); i++)
      perm[i] = i;
    if (k == 0) return;

    ColMat<T> A(m, n);
    A = a;
    Vector<T> tau(k);
    integer lda = m, info = 0, lwork = -1;
    T work0 = 0;
    if (pivoting)
      {
        Array<integer> jpvt(n);
        jpvt = 0;      // all columns free
        if constexpr (is_same<T, double>::value)
          {
            dgeqp3_ (&m, &n, A.Data(), &lda, jpvt.Data(), tau.Data(), &work0, &lwork, &info);
            lwork = WorkSize(work0);
            Array<double> work(lwork);
            dgeqp3_ (&m, &n, A.Data(), &lda, jpvt.Data(), tau.Data(), work.Data(), &lwork, &info);
          }
        else
          {
            Array<double> rwork(2*n);
            zgeqp3_ (&m, &n, A.Data(), &lda, jpvt.Data(), tau.Data(), &work0, &lwork, rwork.Data(), &info);
            lwork = WorkSize(work0);
            Array<Complex> work(lwork);
            zgeqp3_ (&m, &n, A.Data(), &lda, jpvt.Data(), tau.Data(), work.Data(), &lwork, rwork.Data(), &info);
          }
        CheckInfo (info, "DenseQR", "pivoted QR failed");
        for (integer j = 0; j < n; j++)
          perm[j] = jpvt[j] - 1;
      }
    else
      {
        if constexpr (is_same<T, double>::value)
          {
            dgeqrf_ (&m, &n, A.Data(), &lda, tau.Data(), &work0, &lwork, &info);
            lwork = WorkSize(work0);
            Array<double> work(lwork);
            dgeqrf_ (&m, &n, A.Data(), &lda, tau.Data(), work.Data(), &lwork, &info);
          }
        else
          {
            zgeqrf_ (&m, &n, A.Data(), &lda, tau.Data(), &work0, &lwork, &info);
            lwork = WorkSize(work0);
            Array<Complex> work(lwork);
            zgeqrf_ (&m, &n, A.Data(), &lda, tau.Data(), work.Data(), &lwork, &info);
          }
        CheckInfo (info, "DenseQR", "QR failed");
      }

    // R: the upper triangle of the first k rows; then Q from the Householder reflectors
    r = T(0);
    for (integer i = 0; i < k; i++)
      for (integer j = i; j < n; j++)
        r(i,j) = A(i,j);

    lwork = -1;
    if constexpr (is_same<T, double>::value)
      {
        dorgqr_ (&m, &k, &k, A.Data(), &lda, tau.Data(), &work0, &lwork, &info);
        lwork = WorkSize(work0);
        Array<double> work(lwork);
        dorgqr_ (&m, &k, &k, A.Data(), &lda, tau.Data(), work.Data(), &lwork, &info);
      }
    else
      {
        zungqr_ (&m, &k, &k, A.Data(), &lda, tau.Data(), &work0, &lwork, &info);
        lwork = WorkSize(work0);
        Array<Complex> work(lwork);
        zungqr_ (&m, &k, &k, A.Data(), &lda, tau.Data(), work.Data(), &lwork, &info);
      }
    CheckInfo (info, "DenseQR", "forming Q failed");
    for (integer i = 0; i < m; i++)
      for (integer j = 0; j < k; j++)
        q(i,j) = A(i,j);
  }


  template <typename T>
  void T_DenseCholesky (FlatMatrix<T> a, FlatMatrix<T> l)
  {
    static Timer t("DenseCholesky"); RegionTimer reg(t);
    integer n = a.Height();
    if (a.Width() != size_t(n) || l.Height() != size_t(n) || l.Width() != size_t(n))
      throw Exception ("DenseCholesky: needs a square matrix and an n x n output");
    if (n == 0) return;

    ColMat<T> A(n, n);
    A = a;
    char uplo = 'L';
    integer lda = n, info = 0;
    if constexpr (is_same<T, double>::value)
      dpotrf_ (&uplo, &n, A.Data(), &lda, &info);
    else
      zpotrf_ (&uplo, &n, A.Data(), &lda, &info);
    CheckInfo (info, "DenseCholesky", "the matrix is not positive definite");
    l = T(0);
    for (integer j = 0; j < n; j++)
      for (integer i = j; i < n; i++)
        l(i,j) = A(i,j);
  }


  template <typename T>
  void T_DenseSolve (FlatMatrix<T> a, FlatMatrix<T> b, FlatMatrix<T> x)
  {
    static Timer t("DenseSolve"); RegionTimer reg(t);
    integer n = a.Height(), nrhs = b.Width();
    if (a.Width() != size_t(n) || b.Height() != size_t(n) || x.Height() != b.Height() || x.Width() != b.Width())
      throw Exception ("DenseSolve: needs a square n x n matrix, and n x nrhs right-hand side and solution");
    if (n == 0 || nrhs == 0) return;

    ColMat<T> A(n, n), B(n, nrhs);
    A = a;
    B = b;
    Array<integer> ipiv(n);
    integer lda = n, ldb = n, info = 0;
    if constexpr (is_same<T, double>::value)
      dgesv_ (&n, &nrhs, A.Data(), &lda, ipiv.Data(), B.Data(), &ldb, &info);
    else
      zgesv_ (&n, &nrhs, A.Data(), &lda, ipiv.Data(), B.Data(), &ldb, &info);
    CheckInfo (info, "DenseSolve", "the matrix is singular");
    x = B;
  }

#else // no LAPACK

  namespace
  {
    [[noreturn]] void NoLapack (const string & routine)
    {
      throw Exception (routine + " needs NGSolve built with LAPACK (USE_LAPACK=ON)");
    }
  }
  template <typename T> void T_DenseEigh (FlatMatrix<T>, FlatVector<double>, FlatMatrix<T>) { NoLapack("DenseEigh"); }
  template <typename T> void T_DenseEig (FlatMatrix<T>, FlatVector<Complex>, FlatMatrix<Complex>) { NoLapack("DenseEig"); }
  template <typename T> void T_DenseSVD (FlatMatrix<T>, FlatMatrix<T>, FlatVector<double>, FlatMatrix<T>, bool) { NoLapack("DenseSVD"); }
  template <typename T> void T_DenseQR (FlatMatrix<T>, FlatMatrix<T>, FlatMatrix<T>, FlatArray<int>) { NoLapack("DenseQR"); }
  template <typename T> void T_DenseCholesky (FlatMatrix<T>, FlatMatrix<T>) { NoLapack("DenseCholesky"); }
  template <typename T> void T_DenseSolve (FlatMatrix<T>, FlatMatrix<T>, FlatMatrix<T>) { NoLapack("DenseSolve"); }

#endif


  void DenseEigh (FlatMatrix<double> a, FlatVector<double> lam, FlatMatrix<double> evecs) { T_DenseEigh (a, lam, evecs); }
  void DenseEigh (FlatMatrix<Complex> a, FlatVector<double> lam, FlatMatrix<Complex> evecs) { T_DenseEigh (a, lam, evecs); }

  void DenseEig (FlatMatrix<double> a, FlatVector<Complex> lam, FlatMatrix<Complex> evecs) { T_DenseEig (a, lam, evecs); }
  void DenseEig (FlatMatrix<Complex> a, FlatVector<Complex> lam, FlatMatrix<Complex> evecs) { T_DenseEig (a, lam, evecs); }

  void DenseSVD (FlatMatrix<double> a, FlatMatrix<double> u, FlatVector<double> s, FlatMatrix<double> vh, bool full)
  { T_DenseSVD (a, u, s, vh, full); }
  void DenseSVD (FlatMatrix<Complex> a, FlatMatrix<Complex> u, FlatVector<double> s, FlatMatrix<Complex> vh, bool full)
  { T_DenseSVD (a, u, s, vh, full); }

  void DenseQR (FlatMatrix<double> a, FlatMatrix<double> q, FlatMatrix<double> r, FlatArray<int> perm) { T_DenseQR (a, q, r, perm); }
  void DenseQR (FlatMatrix<Complex> a, FlatMatrix<Complex> q, FlatMatrix<Complex> r, FlatArray<int> perm) { T_DenseQR (a, q, r, perm); }

  void DenseCholesky (FlatMatrix<double> a, FlatMatrix<double> l) { T_DenseCholesky (a, l); }
  void DenseCholesky (FlatMatrix<Complex> a, FlatMatrix<Complex> l) { T_DenseCholesky (a, l); }

  void DenseSolve (FlatMatrix<double> a, FlatMatrix<double> b, FlatMatrix<double> x) { T_DenseSolve (a, b, x); }
  void DenseSolve (FlatMatrix<Complex> a, FlatMatrix<Complex> b, FlatMatrix<Complex> x) { T_DenseSolve (a, b, x); }
}
