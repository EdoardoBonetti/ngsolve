from ngsolve.la import InnerProduct, MultiVector
from math import sqrt
from ngsolve import Projector, Norm, Matrix, Vector, IdentityMatrix

def Orthogonalize (vecs, mat):
    mv = []
    for i in range(len(vecs)):
        for j in range(i):
            vecs[i] -= InnerProduct(vecs[i], mv[j]) * vecs[j]
            
        hv = mat.CreateRowVector()
        hv.data = mat * vecs[i]
        norm = sqrt(InnerProduct(vecs[i], hv))
        vecs[i] *= 1/norm
        hv *= 1/norm
        mv.append (hv)


def PINVIT1(mata, matm, pre, num=1, maxit=20, printrates=True, GramSchmidt=False):
    """preconditioned inverse iteration"""
    import scipy.linalg

    r = mata.CreateRowVector()
    Av = mata.CreateRowVector()
    Mv = mata.CreateRowVector()

    uvecs = []
    for i in range(num):
        uvecs.append (mata.CreateRowVector())
    
    vecs = []
    for i in range(2*num):
        vecs.append (mata.CreateRowVector())

    for v in uvecs:
        r.SetRandom()
        v.data = pre * r

    asmall = Matrix(2*num, 2*num)
    msmall = Matrix(2*num, 2*num)
    lams = num * [1]

    for i in range(maxit):
        
        for j in range(num):
            vecs[j].data = uvecs[j]
            r.data = mata * vecs[j] - lams[j] * matm * vecs[j]
            vecs[num+j].data = pre * r

        if GramSchmidt:
            Orthogonalize (vecs, matm)

        for j in range(2*num):
            Av.data = mata * vecs[j]
            Mv.data = matm * vecs[j]
            for k in range(2*num):
                asmall[j,k] = InnerProduct(Av, vecs[k])
                msmall[j,k] = InnerProduct(Mv, vecs[k])

        ev,evec = scipy.linalg.eigh(a=asmall, b=msmall)
        lams[:] = ev[0:num]
        if printrates:
            print (i, ":", lams)
    
        for j in range(num):
            uvecs[j][:] = 0.0
            for k in range(2*num):
                uvecs[j].data += float(evec[k,j]) * vecs[k]

    return lams, uvecs




def PINVIT(mata, matm, pre, num=1, maxit=20, printrates=True, GramSchmidt=True):
    """preconditioned inverse iteration"""
    import scipy.linalg

    r = mata.CreateRowVector()
    
    uvecs = MultiVector(r, num)
    vecs = MultiVector(r, 2*num)
    # hv = MultiVector(r, 2*num)

    for v in vecs[0:num]:
        v.SetRandom()
    uvecs[:] = pre * vecs[0:num]
    lams = Vector(num * [1])
    
    for i in range(maxit):
        vecs[0:num] = mata * uvecs - (matm * uvecs).Scale (lams)
        vecs[num:2*num] = pre * vecs[0:num]
        vecs[0:num] = uvecs

        vecs.Orthogonalize(matm)

        # hv[:] = mata * vecs
        # asmall = InnerProduct (vecs, hv)
        # hv[:] = matm * vecs
        # msmall = InnerProduct (vecs, hv)
        asmall = InnerProduct (vecs, mata * vecs)
        msmall = InnerProduct (vecs, matm * vecs)
    
        ev,evec = scipy.linalg.eigh(a=asmall, b=msmall)
        lams = Vector(ev[0:num])
        if printrates:
            print (i, ":", list(lams))

        uvecs[:] = vecs * Matrix(evec[:,0:num])
    return lams, uvecs


def LOBPCG(mata, matm, pre, num=1, maxit=20, initial=None, printrates=True, largest=False):
    """Knyazev's cg-like extension of PINVIT"""
    import scipy.linalg

    r = mata.CreateRowVector()

    if initial:
        num=len(initial)
        uvecs = initial
    else:
        uvecs = MultiVector(r, num)

    vecs = MultiVector(r, 3*num)
    for v in vecs:
        r.SetRandom()
        v.data = pre * r

    if initial:
         vecs[0:num] = uvecs       
        
    lams = Vector(num * [1])
    
    for i in range(maxit):
        uvecs.data = mata * vecs[0:num] - (matm * vecs[0:num]).Scale (lams)
        vecs[2*num:3*num] = pre * uvecs
        
        vecs.Orthogonalize(matm)

        asmall = InnerProduct (vecs, mata * vecs)
        msmall = InnerProduct (vecs, matm * vecs)
    
        ev,evec = scipy.linalg.eigh(a=asmall, b=msmall)

        if not largest:
            lams = Vector(ev[0:num])
            if printrates:
                print (i, ":", list(lams), flush=True)

            uvecs[:] = vecs * Matrix(evec[:,0:num])
            vecs[num:2*num] = vecs[0:num]
            vecs[0:num] = uvecs
        else:
            lams = Vector(ev[2*num:3*num])
            if printrates:
                print (i, ":", list(lams), flush=True)

            uvecs[:] = vecs * Matrix(evec[:,2*num:3*num])
            vecs[num:2*num] = vecs[0:num]
            vecs[0:num] = uvecs

    return lams, uvecs



def _SortedQZ(Aa, Ba, sigma, nsort):
    """generalized Schur form Aa ZR = QL AA, Ba ZR = QL BB, where the first
    nsort eigenvalues AA[i,i]/BB[i,i] are ordered by increasing distance to sigma"""
    import numpy as np
    import scipy.linalg

    def dist(alpha, beta):
        return np.abs(alpha - sigma*beta) / np.maximum(np.abs(beta), 1e-300)

    AA, BB, QL, ZR = scipy.linalg.qz(Aa, Ba, output='complex')
    for j in range(nsort):
        # moving the j+1 closest eigenvalues to the top keeps the order of the first j
        thres = np.sort(dist(np.diag(AA), np.diag(BB)))[j] * (1+1e-12)
        AA, BB, _, _, Q2, Z2 = scipy.linalg.ordqz(AA, BB, sort=lambda a, b: dist(a, b) <= thres,
                                                  output='complex')
        QL, ZR = QL @ Q2, ZR @ Z2
    return AA, BB, QL, ZR


def _QFreeFactors(RA, RB):
    """triangular MA, MB of the Q-free Schur form A V MB = B V MA, eq. (2.8), (2.11), (2.12)
    in Vecharynski, Yang, Xue (2016)"""
    import numpy as np
    import scipy.linalg

    a, b = np.diag(RA), np.diag(RB)
    small = np.abs(a) < np.abs(b)
    g1 = np.where(small, 0, (1-b) / np.where(small, 1, a))
    g2 = np.where(small, 1 / np.where(small, b, 1), 1)
    GinvRA = scipy.linalg.solve_triangular(RA*g1 + RB*g2, RA)
    MA = g2[:,None] * GinvRA
    MB = np.eye(len(a)) - g1[:,None] * GinvRA
    return MA, MB


def GPLHR(mata, matm, pre, num=1, sigma=0, m=1, blocksize=None, maxit=100, tol=1e-8,
          initial=None, freedofs=None, thick=True, printrates=True, callback=None):
    """Generalized Preconditioned Locally Harmonic Residual method for the
    non-Hermitian generalized eigenvalue problem  mata x = lam matm x.

    Computes the num eigenvalues closest to the target sigma without factorizing
    mata - sigma*matm. Algorithm 2 of E. Vecharynski, C. Yang, F. Xue, "Generalized
    preconditioned locally harmonic residual method for non-Hermitian eigenproblems",
    SIAM J. Sci. Comput. 38(1), A500-A527 (2016):

      - iterates on a partial generalized Schur form (A V = Q RA, B V = Q RB),
        which stays robust for strongly non-normal problems
      - trial space Z = [V, W, S_1..S_m, P], generated by the projected
        preconditioner (I - V V*) pre (I - Q Q*) applied to the residual
      - harmonic Schur-Rayleigh-Ritz extraction with test space (A - sigma B) Z
      - P is the next block of harmonic Schur vectors ("thick" restart)

    Only products with mata, matm and pre are needed, no transposes.

    Vectors must be complex (use a complex FESpace): eigenvalues of non-Hermitian
    problems are complex in general, and real operators applied to complex vectors
    do not give correct results.

    Parameters
    ----------

    mata, matm : BaseMatrix
      The pencil; mata does not need to be symmetric.

    pre : BaseMatrix
      Approximation of (mata - sigma*matm)^-1, e.g. a BDDC preconditioner, or a few
      steps of a Krylov solver preconditioned by it (TFQMRSolver, GMResSolver, ...).
      Interior targets need a stronger pre than exterior ones.

    num : int
      Number of wanted eigenvalues.

    sigma : complex
      Target; should not be an eigenvalue.

    m : int
      Number of additional Krylov blocks S_1..S_m in the trial space.

    blocksize : int
      Number of Schur vectors iterated (>= num), default num.

    maxit : int
      Maximal number of iterations.

    tol : float
      Tolerance for the relative column-wise residual of the Schur form,
      |A v_j - Q RA[:,j]| / |RA[:,j]| + |B v_j - Q RB[:,j]| / |RB[:,j]|.
      Both equations are scaled separately, since |matm| << |mata| for FEM matrices.

    initial : MultiVector
      Initial guess (default: pre applied to random vectors).

    freedofs : BitArray
      Restricts the problem to free dofs; needed if there are Dirichlet dofs.

    thick : bool
      P from the next block of harmonic Schur vectors (True, recommended),
      or LOBPCG-style from the previous update (False).

    printrates : bool
      Print iteration, number of converged eigenvalues, residual and eigenvalues.

    callback : Callable[[int, int, numpy.ndarray, numpy.ndarray], None]
      Called in every iteration with (iteration, number of converged eigenvalues,
      current eigenvalue approximations, residuals of the blocksize Schur vectors).

    Returns
    -------
    (Vector, MultiVector)
      The num eigenvalues closest to sigma, ordered by distance to sigma, and
      the eigenvectors, normalized with respect to matm.
    """
    import numpy as np

    k = blocksize if blocksize else num
    if k < num:
        raise ValueError("blocksize must be at least num")
    tmp = mata.CreateColVector()
    if not tmp.is_complex:
        raise ValueError("GPLHR needs complex vectors, use a complex FESpace")

    proj = Projector(freedofs, True) if freedofs is not None else None
    A, B = (proj @ mata @ proj, proj @ matm @ proj) if proj else (mata, matm)

    def Ev(expr):
        return expr.Evaluate()

    def apply(op, X):   # result in vectors of the type of mata (e.g. Projector creates real ones)
        R = MultiVector(tmp, len(X))
        R[:] = op * X
        return R

    def ip(Y, X):    # Y^H X
        return InnerProduct(X, Y).NumPy()

    def comb(X, C):  # X C
        return Ev(X * Matrix(np.asarray(C, dtype=complex)))

    def colnorms(X):
        return np.sqrt(np.maximum(np.real(np.diag(ip(X, X))), 0))

    def concat(blocks):
        Z = MultiVector(tmp, sum(len(b) for b in blocks))
        i = 0
        for b in blocks:
            if len(b):
                Z[i:i+len(b)] = b
                i += len(b)
        return Z

    def shifted(AX, BX):   # (A - sigma B) X
        return Ev(AX - BX * Matrix(sigma*np.eye(len(BX), dtype=complex)))

    def orth(X, against=[], images=[], drop=1e-10):
        # orthonormal basis of span(X), orthogonal to the orthonormal blocks against[i][0];
        # the same linear combinations are applied to images (e.g. AX, BX) using against[i][1:]
        for sweep in range(2):
            for Qb in against:
                if len(Qb[0]) and len(X):
                    C = Matrix(ip(Qb[0], X).astype(complex))
                    X = Ev(X - Qb[0] * C)
                    images = [Ev(Xi - Qi * C) for Xi, Qi in zip(images, Qb[1:])]
            if len(X) == 0:
                break
            G = ip(X, X)
            d = np.sqrt(np.maximum(np.real(np.diag(G)), 0))
            d[d == 0] = 1
            w, U = np.linalg.eigh((G + G.conj().T) / 2 / np.outer(d, d))
            keep = w > (drop if sweep == 0 else 1e-14) * max(w.max(), 1e-300)
            if not keep.any():
                X, images = MultiVector(tmp, 0), [MultiVector(tmp, 0) for Xi in images]
                break
            C = (U[:, keep] / np.sqrt(w[keep])) / d[:, None]
            X = comb(X, C)
            images = [comb(Xi, C) for Xi in images]
        return (X, *images) if images else X

    def TL(X):   # projected preconditioner (I - V V*) pre (I - Q Q*)
        Y = Ev(X - Q * Matrix(ip(Q, X)))
        if proj: Y = apply(proj, Y)
        Y = apply(pre, Y)
        if proj: Y = apply(proj, Y)
        return Ev(Y - V * Matrix(ip(V, Y)))

    # initial Schur basis
    V = MultiVector(tmp, k)
    nini = min(len(initial), k) if initial is not None else 0
    if nini:
        V[0:nini] = initial[0:nini]
    if nini < k:
        rand = MultiVector(tmp, k-nini)
        for v in rand:
            v.SetRandom()
        V[nini:k] = apply(pre, rand)
    if proj: V = apply(proj, V)
    V = orth(V)
    if len(V) < k:
        raise RuntimeError("GPLHR: initial vectors are linearly dependent")
    AV, BV = apply(A, V), apply(B, V)
    Q = orth(shifted(AV, BV))
    AA, BB, YL, YR = _SortedQZ(ip(Q, AV), ip(Q, BV), sigma, k)
    V, AV, BV, Q = comb(V, YR), comb(AV, YR), comb(BV, YR), comb(Q, YL)
    RA, RB = AA, BB
    MA, MB = _QFreeFactors(RA, RB)
    P = None

    for it in range(maxit+1):
        for check in range(2):
            WA, WB = Ev(AV - Q * Matrix(RA)), Ev(BV - Q * Matrix(RB))
            res = colnorms(WA) / np.maximum(np.linalg.norm(RA, axis=0), 1e-300) \
                + colnorms(WB) / np.maximum(np.linalg.norm(RB, axis=0), 1e-300)
            nconv = 0
            while nconv < k and res[nconv] < tol:
                nconv += 1
            if nconv < num or check == 1:
                break
            # AV, BV are updated by recurrences: confirm convergence with fresh products
            AV, BV = apply(A, V), apply(B, V)
        lams = np.diag(RA) / np.diag(RB)
        if printrates:
            print("GPLHR it {}: {} converged, residual {:.2e}, lams {}".format(
                it, nconv, max(res[:num]), [complex(l) for l in lams[:num]]), flush=True)
        if callback:
            callback(it, nconv, lams, res)
        if nconv >= num or it == maxit:
            break

        # soft locking: converged leading Schur vectors stay in V, but get no new directions
        nlock = nconv
        act = slice(nlock, k)
        ml = min(m*k // (k-nlock), 20)
        MAa, MBa = MA[act, act], MB[act, act]

        # trial subspace Z = [V, W, S_1..S_m, P]
        W = orth(TL(Ev(WA * Matrix(MB[:, act]) - WB * Matrix(MA[:, act]))), against=[(V,)])
        blocks = [(W, apply(A, W), apply(B, W))]
        for l in range(ml):
            S, AS, BS = blocks[-1]
            if len(S) == k-nlock:
                X = Ev(AS * Matrix(MBa) - BS * Matrix(MAa))
            else:
                X = shifted(AS, BS)
            S = orth(TL(X), against=[(V,)] + [(b[0],) for b in blocks])
            if len(S) == 0:
                break
            blocks.append((S, apply(A, S), apply(B, S)))
        if P is not None and len(P):
            P, AP, BP = orth(P, against=[(V, AV, BV)] + blocks, images=[AP, BP])
            if len(P):
                blocks.append((P, AP, BP))

        Z = concat([V] + [b[0] for b in blocks])
        AZ = concat([AV] + [b[1] for b in blocks])
        BZ = concat([BV] + [b[2] for b in blocks])
        s = len(Z)

        # harmonic Schur-Rayleigh-Ritz with test space U = [Q, orth((A - sigma B) Z_new)]
        Qhat = orth(shifted(concat([b[1] for b in blocks]), concat([b[2] for b in blocks])),
                    against=[(Q,)], drop=1e-14)
        U = concat([Q, Qhat])
        if len(U) != s:
            raise RuntimeError("GPLHR: (A - sigma B) Z is rank deficient, is sigma an eigenvalue?")
        nsort = min(2*k, s) if thick else k
        AA, BB, YLs, YRs = _SortedQZ(ip(U, AZ), ip(U, BZ), sigma, nsort)
        YR, YL = YRs[:, :k], YLs[:, :k]
        RA, RB = AA[:k, :k], BB[:k, :k]

        if thick:
            npl = min(k-nlock, s-k)
            P, AP, BP = comb(Z, YRs[:, k:k+npl]), comb(AZ, YRs[:, k:k+npl]), comb(BZ, YRs[:, k:k+npl])
        else:
            Zn, AZn, BZn = Z[k:s], AZ[k:s], BZ[k:s]
            P, AP, BP = comb(Zn, YR[k:, act]), comb(AZn, YR[k:, act]), comb(BZn, YR[k:, act])
        V, AV, BV, Q = comb(Z, YR), comb(AZ, YR), comb(BZ, YR), comb(U, YL)
        MA, MB = _QFreeFactors(RA, RB)

    if nconv < num:
        print("WARNING: GPLHR did not converge, {} of {} eigenvalues converged in {} iterations".format(
            nconv, num, maxit))

    # eigenvectors from the triangular pencil (RA, RB)
    lams = np.diag(RA)[:num] / np.diag(RB)[:num]
    Y = np.zeros((num, num), dtype=complex)
    for j in range(num):
        Y[j,j] = 1
        if j:
            Y[:j,j] = np.linalg.lstsq(RA[:j,:j] - lams[j]*RB[:j,:j],
                                      -(RA[:j,j] - lams[j]*RB[:j,j]), rcond=None)[0]
    vecs = comb(V[0:num], Y)
    scal = np.sqrt(np.abs(np.diag(ip(vecs, apply(B, vecs)))))
    scal[scal == 0] = 1
    vecs = comb(vecs, np.diag(1/scal))
    return Vector(list(lams)), vecs




def Arnoldi (mat, tol=1e-10, maxiter=200):
    import scipy.linalg
    H = Matrix(maxiter,maxiter, complex=mat.is_complex)
    H[:,:] = 0
    v = mat.CreateVector(colvector=False)
    abv = MultiVector(v, 0)
    v.SetRandom()
    v /= Norm(v)

    for i in range(maxiter):
        abv.Append(v)
        v = (mat*v).Evaluate()
        for j in range(i+1):
            H[j,i] = InnerProduct(v, abv[j])
            v -= H[j,i]*abv[j]
        if i+1 < maxiter:
            H[i+1,i] = Norm(v)
        v = 1/Norm(v)*v

    lam,ev = scipy.linalg.eig(H)
    return Vector(lam), (abv*Matrix(ev)).Evaluate()
    
    





# SOAR: A SECOND-ORDER ARNOLDI METHOD FOR THE SOLUTION OF THE QUADRATIC EIGENVALUE PROBLEM
# Z. Bai and Y. Su, SIAM J. Matrix Anal. Appl 26, pp 640-659 (2005)
# author: A. Schoefl

def SOAR (A, B, maxiter=200):
# first version without memory saving and breakdown 

    q = A.CreateVector()
    p = A.CreateVector()
    s = A.CreateVector()
    r = A.CreateVector()

    Q = MultiVector(q,0)
    P = MultiVector(p,0)
    p[:] = 0
    q.SetRandom()
    q.FV().imag = 0      
    q /= Norm(q)
    T = Matrix(maxiter, complex=A.is_complex)
    T[:,:] = 0
    
    for j in range(maxiter):
        Q.Append(q)
        P.Append(p)
        r.data = A*q + B*p
        s.data = q

        for i in range(j+1):

            T[i,j] = InnerProduct(r, Q[i])
            r -= T[i,j]*Q[i]
            s -= T[i,j]*P[i]
        
        if j+1 < maxiter:
            T[j+1,j] = Norm(r)
            if T[j+1,j] == 0:
                print("SOAR stopped at iteration j = ", j)
                break

            q.data = 1/T[j+1,j]*r
            p.data = 1/T[j+1,j]*s

    return Q




# author: A. Schoefl
def TOAR (A, B, maxiter=200):

    r = A.CreateVector()

    tmp_np = np.zeros((len(r), 2))#, dtype=complex)
    r.SetRandom()
    r.FV().imag = 0      
    r /= Norm(r)
    tmp_np[:,0] = r.FV().real
    r.SetRandom()
    r.FV().imag = 0      
    r /= Norm(r)
    tmp_np[:,1] = r.FV().real

    # tmp_np[:,1] = tmp_np[:,0] # just for testing


    Q_np, X_np, perm = la.qr(tmp_np, pivoting=True, mode="economic")
    # print(Q_np)
    # print(Q_np.shape, X_np.shape, X_np)

    Q = MultiVector(r,0)
    # r.FV().NumPy()[:] = Q_np[:,0]
    
    r.FV().real[:] = Q_np[:,0]
    r.FV().imag = 0
    Q.Append(r)
    # assign rank
    if np.isclose(X_np[1,1], 0):
        eta = 1
    else:
        eta = 2

        # r.FV().NumPy()[:] = Q_np[:,1]
        r.FV().real[:] = Q_np[:,1]
        r.FV().imag = 0
        Q.Append(r)

    # print (Q_np)
    # print (Q[0], Q[1])
    print (InnerProduct(Q,Q))
        
    gamma = np.linalg.norm(tmp_np, ord='fro')
    
    U1 = Matrix(eta,1, True) 
    U1.NumPy()[:,0] = X_np[:eta,1]/gamma

    U2 = Matrix(eta,1, True) 
    U2.NumPy()[:,0] = X_np[:eta,0]/gamma
    print ("X_nb = ", X_np[:eta,:])
    print ("U1 = ", U1)
    print ("U2 = ", U2)

    H = Matrix(maxiter, maxiter-1, True)

    # TODO: would be more efficient in C++
    for j in range(maxiter-1):

        # print(U1, "\n", U2)

        r.data = A*(Q*U1[:,j])+B*(Q*U2[:,j])
        s = Vector(eta, True)
        # MGS: orthogonalize r against Q
        for i in range(eta):
            s[i] = InnerProduct(r, Q[i], conjugate=True) # TODO: order of Q[i], r correct? 
            r-=s[i]*Q[i]
        alpha = InnerProduct(r,r, conjugate=True).real
        
        # MGS 
        for i in range(j):
            # TODO: I have to re-read in proof if should be conjugation makes sense
            # (currently everything is real valued)
            H[i,j] = InnerProduct(s, U1[:,i], conjugate=False) + \
                        InnerProduct( U1[:,j], U2[:,i], conjugate=True)
            s -= H[i,j]*U1[:,i]
            U1[:,j] -= H[i,j]*U2[:,i]

        H[j+1,j] = sqrt(alpha+InnerProduct(s,s,conjugate=True).real+
                    InnerProduct(U1[:,j],U1[:,j], conjugate=True).real)

        alpha = sqrt(alpha)

        # breakdown
        if H[j+1,j] == 0: 
            print("breakdown in iteration ", j)
            return Q

        # deflation
        if alpha == 0:
            print("deflation in iteration ", j)
            tmp = U1
            U1 = Matrix(eta, j+2, True) 
            U1[:,:j+1] = tmp
            U1[:,j+1] = 1/H[j+1,j]*s

            tmp = U2
            U2 = Matrix(eta, j+2, True) 
            U2[:,:j+1] = tmp
            U2[:,j+1] = 1/H[j+1,j]*U1[:,j]
        else:
            # update rank
            eta+=1

            tmp = U2
            U2 = Matrix(eta, j+2, True) 
            U2[:-1,:j+1] = tmp
            U2[:-1,j+1] = 1/H[j+1,j]*U1[:,j]
            U2[eta-1, :] = 0

            tmp = U1
            U1 = Matrix(eta, j+2, True) 
            U1[:-1,:j+1] = tmp[:,:]
            U1[:-1,j+1] = 1/H[j+1,j]*s
            U1[eta-1, j+1] = alpha/H[j+1,j]

            Q.Append(1/alpha*r)                        


    return Q

