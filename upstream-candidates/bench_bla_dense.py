"""bench_bla_dense: bla.Matrix dense factorizations (feature-bla-dense-linalg) against numpy.linalg.

Purpose: the fork binds LAPACK's dense SVD / QR / Eigh / Solve to bla.Matrix (no numpy round trip). This measures
the call against numpy.linalg on the same matrix: SVD on 200 x 200 and 2000 x 200 (reduced, numpy with
full_matrices=False), plus QR with pivoting (numpy: plain QR, it has no pivoting), Eigh and Solve on 200 x 200.
The bla times include the copy of the input to column-major storage and of the results back.
The numpy calls are the baseline being measured against, hence their NGS-GAP tags.
Run:  source env.sh fork; python3 bench_bla_dense.py [--reps 20]
Correct: the singular values, eigenvalues and solutions agree with numpy to ~1e-13 relative.
"""
import argparse, time
import numpy as np
from common import save, best_of
from ngsolve.bla import Matrix, Vector

# numpy baseline (what the bla methods replace)
np_svd = lambda a: np.linalg.svd(a, full_matrices=False)    # NGS-GAP: ngs-dense-linalg (numpy baseline)
np_sv = lambda a: np.linalg.svd(a, compute_uv=False)        # NGS-GAP: ngs-dense-linalg (numpy baseline)
np_qr = lambda a: np.linalg.qr(a)                           # NGS-GAP: ngs-dense-linalg (numpy baseline)
np_eigh = lambda a: np.linalg.eigh(a)                       # NGS-GAP: ngs-dense-linalg (numpy baseline)
np_eigvalsh = lambda a: np.linalg.eigvalsh(a)               # NGS-GAP: ngs-dense-linalg (numpy baseline)
np_solve = lambda a, b: np.linalg.solve(a, b)               # NGS-GAP: ngs-dense-linalg (numpy baseline)

p = argparse.ArgumentParser()
p.add_argument("--reps", type=int, default=20)
a_ = p.parse_args()
t0 = time.perf_counter()
rng = np.random.default_rng(0)
out = {"numpy": np.__version__}
try:
    out["numpy_lapack"] = np.show_config(mode="dicts")["Build Dependencies"]["lapack"]["name"]
except Exception:
    pass


def rel(x, ref):
    return float(np.abs(np.asarray(x) - ref).max() / np.abs(ref).max())


for m, n in [(200, 200), (2000, 200)]:
    a = rng.standard_normal((m, n))
    M = Matrix(a)
    t_bla, _ = best_of(lambda: M.SVD(), a_.reps)
    t_np, _ = best_of(lambda: np_svd(a), a_.reps)
    out[f"svd_{m}x{n}"] = {"bla_s": t_bla, "numpy_s": t_np, "ratio_numpy_over_bla": t_np / t_bla,
                           "rel_err_s": rel(M.SVD()[1], np_sv(a))}

a = rng.standard_normal((200, 200))
M = Matrix(a)
sym = a + a.T
Ms = Matrix(sym)
b = rng.standard_normal(200)
rows = {
    "qr_pivoting_200": (lambda: M.QR(pivoting=True), lambda: np_qr(a)),
    "eigh_200": (lambda: Ms.Eigh(), lambda: np_eigh(sym)),
    "solve_200": (lambda: M.Solve(Vector(b)), lambda: np_solve(a, b)),
}
for name, (f_bla, f_np) in rows.items():
    t_bla, _ = best_of(f_bla, a_.reps)
    t_np, _ = best_of(f_np, a_.reps)
    out[name] = {"bla_s": t_bla, "numpy_s": t_np, "ratio_numpy_over_bla": t_np / t_bla}
out["eigh_200"]["rel_err_lam"] = rel(Ms.Eigh()[0], np_eigvalsh(sym))
out["solve_200"]["rel_err_x"] = rel(M.Solve(Vector(b)), np_solve(a, b))
save("bla_dense", out, t0, threads=None)
