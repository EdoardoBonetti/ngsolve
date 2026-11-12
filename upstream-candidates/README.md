# Upstream candidates: what this fork could give back to NGSolve

**Contributing back (Edoardo, 2026-11-09):** no pull requests to NGSolve on GitHub. Bugs and fixes are written down
here and fixed locally in the fork (one local branch per fix, with a test). Later Edoardo asks for merging on the
NGSolve **GitLab**, as merge requests of **cherry-picked** commits, never the whole fork.

Fork branch `ngsolve` at a4b5bfd04 (upstream master merged on 2026-11-09) against pristine upstream
master 6e701b568 (netgen 9d852668). Both built the same way (RelWithDebInfo, no GUI, OCC, UMFPACK,
Accelerate) on the Mac (M1 Max, 10 cores, 32 GB). Everything here reruns with one command:

```bash
./run_ab.sh 2>&1 | tee results/run_ab.log      # tests + benchmarks, upstream then fork
```

`source env.sh fork|upstream` selects the build (`install-dev` or `../../upstream-ref/install`).
Every result is a JSON in `results/` with wall time, peak memory, machine, threads, command and commit.

## Regression tests: master fails, the fork passes

`tests_ab.py` runs the fork's 19 test files, each in its own process. Fork: all pass. Upstream master:

| test file | upstream master | what it shows |
|---|---|---|
| test_tangential_simd_bbnd | 1 of 3 fail | `specialcf.tangential` is 0 in SIMD edge (BBND) forms: the matrix assembles to zero |
| test_edgefacetangential_simd | 1 of 2 fail | `EdgeFaceTangentialVectors` has no SIMD Evaluate; the integrator switches to scalar for good |
| test_other_bnd_assembled | 2 of 2 fail | `u.Other(bnd=g)` boundary-facet term missing from the assembled matrix |
| test_irspace_set | 6 of 6 fail | `IntegrationRuleSpace(order>=6).Set` in 3D: SIMD blocks of 64 points land on wrong dofs (garbage/NaN) |
| test_equivalence_keys | 1 of 1 fail | `code_uses_equivalence_keys=True`: generated C++ does not compile (12 compiler errors) |
| test_periodic | 4 of 8 fail | `.Other()` on periodic facets evaluated at wrong points when the translation reorders vertex numbers |
| test_l2prolongation_refined | 8 of 15 fail | L2/VectorL2/SurfaceL2 `hoprolongation` on spaces created after `Refine` (L2 error 1e-5 → 9e-2) |
| test_device_krylov | 5 of 7 fail | DeviceCG/DeviceTFQMR give NaN after exact convergence between residual checks |
| test_localhoprolongation | 9 of 11 fail | feature: `H1(localhoprolongation=True)` |
| test_hcurlcurl_prolongation, _trig | 15 of 15 fail | feature: HCurlCurl has no prolongation |
| test_refine1d (ngsolve + netgen) | 15 of 15 fail | feature: 1D adaptive bisection and exact L2 prolongation |
| test_solvers | 3 of 6 fail | feature: GPLHR |
| test_tendexlines | error | feature: TendexLines |
| netgen test_split_alfeld | 1 of 3 fail | `ngmesh.SplitAlfeld()` puts the 3D vertex outside the tet (inverted elements, volume 2 on the unit cube) |
| test_inverse_temporary | passes | the crash needs MKL Pardiso (the Linux workstation); not reproducible on the Mac |
| test_multivector_orthogonalize | 30 of 64 fail | `MultiVector.Orthogonalize()` without `ipmat` not orthonormal from 4 vectors on |
| test_real_operator_complex_vector | 1 of 6 fail | a real BDDC applied to a complex vector raises (matrix, Cholesky, UMFPACK, Jacobi already work on master) |

## Measured speed-ups (same machine, same options)

| item | setup | upstream master | fork | gain |
|---|---|---|---|---|
| Field lines (netgen `FieldLineCalc`) | 2000 lines, 46,531 tets, `bench_fieldlines.py` | 9.26 s | 0.49 s (10 threads); 1.91 s (1 thread) | **19×**; 4.9× from the element hint alone |
| H1 high-order transfer under adaptive refinement | L-shape, p = 4, 5 levels to 2.28 M dofs, `bench_hoprol.py` | 1.23 s, 636 MB (`hoprolongation`) | 0.47 s, 407 MB (`localhoprolongation`) | **2.6×**, −36 % memory; exact (1.4e-14) |
| Non-Hermitian eigenvalues near a target | −Δu + b·∇u = λu, unit cube, H1 p = 2, 4 eigenvalues, `bench_gplhr.py` | Arnoldi + UMFPACK, 20 vectors: 83.4 s, 10.5 GB | GPLHR + BDDC: 48.0 s, 9.5 GB | **1.7×** at 147k dofs; slower below ~50k dofs |
| Edge form with `EdgeFaceTangentialVectors` | VectorH1 p = 3, 137k dofs, `bench_simd_edgeface.py` | 0.172 s (scalar fallback) | 0.106 s (SIMD) | **1.6×**, identical matrix |
| Complex system, real BDDC preconditioner | 62,821 dofs, GMRES, `bench_complex_precond.py` | raises `SparseCholesky::MultAdd: illegal combination of scalar types` | 17 iterations, true residual 2.8e-11 | works vs. does not |
| `mesh.Refine()` (netgen cluster update skipped) | 3D, `bench_refine.py` | 1.42 s (3.2 M tets), 15.07 s (20.4 M) | 1.36 s, 14.58 s | 3–4 % |

Notes on the numbers:
- Field lines: the traced length agrees (1429.105 vs 1428.993); the fork has 3.6 % more segments because
  upstream carries the RK step size and critical value from one line into the next. The fork's output is the
  same at 1 and 10 threads.
- GPLHR: GPLHR and Arnoldi(20) agree on 58.609 and the triple 88.217 at every size. Times (Arnoldi / GPLHR):
  1.9 / 3.4 s at 19k dofs, 5.9 / 8.8 s at 37k, 15.3 / 13.4 s at 63k, 83.4 / 48.0 s at 147k. Memory is close
  because BDDC's own factorization dominates; a multigrid preconditioner would be needed to show a memory gain.
  With only 5 vectors Arnoldi returned values that are not eigenvalues (119.5, 133.5, 5256; exact: 88.22 ×3,
  117.83 ×3), since it has no convergence test. First run with num = 5: `results/gplhr_n5_first_run/`.
- Equivalence keys: correctness only here (master fails to compile, fork compiles; compile 2.17 s without keys,
  2.04 s with). The earlier measurement on the XCTS source term in the NR repo was 46.1 s → 13.9 s with keys.
- `mesh.Refine` at 154 M tets ran 248 s vs 164 s, but that level was near the memory limit; not claimed.
- Also found: `ngsolve.webgui.FieldLines(start_points=<numpy array>)` fails on its own `assert`
  (truth value of an array). One-line fix: `start_points is not None`.

## Small / medium / large on MUSICA (2026-11-09)

Proposal PDF for Joachim: `report/proposal.pdf` (2 pages; `report/proposal.tex`, built with
`latexmk -lualatex`). Figure and table: `python3 plot_sweep.py` from `results/musica/*.json`.

Both variants built on MUSICA with identical options (Release, CPU only, UMFPACK; `musica/remote-setup-ab.sh`)
from pinned clones (fork ddcd4c270, master 6e701b568; `musica/ab.sh`). One CPU node share per job (48 cores,
192 GB); each upstream/fork pair runs back to back in the same job. The regular CPU queue had multi-day waits,
so the sweep ran as eleven 10-minute jobs on the dev QOS (`musica/submit_sweep.sh`, chunks in
`musica/sweep_ab.py`), several sharing a node: expect a few % of timing noise. Times are the operation only.
Commit names: the fork commits were rewritten before they were published (sources unchanged), and the
result JSONs keep the names of the run: `ngsolve_version` …-g**cfd8f92b6** is ddcd4c270, and
`script_commit` **bdcda8481** is 77e38beb8.

| item | small | medium | large |
|---|---|---|---|
| Field lines, 48 threads | 0.774 → 0.0137 s, **56×** | 5.45 → 0.0747 s, **73×** | 75.2 → 0.976 s, **77×** |
| H1 p = 4 transfer, `localhoprolongation` | 0.0084 → 0.0048 s, **1.8×** | 0.162 → 0.103 s, **1.6×** | 1.09 → 0.605 s, **1.8×** (2.4 → 1.3 GB) |
| 4 eigenvalues: Arnoldi(20) + UMFPACK → GPLHR + geometric multigrid | 2.10 → 2.62 s, **0.80×** | 15.1 → 7.18 s, **2.1×** | 135 → 21.7 s, **6.2×** (14.6 → 1.8 GB) |
| SIMD edge form | 0.0069 → 0.0042 s, **1.6×** | 0.032 → 0.026 s, **1.25×** | 0.185 → 0.132 s, **1.4×** |
| complex system, best complex BDDC → real BDDC | 0.045 → 0.053 s, **0.85×** | 0.373 → 0.341 s, **1.09×** | 7.11 → 4.59 s, **1.55×** (5.5 → 3.9 GB) |
| `mesh.Refine()` | 1.01 → 0.963 s, **1.05×** | 11.1 → 10.6 s, **1.05×** | 121 → 116 s, **1.05×** |

Sizes: field lines 500 / 2 000 / 10 000 lines on 6 k / 47 k / 379 k tets; H1 transfer to 0.14 / 2.3 / 14.3 M dofs;
eigenproblem H1 p = 2 with 20 / 83 / 287 k dofs (and 5 k: 0.26 vs 0.77 s); edge form VectorH1 p = 3 with
18 / 137 / 987 k dofs; complex system H1 p = 3 with 9 / 63 / 443 k dofs; `Refine` to 3.2 / 20 / 154 M tets
(levels of one run; the separate large refine run exceeds 10 minutes and timed out by design).

Notes:
- Field lines on one thread (fork): 0.198 / 1.51 / 24.8 s, i.e. 3.9× / 3.6× / 3.0× from the element hint alone.
- GPLHR: at order 2 every H1 dof is a vertex or edge dof, so BDDC's wirebasket is the whole space and BDDC costs a
  full direct solve: that is why GPLHR + BDDC lost below 50k dofs on the Mac. With geometric multigrid (nested
  meshes, `--pre multigrid --levels k`) GPLHR wins from about 30k dofs and uses less memory at every size
  (−37 % at 20k, −66 % at 83k, −88 % at 287k). Eigenvalues agree to the printed digits at every size. On the Mac
  at 20k dofs the two tie with 4 threads (1.69 vs 1.65 s).
- Complex system: master's two workarounds (complex BDDC of the same form; BDDC of the complex system) need the
  same iterations (16 / 17 / 19) as the fork's real BDDC; the fork saves the complex factorizations.
- Every fork result matches master: matrix norms identical, eigenvalues equal, interpolation errors 1e-14 in both.

## Does each change apply to today's master?

Cherry-pick of the original commits onto 6e701b568 (2026-11-09):
applies cleanly: Other(bnd=g), Inverse keep_alive, IRSpace split rules, ZeroCF key, periodic facets, DeviceCG SafeDiv,
L2 prolongation on refined meshes, GPLHR, local H1 prolongation, HCurlCurl prolongation, HDiv 2D prolongation,
AdjMatrix, 1D refinement; netgen: SplitAlfeld, parallel field lines, Refine cluster skip, 1D bisection.
Needs a small rebase: SIMD edge forms (upstream reworked `intrule.cpp` on 2026-11-03), real operators on complex
vectors (sparsecholesky/umfpack), block-selective smoothing (blockjacobi).
Already upstream: OCC < 7.8 shape hasher, NG_TCL macros.

## Fixed in the fork today because of the upstream merge
- `ComputeNormalAndMeasureET` (new in master, ba256c2e2) sets the edge tangent only for `double`, so the SIMD
  tangent is 0 again: 8e7f19d9f. **This is a bug in current master.**
- Master merges the facet rules of an element into one SIMD rule (11e5bb11c); the fork's SIMD
  `EdgeFaceTangentialVectors` now takes the edge per point: a4b5bfd04.
- HDiv `RT` became per element type (`et_RT[]`); the fork's HDiv hoprolongation follows: b7308a62d.
- Master caches the inverse duality matrix per vertex-order class (b97daf6ac) and picks the class from the mesh
  vertex numbers, which `PeriodicFESpace::GetFE` then renumbers: `Set(cf, dual=True)` and `Interpolate` on
  `Periodic(H1/HCurl/HDiv/HCurlCurl)` are wrong from order 3/2/2/1. Fix on branch `fix-periodic-dual-inverse`
  (b878c4c28, on master), merged here; test `test_periodic.py::test_periodic_dual_set` (8 of 10 fail on master).
  Merge-request text (for GitLab, later): `pr-periodic-dual-inverse.md`. **This is a bug in current master.**
  Sent as GitHub PR NGSolve/ngsolve#102 on 2026-11-09: leave it as it is (do not modify). Later fixes follow the rule above.
- `MultiVector.Orthogonalize()` **without `ipmat`** (found 2026-11-09 on 6.2.2608-332): the block-recursive branch of
  `MultiVector::T_Orthogonalize` (`linalg/multivector.cpp`, `ipmat == nullptr`) passed `InnerProduct(mv1, mv2)`, which
  is (size2 × size1), to `mv2->Add(mv1, ·)`, which expects (size1 × size2): the projection used the transposed block.
  With one vector in the first half (n = 2, 3) the transpose has the same memory layout, so the vectors stayed
  orthonormal there; from n = 4 they did not. The R factor got the block below the diagonal, so V = Q R failed
  already for n = 2, 3. Fix on branch `fix-multivector-orthogonalize` (5643b8bb7, on master 6e701b568), merged here
  (c7460854f); fixed in the fork from 6.2.2608-340. Test `tests/pytest/test_multivector_orthogonalize.py`
  (n = 1…16, real and complex, with and without `ipmat`; Gram error and V = Q R both < 1e-12).
  Before (6.2.2608-337): 30 of 64 cases fail, every case without `ipmat` and n ≥ 2; max |G − I| from 4.75e-02 (n = 4)
  to 3.46e-01 (n ≤ 16). After (6.2.2608-340): 64 of 64 pass; over n = 1…16 the worst max |G − I| is 1.1e-15 (real),
  6.7e-16 (complex), and the worst |V_old − V_new R| / |V_old| is 4.0e-16. **This is a bug in current master.**
  Not sent anywhere (rule above).

## New features on local branches
- **Dense linear algebra on `bla.Matrix`** (2026-11-09, gap `ngs-dense-linalg`): branch `feature-bla-dense-linalg`
  (356a2a604 + test fix f6e645036, on master 6e701b568), merged here (18694a474); in the fork from 6.2.2608-343.
  New `basiclinalg/dense_linalg.{hpp,cpp}` (C++: `DenseEigh`, `DenseEig`, `DenseSVD`, `DenseQR`, `DenseCholesky`,
  `DenseSolve`, double and Complex) on top of LAPACK, and the Python methods of `FlatMatrixD`/`FlatMatrixC` (so of
  `MatrixD`/`MatrixC`), with numpy's conventions and bla results:
  - `Eigh()` → (lam ascending, V), A = V diag(lam) V^H (`?syevd`/`?heevd`);
  - `Eig()` → complex (lam, V), A V = V diag(lam) (`?geev`);
  - `SVD(full_matrices=False)` → (U, s, Vh) (`?gesdd`; reduced by default, unlike numpy);
  - `QR(pivoting=False)` → (Q, R) reduced; `pivoting=True` → (Q, R, perm), A[:, perm] = Q R, rank revealing
    (`?geqp3`), which numpy does not offer;
  - `Cholesky()` → L lower, A = L L^H (`?potrf`); `Solve(b)` for a vector or matrix b (`?gesv`).

  The input is never modified. Test `tests/pytest/test_bla_dense_linalg.py`: 61 cases (real and complex, square
  and rectangular, rank-deficient pivoted QR, error cases), all agree with numpy to 1e-12, 0.7 s.
  Speed (`bench_bla_dense.py`, `results/bla_dense_fork.json`, best of 20, the Mac; numpy 2.2.6 also on Accelerate):
  parity, so the gain is NGSolve-native code and pivoted QR, not time.

  | call | bla.Matrix | numpy.linalg | agreement |
  |---|---|---|---|
  | SVD 200 × 200 | 4.46 ms | 4.33 ms (`full_matrices=False`) | s to 1.5e-15 |
  | SVD 2000 × 200 | 21.7 ms | 21.2 ms | s to 5.7e-15 |
  | QR with pivoting 200 × 200 | 1.21 ms | 0.90 ms (plain QR, no pivoting) | – |
  | Eigh 200 × 200 | 2.59 ms | 2.63 ms | lam to 3.6e-15 |
  | Solve 200 × 200 | 0.165 ms | 0.182 ms | x to 1.2e-14 |

## Fixed on local branches (not merged into `ngsolve` yet)
- **Symbolic integrators build the trial Jacobian of the integrand on first use** (2026-11-12, performance).
  With `ngsglobals.symbolic_integrator_uses_diff = True` (set by `python/utils.py` since 1fcae9b0e, 2026-10-17) the
  constructors of `SymbolicBilinearFormIntegrator` and `SymbolicFacetBilinearFormIntegrator`
  (`fem/symbolicintegrator.cpp`) differentiated the integrand by the test functions (`dcf_dtest`, which `Apply`
  evaluates once instead of once per test component) and also by the test and trial functions
  (`ddcf_dtest_dtrial`, used only by element matrices and linearizations). `CompiledCoefficientFunction::DiffJacobi`
  compiles each derivative with the parent's `realcompile` flag, so every realcompiled integrand paid a C++ compile of
  its trial Jacobian when the form was built, also in a `nonassemble` form that is only applied. Found with a
  realcompiled 50-component DG face flux: its Jacobian was 140 MB of C++, which gcc 13 (-O2) compiled for two days
  before an internal compiler error. On the Mac (clang), for the 24-component integrand of the test below: integrand
  2.5 s, `dcf_dtest` 2.1 s, trial Jacobian 75.5 s. **Present in current master** (same constructors).
  Fix: `ddcf_dtest_dtrial` is built on the first call of `DDcfDtestDtrial(l1, k1)` (`std::call_once`, so it is safe
  inside the parallel assembly); all seven users go through it; `dcf_dtest` stays in the constructor. Branch
  `lazy-trial-jacobian` (b89c87e30, on `ngsolve` 1a293621e); only `fem/symbolicintegrator.{hpp,cpp}` and the test.
  The class layout changes, so add-ons that derive from these integrators need a rebuild.
  Test `tests/pytest/test_lazy_trial_jacobian.py` (a logging `ngscxx` in front of the real one counts the compiles):
  (1) `L2**24`, realcompiled nonlinear volume term and inner-facet jump term, `nonassemble`: building the form compiles
  at most one derivative per integrator, `Apply` compiles nothing and equals the result with `uses_diff` off to 1e-12;
  (2) `L2**4`: `AssembleLinearization` and `Assemble` under `TaskManager` compile the Jacobians on first use and
  give the matrices of `uses_diff` off to 1e-12. Before (dev build without the fix): (1) fails with 5 derivative
  compiles while the form is built (2 `dcf_dtest` + 3 Jacobian entries). After (dev build 6.2.2608-355): 2 of 2 pass
  in 40 s; full `tests/pytest` (without `mpi_only`) 685 passed, 33 skipped.
  A/B with `bench_lazy_jacobian.py` (`results/lazy_jacobian_N*_p2_*.json`; L2**N, p = 2, 8 threads; Apply = median of
  150 calls, the two variants interleaved): N = 24 (367 200 unknowns): building the `nonassemble` form with
  `uses_diff` on takes 127.1 s on master (it compiles the Jacobians) and 6.4 s on the branch (2 `dcf_dtest`
  compiles); on the branch the first `AssembleLinearization` compiles the Jacobians (121.9 s; master 0.69 s). Apply
  with `uses_diff` on / off: master 1.47 / 3.96 ms, branch 1.48 / 3.61 ms. N = 50 (282 600 unknowns, branch only):
  build 49.5 s, Apply 1.25 ms on vs 8.30 ms off (6.6×). Every Apply equals the `uses_diff`-off result (rel_diff 0).
  So a realcompiled matrix-free form can keep `uses_diff` on and get the 2.4× (N = 24) to 6.6× (N = 50) faster Apply
  without compiling the Jacobian.
  Dev build only (`install-dev`); not merged, not in Netgen.app, not sent anywhere (rule above).
  Earlier attempt, superseded: branch `fix-realcompile-diffjacobi` (e41856bb7, 2026-11-02) turned the derivatives off
  for realcompiled integrands, which also loses the `dcf_dtest` path of `Apply`.
- **`SymbolicEnergy` builds the Hessian of the energy on first use** (2026-11-12, performance; found with the item
  above). With `symbolic_integrator_uses_diff` set, `SymbolicEnergy::SymbolicEnergy` (`fem/symbolicintegrator.cpp`)
  computed the gradient blocks `dcf` (reverse mode, used by `Apply` through `EnergyGradient`) and also the Hessian
  blocks `ddcf` (used only by the SIMD `AddLinearizedElementMatrix`), so a realcompiled `Variation(E * dx)` compiled
  its Hessian when the form was built, also in a `nonassemble` form that is only applied. **Present in current
  master** (6e701b568, same constructor); b89c87e30 and then 6d80a8ad3 cherry-pick onto it without conflicts
  (`git merge-tree`, 2026-11-12; not built there).
  Fix: `ddcf` is built on the first call of `DDcf(i, j)` (`std::call_once`, safe inside the parallel assembly), and
  only if every gradient block was built (as before: an exception there skipped the Hessian); its one reader goes
  through it; `dcf` stays in the constructor. Branch `lazy-energy-hessian` (6d80a8ad3, on `lazy-trial-jacobian`
  cb4abee5c); only `fem/symbolicintegrator.{hpp,cpp}` and the test. The class layout changes, so add-ons that derive
  from `SymbolicEnergy` need a rebuild.
  Test: two cases added to `tests/pytest/test_lazy_trial_jacobian.py`, a realcompiled energy on `H1**N` (order 2)
  whose trial proxies `u` and `Grad(u)` interact (Hessian blocks (u, u), (Grad u, u), (Grad u, Grad u)):
  (a) `N = 8`, `nonassemble`: building the form compiles at most one derivative per trial proxy, `Apply` compiles
  nothing, `Apply` and `Energy` equal the `uses_diff`-off results to 1e-12; (b) `N = 3`: the first
  `AssembleLinearization` under `TaskManager` compiles the Hessian and gives the `uses_diff`-off matrix to 1e-12.
  Before (dev build 6.2.2608-355-gb89c87e30): (a) fails with 5 compiles while the form is built (2 gradient + 3
  Hessian blocks), (b) fails (nothing left to compile on first use). After (dev build 6.2.2608-358-g6d80a8ad3): the
  file's 4 tests pass in 62 s; full `tests/pytest` (without `mpi_only`) 687 passed, 33 skipped
  (`~/Applications/ngsuite/update_20261112_lazyenergy_dev_ngsolve_pytest.log`).
  A/B with `bench_lazy_energy.py` (`results/lazy_energy_N8_p2_{fork,upstream}.json`, script 02195ad3c; the test
  energy with N = 8, order 2, maxh 0.3, 488 unknowns, 8 threads, clang): energy compile 2.6 s (fork) / 2.4 s
  (master); building the `nonassemble` form with `uses_diff` on takes 65.1 s on master (2 gradient + 3 Hessian
  compiles) and 6.0 s on the branch (2 gradient compiles); the branch compiles the Hessian in the first
  `AssembleLinearization` (59.4 s; master 5.5 ms; second call 1.3 ms; with `uses_diff` off 4.1 ms). On both builds
  `Apply`, `Energy` and the linearized matrix with `uses_diff` on equal the off results (rel_diff 1.2e-18, 0, 9.0e-17).
  Dev build only (`install-dev`); not merged, not in Netgen.app, not sent anywhere (rule above).

## Found, not fixed yet (local fix to do)
- **`ContactIntegrator` builds both derivatives of its integrand eagerly** (found 2026-11-12 with the lazy Jacobian
  items above): `comp/contact.cpp` computes `dcf_dtest` and `ddcf_dtest_dtrial` in the constructor regardless of
  `symbolic_integrator_uses_diff`, so a realcompiled contact integrand compiles both. Same lazy fix would apply.
