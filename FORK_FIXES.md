# Fork fixes (branch `fork-fixes-2026-10-11`)

Fixes for problems found while running the numerical-relativity practice
scripts on asc203 (2026-10-10/11). Each item has a regression test; all tests
fail (or crash) on the previous build and pass with this branch.

## 1. SIMD Evaluate for `specialcf.EdgeFaceTangentialVectors`

`cl_EdgeFaceTangentialVectorsCF` (`fem/coefficient_geo.cpp`) had only a scalar
`Evaluate`. The SIMD call fell back to the base class, which throws
`ExceptionNOSIMD`; the symbolic integrators catch it and then switch
`simd_evaluate` off for good, so every later element of that form (e.g. the
whole SymbolicEnergy of a Bartnik/DeTurck form) was assembled in scalar mode.
The reference vectors depend only on the edge and the vertex order, so they are
now computed once per rule (`RefVectors`, shared with the scalar version), and
the SIMD version maps, orthonormalizes and orders them per lane with the same
`det(t, t1, t2) > 0` rule. Test: `tests/pytest/test_edgefacetangential_simd.py`
(the integrator keeps `simd_evaluate == True`, SIMD and scalar matrices agree,
t1/t2 are unit vectors orthogonal to the edge).

## 2. `u.Other(bnd=g)` in `dx(element_boundary=True)` on boundary facets

The cause: `SymbolicFacetBilinearFormIntegrator::T_CalcFacetMatrix` for boundary
facets skipped every term with an `Other` proxy, and the boundary values `g`
were only used in the matrix-free `ApplyFacetMatrix`. So an assembled matrix
silently had a zero ghost value. Now, on boundary facets, the term
`d cf / d u.Other` is multiplied with the part of `g` that is linear in the
element's own trial functions (`d g / d u`, with `g` evaluated on the surface
element as in `ApplyFacetMatrix`). Example: `g = 5*u` or
`mesh.BoundaryCF({"left": 5*u, ...})` (reflection/extrapolation ghosts). A
constant part of `g` belongs to the right-hand side; a bilinear form can't hold it,
so it is dropped. Use a LinearForm term for it. Not covered:
`CalcLinearizedFacetMatrix` (nonlinear `Apply`/`AssembleLinearization` with
`Other(bnd=g)`); that path still skips `Other` terms on boundary facets.
Test: `tests/pytest/test_other_bnd_assembled.py`.

## 3. Inverse of a temporary sparse matrix

`SparseFactorization` (the base of MKL Pardiso, sparsecholesky...) keeps only a
`weak_ptr` to the matrix, and `VHeight/VWidth` (and `Smooth`, `Update`)
dereference it. In `(M - dt/2*L).CreateSparseMatrix().Inverse(...)` the matrix is
a temporary, so `inv * v` worked but `inv @ B` (which asks for the size) read a
null pointer and segfaulted. The Python binding `BaseMatrix.Inverse` now has
`py::keep_alive<0,1>`, so the inverse keeps its matrix alive. Test:
`tests/pytest/test_inverse_temporary.py` (default inverse, pardiso, umfpack,
sparsecholesky; the default on asc203 is MKL Pardiso).

## 4. netgen: OpenCascade < 7.8, submodule at 766b338e

netgen commit `fbfaed1c` (branch `fork-fixes-2026-10-11` in the netgen
submodule) adds `NgShapeMapHasher`. It aliases `TopTools_ShapeMapHasher` on
OCC >= 7.8 and falls back to `HashCode`/`IsSame` on older versions (asc203 has OCC
7.6.3). The submodule now points to the netgen branch head, which is
766b338e (FieldLineCalc, needed for TendexLines) plus the commits of items 4
and 5. Test: `tests/pytest/test_occ_shapehash.py` in netgen.

## 5. netgen: `ngmesh.SplitAlfeld()` vertex position

`libsrc/meshing/python_mesh.cpp` passed `fac = 0.5` to
`HPRefinement(SPLIT_ALFELD)`. The new element point of `HP_TET_ALFELD` is
`(1-3 fac) p0 + fac (p1+p2+p3)`, so in 3D the vertex was at `-p0/2 + (p1+p2+p3)/2`,
which gives inverted and overlapping tets (volume 2 on the unit cube, no space is
conforming). It now passes `1/(dim+1)`, like `Ng_SplitAlfeld` in
`nginterface_v2.cpp`, which is what NGSolve's `mesh.SplitElements_Alfeld()`
uses. In 2D the Alfeld point is a face point with `min(1/3, fac) = 1/3`, so
2D was already correct. `SplitPowellSabin` exists only in 2D: the edge points
are at `fac = 0.5` (midpoints) and the face point at the barycentre, so it is
correct and unchanged. Test: `tests/pytest/test_split_alfeld.py` in netgen
(volume/area, no inverted elements, every interior edge shared by two triangles).

## 6. `IntegrationRuleSpace(order >= 6).Set(...)` in 3D

The cause: `SymbolicBFI::T_CalcElementMatrixAdd` evaluates a big SIMD rule in
blocks of 64 SIMD points (256 points with AVX2), via `SIMD_IntegrationRule::Range`. The
SIMD evaluator of the IR space (`IRDiffOp` in `comp/irspace.cpp`) assumes that the
rule starts at point 0. So from the second block on, the identity rows and the
values landed on the wrong dofs. Only 3D at order >= 6 has more than 256
points per element (hex 7^3 = 343), which explains the threshold. `Set` solves
with that wrong mass matrix, which gave the garbage/NaN; `Interpolate` uses no
SIMD block and was correct. Now `SIMD_IntegrationRule` carries the number
of its first point (`FirstNr()`, set by `Range` and kept by the mapped rule),
and `IRDiffOp` uses it. Test: `tests/pytest/test_irspace_set.py` (Set equals
Interpolate on hexes and tets, orders 5, 6, 8).

## Build and test notes

Built in `~/ngsuite/build-fixes`, installed to `~/ngsuite/install-fixes`, with
`LAPACK_LIBRARIES` as in the production build (reference `liblapack.so` before
MKL). On asc203, any MKL call from Python (the `| Lapack` products, Pardiso)
aborts with `libmkl_avx2.so: undefined symbol mkl_sparse_optimize_bsr_trsm_i8`.
This also happens in the production build, independent of these fixes. MKL is
loaded with `RTLD_LOCAL`; preloading it works around this:

    LD_PRELOAD="/lib/x86_64-linux-gnu/libmkl_core.so /lib/x86_64-linux-gnu/libmkl_intel_thread.so \
      /lib/x86_64-linux-gnu/libmkl_intel_lp64.so /lib/x86_64-linux-gnu/libiomp5.so" \
    PATH=~/ngsuite/install-fixes/bin:$PATH \
    PYTHONPATH=~/ngsuite/install-fixes/lib/python3.12/dist-packages python -m pytest tests/pytest/test_...py

`PATH` matters: `Compile(True)` runs the `ngscxx` it finds on `PATH`. Item 6 changes the
layout of `SIMD_IntegrationRule`, so code compiled with the production `ngscxx`
(old headers) crashes in the new build. After installing this branch, rebuild
everything that compiles against NGSolve headers (JIT-compiled CFs are compiled per
session, but C++ add-ons need a rebuild).

Regression run with this setup (19 existing test files plus the new ones): 84 passed,
5 skipped, 2 failed. The 2 failures are `test_periodic.py::test_periodicH1` and
`::test_quasiperiodic`, which fail the same way in the production build.
