# Cross-version regression tests

Checks that a modified NGSolve/Netgen build gives the same results as a
reference build on the major features. The features are:

- mesh generation (CSG, geom2d, OCC 2D/3D)
- uniform and adaptive refinement (`onlyonce` on and off), with parent maps and topology
- curved geometry
- dof counts of all the standard spaces
- assembly (H1, HCurl, mixed HDiv-L2, DG with skeleton terms, static condensation)
- direct and iterative solvers
- multigrid over refined levels
- GridFunction autoupdate with low- and high-order prolongation
- Set, Interpolate and Integrate
- eigenvalues, Newton, a ZZ adaptive loop, heat-equation time stepping
- JIT compilation, pickling and mesh I/O

Each feature returns arrays, and their wall times are recorded too.

```bash
# 1. record with the reference build on PYTHONPATH
python fingerprints.py ref.npz
# 2. record with the build under test and compare (table with timings)
python fingerprints.py new.npz
python compare.py ref.npz new.npz --md report.md
# or as part of the pytest suite
NGS_CROSSVERSION_REF=ref.npz python -m pytest test_crossversion.py
```

## Tolerances

- **Integer data** (meshes, element vertices, topology, dof counts, iteration
  counts) must be identical.
- **Floating-point data** must agree to 1e-8, elementwise relative. Threaded
  assembly and the sparse direct solvers vary by up to about 1e-9 between two runs
  of the *same* build.
- **Bit-for-bit check:** record both builds with `NGS_FP_SERIAL=1`, which runs
  without TaskManager, and compare with `--exact`. Every feature must then match
  bit for bit, except those in `fingerprints.NOT_REPRODUCIBLE`. Those use threaded
  BLAS inside the factorization and keep the tolerance.
