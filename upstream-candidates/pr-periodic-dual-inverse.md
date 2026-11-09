# Draft merge request (NGSolve GitLab, later): cached dual inverse is wrong on Periodic spaces

Branch: `fix-periodic-dual-inverse` (one commit on master 6e701b568). To be cherry-picked into a GitLab merge request
when Edoardo asks for it. (Already sent as GitHub PR NGSolve/ngsolve#102 on 2026-11-09: do not modify it.)

---

**Title:** Cached dual inverse: take the vertex-order class from the element's vertex numbers (fixes dual Set on Periodic spaces)

**Problem.** Since b97daf6ac ("store dualinverse for some geom-free FESpaces"), `GridFunction.Set(cf, dual=True)`
and `Interpolate` are wrong on `Periodic(...)` of every space that caches D⁻¹: H1, HCurl, HDiv, HCurlCurl.
The space picks the cache slot in its `GetFE` from the mesh vertex numbers, but `PeriodicFESpace::GetFE` then
renumbers the element's vertices (`vertex_orientation`). The element computes D⁻¹ with the new numbers, stores it
in the slot of the old class, and reads D⁻¹ of another orientation for every other element of that class.

Smallest example (3D periodic box, constant symmetric K, so the exact answer is K):

```python
from netgen.occ import Box, IdentificationType, OCCGeometry, X, Y, Z
from ngsolve import *
box = Box((0, 0, 0), (1, 1.2, 1.5))
for d, name in ((X, "px"), (Y, "py"), (Z, "pz")):
    box.faces.Min(d).Identify(box.faces.Max(d), name, IdentificationType.PERIODIC)
mesh = Mesh(OCCGeometry(box).GenerateMesh(maxh=0.6))
K = CF((1, 0.2, 0, 0.2, 0, 0, 0, 0, 0), dims=(3, 3))
gf = GridFunction(Periodic(HCurlCurl(mesh, order=1)))
gf.Set(K, dual=True)
print(sqrt(Integrate(InnerProduct(gf - K, gf - K), mesh)))   # 1.16 on master, 1e-16 with this PR
```

Where it shows (dual Set on `Periodic(V)` against dual Set on `V`, smooth periodic field, master 6e701b568):

| space | wrong from order | ‖periodic − plain‖ at order 2 (2D / 3D) |
|---|---|---|
| H1 | 3 | — (order 3: 8.0e-2 / 3.4e-2) |
| HCurl | 2 | 6.2e-2 / 3.6e-1 |
| HDiv | 2 | 3.1e-1 / 9.3e-1 |
| HCurlCurl | 1 | 4.5e-1 / 2.6 |
| HDivDiv (no cache) | never | 4e-16 / 1e-14 |

Lower orders are right because their D⁻¹ does not depend on the orientation.

**Fix.** The class is a property of the vertex numbers the element finally carries, so the element picks the slot
itself, at solve time:
- `VertexOrientedFE::VertexOrderClass()` (fe_interfaces.hpp): the class from `vnums`, the same bit pattern as before;
  -1 if two vertex numbers coincide (then the class does not determine the orientation, and the element falls back to
  the assembled duality matrix).
- The H1, HCurl, HDiv and HCurlCurl elements get the space's table of 64 slots instead of one slot, and
  `SolveDuality` / `SolveDualityTrans` use `dual_inverse[VertexOrderClass()]`.
- The spaces no longer compute the class; they pass `dual_inverse[row]`. Net simplification in comp/.

No change for non-periodic spaces (same class, same slot); the class costs a handful of integer comparisons per
`SolveDuality` call.

**Test.** `tests/pytest/test_periodic.py::test_periodic_dual_set`: 2D and 3D periodic meshes, H1/HCurl/HDiv/HCurlCurl/
HDivDiv, orders 0 (1 for H1) to 3, compares the dual Set on `Periodic(V)` with the one on `V`
(< 1e-10; 8 of 10 cases fail on master). About 1 s.

Tested on the fork build (master 6e701b568 plus the fork's patches; the code touched here is identical in master),
Mac M1 Max: the full `tests/pytest` (without mpi_only) passes, 552 passed and 33 skipped (542 before plus the
10 new cases). The add-on reggeminus, whose periodic Regge kernel test found the bug, passes again
(`test_B1_curl_kernel_is_hess_lag[periodic-R1]`, `[periodic-R2]`). Not built on pristine master on its own.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
