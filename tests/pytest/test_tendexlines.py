"""ngsolve.webgui.TendexLines: eigenvector lines of a symmetric 3x3 CoefficientFunction."""
import numpy as np
from ngsolve import CF, OuterProduct, Id, sqrt, x, y, z
from ngsolve.meshes import MakeStructured3DMesh
from ngsolve.webgui import TendexLines


def _mesh():
    return MakeStructured3DMesh(False, nx=3, ny=3, nz=3, mapping=lambda a, b, c: (4 * a - 2, 4 * b - 2, 4 * c - 2))


def _radial():
    r = sqrt(x * x + y * y + z * z)
    n = CF((x, y, z)) / r
    return OuterProduct(n, n)                      # eigenvalues 0, 0, 1; eigenvector n for 1


def test_radial_lines():
    seeds = np.array([[0.3, 0.2, 0.1], [-0.2, 0.4, 0.3], [0.1, -0.3, -0.4]])
    d = TendexLines(_radial().Compile(), _mesh(), seeds, family=2, length=2.0)
    ps, pe = np.reshape(d["pstart"], (-1, 3)), np.reshape(d["pend"], (-1, 3))
    assert len(ps) > 20
    t = (pe - ps) / np.linalg.norm(pe - ps, axis=1)[:, None]
    n = ps / np.linalg.norm(ps, axis=1)[:, None]
    assert np.abs(np.abs(np.einsum("ni,ni->n", t, n)) - 1).max() < 1e-6       # straight radial segments
    assert np.allclose(d["eigenvalue"], 1)


def test_degenerate_family_gives_no_lines():
    d = TendexLines(_radial(), _mesh(), np.array([[0.3, 0.2, 0.1]]), family=0, length=1.0)
    assert len(d["pstart"]) == 0                                                # eigenvalue 0 is double


def test_metric_and_plane():
    # with metric 4*Id the eigenvectors are rescaled, the lines are the same; plane_normal keeps them in y = 0
    seeds = np.array([[0.5, 0.0, 0.3], [-0.4, 0.0, 0.6]])
    d = TendexLines(_radial(), _mesh(), seeds, metric=4 * Id(3), plane_normal=(0, 1, 0), which="max", length=1.0)
    ps = np.reshape(d["pstart"], (-1, 3))
    assert len(ps) > 10 and np.abs(ps[:, 1]).max() < 1e-10
