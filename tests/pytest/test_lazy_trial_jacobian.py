"""Lazy trial Jacobian of the symbolic bilinear-form integrators.

With ngsglobals.symbolic_integrator_uses_diff (set by ngsolve/utils.py), SymbolicBFI and SymbolicFacetBFI
differentiate the integrand by the test functions in the constructor (dcf_dtest, used by Apply), and by the test and
trial functions (ddcf_dtest_dtrial, used by element matrices and linearizations) only on first use. For an integrand
compiled with Compile(realcompile=True) every derivative is another C++ compile, and the trial Jacobian of a
many-component integrand is by far the largest one (N = 24 below: about 2 s for the integrand or dcf_dtest, 80 s for
the Jacobian, clang on an M1). A matrix-free form that is only applied must not compile it.

Correct: building a nonassemble form compiles dcf_dtest only (one compile per integrator) and Apply compiles nothing;
the Jacobian is compiled when a linearization or an element matrix first needs it. Apply, AssembleLinearization and
Assemble give the same results (to rounding) as with symbolic_integrator_uses_diff off.

Run: python3 -m pytest tests/pytest/test_lazy_trial_jacobian.py   (about 45 s, mostly C++ compiles)
"""

import os
import shutil
import sys

import pytest
from netgen.geom2d import unit_square
from ngsolve import *


@pytest.fixture
def compile_count(tmp_path, monkeypatch):
    """Put a logging ngscxx in front of the real one; return a function that counts the C++ compiles so far.

    CompileCode compiles each CoefficientFunction in a new temporary directory, so the number of distinct
    working directories of ngscxx calls is the number of compiles (a library cache hit calls no compiler)."""
    if sys.platform == "win32":
        pytest.skip("the logging compiler wrapper is a shell script")
    real = shutil.which("ngscxx")
    if real is None:
        pytest.skip("ngscxx not on PATH")
    log = tmp_path / "ngscxx.log"
    fake = tmp_path / "bin" / "ngscxx"
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\necho "$PWD" >> "{log}"\nexec "{real}" "$@"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake.parent) + os.pathsep + os.environ["PATH"])

    def count():
        return len(set(log.read_text().splitlines())) if log.exists() else 0

    return count


@pytest.fixture
def uses_diff():
    """Set ngsglobals.symbolic_integrator_uses_diff while integrators are built; restore it afterwards."""
    saved = ngsglobals.symbolic_integrator_uses_diff

    def set_flag(value):
        ngsglobals.symbolic_integrator_uses_diff = value

    yield set_flag
    ngsglobals.symbolic_integrator_uses_diff = saved


def setup(N, names):
    """L2**N on a coarse mesh, the realcompiled integrands in names, and a smooth state.

    vol:  F(u) . v with F_i(u) = u_i u_{i+1} + u_i^3 / 3 (nonlinear, couples neighbouring components)
    jump: B (u - u.Other()) . v on the interior facets, lin: B u . v, with B = identity + one off-diagonal."""
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = L2(mesh, order=1, dgjumps=True) ** N
    u, v = fes.TnT()
    flux = CF(tuple(u[i] * u[(i + 1) % N] + u[i] ** 3 / 3 for i in range(N)))
    B = CF(tuple(1.0 if j == i else (0.5 if j == (i + 1) % N else 0.0) for i in range(N) for j in range(N)),
           dims=(N, N))
    integrands = {"vol": InnerProduct(flux, v),
                  "jump": InnerProduct(B * (u - u.Other()), v),
                  "lin": InnerProduct(B * u, v)}
    state = GridFunction(fes)
    state.Set(CF(tuple(0.5 * sin((i + 1) * x) * cos(y + i) for i in range(N))))
    return fes, [integrands[n].Compile(realcompile=True, wait=True) for n in names], state.vec


def form(fes, terms, nonassemble=False):
    a = BilinearForm(fes, nonassemble=nonassemble)
    for integrand, measure in terms:
        a += integrand * measure
    return a


def rel_diff(a, b):
    return (a - b).Norm() / b.Norm()


def test_apply_compiles_no_trial_jacobian(compile_count, uses_diff):
    fes, (vol, jump), x = setup(24, ["vol", "jump"])
    terms =[(vol, dx), (jump, dx(skeleton=True))]
    n0 = compile_count()  # the integrands themselves

    uses_diff(False)
    y_ref = x.CreateVector()
    form(fes, terms, nonassemble=True).Apply(x, y_ref)
    assert compile_count() == n0  # without diff nothing is differentiated

    uses_diff(True)
    a = form(fes, terms, nonassemble=True)
    # dcf_dtest only, one compile per integrator. A trial Jacobian built in the constructor would add one compile per
    # (test, trial) proxy pair: 1 for vol, 2 for jump (u and u.Other()).
    n_built = compile_count()
    assert n_built - n0 <= 2
    y = x.CreateVector()
    a.Apply(x, y)
    assert compile_count() == n_built  # Apply uses dcf_dtest, no Jacobian
    assert rel_diff(y, y_ref) < 1e-12


def test_linearization_and_assembly_build_jacobian_on_first_use(compile_count, uses_diff):
    fes, (vol, jump, lin), x = setup(4, ["vol", "jump", "lin"])
    residual = [(vol, dx), (jump, dx(skeleton=True))]   # CalcLinearizedElementMatrix, inner-facet CalcFacetMatrix
    linear = [(lin, dx), (jump, dx(skeleton=True))]      # CalcElementMatrix, CalcFacetMatrix

    uses_diff(False)
    a_ref = form(fes, residual)
    a_ref.AssembleLinearization(x)
    m_ref = form(fes, linear)
    m_ref.Assemble()

    uses_diff(True)
    a = form(fes, residual)
    m = form(fes, linear)
    n_built = compile_count()
    with TaskManager():  # the first use, and so the build of the Jacobian, happens inside the parallel assembly
        a.AssembleLinearization(x)
        m.Assemble()
    assert compile_count() > n_built  # the Jacobians are compiled now
    assert rel_diff(a.mat.AsVector(), a_ref.mat.AsVector()) < 1e-12
    assert rel_diff(m.mat.AsVector(), m_ref.mat.AsVector()) < 1e-12
