"""
ngsjit as ngsolve.jit - the package lives in the submodule
external_dependencies/ngsolve-jit and is installed as top-level ngsjit.

    from ngsolve.jit import compile

re-exports ngsjit, so ngsolve.jit.compile is ngsjit.compile (one registry,
one cache); submodules resolve as ngsolve.jit.kernels -> ngsjit.kernels.
"""
import importlib as _importlib

from ngsjit import *          # noqa: F401,F403
from ngsjit import __all__    # noqa: F401


def __getattr__(name):
    try:
        return _importlib.import_module("ngsjit." + name)
    except ModuleNotFoundError as e:
        if e.name != "ngsjit." + name:
            raise
        raise AttributeError(f"module 'ngsolve.jit' has no attribute {name!r}") from None
