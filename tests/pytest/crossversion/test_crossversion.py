"""Cross-version regression test: every major feature must give the same
results as a reference build.

Record a reference once with the reference build on PYTHONPATH:
    python fingerprints.py ref.npz
then, with the build under test:
    NGS_CROSSVERSION_REF=ref.npz python -m pytest test_crossversion.py
Without NGS_CROSSVERSION_REF the comparison tests are skipped (the features
are still run once, so they also work as smoke tests).
"""
import os

import numpy as np
import pytest

import fingerprints as fp
from compare import diff

REF = os.environ.get("NGS_CROSSVERSION_REF")


@pytest.fixture(scope="module")
def reference():
    if not REF:
        return None
    d = np.load(REF)
    ref = {}
    for k in d.files:
        if "::" in k:
            f, key = k.split("::", 1)
            ref.setdefault(f, {})[key] = d[k]
    return ref


@pytest.mark.parametrize("name", list(fp.FEATURES))
def test_feature(name, reference):
    res = {k: np.asarray(v) for k, v in fp.FEATURES[name]().items()}
    assert res, "feature produced no output"
    if reference is None:
        pytest.skip("smoke run only: set NGS_CROSSVERSION_REF to compare")
    if name not in reference or str(reference[name].get("__status")) != "ok":
        pytest.skip("feature not available in the reference build")
    ref = {k: v for k, v in reference[name].items() if not k.startswith("__")}
    assert set(ref) == set(res)
    bad = []
    for k in ref:
        ok, rel = diff(ref[k], res[k])
        if not ok:
            bad.append(f"{k}: rel diff {rel:.2e}")
    assert not bad, "\n".join(bad)
