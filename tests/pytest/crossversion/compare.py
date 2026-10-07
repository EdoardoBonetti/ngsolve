"""Compare two fingerprint recordings (see fingerprints.py).

    python compare.py REF.npz NEW.npz [--md report.md] [--exact]

--exact requires bitwise equality (use with two NGS_FP_SERIAL=1 recordings),
except for the features fingerprints.NOT_REPRODUCIBLE, which keep the tolerance.

Integer arrays must be identical; floating point arrays must agree to a
relative tolerance (assembly/solves run threaded, so summation order may differ).
Prints per feature: status, number of compared arrays, worst relative
difference, time in both builds.
"""
import sys

import numpy as np

# Threaded assembly and the sparse direct solvers are reproducible only up to
# round-off: two runs of the same build differ by up to ~1e-9 (direct solves),
# so the default tolerance is 1e-8.
RTOL = 1e-8
EXACT = False


def load(path):
    d = np.load(path, allow_pickle=False)
    feats = {}
    for k in d.files:
        if "::" not in k:
            continue
        f, key = k.split("::", 1)
        feats.setdefault(f, {})[key] = d[k]
    meta = {k: str(d[k]) for k in d.files if k.startswith("__")}
    return feats, meta


def diff(a, b, rtol=None):
    """Return (ok, relative difference) for two arrays."""
    rtol = RTOL if rtol is None else rtol
    if a.shape != b.shape:
        return False, np.inf
    if a.dtype.kind in "iub" or b.dtype.kind in "iub" or a.dtype.kind in "US":
        return bool(np.array_equal(a, b)), 0.0 if np.array_equal(a, b) else np.inf
    if not a.size:
        return True, 0.0
    # elementwise relative, with a small floor (relative to the array) for entries near zero
    floor = 1e-6 * float(np.max(np.abs(a))) + 1e-300
    rel = float(np.max(np.abs(a - b) / (np.abs(a) + floor)))
    return rel <= rtol, rel


def compare(ref_path, new_path):
    ref, rmeta = load(ref_path)
    new, nmeta = load(new_path)
    rows = []
    for f in sorted(set(ref) | set(new)):
        R, N = ref.get(f, {}), new.get(f, {})
        bad, worst, n = [], 0.0, 0
        for k in sorted(set(R) | set(N)):
            if k.startswith("__"):
                continue
            if k not in R or k not in N:
                bad.append(f"{k}: missing in {'ref' if k not in R else 'new'}")
                continue
            exact = EXACT and bool(R.get("__reproducible", True))
            ok, rel = diff(R[k], N[k], 0.0 if exact else RTOL)
            n += 1
            worst = max(worst, rel)
            if not ok:
                bad.append(f"{k}: rel diff {rel:.2e} shapes {R[k].shape} vs {N[k].shape}")
        rs, ns = str(R.get("__status", "missing")), str(N.get("__status", "missing"))
        if rs != "ok" or ns != "ok":
            bad.insert(0, f"status ref={rs!r} new={ns!r}")
        rows.append(dict(feature=f, n=n, worst=worst, bad=bad,
                         tref=float(R.get("__time", np.nan)), tnew=float(N.get("__time", np.nan))))
    return rows, rmeta, nmeta


def markdown(rows, rmeta, nmeta):
    lines = [f"Reference: `{rmeta.get('__path')}` ({rmeta.get('__version')})  ",
             f"New: `{nmeta.get('__path')}` ({nmeta.get('__version')})", "",
             "| feature | arrays | result | worst rel. diff | t ref [s] | t new [s] | new/ref |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        res = "identical" if not r["bad"] and r["worst"] == 0 else ("same (fp)" if not r["bad"] else "**DIFFERENT**")
        lines.append(f"| {r['feature']} | {r['n']} | {res} | {r['worst']:.1e} | {r['tref']:.2f} | "
                     f"{r['tnew']:.2f} | {r['tnew'] / r['tref']:.2f} |")
    for r in rows:
        if r["bad"]:
            lines += ["", f"**{r['feature']}**:"] + [f"- {b}" for b in r["bad"][:15]]
    return "\n".join(lines)


if __name__ == "__main__":
    EXACT = "--exact" in sys.argv
    rows, rm, nm = compare(sys.argv[1], sys.argv[2])
    md = markdown(rows, rm, nm)
    print(md)
    if "--md" in sys.argv:
        open(sys.argv[sys.argv.index("--md") + 1], "w").write(md + "\n")
    sys.exit(1 if any(r["bad"] for r in rows) else 0)
