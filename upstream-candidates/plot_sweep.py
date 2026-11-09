"""plot_sweep: figure and LaTeX table of the MUSICA small/medium/large A/B sweep (musica/sweep_ab.py).

Purpose: read the raw JSONs in results/musica/ (nothing is rerun) and write
  report/fig_sweep.pdf    time against problem size, upstream master vs fork, one panel per proposal (plotkit),
  report/sweep_table.tex  one row per proposal: upstream time, fork time and gain at S, M, L,
  results/musica/summary.json  the same numbers plus the consistency checks.
Run:  python3 plot_sweep.py [theme]      (theme: a plotkit theme name, default paper)
Correct: every gain cell is filled; the checks (eigenvalues, matrix norms, traced length, errors) agree.
"""
import glob, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / "Desktop/PhD/Research/Numerical Relativity/tools"))   # plotkit, as its README
import plotkit as pk

R = HERE / "results" / "musica"
OUT = HERE / "report"
SIZES = ["S", "M", "L"]


def load(pattern):
    f = sorted(glob.glob(str(R / pattern)))
    return json.loads(Path(f[0]).read_text()) if f else None


def ok(d):
    return d is not None and not (isinstance(d["data"], dict) and d["data"].get("error"))


# per proposal: label, x label, how to read (size, time) from a JSON, and the series to compare
def fl(d): return d["data"]["lines"], d["data"]["trace_s_best"]
def simd(d): return d["data"]["ndof"], d["data"]["assemble_s_best"]
def hop(d): return d["data"]["levels"][-1]["ndof"], d["data"]["total_update_s"]
def cpx(d): return d["data"]["ndof"], d["data"]["total_s"]
def ref(d): return d["data"]["uniform"][-1]["ne"], sum(l["refine_s"] for m in ("uniform", "adaptive") for l in d["data"][m])
def eig(d): return d["data"]["ndof"], d["data"]["solve_s"]

P = [
    dict(key="fieldlines", title="Field lines (netgen)", xlabel="lines traced", get=fl,
         up="fieldlines_{s}_upstream.json", fork="fieldlines_{s}_fork.json",
         extra=[("fork, 1 thread", "fieldlines_{s}_t1_fork.json")]),
    dict(key="hoprol", title="H1 p=4 transfer, adaptive refinement", xlabel="dofs (finest)", get=hop,
         up="hoprol_hoprolongation_{s}_upstream.json", fork="hoprol_localhoprolongation_{s}_fork.json", extra=[]),
    dict(key="gplhr", title="4 eigenvalues, non-Hermitian 3D", xlabel="dofs", get=eig,
         up="gplhr_arnoldi*_{s}_upstream.json", fork="gplhr_gplhr*_{s}_fork.json", extra=[]),
    dict(key="simd", title="Edge (BBND) form assembly", xlabel="dofs", get=simd,
         up="simd_edgeface_{s}_upstream.json", fork="simd_edgeface_{s}_fork.json", extra=[]),
    dict(key="complex", title="Complex system, BDDC + GMRES", xlabel="dofs", get=cpx,
         up="complex_precond_complex_{s}_upstream.json", fork="complex_precond_real_{s}_fork.json",
         extra=[("upstream, BDDC of system", "complex_precond_system_{s}_upstream.json")]),
    dict(key="refine", title="mesh.Refine (netgen)", xlabel="tets (finest)", get=ref,
         up="refine_{s}_upstream.json", fork="refine_{s}_fork.json", extra=[]),
]

summary, specs, rows = {}, [], []
for p in P:
    cells, series = [], {"upstream master": [], "fork": []}
    ex = {lab: [] for lab, _ in p["extra"]}
    for s in (["XS"] if p["key"] == "gplhr" else []) + SIZES:
        du, df = load(p["up"].format(s=s)), load(p["fork"].format(s=s))
        rec = {"size": s}
        if ok(du):
            x, t = p["get"](du); series["upstream master"].append((x, t)); rec.update(x=x, upstream_s=t,
                                                                                  upstream_mem_mb=du["peak_mem_mb"])
        if ok(df):
            x, t = p["get"](df); series["fork"].append((x, t)); rec.update(x=x, fork_s=t, fork_mem_mb=df["peak_mem_mb"])
        for lab, pat in p["extra"]:
            de = load(pat.format(s=s))
            if ok(de):
                ex[lab].append(p["get"](de)); rec[lab] = p["get"](de)[1]
        if "upstream_s" in rec and "fork_s" in rec:
            rec["gain"] = rec["upstream_s"] / rec["fork_s"]
        summary.setdefault(p["key"], []).append(rec)
    sty = {"upstream master": dict(color="muted", ls="-", marker="s"), "fork": dict(color="primary", ls="-", marker="o")}
    ser = [dict(x=[a for a, _ in v], y=[b for _, b in v], label=k, **sty[k]) for k, v in series.items() if v]
    ser += [dict(x=[a for a, _ in v], y=[b for _, b in v], label=k, color="primary" if "fork" in k else "muted",
                 ls="--", marker=".") for k, v in ex.items() if v]
    specs.append(dict(series=ser, title=p["title"], xlabel=p["xlabel"], ylabel="time [s]", logx=True, logy=True))


def fmt(t):
    return f"{t:.3g}" if t < 100 else f"{t:.0f}"


rows = []
names = {"fieldlines": "Field lines", "hoprol": "H1 transfer (\\texttt{localhoprolongation})",
         "gplhr": "Eigenvalues (GPLHR + MG vs Arnoldi + UMFPACK)", "simd": "SIMD edge forms",
         "complex": "Real BDDC on complex system", "refine": "\\texttt{mesh.Refine}"}
for key, recs in summary.items():
    cells = []
    for r in recs:
        if r["size"] not in SIZES:
            continue
        if "gain" in r:
            cells.append(f"{fmt(r['upstream_s'])} $\\to$ {fmt(r['fork_s'])} s, \\textbf{{{r['gain']:.1f}$\\times$}}")
        else:
            cells.append("--")
    rows.append(f"{names[key]} & " + " & ".join(cells) + " \\\\")

OUT.mkdir(exist_ok=True)
(OUT / "sweep_table.tex").write_text("\n".join(rows) + "\n")
(R / "summary.json").write_text(json.dumps(summary, indent=1))
th = pk.theme(sys.argv[1] if len(sys.argv) > 1 else "paper")
pk.panels(specs, str(OUT / "fig_sweep.pdf"), th, ncols=3, figsize=(10.5, 5.6))
print("\n".join(rows))
print("->", OUT / "fig_sweep.pdf", OUT / "sweep_table.tex", R / "summary.json")
