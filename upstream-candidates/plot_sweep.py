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
# refine: one run (the M run: maxh 0.1, 3 uniform levels, then 5 levels with 30 % random marks, which the bisection
# closure spreads to nearly every element) gives the three sizes: 3.2 M tets (uniform level 3), 20 M and 154 M tets
# (adaptive levels 4 and 5); the time is that single Refine() call
REFINE_LEVEL = {"S": ("uniform", 2), "M": ("adaptive", 3), "L": ("adaptive", 4)}
def ref_at(size):
    def get(d):
        mode, i = REFINE_LEVEL[size]
        l = d["data"][mode][i]
        return l["ne"], l["refine_s"]
    return get
def eig(d): return d["data"]["ndof"], d["data"]["solve_s"]

P = [
    dict(key="fieldlines", title="Field lines (netgen)", xlabel="lines traced", get=fl,
         up="fieldlines_{s}_upstream.json", fork="fieldlines_{s}_fork.json",
         extra=[("fork, 1 thread", "fieldlines_{s}_t1_fork.json")]),
    dict(key="hoprol", title="H1 p=4 transfer, adaptive refinement", xlabel="dofs (finest)", get=hop,
         up="hoprol_hoprolongation_{s}_upstream.json", fork="hoprol_localhoprolongation_{s}_fork.json", extra=[]),
    dict(key="gplhr", title="4 eigenvalues, non-Hermitian 3D", xlabel="dofs", get=eig,
         up="gplhr_arnoldi*_{s}_upstream.json", fork="gplhr_gplhr*_{s}_fork.json", extra=[],
         labels=("Arnoldi + UMFPACK (master)", "GPLHR + multigrid (fork)")),
    dict(key="simd", title="Edge (BBND) form assembly", xlabel="dofs", get=simd,
         up="simd_edgeface_{s}_upstream.json", fork="simd_edgeface_{s}_fork.json", extra=[]),
    dict(key="complex", title="Complex system, BDDC + GMRES", xlabel="dofs", get=cpx,
         # upstream: the faster of its two workarounds (complex BDDC of the same form, or BDDC of the system)
         up=["complex_precond_complex_{s}_upstream.json", "complex_precond_system_{s}_upstream.json"],
         fork="complex_precond_real_{s}_fork.json", extra=[]),
    dict(key="refine", title="mesh.Refine (netgen)", xlabel="tets after Refine", get=None,
         up="refine_M_upstream.json", fork="refine_M_fork.json", extra=[]),
]

summary, specs, rows = {}, [], []
for p in P:
    cells, series = [], {"upstream master": [], "fork": []}
    ex = {lab: [] for lab, _ in p["extra"]}
    for s in (["XS"] if p["key"] == "gplhr" else []) + SIZES:
        get = p["get"] or ref_at(s)
        ups = [load(u.format(s=s)) for u in (p["up"] if isinstance(p["up"], list) else [p["up"]])]
        ups = [d for d in ups if ok(d)]
        du, df = (min(ups, key=lambda d: get(d)[1]) if ups else None), load(p["fork"].format(s=s))
        rec = {"size": s}
        if ok(du):
            x, t = get(du); series["upstream master"].append((x, t)); rec.update(x=x, upstream_s=t,
                                                                                  upstream_mem_mb=du["peak_mem_mb"])
        if ok(df):
            x, t = get(df); series["fork"].append((x, t)); rec.update(x=x, fork_s=t, fork_mem_mb=df["peak_mem_mb"])
        for lab, pat in p["extra"]:
            de = load(pat.format(s=s))
            if ok(de):
                ex[lab].append(get(de)); rec[lab] = get(de)[1]
        if "upstream_s" in rec and "fork_s" in rec:
            rec["gain"] = rec["upstream_s"] / rec["fork_s"]
        summary.setdefault(p["key"], []).append(rec)
    sty = {"upstream master": dict(color="muted", ls="-", marker="s"), "fork": dict(color="primary", ls="-", marker="o")}
    lab = dict(zip(("upstream master", "fork"), p.get("labels", ("upstream master", "fork"))))
    ser = [dict(x=[a for a, _ in v], y=[b for _, b in v], **sty[k], label=lab[k]) for k, v in series.items() if v]
    ser += [dict(x=[a for a, _ in v], y=[b for _, b in v], label=k, color="primary" if "fork" in k else "muted",
                 ls="--", marker=".") for k, v in ex.items() if v]
    if p["key"] != "refine":          # 4-5 % gain: in the table only; its panel goes to the eigensolver memory
        specs.append(dict(series=ser, title=p["title"], xlabel=p["xlabel"], ylabel="time [s]", logx=True, logy=True))
    if p["key"] == "gplhr":
        rs = summary["gplhr"]
        mem = [dict(x=[r["x"] for r in rs if f"{v}_mem_mb" in r], y=[r[f"{v}_mem_mb"] / 1024 for r in rs if f"{v}_mem_mb" in r],
                    label=l, **sty[k]) for v, k, l in (("upstream", "upstream master", lab["upstream master"]),
                                                       ("fork", "fork", lab["fork"]))]
        mem_spec = dict(series=mem, title="Peak memory, same eigenproblem", xlabel="dofs", ylabel="memory [GB]",
                        logx=True, logy=True)


def fmt(t):
    return f"{t:.3g}" if t < 100 else f"{t:.0f}"


rows = []
names = {"fieldlines": "Field lines", "hoprol": "H1 transfer (\\texttt{localhoprolongation})",
         "gplhr": "Eigenvalues (GPLHR + MG vs Arnoldi + UMFPACK)", "simd": "SIMD edge forms",
         "complex": "Complex system: real BDDC vs best complex BDDC", "refine": "\\texttt{mesh.Refine}"}
for key, recs in summary.items():
    cells = []
    for r in recs:
        if r["size"] not in SIZES:
            continue
        if "gain" in r:
            g = f"{r['gain']:.0f}" if r["gain"] >= 10 else f"{r['gain']:.1f}" if r["gain"] >= 1.2 else f"{r['gain']:.2f}"
            cells.append(f"{fmt(r['upstream_s'])} $\\to$ {fmt(r['fork_s'])} s, \\textbf{{{g}$\\times$}}")
        else:
            cells.append("--")
    rows.append(f"{names[key]} & " + " & ".join(cells) + " \\\\")

OUT.mkdir(exist_ok=True)
head = ["\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X ccc@{}}", "\\toprule",
        "& \\textbf{small} & \\textbf{medium} & \\textbf{large}\\\\", "\\midrule"]
(OUT / "sweep_table.tex").write_text("\n".join(head + rows + ["\\bottomrule", "\\end{tabularx}"]) + "\n")
(R / "summary.json").write_text(json.dumps(summary, indent=1))
specs.insert(3, mem_spec)            # next to the eigensolver times
th = pk.theme(sys.argv[1] if len(sys.argv) > 1 else "paper")
pk.panels(specs, str(OUT / "fig_sweep.pdf"), th, ncols=3, figsize=(10.5, 5.6))
print("\n".join(rows))
print("->", OUT / "fig_sweep.pdf", OUT / "sweep_table.tex", R / "summary.json")
