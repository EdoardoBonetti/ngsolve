"""Shared helpers for the fork-vs-upstream A/B runs: timing, peak memory and the provenance JSON
required by WORKING_RULES.md (wall time, peak memory, machine, cores, command, git commit)."""
import json, os, resource, socket, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def build_info():
    import ngsolve, netgen
    return {"variant": os.environ.get("NGS_VARIANT", "unknown"),
            "ngsolve_version": ngsolve.__version__,
            "netgen_version": getattr(netgen, "__version__", None),
            "ngsolve_path": str(Path(ngsolve.__file__).parent)}


def script_commit():
    try:
        return subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def peak_mem_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10   # bytes on macOS, KiB on Linux


def save(name, data, t_start, threads):
    out = {"name": name, **build_info(), "data": data,
           "wall_s": time.perf_counter() - t_start, "peak_mem_mb": peak_mem_mb(),
           "machine": socket.gethostname(), "threads": threads, "cpu_count": os.cpu_count(),
           "command": " ".join([sys.executable.split("/")[-1]] + sys.argv),
           "script_commit": script_commit(),
           "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    RESULTS.mkdir(exist_ok=True)
    f = RESULTS / f"{name}_{out['variant']}.json"
    f.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["data"], indent=1))
    print("->", f)
    return out


def best_of(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts), ts
