"""
Analyze an NVIDIA Nsight Systems profile (.nsys-rep or .sqlite) and print a
structured report.

Usage:
    python analyze_nsys.py <profile.nsys-rep>
    python analyze_nsys.py <profile.sqlite>
    python analyze_nsys.py <profile.nsys-rep> --top 30
    python analyze_nsys.py <profile.nsys-rep> --save report.md

The script reads the SQLite export that nsys creates alongside the .nsys-rep
file (or exports it on the fly) and produces:
  1. GPU device info
  2. GPU kernel summary — top kernels by total time, with type classification
  3. GPU utilization — kernel density and idle-gap statistics
  4. CUDA runtime API summary — syncs, launches, graph launches
  5. NVTX range summary — DALI pipeline, user annotations
  6. Memory operations — memcpy / memset
  7. Detected bottlenecks with severity ratings
"""

import argparse
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def resolve_sqlite(path: str) -> str:
    """Return path to the SQLite file, exporting from .nsys-rep if needed."""
    p = Path(path)
    if p.suffix == ".sqlite":
        if not p.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return str(p)
    # .nsys-rep — look for an adjacent .sqlite
    sqlite_path = p.with_suffix(".sqlite")
    if sqlite_path.exists():
        print(f"Using existing SQLite: {sqlite_path}", file=sys.stderr)
        return str(sqlite_path)
    # Export
    print(f"Exporting {p.name} → {sqlite_path.name} …", file=sys.stderr)
    result = subprocess.run(
        ["nsys", "export", "--type=sqlite", "--force-overwrite=true",
         "--output", str(sqlite_path), str(p)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"nsys export failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return str(sqlite_path)


class DB:
    def __init__(self, path: str):
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row
        # Build StringIds lookup once
        rows = self.con.execute("SELECT id, value FROM StringIds").fetchall()
        self._strings: dict[int, str] = {r["id"]: r["value"] for r in rows}

    def resolve(self, sid: int | None) -> str:
        if sid is None:
            return ""
        return self._strings.get(sid, f"<{sid}>")

    def q(self, sql: str, params=()):
        return self.con.execute(sql, params).fetchall()

    def close(self):
        self.con.close()


# ---------------------------------------------------------------------------
# Analysis modules
# ---------------------------------------------------------------------------

def gpu_info(db: DB) -> list[dict]:
    rows = db.q("SELECT name, totalMemory, clockRate, smCount, computeMajor, computeMinor FROM TARGET_INFO_GPU")
    return [dict(r) for r in rows]


def kernel_summary(db: DB, top_n: int) -> dict:
    rows = db.q("""
        SELECT shortName, demangledName, COUNT(*) AS cnt,
               SUM(end - start) AS total_ns,
               AVG(end - start) AS avg_ns,
               MIN(end - start) AS min_ns,
               MAX(end - start) AS max_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY shortName
        ORDER BY total_ns DESC
    """)
    total_ns = sum(r["total_ns"] for r in rows)
    ranked = []
    for r in rows:
        name = db.resolve(r["demangledName"]) or db.resolve(r["shortName"])
        ranked.append({
            "name": name,
            "count": r["cnt"],
            "total_ns": r["total_ns"],
            "avg_ns": r["avg_ns"],
            "min_ns": r["min_ns"],
            "max_ns": r["max_ns"],
        })
    return {"total_ns": total_ns, "ranked": ranked, "top": ranked[:top_n]}


def kernel_type_summary(ranked: list[dict]) -> dict:
    """Classify kernels into broad categories."""
    buckets: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_ns": 0.0})
    for k in ranked:
        low = k["name"].lower()
        if "flash_bwd" in low:
            t = "FlashAttention backward"
        elif "flash_fwd" in low:
            t = "FlashAttention forward"
        elif "gemm" in low or "cutlass" in low:
            t = "GEMM / matmul"
        elif "triton" in low:
            t = "Triton fused"
        elif "multi_tensor_apply" in low:
            t = "Optimizer"
        elif "fillfunctor" in low or "fill" in low:
            t = "Fill / zero_grad"
        elif "memcpy" in low or "memset" in low:
            t = "Memory ops"
        elif "softmax" in low or "reduce" in low:
            t = "Reduction / softmax"
        elif "dali" in low or "slicenormalize" in low or "sliceflip" in low:
            t = "DALI"
        elif "elementwise" in low or "vectorized" in low:
            t = "Elementwise"
        else:
            t = "Other"
        buckets[t]["count"] += k["count"]
        buckets[t]["total_ns"] += k["total_ns"]
    return dict(buckets)


def gpu_utilization(db: DB) -> dict:
    rows = db.q("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
    if not rows:
        return {"kernel_count": 0}

    kernels = [(r["start"], r["end"]) for r in rows]
    total_ns = sum(e - s for s, e in kernels)
    span_start = kernels[0][0]
    span_end = max(e for _, e in kernels)
    span_ns = span_end - span_start

    gaps = []
    cursor = kernels[0][1]
    for s, e in kernels[1:]:
        if s > cursor:
            gaps.append(s - cursor)
        cursor = max(cursor, e)

    big_gaps = sorted([g for g in gaps if g > 50_000], reverse=True)  # >50µs
    total_gap = sum(gaps)

    return {
        "kernel_count": len(kernels),
        "total_ns": total_ns,
        "span_ns": span_ns,
        "density_pct": total_ns / span_ns * 100 if span_ns else 0,
        "total_gap_ns": total_gap,
        "gap_count": len(gaps),
        "avg_gap_ns": total_gap / len(gaps) if gaps else 0,
        "big_gaps_count": len(big_gaps),
        "big_gaps_total_ns": sum(big_gaps),
        "top_gaps_ns": big_gaps[:10],
    }


def runtime_summary(db: DB, top_n: int) -> dict:
    rows = db.q("""
        SELECT nameId, COUNT(*) AS cnt,
               SUM(end - start) AS total_ns,
               AVG(end - start) AS avg_ns,
               MAX(end - start) AS max_ns
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        GROUP BY nameId
        ORDER BY total_ns DESC
    """)
    total_ns = sum(r["total_ns"] for r in rows)
    ranked = []
    for r in rows:
        ranked.append({
            "name": db.resolve(r["nameId"]).replace("_v3020", "").replace("_v7000", "")
                       .replace("_v10000", "").replace("_v11010", "").replace("_v11040", "")
                       .replace("_v12000", "").replace("_v11030", ""),
            "count": r["cnt"],
            "total_ns": r["total_ns"],
            "avg_ns": r["avg_ns"],
            "max_ns": r["max_ns"],
        })
    return {"total_ns": total_ns, "ranked": ranked, "top": ranked[:top_n]}


def nvtx_summary(db: DB, top_n: int) -> dict:
    # eventType 59 = NvtxPushPopRange (complete ranges with start+end)
    rows = db.q("""
        SELECT
            COALESCE(text, '') AS label,
            textId,
            COUNT(*) AS cnt,
            SUM(end - start) AS total_ns,
            AVG(end - start) AS avg_ns,
            MIN(end - start) AS min_ns,
            MAX(end - start) AS max_ns
        FROM NVTX_EVENTS
        WHERE eventType = 59 AND end IS NOT NULL AND end > start
        GROUP BY COALESCE(text, textId)
        ORDER BY total_ns DESC
    """)
    total_ns = sum(r["total_ns"] for r in rows)
    ranked = []
    for r in rows:
        label = r["label"] or db.resolve(r["textId"])
        ranked.append({
            "name": label,
            "count": r["cnt"],
            "total_ns": r["total_ns"],
            "avg_ns": r["avg_ns"],
            "min_ns": r["min_ns"],
            "max_ns": r["max_ns"],
        })
    return {"total_ns": total_ns, "ranked": ranked, "top": ranked[:top_n]}


def memcpy_summary(db: DB) -> dict:
    rows = db.q("SELECT COUNT(*) AS cnt, SUM(end-start) AS total_ns, SUM(bytes) AS total_bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    r = rows[0]
    rows2 = db.q("SELECT COUNT(*) AS cnt, SUM(end-start) AS total_ns FROM CUPTI_ACTIVITY_KIND_MEMSET")
    r2 = rows2[0]
    return {
        "memcpy_count": r["cnt"] or 0,
        "memcpy_ns": r["total_ns"] or 0,
        "memcpy_bytes": r["total_bytes"] or 0,
        "memset_count": r2["cnt"] or 0,
        "memset_ns": r2["total_ns"] or 0,
    }


def detect_bottlenecks(gpu_util: dict, runtime: dict, kernels: dict) -> list[dict]:
    issues = []

    # GPU kernel density
    density = gpu_util.get("density_pct", 100)
    if density < 70:
        issues.append({"severity": "HIGH",
                       "title": f"Low GPU kernel density ({density:.1f}%)",
                       "detail": f"GPU idle {gpu_util['total_gap_ns']/1e6:.1f} ms of {gpu_util['span_ns']/1e6:.1f} ms active span."})
    elif density < 90:
        issues.append({"severity": "MEDIUM",
                       "title": f"Moderate GPU kernel density ({density:.1f}%)",
                       "detail": f"{gpu_util['big_gaps_count']} gaps >50µs, totalling {gpu_util['big_gaps_total_ns']/1e6:.1f} ms."})

    # Excessive DeviceSynchronize
    for r in runtime["ranked"]:
        if "DeviceSynchronize" in r["name"] and r["count"] > 20:
            sev = "HIGH" if r["count"] > 200 else "MEDIUM"
            issues.append({"severity": sev,
                           "title": f"cudaDeviceSynchronize: {r['count']} calls, {r['total_ns']/1e6:.1f} ms",
                           "detail": f"Avg {r['avg_ns']/1e3:.1f} µs, max {r['max_ns']/1e6:.1f} ms. Hard CPU↔GPU syncs block pipelining."})

    # High StreamSynchronize (potential DALI or loss sync)
    for r in runtime["ranked"]:
        if "StreamSynchronize" in r["name"] and r["count"] > 500:
            issues.append({"severity": "MEDIUM",
                           "title": f"cudaStreamSynchronize: {r['count']} calls, {r['total_ns']/1e6:.1f} ms",
                           "detail": "Typically from DALI internal multi-stream sync. Not directly controllable."})

    # No CUDA graph launches
    graph_calls = sum(r["count"] for r in runtime["ranked"] if "GraphLaunch" in r["name"])
    if graph_calls == 0:
        issues.append({"severity": "MEDIUM",
                       "title": "No CUDA graph launches detected",
                       "detail": "torch.compile(mode='reduce-overhead') captures forward/backward into CUDA graphs."})
    else:
        issues.append({"severity": "INFO",
                       "title": f"CUDA graphs active — {graph_calls} graph launches",
                       "detail": "Compiled forward+backward+optimizer running via cudaGraphLaunch."})

    # Many individual kernel launches (outside graphs)
    for r in runtime["ranked"]:
        if "LaunchKernel" in r["name"] and r["count"] > 1000:
            issues.append({"severity": "INFO",
                           "title": f"cudaLaunchKernel: {r['count']} individual launches ({r['total_ns']/1e6:.1f} ms CPU)",
                           "detail": "These are ops outside CUDA graphs (DALI, masking, zero_grad). Normal alongside graph launches."})

    # Fill kernel dominance
    for k in kernels["top"][:5]:
        if "FillFunctor" in k["name"] and k["total_ns"] / kernels["total_ns"] > 0.10:
            pct = k["total_ns"] / kernels["total_ns"] * 100
            issues.append({"severity": "MEDIUM",
                           "title": f"FillFunctor kernel at {pct:.1f}% of GPU time ({k['count']} calls)",
                           "detail": "Likely from random masking (torch.ones/zeros) or DALI internal fills outside CUDA graph."})

    # DALI stalls (NVTX max >> median would be visible from nvtx data)

    if not issues:
        issues.append({"severity": "INFO", "title": "No major bottlenecks detected", "detail": ""})

    return issues


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(ns: float) -> str:
    if ns >= 1e9:
        return f"{ns/1e9:.2f} s"
    if ns >= 1e6:
        return f"{ns/1e6:.1f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.1f} µs"
    return f"{ns:.0f} ns"


def fmt_pct(part: float, whole: float) -> str:
    if whole == 0:
        return "  N/A"
    return f"{part/whole*100:5.1f}%"


def generate_report(sqlite_path: str, db: DB, top_n: int) -> str:
    gpus = gpu_info(db)
    kernels = kernel_summary(db, top_n)
    ktypes = kernel_type_summary(kernels["ranked"])
    util = gpu_utilization(db)
    runtime = runtime_summary(db, top_n)
    nvtx = nvtx_summary(db, top_n)
    mem = memcpy_summary(db)
    bottlenecks = detect_bottlenecks(util, runtime, kernels)

    lines: list[str] = []

    def out(s=""):
        lines.append(s)

    def header(s):
        out(f"\n{'='*72}")
        out(f"  {s}")
        out(f"{'='*72}")

    out("Nsight Systems Profile Analysis")
    out(f"SQLite: {sqlite_path}")

    # GPU info
    header("GPU DEVICE INFO")
    for g in gpus:
        mem_gb = g["totalMemory"] / 1024**3
        clock_ghz = g["clockRate"] / 1e9
        out(f"  {g['name']}  |  {mem_gb:.1f} GB  |  {g['smCount']} SMs  |  "
            f"SM {g['computeMajor']}.{g['computeMinor']}  |  {clock_ghz:.2f} GHz boost")

    # Bottlenecks
    header("DETECTED BOTTLENECKS")
    for b in bottlenecks:
        out(f"\n  [{b['severity']}] {b['title']}")
        if b["detail"]:
            detail = b["detail"]
            while len(detail) > 70:
                split = detail[:70].rfind(" ")
                if split == -1:
                    split = 70
                out(f"    {detail[:split]}")
                detail = detail[split:].lstrip()
            if detail:
                out(f"    {detail}")

    # GPU utilization
    header("GPU UTILIZATION")
    if util["kernel_count"] == 0:
        out("  No GPU kernels found.")
    else:
        out(f"  Kernel count:       {util['kernel_count']:,}")
        out(f"  Total kernel time:  {fmt(util['total_ns'])}")
        out(f"  GPU active span:    {fmt(util['span_ns'])}")
        out(f"  Kernel density:     {util['density_pct']:.1f}%")
        out(f"  Total idle gaps:    {fmt(util['total_gap_ns'])}  across {util['gap_count']:,} gaps")
        out(f"  Avg gap:            {fmt(util['avg_gap_ns'])}")
        out(f"  Gaps > 50µs:        {util['big_gaps_count']:,}  ({fmt(util['big_gaps_total_ns'])} total)")
        if util["top_gaps_ns"]:
            out(f"  Top gaps (ms):      [{', '.join(f'{g/1e6:.1f}' for g in util['top_gaps_ns'])}]")

    # Kernel type breakdown
    header("GPU KERNEL SUMMARY BY TYPE")
    out(f"  Total GPU kernel time: {fmt(kernels['total_ns'])}")
    out()
    for ktype, info in sorted(ktypes.items(), key=lambda x: -x[1]["total_ns"]):
        out(f"  {fmt_pct(info['total_ns'], kernels['total_ns'])} | "
            f"{fmt(info['total_ns']):>10} | x{info['count']:>6,} | {ktype}")

    # Top kernels
    header(f"TOP {top_n} GPU KERNELS")
    out(f"  {'%':>6}  {'Total':>10}  {'Count':>7}  {'Avg':>9}  {'Max':>9}  Name")
    out(f"  {'-'*6}  {'-'*10}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*50}")
    for k in kernels["top"]:
        pct = fmt_pct(k["total_ns"], kernels["total_ns"])
        out(f"  {pct}  {fmt(k['total_ns']):>10}  {k['count']:>7,}  "
            f"{fmt(k['avg_ns']):>9}  {fmt(k['max_ns']):>9}  {k['name'][:80]}")

    # CUDA runtime
    header(f"TOP {top_n} CUDA RUNTIME API CALLS")
    out(f"  {'%':>6}  {'Total':>10}  {'Count':>7}  {'Avg':>9}  {'Max':>9}  Name")
    out(f"  {'-'*6}  {'-'*10}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*40}")
    for r in runtime["top"]:
        pct = fmt_pct(r["total_ns"], runtime["total_ns"])
        out(f"  {pct}  {fmt(r['total_ns']):>10}  {r['count']:>7,}  "
            f"{fmt(r['avg_ns']):>9}  {fmt(r['max_ns']):>9}  {r['name']}")

    # NVTX
    header(f"TOP {top_n} NVTX RANGES")
    out(f"  {'%':>6}  {'Total':>10}  {'Count':>6}  {'Avg':>9}  {'Max':>9}  Name")
    out(f"  {'-'*6}  {'-'*10}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*50}")
    for r in nvtx["top"]:
        pct = fmt_pct(r["total_ns"], nvtx["total_ns"])
        out(f"  {pct}  {fmt(r['total_ns']):>10}  {r['count']:>6,}  "
            f"{fmt(r['avg_ns']):>9}  {fmt(r['max_ns']):>9}  {r['name'][:80]}")

    # Memory
    header("MEMORY OPERATIONS")
    out(f"  Memcpy:  {mem['memcpy_count']:,} ops  |  {fmt(mem['memcpy_ns'])}  |  "
        f"{mem['memcpy_bytes']/1024**2:.1f} MB transferred")
    out(f"  Memset:  {mem['memset_count']:,} ops  |  {fmt(mem['memset_ns'])}")

    out()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze an nsys .nsys-rep or .sqlite file and report bottlenecks.",
    )
    parser.add_argument("profile", help="Path to .nsys-rep or .sqlite file")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top entries per section (default: 20)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save report to a file")
    args = parser.parse_args()

    if not Path(args.profile).exists():
        print(f"Error: file not found: {args.profile}", file=sys.stderr)
        sys.exit(1)

    sqlite_path = resolve_sqlite(args.profile)
    print(f"Analyzing {sqlite_path} …", file=sys.stderr)

    db = DB(sqlite_path)
    report = generate_report(sqlite_path, db, top_n=args.top)
    db.close()

    if args.save:
        Path(args.save).write_text(report, encoding="utf-8")
        print(f"Report saved to: {args.save}", file=sys.stderr)

    print(report)


if __name__ == "__main__":
    main()
