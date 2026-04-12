"""
Analyze a PyTorch profiler trace JSON file and print a structured report.

Usage:
    python analyze_trace.py <trace.json>
    python analyze_trace.py <trace.json> --top 30
    python analyze_trace.py <trace.json> --save report.md

The script parses Chrome Trace Format events emitted by torch.profiler and
produces:
  1. Step timing breakdown (forward / backward / optimizer)
  2. GPU utilization and idle-gap analysis
  3. Top GPU kernels aggregated by name
  4. Top CPU operations aggregated by name
  5. CUDA runtime overhead (kernel launches, syncs, graph launches)
  6. Memory operation summary
  7. Detected bottlenecks with severity ratings
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def load_trace(path: str) -> list[dict]:
    """Load trace events from a PyTorch profiler JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("traceEvents", [])


def duration_events(events: list[dict]) -> list[dict]:
    """Filter to complete duration events only (ph == 'X')."""
    return [e for e in events if e.get("ph") == "X"]


# ---------------------------------------------------------------------------
# Analysis modules — each returns a dict of structured results
# ---------------------------------------------------------------------------

def analyze_categories(events: list[dict]) -> dict[str, int]:
    """Count events per category."""
    cats: dict[str, int] = defaultdict(int)
    for e in events:
        if "cat" in e:
            cats[e["cat"]] += 1
    return dict(sorted(cats.items()))


def analyze_gpu_kernels(events: list[dict], top_n: int) -> dict:
    """Aggregate GPU kernel durations by (shortened) name."""
    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    total = 0.0

    for e in events:
        if e.get("cat") != "kernel":
            continue
        dur = e.get("dur", 0)
        # Truncate to first 120 chars for grouping — keeps unique kernel
        # signatures while trimming template noise.
        name = e.get("name", "")[:120]
        agg[name]["count"] += 1
        agg[name]["total_us"] += dur
        total += dur

    ranked = sorted(agg.items(), key=lambda x: x[1]["total_us"], reverse=True)
    return {"total_us": total, "ranked": ranked[:top_n], "all": ranked}


def analyze_cpu_ops(events: list[dict], top_n: int) -> dict:
    """Aggregate CPU operator durations by name."""
    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    total = 0.0

    for e in events:
        if e.get("cat") != "cpu_op":
            continue
        dur = e.get("dur", 0)
        name = e.get("name", "")
        agg[name]["count"] += 1
        agg[name]["total_us"] += dur
        total += dur

    ranked = sorted(agg.items(), key=lambda x: x[1]["total_us"], reverse=True)
    return {"total_us": total, "ranked": ranked[:top_n], "all": ranked}


def analyze_cuda_runtime(events: list[dict]) -> dict:
    """Aggregate CUDA runtime calls (kernel launches, syncs, graph launches)."""
    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_us": 0.0})

    for e in events:
        if e.get("cat") != "cuda_runtime":
            continue
        dur = e.get("dur", 0)
        name = e.get("name", "")
        agg[name]["count"] += 1
        agg[name]["total_us"] += dur

    ranked = sorted(agg.items(), key=lambda x: x[1]["total_us"], reverse=True)
    return {"ranked": ranked}


def analyze_annotations(events: list[dict]) -> list[dict]:
    """Extract user_annotation and gpu_user_annotation duration events."""
    results = []
    for e in events:
        if e.get("cat") in ("user_annotation", "gpu_user_annotation") and e.get("dur", 0) > 100:
            results.append({
                "name": e["name"],
                "cat": e["cat"],
                "ts": e.get("ts", 0),
                "dur": e.get("dur", 0),
            })
    results.sort(key=lambda x: x["ts"])
    return results


def analyze_gpu_utilization(events: list[dict]) -> dict:
    """Compute GPU kernel density and idle-gap statistics."""
    kernels = [
        (e.get("ts", 0), e.get("dur", 0))
        for e in events
        if e.get("cat") == "kernel"
    ]
    if not kernels:
        return {"kernel_count": 0}

    kernels.sort()
    total_kernel_us = sum(d for _, d in kernels)
    span_start = min(ts for ts, _ in kernels)
    span_end = max(ts + dur for ts, dur in kernels)
    span_us = span_end - span_start

    # Gap analysis
    gaps = []
    for i in range(1, len(kernels)):
        gap = kernels[i][0] - (kernels[i - 1][0] + kernels[i - 1][1])
        if gap > 0:
            gaps.append(gap)

    total_gap = sum(gaps) if gaps else 0
    big_gaps = sorted([g for g in gaps if g > 50], reverse=True)

    return {
        "kernel_count": len(kernels),
        "total_kernel_us": total_kernel_us,
        "span_us": span_us,
        "density_pct": (total_kernel_us / span_us * 100) if span_us else 0,
        "total_gap_us": total_gap,
        "gap_count": len(gaps),
        "avg_gap_us": (total_gap / len(gaps)) if gaps else 0,
        "big_gaps_count": len(big_gaps),
        "big_gaps_total_us": sum(big_gaps),
        "top_10_gaps": big_gaps[:10],
    }


def analyze_memory_ops(events: list[dict]) -> dict:
    """Summarize GPU memcpy and memset operations."""
    memcpy = [e for e in events if e.get("cat") == "gpu_memcpy"]
    memset = [e for e in events if e.get("cat") == "gpu_memset"]
    return {
        "memcpy_count": len(memcpy),
        "memcpy_total_us": sum(e.get("dur", 0) for e in memcpy),
        "memset_count": len(memset),
        "memset_total_us": sum(e.get("dur", 0) for e in memset),
    }


def analyze_phase_split(events: list[dict]) -> dict:
    """
    Split kernels into forward / backward / optimizer phases using
    gpu_user_annotation boundaries.  Falls back to a simple 'unknown' bucket
    if no annotations are found.
    """
    # Collect GPU annotation spans sorted by timestamp
    annotations = []
    for e in events:
        if e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X":
            annotations.append({
                "name": e["name"],
                "start": e.get("ts", 0),
                "end": e.get("ts", 0) + e.get("dur", 0),
                "dur": e.get("dur", 0),
            })
    annotations.sort(key=lambda x: x["start"])

    # Heuristic labeling: look for known patterns in annotation names
    def label_annotation(name: str) -> str:
        low = name.lower()
        if "optimizer" in low:
            return "optimizer"
        return "compiled_graph"

    # Collect all kernel (ts, dur) pairs
    kernels = [
        (e.get("ts", 0), e.get("dur", 0))
        for e in events
        if e.get("cat") == "kernel"
    ]

    if not annotations:
        total = sum(d for _, d in kernels)
        return {"phases": {"unknown": {"kernel_count": len(kernels), "total_us": total}},
                "annotations": []}

    # Assign kernels to annotation spans.  If a kernel falls in multiple
    # spans we attribute it to the first matching one.
    phase_kernels: dict[str, dict] = defaultdict(lambda: {"kernel_count": 0, "total_us": 0.0, "gpu_span_us": 0.0})
    unassigned = {"kernel_count": 0, "total_us": 0.0}

    for kts, kdur in kernels:
        assigned = False
        for ann in annotations:
            if ann["start"] <= kts < ann["end"]:
                label = ann["name"][:100]
                phase_kernels[label]["kernel_count"] += 1
                phase_kernels[label]["total_us"] += kdur
                phase_kernels[label]["gpu_span_us"] = ann["dur"]
                assigned = True
                break
        if not assigned:
            unassigned["kernel_count"] += 1
            unassigned["total_us"] += kdur

    phases = dict(phase_kernels)
    if unassigned["kernel_count"]:
        phases["(unassigned)"] = unassigned

    return {"phases": phases, "annotations": annotations}


def detect_bottlenecks(
    gpu_util: dict,
    cuda_rt: dict,
    cpu_ops: dict,
    phase_split: dict,
    annotations: list[dict],
) -> list[dict]:
    """Run heuristic checks and return a list of detected issues."""
    issues = []

    # 1. cudaDeviceSynchronize in optimizer (bitsandbytes pattern)
    for name, info in cuda_rt["ranked"]:
        if "Synchronize" in name and info["count"] > 50:
            issues.append({
                "severity": "HIGH" if info["count"] > 100 else "MEDIUM",
                "title": f"Excessive cudaDeviceSynchronize ({info['count']} calls)",
                "detail": (
                    f"{info['count']} sync calls totalling {info['total_us']:,.0f} us. "
                    f"Avg {info['total_us']/info['count']:.1f} us each. "
                    "This forces CPU-GPU serialization and prevents kernel pipelining. "
                    "Common cause: bitsandbytes 8-bit optimizer syncs per parameter."
                ),
            })

    # 2. Low GPU kernel density
    density = gpu_util.get("density_pct", 0)
    if density < 70:
        issues.append({
            "severity": "HIGH",
            "title": f"Low GPU kernel density ({density:.1f}%)",
            "detail": (
                f"GPU kernels occupy only {density:.1f}% of the active span. "
                f"Total idle gaps: {gpu_util['total_gap_us']:,.0f} us. "
                "Consider using torch.compile with CUDA graphs, or check for "
                "CPU-side bottlenecks (data loading, Python overhead)."
            ),
        })
    elif density < 85:
        issues.append({
            "severity": "MEDIUM",
            "title": f"Moderate GPU kernel density ({density:.1f}%)",
            "detail": (
                f"GPU kernels occupy {density:.1f}% of the active span. "
                f"Total idle gaps: {gpu_util['total_gap_us']:,.0f} us across "
                f"{gpu_util['gap_count']} gaps."
            ),
        })

    # 3. Many individual kernel launches (not using CUDA graphs)
    for name, info in cuda_rt["ranked"]:
        if "LaunchKernel" in name and info["count"] > 200:
            issues.append({
                "severity": "MEDIUM",
                "title": f"High kernel launch count ({info['count']} cudaLaunchKernel calls)",
                "detail": (
                    f"{info['count']} individual kernel launches totalling "
                    f"{info['total_us']:,.0f} us CPU time. "
                    "If forward/backward are not using CUDA graphs, consider "
                    "torch.compile(mode='reduce-overhead')."
                ),
            })

    # 4. Large single CPU op (blocking on GPU)
    # Ops that appear large because they contain a blocking CUDA call
    # while the GPU runs a compiled graph — not real bottlenecks.
    blocking_patterns = ("AccumulateGrad", "aten::add_", "aten::copy_")
    for name, info in cpu_ops["ranked"][:5]:
        if info["total_us"] > 100_000:
            if any(pat in name for pat in blocking_patterns):
                issues.append({
                    "severity": "INFO",
                    "title": f"CPU blocked in {name} ({info['total_us']/1000:.1f} ms)",
                    "detail": (
                        "This is typically the CPU waiting for the GPU to finish a "
                        "CUDA graph. Not a real bottleneck if GPU utilization is high."
                    ),
                })
            else:
                issues.append({
                    "severity": "HIGH",
                    "title": f"Large CPU op: {name} ({info['total_us']/1000:.1f} ms)",
                    "detail": (
                        f"Aggregated {info['total_us']:,.0f} us across {info['count']} calls. "
                        "Investigate whether this blocks GPU execution."
                    ),
                })

    # 5. No CUDA graph launches detected
    graph_launches = sum(
        info["count"]
        for name, info in cuda_rt["ranked"]
        if "GraphLaunch" in name
    )
    if graph_launches == 0:
        issues.append({
            "severity": "MEDIUM",
            "title": "No CUDA graph launches detected",
            "detail": (
                "torch.compile with mode='reduce-overhead' captures forward/backward "
                "into CUDA graphs, reducing CPU dispatch overhead to ~1 ms per phase. "
                "Consider enabling it."
            ),
        })

    # 6. Significant memcpy time
    # (checked externally via memory_ops, but we can add here if needed)

    if not issues:
        issues.append({
            "severity": "INFO",
            "title": "No major bottlenecks detected",
            "detail": "The trace looks healthy. Check the detailed breakdown for minor optimizations.",
        })

    return issues


# ---------------------------------------------------------------------------
# Formatting / report generation
# ---------------------------------------------------------------------------

def fmt_us(us: float) -> str:
    """Format microseconds into a human-readable string."""
    if us >= 1_000_000:
        return f"{us/1_000_000:.2f} s"
    if us >= 1_000:
        return f"{us/1_000:.1f} ms"
    return f"{us:.1f} us"


def fmt_pct(part: float, whole: float) -> str:
    if whole == 0:
        return "  N/A"
    return f"{part/whole*100:5.1f}%"


def generate_report(
    trace_path: str,
    events: list[dict],
    top_n: int,
) -> str:
    """Run all analyses and build a formatted text report."""
    dur_events = duration_events(events)
    categories = analyze_categories(events)
    gpu_kernels = analyze_gpu_kernels(dur_events, top_n)
    cpu_ops = analyze_cpu_ops(dur_events, top_n)
    cuda_rt = analyze_cuda_runtime(dur_events)
    annotations = analyze_annotations(dur_events)
    gpu_util = analyze_gpu_utilization(dur_events)
    mem_ops = analyze_memory_ops(dur_events)
    phase_split = analyze_phase_split(dur_events)
    bottlenecks = detect_bottlenecks(gpu_util, cuda_rt, cpu_ops, phase_split, annotations)

    lines: list[str] = []
    def out(s: str = ""):
        lines.append(s)
    def header(s: str):
        out(f"\n{'='*70}")
        out(f"  {s}")
        out(f"{'='*70}")

    # Title
    out(f"PyTorch Profiler Trace Analysis")
    out(f"Trace: {trace_path}")
    out(f"Total events: {len(events):,}")
    out(f"Duration events: {len(dur_events):,}")
    out(f"Categories: {', '.join(categories.keys())}")

    # --- Bottlenecks (up front) ---
    header("DETECTED BOTTLENECKS")
    for b in bottlenecks:
        out(f"\n  [{b['severity']}] {b['title']}")
        # Wrap detail text
        detail = b["detail"]
        while len(detail) > 72:
            split = detail[:72].rfind(" ")
            if split == -1:
                split = 72
            out(f"    {detail[:split]}")
            detail = detail[split:].lstrip()
        if detail:
            out(f"    {detail}")

    # --- GPU Utilization ---
    header("GPU UTILIZATION")
    if gpu_util["kernel_count"] == 0:
        out("  No GPU kernels found in trace.")
    else:
        out(f"  Kernel count:       {gpu_util['kernel_count']:,}")
        out(f"  Total kernel time:  {fmt_us(gpu_util['total_kernel_us'])}")
        out(f"  GPU active span:    {fmt_us(gpu_util['span_us'])}")
        out(f"  Kernel density:     {gpu_util['density_pct']:.1f}%")
        out(f"  Total idle gaps:    {fmt_us(gpu_util['total_gap_us'])} across {gpu_util['gap_count']} gaps")
        out(f"  Avg gap:            {fmt_us(gpu_util['avg_gap_us'])}")
        out(f"  Gaps > 50us:        {gpu_util['big_gaps_count']} ({fmt_us(gpu_util['big_gaps_total_us'])} total)")
        if gpu_util["top_10_gaps"]:
            top_gaps_str = ", ".join(f"{g:.0f}" for g in gpu_util["top_10_gaps"])
            out(f"  Top 10 gaps (us):   [{top_gaps_str}]")

    # --- Phase Split ---
    header("PHASE BREAKDOWN (GPU)")
    phases = phase_split["phases"]
    phase_total = sum(p["total_us"] for p in phases.values())
    # Label compiled graphs by order: 1st = forward, 2nd = backward (common pattern)
    graph_names = [
        name for name in phases
        if "CompiledFxGraph" in name and name != "(unassigned)"
    ]
    # Sort by first kernel timestamp to infer execution order
    graph_order = {}
    if len(graph_names) >= 2:
        ann_by_name = {a["name"][:100]: a["start"] for a in phase_split.get("annotations", [])}
        ordered = sorted(graph_names, key=lambda n: ann_by_name.get(n, 0))
        graph_order[ordered[0]] = "(likely forward)"
        graph_order[ordered[1]] = "(likely backward)"

    for name, info in sorted(phases.items(), key=lambda x: -x[1]["total_us"]):
        pct = fmt_pct(info["total_us"], phase_total)
        span = f" (span: {fmt_us(info['gpu_span_us'])})" if info.get("gpu_span_us") else ""
        label = f" {graph_order[name]}" if name in graph_order else ""
        out(f"  {pct} | {info['kernel_count']:>5} kernels | {fmt_us(info['total_us']):>10}{span}")
        out(f"         {name}{label}")

    # --- Annotations ---
    header("ANNOTATIONS (NVTX / user ranges)")
    if not annotations:
        out("  No annotations found.")
    else:
        for a in annotations:
            out(f"  {fmt_us(a['dur']):>10} | {a['cat']:25s} | {a['name'][:90]}")

    # --- Top GPU Kernels ---
    header(f"TOP {top_n} GPU KERNELS (aggregated)")
    out(f"  Total GPU kernel time: {fmt_us(gpu_kernels['total_us'])}")
    out()
    for name, info in gpu_kernels["ranked"]:
        pct = fmt_pct(info["total_us"], gpu_kernels["total_us"])
        out(f"  {pct} | {fmt_us(info['total_us']):>10} | x{info['count']:>3} | {name[:90]}")

    # --- Kernel type summary ---
    header("GPU KERNEL SUMMARY BY TYPE")
    type_agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    for name, info in gpu_kernels["all"]:
        low = name.lower()
        if "gemm" in low or "cutlass" in low:
            ktype = "GEMM (matmul)"
        elif "flash_bwd" in low:
            ktype = "FlashAttention backward"
        elif "flash_fwd" in low:
            ktype = "FlashAttention forward"
        elif "triton" in low:
            ktype = "Triton fused"
        elif "optimizer" in low or "optim" in low:
            ktype = "Optimizer"
        elif "memcpy" in low or "memset" in low:
            ktype = "Memory ops"
        elif "elementwise" in low or "vectorized" in low:
            ktype = "Elementwise"
        elif "reduce" in low or "softmax" in low:
            ktype = "Reduction/Softmax"
        else:
            ktype = "Other"
        type_agg[ktype]["count"] += info["count"]
        type_agg[ktype]["total_us"] += info["total_us"]

    total_k = gpu_kernels["total_us"]
    for ktype, info in sorted(type_agg.items(), key=lambda x: -x[1]["total_us"]):
        pct = fmt_pct(info["total_us"], total_k)
        out(f"  {pct} | {fmt_us(info['total_us']):>10} | x{info['count']:>4} | {ktype}")

    # --- Top CPU Ops ---
    header(f"TOP {top_n} CPU OPERATIONS (aggregated)")
    out(f"  Total CPU op time: {fmt_us(cpu_ops['total_us'])}")
    out()
    for name, info in cpu_ops["ranked"]:
        pct = fmt_pct(info["total_us"], cpu_ops["total_us"])
        out(f"  {pct} | {fmt_us(info['total_us']):>10} | x{info['count']:>3} | {name[:90]}")

    # --- CUDA Runtime ---
    header("CUDA RUNTIME CALLS")
    for name, info in cuda_rt["ranked"][:15]:
        avg = info["total_us"] / info["count"] if info["count"] else 0
        out(f"  {fmt_us(info['total_us']):>10} | x{info['count']:>5} | avg {fmt_us(avg):>8} | {name}")

    # --- Memory ---
    header("MEMORY OPERATIONS")
    out(f"  GPU memcpy:  {mem_ops['memcpy_count']} events, {fmt_us(mem_ops['memcpy_total_us'])} total")
    out(f"  GPU memset:  {mem_ops['memset_count']} events, {fmt_us(mem_ops['memset_total_us'])} total")

    out()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a PyTorch profiler trace JSON and report bottlenecks.",
    )
    parser.add_argument("trace", help="Path to .pt.trace.json file")
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of top entries to show per section (default: 20)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save the report to a file (markdown-compatible text)",
    )
    args = parser.parse_args()

    trace_path = args.trace
    if not Path(trace_path).exists():
        print(f"Error: file not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading trace: {trace_path} ...", file=sys.stderr)
    events = load_trace(trace_path)
    print(f"Loaded {len(events):,} events. Analyzing...", file=sys.stderr)

    report = generate_report(trace_path, events, top_n=args.top)

    if args.save:
        Path(args.save).write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {args.save}", file=sys.stderr)

    print(report)


if __name__ == "__main__":
    main()
