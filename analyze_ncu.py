#!/usr/bin/env python3
"""CLI tool: analyze Nsight Compute .ncu-rep files."""

import argparse
import csv
import collections
import subprocess
import sys
import tempfile
import os


METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
]

OCCUPANCY_METRICS = [
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]


def run_ncu_csv(rep_path: str, metrics: list[str]) -> str:
    cmd = [
        "ncu", "--import", rep_path,
        "--csv",
        "--metrics", ",".join(metrics),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ncu error: {result.stderr[:500]}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def parse_csv(text: str) -> dict:
    """Returns dict keyed by (id, kernel_name) -> {metric: value}."""
    reader = csv.DictReader(text.splitlines())
    data: dict = collections.defaultdict(dict)
    for row in reader:
        kid = row.get("ID", "")
        name = row.get("Kernel Name", "")
        metric = row.get("Metric Name", "")
        val_str = row.get("Metric Value", "").replace(",", ".")
        unit = row.get("Metric Unit", "")
        try:
            val = float(val_str)
        except ValueError:
            continue
        key = (kid, name)
        data[key]["name"] = name
        if metric == "Duration" and unit == "us":
            data[key]["dur_us"] = val
        elif metric == "Compute (SM) Throughput" and unit == "%":
            data[key]["sm_pct"] = val
        elif metric == "DRAM Throughput" and unit == "%":
            data[key]["dram_pct"] = val
        elif metric == "Memory Throughput" and unit == "%":
            data[key]["mem_pct"] = val
    return data


def categorize(name: str) -> str:
    if "flash_fwd" in name:
        return "FlashAttn fwd"
    if "flash_bwd" in name:
        return "FlashAttn bwd"
    if "splitKreduce" in name:
        return "GEMM split-K"
    if "gemm" in name.lower() or "s16816" in name or "s1688" in name:
        return "GEMM"
    if "cutlass" in name.lower():
        return "CUTLASS"
    if "GammaBeta" in name or "layer_norm_grad" in name or "vectorized_layer_norm" in name:
        return "LayerNorm"
    if "multi_tensor_apply" in name and ("Adam" in name or "FusedAdam" in name):
        return "FusedAdamW"
    if "multi_tensor_apply" in name and "LpNorm" in name:
        return "GradNorm"
    if "multi_tensor_apply" in name:
        return "MultiTensor"
    if "reduce_kernel" in name:
        return "Reduce"
    if "scatter_gather" in name:
        return "Scatter/Gather"
    if "Gelu" in name or "gelu" in name:
        return "GELU"
    if "distribution" in name or "normal_kernel" in name:
        return "RNG"
    if "radixSort" in name:
        return "Sort"
    if "nchwToNhwc" in name or "nhwcToNchw" in name:
        return "Transpose"
    if "CatArray" in name:
        return "Cat"
    if "elementwise" in name or "vectorized" in name or "unrolled" in name or "FillFunctor" in name:
        return "Elementwise"
    return "Other"


def shorten_name(name: str) -> str:
    if "flash_fwd_kernel" in name:
        return "FlashAttn fwd"
    if "flash_bwd" in name:
        return "FlashAttn bwd"
    if "multi_tensor_apply" in name and "Adam" in name:
        return "FusedAdamW"
    if "multi_tensor_apply" in name and "LpNorm" in name:
        return "GradNorm (LpNorm)"
    if "multi_tensor_apply" in name:
        return "MultiTensor"
    if "vectorized_layer_norm" in name:
        return "LayerNorm fwd"
    if "layer_norm_grad_input" in name:
        return "LayerNorm bwd (dX)"
    if "GammaBetaBackward" in name:
        return "LayerNorm bwd (dγ/dβ)"
    if "reduce_kernel" in name and "BFloat16" in name:
        return "Reduce (bf16)"
    if "reduce_kernel" in name:
        return "Reduce (fp32)"
    if "scatter_gather" in name and "<0," in name:
        return "Scatter (mask write)"
    if "scatter_gather" in name and "<1," in name:
        return "Gather (mask read)"
    if "distribution" in name or "normal_kernel" in name:
        return "RNG (normal)"
    if "bfloat16_copy" in name:
        return "Copy bf16"
    if "direct_copy" in name:
        return "Copy fp32"
    if "FillFunctor<float>" in name:
        return "Fill fp32"
    if "FillFunctor<c10::BFloat16>" in name:
        return "Fill bf16"
    if "CUDAFunctor_add<float>" in name:
        return "Add fp32"
    if "CUDAFunctor_add<c10::BFloat16>" in name:
        return "Add bf16"
    if "GeluCUDA" in name and "Backward" not in name:
        return "GELU fwd"
    if "GeluBackward" in name:
        return "GELU bwd"
    if "MulFunctor" in name:
        return "Mul elementwise"
    if "CatArray" in name:
        return "Cat (concat)"
    if "radixSort" in name:
        return "RadixSort"
    if "nchwToNhwc" in name:
        return "nchw→nhwc"
    if "splitKreduce" in name:
        return "GEMM split-K reduce"
    if "ampere_bf16" in name:
        parts = name.split("_")
        tile = next((p for p in parts if "x" in p and p[0].isdigit()), "?")
        return f"GEMM bf16 {tile}"
    if "s1688gemm" in name:
        tag = "relu" if "relu" in name else "lin"
        return f"GEMM s1688 {tag}"
    if "cutlass" in name.lower():
        return "CUTLASS GEMM"
    return name[:52]


def hr(char: str = "-", width: int = 76) -> str:
    return char * width


def section(title: str) -> None:
    print(f"\n{'='*76}")
    print(f"  {title}")
    print(f"{'='*76}")


def print_category_table(data: dict) -> None:
    section("Kernel Time by Category")
    by_cat: dict = collections.defaultdict(
        lambda: {"total_us": 0.0, "count": 0, "sm_sum": 0.0, "dram_sum": 0.0, "n": 0}
    )
    for v in data.values():
        if "dur_us" not in v:
            continue
        cat = categorize(v["name"])
        by_cat[cat]["total_us"] += v["dur_us"]
        by_cat[cat]["count"] += 1
        if "sm_pct" in v:
            by_cat[cat]["sm_sum"] += v["sm_pct"]
            by_cat[cat]["dram_sum"] += v.get("dram_pct", 0)
            by_cat[cat]["n"] += 1

    total_us = sum(d["total_us"] for d in by_cat.values())
    total_ms = total_us / 1000
    n_kernels = sum(d["count"] for d in by_cat.values())

    print(f"\n  Total GPU time profiled: {total_ms:.2f} ms  |  Kernel invocations: {n_kernels}\n")
    hdr = f"  {'Category':<22} {'ms':>8} {'Share':>7} {'Cnt':>6} {'Avg SM%':>8} {'Avg DRAM%':>10}"
    print(hdr)
    print("  " + hr())
    for cat, d in sorted(by_cat.items(), key=lambda x: -x[1]["total_us"]):
        avg_sm = d["sm_sum"] / d["n"] if d["n"] else 0
        avg_dram = d["dram_sum"] / d["n"] if d["n"] else 0
        pct = d["total_us"] / total_us * 100
        print(f"  {cat:<22} {d['total_us']/1000:>8.3f} {pct:>6.1f}% {d['count']:>6} {avg_sm:>8.1f} {avg_dram:>10.1f}")


def print_top_kernels(data: dict, top_n: int = 25) -> None:
    section(f"Top {top_n} Individual Kernels by GPU Time")
    rows = [
        (v["dur_us"], v.get("sm_pct", 0), v.get("dram_pct", 0), v["name"])
        for v in data.values()
        if "dur_us" in v and "name" in v
    ]
    rows.sort(reverse=True)
    total_us = sum(r[0] for r in rows)

    print(f"\n  {'Kernel':<52} {'ms':>7} {'SM%':>5} {'DRAM%':>6}")
    print("  " + hr())
    for dur, sm, dram, name in rows[:top_n]:
        pct = dur / total_us * 100
        label = shorten_name(name)
        print(f"  {label:<52} {dur/1000:>7.3f} {sm:>5.1f} {dram:>6.1f}")


def print_low_utilization(data: dict, sm_threshold: float = 20, min_ms: float = 0.05) -> None:
    section(f"Low SM Utilization Warnings (SM < {sm_threshold}%, dur > {min_ms:.0f}µs)")
    rows = [
        (v["dur_us"], v.get("sm_pct", 0), v.get("dram_pct", 0), v["name"])
        for v in data.values()
        if "dur_us" in v and v.get("sm_pct", 100) < sm_threshold and v["dur_us"] > min_ms * 1000
    ]
    rows.sort(key=lambda x: -x[0])
    if not rows:
        print("\n  None found.")
        return

    total_us = sum(r[0] for r in rows)
    print(f"\n  {len(rows)} kernels, total: {total_us/1000:.3f} ms\n")
    print(f"  {'Kernel':<52} {'ms':>7} {'SM%':>5} {'DRAM%':>6}")
    print("  " + hr())
    for dur, sm, dram, name in rows[:20]:
        print(f"  {shorten_name(name):<52} {dur/1000:>7.3f} {sm:>5.1f} {dram:>6.1f}")


def print_occupancy_summary(data: dict) -> None:
    section("Occupancy Summary (SM% proxy)")
    sm_vals = [v["sm_pct"] for v in data.values() if "sm_pct" in v]
    if not sm_vals:
        print("  No SM% data found.")
        return
    weighted_sm = sum(
        v["sm_pct"] * v["dur_us"]
        for v in data.values()
        if "sm_pct" in v and "dur_us" in v
    )
    total_us = sum(v["dur_us"] for v in data.values() if "sm_pct" in v and "dur_us" in v)
    weighted_avg = weighted_sm / total_us if total_us else 0

    buckets = {"<10%": 0, "10-25%": 0, "25-50%": 0, "50-75%": 0, ">75%": 0}
    for v in sm_vals:
        if v < 10:
            buckets["<10%"] += 1
        elif v < 25:
            buckets["10-25%"] += 1
        elif v < 50:
            buckets["25-50%"] += 1
        elif v < 75:
            buckets["50-75%"] += 1
        else:
            buckets[">75%"] += 1

    print(f"\n  Weighted-avg SM throughput: {weighted_avg:.1f}%")
    print(f"  Min: {min(sm_vals):.1f}%  Max: {max(sm_vals):.1f}%  Mean: {sum(sm_vals)/len(sm_vals):.1f}%\n")
    print(f"  {'SM% bucket':<12} {'Count':>8} {'Bar'}")
    print("  " + hr(width=50))
    n_total = len(sm_vals)
    for bucket, count in buckets.items():
        bar = "█" * int(count / n_total * 40)
        print(f"  {bucket:<12} {count:>8}   {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Nsight Compute .ncu-rep files")
    parser.add_argument("rep_file", help="Path to .ncu-rep file")
    parser.add_argument("--top", type=int, default=25, help="Number of top kernels to show (default: 25)")
    parser.add_argument("--sm-threshold", type=float, default=20.0, help="SM%% threshold for low-util warning (default: 20)")
    args = parser.parse_args()

    if not os.path.exists(args.rep_file):
        print(f"File not found: {args.rep_file}", file=sys.stderr)
        sys.exit(1)

    print(f"\nAnalyzing: {args.rep_file}")

    csv_text = run_ncu_csv(args.rep_file, METRICS)
    data = parse_csv(csv_text)

    if not data:
        print("No kernel data found in report.", file=sys.stderr)
        sys.exit(1)

    print_category_table(data)
    print_top_kernels(data, top_n=args.top)
    print_occupancy_summary(data)
    print_low_utilization(data, sm_threshold=args.sm_threshold)
    print()


if __name__ == "__main__":
    main()
