"""
Heston Monte Carlo Benchmark — MLX vs PyTorch MPS
===================================================
Scales from 10k to 2M paths, 252 timesteps, 5 runs each.
Keeps computation single-threaded per run (FastAPI-friendly).
Run with:  python heston_benchmark.py
"""

import time
import gc
import sys
import platform
import subprocess
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Attempt imports ────────────────────────────────────────────────────────────
try:
    import mlx.core as mx
    import mlx.core.random as mx_rand
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("WARNING: mlx not found. Install with: pip install mlx")

try:
    import torch
    TORCH_AVAILABLE = True
    MPS_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    print("WARNING: torch not found. Install with: pip install torch")

# ── Calibrated Heston params (realistic equity surface fit) ───────────────────
HESTON = dict(
    S0    = 100.0,   # spot
    v0    = 0.04,    # initial variance (~20% vol)
    kappa = 2.0,     # mean reversion speed
    theta = 0.04,    # long-run variance
    xi    = 0.3,     # vol-of-vol
    rho   = -0.7,    # spot-vol correlation
    r     = 0.05,    # risk-free rate
    T     = 1.0,     # 1 year
)

N_STEPS   = 252      # daily steps
N_RUNS    = 5        # repeat runs per path count for timing stability
PATH_COUNTS = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]


# ══════════════════════════════════════════════════════════════════════════════
#  MEMORY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def estimate_memory(n_paths: int, n_steps: int, dtype_bytes: int = 4) -> float:
    """Estimate peak GPU/unified memory in GB for one simulation run."""
    # Two path arrays (S, v) + two correlated BM matrices (z1, z2)
    # z matrices: n_paths × n_steps, paths: n_paths (scalar per step)
    bm_bytes  = 2 * n_paths * n_steps * dtype_bytes
    path_bytes = 2 * n_paths * dtype_bytes
    return (bm_bytes + path_bytes) / 1e9


def get_system_memory_gb() -> float:
    """Return total unified memory in GB (macOS)."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True
        )
        return int(result.stdout.strip()) / 1e9
    except Exception:
        return 0.0


def memory_preflight():
    total_gb = get_system_memory_gb()
    print("=" * 62)
    print("  MEMORY PREFLIGHT CHECK")
    print("=" * 62)
    print(f"  System unified memory : {total_gb:.1f} GB")
    print(f"  Dtype                 : FP32 (4 bytes)")
    print(f"  Timesteps             : {N_STEPS}")
    print()
    print(f"  {'Paths':>10}  {'Peak est. (GB)':>14}  {'Status':>10}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*10}")

    all_ok = True
    for n in PATH_COUNTS:
        est = estimate_memory(n, N_STEPS)
        # Apple Silicon shares memory; conservatively flag if >40% of total
        budget = total_gb * 0.40
        status = "OK" if est < budget else "WARN — may be tight"
        if est >= budget:
            all_ok = False
        print(f"  {n:>10,}  {est:>14.3f}  {status:>10}")

    print()
    if all_ok:
        print("  All path counts within conservative 40% memory budget.")
    else:
        print("  WARNING: some counts may be tight. Script will continue.")
    print("=" * 62)
    print()
    return total_gb


# ══════════════════════════════════════════════════════════════════════════════
#  MLX SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def heston_mlx(n_paths: int, params: dict, n_steps: int) -> np.ndarray:
    """
    Euler-Maruyama Heston simulation in MLX.
    Pre-generates all Brownian increments in one batch for GPU efficiency.
    Returns terminal stock prices as a numpy array.
    """
    S0, v0    = params["S0"],    params["v0"]
    kappa     = params["kappa"]
    theta     = params["theta"]
    xi        = params["xi"]
    rho       = params["rho"]
    r         = params["r"]
    T         = params["T"]

    dt      = T / n_steps
    sqrt_dt = float(np.sqrt(dt))

    # Pre-generate correlated BM increments — shape (n_paths, n_steps)
    z1 = mx_rand.normal(shape=(n_paths, n_steps))
    z2 = mx_rand.normal(shape=(n_paths, n_steps))
    w_S = z1
    w_v = mx.array(rho) * z1 + mx.array(float(np.sqrt(1 - rho**2))) * z2

    # Initialise state vectors
    S = mx.ones((n_paths,), dtype=mx.float32) * float(S0)
    v = mx.ones((n_paths,), dtype=mx.float32) * float(v0)

    # Euler-Maruyama stepping
    for i in range(n_steps):
        v_pos = mx.maximum(v, mx.array(0.0))   # full truncation scheme
        sqrt_v = mx.sqrt(v_pos)

        S = S * mx.exp(
            (float(r) - 0.5 * v_pos) * float(dt)
            + sqrt_v * float(sqrt_dt) * w_S[:, i]
        )
        v = (v
             + float(kappa) * (float(theta) - v_pos) * float(dt)
             + float(xi) * sqrt_v * float(sqrt_dt) * w_v[:, i])

    # Force evaluation before timing stops
    mx.eval(S)
    return np.array(S)


def time_mlx(n_paths: int, n_runs: int) -> dict:
    times = []
    prices = None
    for run in range(n_runs):
        # Warmup on first run to exclude JIT compilation
        if run == 0 and n_paths <= 50_000:
            heston_mlx(min(n_paths, 10_000), HESTON, N_STEPS)

        t0 = time.perf_counter()
        out = heston_mlx(n_paths, HESTON, N_STEPS)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if run == 0:
            prices = out

        # Light GC between runs
        gc.collect()

    call_price = float(np.mean(np.maximum(prices - HESTON["S0"], 0))
                       * np.exp(-HESTON["r"] * HESTON["T"]))
    return {
        "times":      times,
        "mean":       float(np.mean(times)),
        "std":        float(np.std(times)),
        "median":     float(np.median(times)),
        "call_price": call_price,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PYTORCH MPS SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def heston_torch(n_paths: int, params: dict, n_steps: int,
                 device: torch.device) -> np.ndarray:
    """
    Euler-Maruyama Heston simulation in PyTorch.
    Pre-generates all Brownian increments in one batch.
    """
    S0, v0 = params["S0"], params["v0"]
    kappa  = params["kappa"]
    theta  = params["theta"]
    xi     = params["xi"]
    rho    = params["rho"]
    r      = params["r"]
    T      = params["T"]

    dt      = T / n_steps
    sqrt_dt = float(np.sqrt(dt))
    rho_c   = float(np.sqrt(1 - rho**2))

    z1 = torch.randn(n_paths, n_steps, dtype=torch.float32, device=device)
    z2 = torch.randn(n_paths, n_steps, dtype=torch.float32, device=device)
    w_S = z1
    w_v = rho * z1 + rho_c * z2

    S = torch.ones(n_paths, dtype=torch.float32, device=device) * S0
    v = torch.ones(n_paths, dtype=torch.float32, device=device) * v0

    for i in range(n_steps):
        v_pos  = torch.clamp(v, min=0.0)
        sqrt_v = torch.sqrt(v_pos)

        S = S * torch.exp(
            (r - 0.5 * v_pos) * dt
            + sqrt_v * sqrt_dt * w_S[:, i]
        )
        v = (v
             + kappa * (theta - v_pos) * dt
             + xi * sqrt_v * sqrt_dt * w_v[:, i])

    # Synchronise MPS before stopping timer
    if device.type == "mps":
        torch.mps.synchronize()

    return S.cpu().numpy()


def time_torch(n_paths: int, n_runs: int, device: torch.device) -> dict:
    times = []
    prices = None
    for run in range(n_runs):
        if run == 0 and n_paths <= 50_000:
            heston_torch(min(n_paths, 10_000), HESTON, N_STEPS, device)

        t0 = time.perf_counter()
        out = heston_torch(n_paths, HESTON, N_STEPS, device)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if run == 0:
            prices = out

        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    call_price = float(np.mean(np.maximum(prices - HESTON["S0"], 0))
                       * np.exp(-HESTON["r"] * HESTON["T"]))
    return {
        "times":      times,
        "mean":       float(np.mean(times)),
        "std":        float(np.std(times)),
        "median":     float(np.median(times)),
        "call_price": call_price,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmarks() -> dict:
    results = {n: {} for n in PATH_COUNTS}
    device_mps = torch.device("mps") if MPS_AVAILABLE else None

    for n in PATH_COUNTS:
        print(f"  Paths: {n:>9,}", end="", flush=True)

        if MLX_AVAILABLE:
            r = time_mlx(n, N_RUNS)
            results[n]["mlx"] = r
            print(f"  |  MLX  {r['median']:.3f}s (±{r['std']:.3f})", end="", flush=True)

        if TORCH_AVAILABLE and MPS_AVAILABLE:
            r = time_torch(n, N_RUNS, device_mps)
            results[n]["torch_mps"] = r
            print(f"  |  MPS  {r['median']:.3f}s (±{r['std']:.3f})", end="", flush=True)

        print()

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "mlx":       "#1D9E75",   # teal
    "torch_mps": "#378ADD",   # blue
}
LABELS = {
    "mlx":       "MLX",
    "torch_mps": "PyTorch MPS",
}


def plot_results(results: dict, output_path: str = "heston_benchmark.png"):
    backends = [b for b in ["mlx", "torch_mps"]
                if any(b in results[n] for n in PATH_COUNTS)]
    ns = PATH_COUNTS

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    fig.suptitle(
        f"Heston MC Benchmark — MLX vs PyTorch MPS\n"
        f"{N_STEPS} timesteps · {N_RUNS} runs each · FP32 · Apple Silicon",
        fontsize=13, fontweight="normal", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])   # scaling — wall time
    ax2 = fig.add_subplot(gs[0, 1])   # throughput (paths/s)
    ax3 = fig.add_subplot(gs[1, 0])   # head-to-head speedup
    ax4 = fig.add_subplot(gs[1, 1])   # run variability (violin-like)

    # ── 1. Scaling: wall time ────────────────────────────────────────────────
    for b in backends:
        medians = [results[n][b]["median"] for n in ns if b in results[n]]
        ns_b    = [n for n in ns if b in results[n]]
        stds    = [results[n][b]["std"]    for n in ns if b in results[n]]
        ax1.plot(ns_b, medians, "o-", color=COLORS[b], label=LABELS[b],
                 linewidth=2, markersize=5)
        ax1.fill_between(ns_b,
                         [m - s for m, s in zip(medians, stds)],
                         [m + s for m, s in zip(medians, stds)],
                         color=COLORS[b], alpha=0.15)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of paths")
    ax1.set_ylabel("Wall time (s)")
    ax1.set_title("Scaling: wall time")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, which="both", alpha=0.2)
    ax1.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    # ── 2. Throughput (paths / second) ───────────────────────────────────────
    for b in backends:
        ns_b    = [n for n in ns if b in results[n]]
        thru    = [n / results[n][b]["median"] for n in ns_b]
        ax2.plot(ns_b, thru, "o-", color=COLORS[b], label=LABELS[b],
                 linewidth=2, markersize=5)

    ax2.set_xscale("log")
    ax2.set_xlabel("Number of paths")
    ax2.set_ylabel("Paths / second")
    ax2.set_title("GPU throughput")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(True, which="both", alpha=0.2)
    ax2.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"
        )
    )

    # ── 3. Head-to-head speedup (MLX / MPS) ─────────────────────────────────
    if "mlx" in backends and "torch_mps" in backends:
        shared_ns = [n for n in ns
                     if "mlx" in results[n] and "torch_mps" in results[n]]
        speedups = [results[n]["torch_mps"]["median"] / results[n]["mlx"]["median"]
                    for n in shared_ns]

        bar_colors = ["#1D9E75" if s >= 1 else "#E24B4A" for s in speedups]
        bars = ax3.bar(range(len(shared_ns)), speedups, color=bar_colors,
                       width=0.6, alpha=0.85)
        ax3.axhline(1.0, color="#888", linewidth=1, linestyle="--")
        ax3.set_xticks(range(len(shared_ns)))
        ax3.set_xticklabels(
            [f"{n//1000}k" if n < 1_000_000 else f"{n//1_000_000}M"
             for n in shared_ns],
            fontsize=8
        )
        ax3.set_xlabel("Number of paths")
        ax3.set_ylabel("Speedup (×)")
        ax3.set_title("MLX speedup over PyTorch MPS\n(>1 = MLX faster)")
        ax3.grid(True, axis="y", alpha=0.2)
        for bar, s in zip(bars, speedups):
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.02,
                     f"{s:.2f}×", ha="center", va="bottom", fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Need both backends\nfor speedup chart",
                 ha="center", va="center", transform=ax3.transAxes,
                 color="gray", fontsize=10)
        ax3.set_title("MLX speedup over PyTorch MPS")

    # ── 4. Run variability — all individual run times ────────────────────────
    # Show for a selection of path counts to keep it readable
    sample_ns = [n for n in [100_000, 500_000, 1_000_000, 2_000_000]
                 if n in PATH_COUNTS]

    x_positions = np.arange(len(sample_ns))
    bar_width   = 0.35
    offsets     = {"mlx": -bar_width/2, "torch_mps": bar_width/2}

    for b in backends:
        offset  = offsets[b]
        medians = []
        stds    = []
        for n in sample_ns:
            if b in results[n]:
                medians.append(results[n][b]["median"])
                stds.append(results[n][b]["std"])
            else:
                medians.append(0)
                stds.append(0)

        ax4.bar(x_positions + offset, medians, width=bar_width,
                color=COLORS[b], label=LABELS[b], alpha=0.85)
        ax4.errorbar(x_positions + offset, medians, yerr=stds,
                     fmt="none", color="black", capsize=3, linewidth=1)

    ax4.set_xticks(x_positions)
    ax4.set_xticklabels(
        [f"{n//1000}k" if n < 1_000_000 else f"{n//1_000_000}M"
         for n in sample_ns],
        fontsize=9
    )
    ax4.set_xlabel("Number of paths")
    ax4.set_ylabel("Median time (s) ± 1 std")
    ax4.set_title("Timing variability (selected path counts)")
    ax4.legend(frameon=False, fontsize=9)
    ax4.grid(True, axis="y", alpha=0.2)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    print(f"\n  Plot saved to: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict):
    backends = [b for b in ["mlx", "torch_mps"]
                if any(b in results[n] for n in PATH_COUNTS)]

    print()
    print("=" * 78)
    print("  BENCHMARK SUMMARY")
    print("=" * 78)
    print(f"  Heston params: S0={HESTON['S0']}, v0={HESTON['v0']}, "
          f"κ={HESTON['kappa']}, θ={HESTON['theta']}, "
          f"ξ={HESTON['xi']}, ρ={HESTON['rho']}")
    print(f"  Steps: {N_STEPS}  |  Runs per config: {N_RUNS}  |  Dtype: FP32")
    print()

    # Header
    col = 14
    header = f"  {'Paths':>10}"
    for b in backends:
        header += f"  {LABELS[b]:>{col}}(med±std)"
    if len(backends) == 2:
        header += f"  {'MLX speedup':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for n in PATH_COUNTS:
        row = f"  {n:>10,}"
        for b in backends:
            if b in results[n]:
                r = results[n][b]
                row += f"  {r['median']:>8.3f}s ±{r['std']:>5.3f}"
            else:
                row += f"  {'N/A':>{col+9}}"
        if len(backends) == 2 and all(b in results[n] for b in backends):
            spd = results[n]["torch_mps"]["median"] / results[n]["mlx"]["median"]
            row += f"  {spd:>10.2f}×"
        print(row)

    print()
    print("  ATM call price (MC estimate, strike=S0):")
    for b in backends:
        prices = [results[n][b]["call_price"] for n in PATH_COUNTS
                  if b in results[n]]
        if prices:
            print(f"    {LABELS[b]:15s}: {prices[-1]:.4f}  "
                  f"(range across path counts: {min(prices):.4f}–{max(prices):.4f})")

    print()
    print("  Throughput at 1M paths:")
    for b in backends:
        if 1_000_000 in PATH_COUNTS and b in results[1_000_000]:
            t = results[1_000_000][b]["median"]
            print(f"    {LABELS[b]:15s}: {1_000_000/t:,.0f} paths/s  ({t:.3f}s)")

    print("=" * 78)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 62)
    print("  Heston MC Benchmark — MLX vs PyTorch MPS")
    print("=" * 62)
    print(f"  Platform : {platform.platform()}")
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  MLX      : {'available' if MLX_AVAILABLE else 'NOT FOUND'}")
    print(f"  PyTorch  : {'available' if TORCH_AVAILABLE else 'NOT FOUND'}")
    print(f"  MPS      : {'available' if MPS_AVAILABLE else 'NOT FOUND'}")
    print("=" * 62)
    print()

    if not MLX_AVAILABLE and not (TORCH_AVAILABLE and MPS_AVAILABLE):
        print("ERROR: No GPU backends available. Exiting.")
        sys.exit(1)

    # Memory check
    memory_preflight()

    # Run benchmarks
    print("Running benchmarks...")
    print(f"  (path counts: {PATH_COUNTS})")
    print(f"  ({N_RUNS} runs each, {N_STEPS} steps)")
    print()
    results = run_benchmarks()

    # Output
    print_summary(results)
    plot_results(results, output_path="heston_benchmark.png")
    print()


if __name__ == "__main__":
    main()