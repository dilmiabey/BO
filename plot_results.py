"""
plot_results.py
===============
Aggregates results across all seeds and produces 4 publication-quality plots:

  1. Sine     × homoscedastic noise
  2. Sine     × heteroscedastic noise
  3. Branin   × homoscedastic noise
  4. Branin   × heteroscedastic noise

Each plot shows two trajectories (GP-UCB vs GP-UCB-PV) with mean ± shaded
variance across 5 seeds.

Usage:
    python plot_results.py --results_dir results --plots_dir plots --n_seeds 5
"""

import argparse
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np

BENCHMARKS  = ["sine", "branin"]
NOISE_TYPES = ["homo", "hetero"]
ALGORITHMS  = ["gpucb", "gpucb_pv"]

ALGO_LABELS = {
    "gpucb":    "GP-UCB",
    "gpucb_pv": "GP-UCB-PV",
}
ALGO_COLORS = {
    "gpucb":    "#1f77b4",   # blue
    "gpucb_pv": "#d62728",   # red
}

BENCH_LABELS = {
    "sine":   "Sine Benchmark",
    "branin": "Branin Benchmark",
}
NOISE_LABELS = {
    "homo":   "Homoscedastic Noise",
    "hetero": "Heteroscedastic Noise",
}


def load_runs(results_dir: str, seeds: list,
              benchmark: str, noise: str, algorithm: str) -> Optional[np.ndarray]:
    """
    Load f(x_t) arrays for all seeds.
    Returns array of shape (n_seeds, n_iter), or None if any seed is missing.
    """
    arrays = []
    for seed in seeds:
        path = os.path.join(results_dir, f"seed_{seed}",
                            f"{benchmark}_{noise}_{algorithm}.npy")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            return None
        arrays.append(np.load(path))

    runs = np.stack(arrays, axis=0)  # (n_seeds, n_iter)
    return runs


def plot_panel(ax, results_dir, seeds, benchmark, noise):
    """Draw one panel (benchmark × noise) onto the provided axes."""
    ax.set_title(
        f"{BENCH_LABELS[benchmark]}\n{NOISE_LABELS[noise]}",
        fontsize=11, pad=6
    )

    for algo in ALGORITHMS:
        runs = load_runs(results_dir, seeds, benchmark, noise, algo)
        if runs is None:
            print(f"  Skipping {benchmark}/{noise}/{algo}: missing data")
            continue

        n_iter = runs.shape[1]
        iters  = np.arange(1, n_iter + 1)

        mean = runs.mean(axis=0)
        std  = runs.std(axis=0)

        color = ALGO_COLORS[algo]
        label = ALGO_LABELS[algo]

        ax.plot(iters, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(iters,
                        mean - std,
                        mean + std,
                        color=color, alpha=0.20)

    ax.set_xlabel("Iteration $t$", fontsize=10)
    ax.set_ylabel("$f(x_t)$",     fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--plots_dir",   type=str, default="plots")
    parser.add_argument("--n_seeds",     type=int, default=5)
    args = parser.parse_args()

    seeds = list(range(args.n_seeds))
    os.makedirs(args.plots_dir, exist_ok=True)

    # ── 4 separate figures ─────────────────────────────────────────────────
    for benchmark in BENCHMARKS:
        for noise in NOISE_TYPES:
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_panel(ax, args.results_dir, seeds, benchmark, noise)

            title = (f"{BENCH_LABELS[benchmark]} — {NOISE_LABELS[noise]}\n"
                     f"GP-UCB vs GP-UCB-PV  (mean ± std, {args.n_seeds} seeds)")
            fig.suptitle(title, fontsize=10, y=1.02)
            fig.tight_layout()

            fname = f"{benchmark}_{noise}.png"
            fpath = os.path.join(args.plots_dir, fname)
            fig.savefig(fpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {fpath}")

    # ── combined 2×2 figure (bonus) ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i, benchmark in enumerate(BENCHMARKS):
        for j, noise in enumerate(NOISE_TYPES):
            plot_panel(axes[i, j], args.results_dir, seeds, benchmark, noise)

    fig.suptitle(
        f"GP-UCB vs GP-UCB-PV  (mean ± std, {args.n_seeds} seeds)",
        fontsize=13, y=1.01
    )
    fig.tight_layout()
    combined_path = os.path.join(args.plots_dir, "all_results_2x2.png")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined: {combined_path}")


if __name__ == "__main__":
    main()