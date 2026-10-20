"""
plot_results.py
===============
Aggregates results for a given experiment config and produces plots.
"""

import argparse
import os
from typing import Optional

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALGO_LABELS = {
    "gpucb":            "GP-UCB",
    "gpucb_pv":         "GP-UCB-PV",
    "gpucb_pv_tabpfn":  "GP-UCB-PV (TabPFN-PPD)",
    "gpucb_tabpfn_clt": "GP-UCB (TabPFN-CLT)",
}
ALGO_COLORS = {
    "gpucb":            "#1f77b4",   # blue
    "gpucb_pv":         "#d62728",   # red
    "gpucb_pv_tabpfn":  "#2ca02c",   # green
    "gpucb_tabpfn_clt": "#9467bd",   # purple
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
              benchmark: str, noise: str,
              algorithm: str) -> Optional[np.ndarray]:
    arrays = []
    for seed in seeds:
        path = os.path.join(results_dir, f"seed_{seed}",
                            f"{benchmark}_{noise}_{algorithm}.npy")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            return None
        arrays.append(np.load(path))
    return np.stack(arrays, axis=0)


def plot_panel(ax, results_dir, seeds, benchmark, noise, algorithms):
    ax.set_title(f"{BENCH_LABELS[benchmark]}\n{NOISE_LABELS[noise]}",
                 fontsize=11, pad=6)
    for algo in algorithms:
        runs = load_runs(results_dir, seeds, benchmark, noise, algo)
        if runs is None:
            continue
        n_iter = runs.shape[1]
        iters  = np.arange(1, n_iter + 1)
        mean   = runs.mean(axis=0)
        std    = runs.std(axis=0)
        color  = ALGO_COLORS.get(algo, "#333333")
        label  = ALGO_LABELS.get(algo, algo)
        ax.plot(iters, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(iters, mean - std, mean + std,
                        color=color, alpha=0.20)
    ax.set_xlabel("Iteration $i$", fontsize=10)
    ax.set_ylabel("$f(x_{n+1})$", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--plots_dir",   type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_name    = cfg["experiment"]["name"]
    seeds       = cfg.get("seeds", list(range(5)))
    benchmarks  = cfg.get("benchmarks", ["sine", "branin"])
    noise_types = cfg.get("noise_types", ["homo", "hetero"])
    algorithms  = cfg.get("algorithms", ["gpucb", "gpucb_pv"])

    results_dir = args.results_dir or os.path.join("results", exp_name)
    if args.plots_dir:
        plots_dir = os.path.join(args.plots_dir, exp_name)
    else:
        plots_dir = os.path.join("plots", exp_name)
    os.makedirs(plots_dir, exist_ok=True)

    n_seeds = len(seeds)

    for bench in benchmarks:
        for noise in noise_types:
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_panel(ax, results_dir, seeds, bench, noise, algorithms)
            algo_names = " vs ".join(ALGO_LABELS.get(a, a) for a in algorithms)
            title = (f"{BENCH_LABELS[bench]} — {NOISE_LABELS[noise]}\n"
                     f"{algo_names}  (mean ± std, {n_seeds} seeds)")
            fig.suptitle(title, fontsize=10, y=1.02)
            fig.tight_layout()
            fname = f"{bench}_{noise}.png"
            fpath = os.path.join(plots_dir, fname)
            fig.savefig(fpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {fpath}")

    if len(benchmarks) >= 2 and len(noise_types) >= 2:
        fig, axes = plt.subplots(len(benchmarks), len(noise_types),
                                 figsize=(12, 4 * len(benchmarks)))
        for i, bench in enumerate(benchmarks):
            for j, noise in enumerate(noise_types):
                plot_panel(axes[i][j], results_dir, seeds,
                           bench, noise, algorithms)
        fig.suptitle(f"{exp_name}  (mean ± std, {n_seeds} seeds)",
                     fontsize=13, y=1.01)
        fig.tight_layout()
        combined_path = os.path.join(plots_dir, "all_results_2x2.png")
        fig.savefig(combined_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined: {combined_path}")


if __name__ == "__main__":
    main()