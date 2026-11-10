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
    "ei":               "EI",
    "ei_pv":            "EI-PV",
    "ei_pv_tabpfn":     "EI-PV (TabPFN-PPD)",
    "ei_tabpfn_clt":    "EI (TabPFN-CLT)",
}
ALGO_COLORS = {
    "gpucb":            "#1f77b4",   # blue
    "gpucb_pv":         "#d62728",   # red
    "gpucb_pv_tabpfn":  "#2ca02c",   # green
    "gpucb_tabpfn_clt": "#9467bd",   # purple
    "ei":               "#1f77b4",   # blue   (same model/variance as gpucb)
    "ei_pv":            "#d62728",   # red
    "ei_pv_tabpfn":     "#2ca02c",   # green
    "ei_tabpfn_clt":    "#9467bd",   # purple
}
BENCH_LABELS = {
    "sine":   "Sine Benchmark",
    "branin": "Branin Benchmark",
}
NOISE_LABELS = {
    "homo":   "Homoscedastic Noise",
    "hetero": "Heteroscedastic Noise",
    "zero":   "Zero Noise",
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


def auto_zoom_ylim(lows, highs, pct=10.0, pad_frac=0.05):
    """Robust y-limits from the mean +/- std bands of all curves.

    Uses the `pct` percentile of the lower band edges and the (100 - pct)
    percentile of the upper band edges, so isolated spikes (e.g. one bad
    seed at one iteration) do not stretch the axis. Padded by `pad_frac`.
    """
    lo = np.percentile(np.concatenate(lows), pct)
    hi = np.percentile(np.concatenate(highs), 100.0 - pct)
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0) * 0.05
    return lo - pad_frac * span, hi + pad_frac * span


def resolve_zoom(cfg_zoom, bench, cli_xlim, cli_ylim):
    """CLI limits override the per-benchmark `zoom:` section of the config."""
    bench_cfg = (cfg_zoom or {}).get(bench, {}) or {}
    xlim = cli_xlim or bench_cfg.get("xlim")
    ylim = cli_ylim or bench_cfg.get("ylim")
    return xlim, ylim


def plot_panel(ax, results_dir, seeds, benchmark, noise, algorithms,
               xlim=None, ylim=None, zoom=False, zoom_pct=10.0):
    """Plot one panel. Returns the list of mean curves that were drawn.

    zoom=True: apply xlim/ylim; if ylim is None, choose it automatically
    from the mean curves.
    """
    ax.set_title(f"{BENCH_LABELS[benchmark]}\n{NOISE_LABELS[noise]}",
                 fontsize=11, pad=6)
    means, lows, highs = [], [], []
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
        means.append(mean)
        lows.append(mean - std)
        highs.append(mean + std)
    if zoom and means:
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is None:
            ylim = auto_zoom_ylim(lows, highs, pct=zoom_pct)
        ax.set_ylim(*ylim)
    ax.set_xlabel("Iteration $i$", fontsize=10)
    ax.set_ylabel("$f(x_{n+1})$", fontsize=10)
    ax.legend(fontsize=9, loc="best" if zoom else "lower right",
              framealpha=0.8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=9)
    return means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--plots_dir",   type=str, default=None)
    parser.add_argument("--no_zoom", action="store_true",
                        help="Do not produce the extra *_zoom.png plots.")
    parser.add_argument("--zoom_xlim", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="x-limits of the zoomed plots (all benchmarks).")
    parser.add_argument("--zoom_ylim", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="y-limits of the zoomed plots (all benchmarks). "
                             "Default: config `zoom:` section, else automatic.")
    parser.add_argument("--zoom_pct", type=float, default=10.0,
                        help="Automatic zoom clips the lowest/highest PCT %% of "
                             "the mean+/-std band values (default 10). "
                             "Smaller = wider view.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_name    = cfg["experiment"]["name"]
    seeds       = cfg.get("seeds", list(range(5)))
    benchmarks  = cfg.get("benchmarks", ["sine", "branin"])
    noise_types = cfg.get("noise_types", ["homo", "hetero"])
    algorithms  = cfg.get("algorithms", ["gpucb", "gpucb_pv"])
    cfg_zoom    = cfg.get("zoom", {})   # optional: {bench: {xlim:[..], ylim:[..]}}

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

            if args.no_zoom:
                continue
            xlim, ylim = resolve_zoom(cfg_zoom, bench,
                                      args.zoom_xlim, args.zoom_ylim)
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_panel(ax, results_dir, seeds, bench, noise, algorithms,
                       xlim=xlim, ylim=ylim, zoom=True,
                       zoom_pct=args.zoom_pct)
            fig.suptitle(title + "  [zoomed]", fontsize=10, y=1.02)
            fig.tight_layout()
            zpath = os.path.join(plots_dir, f"{bench}_{noise}_zoom.png")
            fig.savefig(zpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved zoomed: {zpath}  (ylim={ax.get_ylim()})")

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