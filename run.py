"""
run.py
======
Reads a YAML experiment config and runs all combinations of
(benchmark x noise_type x algorithm x seed) sequentially.

Usage:
    python run.py --config configs/warm_candidate_current.yaml
"""

import argparse
import os
import time
import yaml
import numpy as np

from src.trial import run_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML experiment config")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Override results base directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_name    = cfg["experiment"]["name"]
    seeds       = cfg.get("seeds", [0, 1, 2, 3, 4])
    benchmarks  = cfg.get("benchmarks", ["sine", "branin"])
    noise_types = cfg.get("noise_types", ["homo", "hetero"])
    algorithms  = cfg.get("algorithms", ["gpucb", "gpucb_pv"])

    base_dir = args.results_dir or os.path.join("results", exp_name)

    total = len(seeds) * len(benchmarks) * len(noise_types) * len(algorithms)
    done = 0

    for bench in benchmarks:
        for noise in noise_types:
            for algo in algorithms:
                for seed in seeds:
                    tag = f"{bench}_{noise}_{algo}"
                    out_dir = os.path.join(base_dir, f"seed_{seed}")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"{tag}.npy")

                    if os.path.exists(out_path):
                        done += 1
                        print(f"[seed={seed}] SKIP {tag} (exists)  "
                              f"[{done}/{total}]")
                        continue

                    print(f"[seed={seed}] START {tag}  [{done+1}/{total}]")
                    t0 = time.time()

                    trial_cfg = {
                        "benchmark":       bench,
                        "noise_type":      noise,
                        "algorithm":       algo,
                        "seed":            seed,
                        "start_type":      cfg["start"]["type"],
                        "n_init":          cfg["start"].get("n_init", 5),
                        "n_iter":          cfg["experiment"]["n_iter"],
                        "noise_source":    cfg["noise"]["source"],
                        "opt_method":      cfg["optimization"]["method"],
                        "n_restarts":      cfg["optimization"].get("n_restarts", 10),
                        "lbfgs_steps":     cfg["optimization"].get("lbfgs_steps", 100),
                        "n_candidates":    cfg["optimization"].get("n_candidates", 2000),
                        "gp_train_iter":   cfg["surrogate"]["gp_train_iter"],
                        "gp_lr":           cfg["surrogate"].get("learning_rate", 0.1),
                        "prior_mean":      cfg["surrogate"].get("prior_mean", 0.0),
                        "prior_var":       cfg["surrogate"].get("prior_var", 1.0),
                        "beta_type":       cfg["beta"].get("type", "srinivas"),
                        "beta_value":      cfg["beta"].get("value", 2.0),
                        "beta_delta":      cfg["beta"].get("delta", 0.1),
                    }

                    f_vals = run_trial(trial_cfg)
                    np.save(out_path, f_vals)

                    done += 1
                    elapsed = time.time() - t0
                    print(f"[seed={seed}] DONE  {tag}  ({elapsed:.1f}s)  "
                          f"best_f={f_vals.max():.4f}  [{done}/{total}]")

    print(f"\nExperiment '{exp_name}' complete: {done}/{total} trials.")


if __name__ == "__main__":
    main()