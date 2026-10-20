"""
run.py
======
Reads a YAML config and runs all (benchmark x noise_type x algorithm x seed)
combinations sequentially.

Usage:
    python run.py --config configs/xxx.yaml                 # all seeds in cfg
    python run.py --config configs/xxx.yaml --seed 2        # only seed 2
"""

import argparse
import os
import time
import yaml
import numpy as np

from src.trial import run_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="Run only this seed (overrides cfg.seeds)")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_name    = cfg["experiment"]["name"]
    benchmarks  = cfg.get("benchmarks", ["sine", "branin"])
    noise_types = cfg.get("noise_types", ["homo", "hetero"])
    algorithms  = cfg.get("algorithms", ["gpucb", "gpucb_pv"])

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = cfg.get("seeds", [0, 1, 2, 3, 4])

    base_dir = args.results_dir or os.path.join("results", exp_name)
    total = len(benchmarks) * len(noise_types) * len(algorithms) * len(seeds)
    done  = 0

    for bench in benchmarks:
        for noise in noise_types:
            for algo in algorithms:
                for seed in seeds:
                    tag     = f"{bench}_{noise}_{algo}"
                    out_dir = os.path.join(base_dir, f"seed_{seed}")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"{tag}.npy")

                    if os.path.exists(out_path):
                        done += 1
                        print(f"[seed={seed}] SKIP {tag}  [{done}/{total}]")
                        continue

                    print(f"[seed={seed}] START {tag}  [{done+1}/{total}]",
                          flush=True)
                    t0 = time.time()

                    trial_cfg = {
                        "benchmark":    bench,
                        "noise_type":   noise,
                        "algorithm":    algo,
                        "seed":         seed,
                        "start_type":   cfg["start"]["type"],
                        "n_init":       cfg["start"].get("n_init", 5),
                        "n_iter":       cfg["experiment"]["n_iter"],
                        "noise_source": cfg["noise"]["source"],
                        "opt_method":   cfg["optimization"]["method"],
                        "n_candidates": cfg["optimization"].get("n_candidates", 2000),
                        "output_scale": cfg["surrogate"].get("output_scale", 1.0),
                        "lengthscale":  cfg["surrogate"].get("lengthscale",  1.0),
                        "prior_mean":   cfg["surrogate"].get("prior_mean",   0.0),
                        "prior_var":    cfg["surrogate"].get("prior_var",    1.0),
                        "beta_type":    cfg["beta"].get("type",  "srinivas"),
                        "beta_value":   cfg["beta"].get("value", 2.0),
                        "beta_delta":   cfg["beta"].get("delta", 0.1),
                    }

                    f_vals = run_trial(trial_cfg)
                    np.save(out_path, f_vals)

                    done += 1
                    elapsed = time.time() - t0
                    print(f"[seed={seed}] DONE  {tag}  ({elapsed:.1f}s)  "
                          f"best_f={f_vals.max():.4f}  [{done}/{total}]",
                          flush=True)

    print(f"\nExperiment '{exp_name}' complete: {done}/{total} trials.")


if __name__ == "__main__":
    main()