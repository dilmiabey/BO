"""
run_seed.py
===========
Runs all 8 configurations for a single seed and saves results to disk.

Usage:
    python run_seed.py --seed 0 --results_dir results --n_iter 50

The 8 configurations are:
    benchmark  × noise_type × algorithm
    sine       × homo       × gpucb
    sine       × homo       × gpucb_pv
    sine       × hetero     × gpucb
    sine       × hetero     × gpucb_pv
    branin     × homo       × gpucb
    branin     × homo       × gpucb_pv
    branin     × hetero     × gpucb
    branin     × hetero     × gpucb_pv

Results are stored as:
    results_dir/seed_{seed}/{benchmark}_{noise}_{algorithm}.npy
"""

import argparse
import os
import time

import numpy as np

from experiment import TrialConfig, run_trial

BENCHMARKS = ["sine", "branin"]
NOISE_TYPES = ["homo", "hetero"]
ALGORITHMS  = ["gpucb", "gpucb_pv"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--n_init",      type=int, default=5)
    parser.add_argument("--n_iter",      type=int, default=50)
    parser.add_argument("--n_cand",      type=int, default=2000)
    parser.add_argument("--gp_train_iter", type=int, default=50)
    args = parser.parse_args()

    out_dir = os.path.join(args.results_dir, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    configs = [
        (b, n, a)
        for b in BENCHMARKS
        for n in NOISE_TYPES
        for a in ALGORITHMS
    ]

    for bench, noise, algo in configs:
        tag = f"{bench}_{noise}_{algo}"
        out_path = os.path.join(out_dir, f"{tag}.npy")

        if os.path.exists(out_path):
            print(f"[seed={args.seed}] SKIP {tag} (already exists)")
            continue

        print(f"[seed={args.seed}] START {tag}")
        t0 = time.time()

        cfg = TrialConfig(
            benchmark=bench,
            noise_type=noise,
            algorithm=algo,
            seed=args.seed,
            n_init=args.n_init,
            n_iter=args.n_iter,
            n_cand=args.n_cand,
            gp_train_iter=args.gp_train_iter,
        )

        f_vals = run_trial(cfg)
        np.save(out_path, f_vals)

        elapsed = time.time() - t0
        print(f"[seed={args.seed}] DONE  {tag}  ({elapsed:.1f}s)  "
              f"best_f={f_vals.max():.4f}")

    print(f"\n[seed={args.seed}] All configurations complete.")


if __name__ == "__main__":
    main()