"""
diagnose_variance.py
====================
Diagnostic script to verify why GP-UCB and GP-UCB-PV diverge on Sine
but overlap on Branin under homoscedastic noise (rho_0 = 0.01).

Hypothesis:  For Sine, sigma^2_n(x) is comparable to rho_0, so adding
             rho_0 inside the sqrt changes the acquisition argmax.
             For Branin, sigma^2_n(x) >> rho_0, so the addition is
             negligible and both algorithms select the same point.

This script runs GP-UCB (warm start, candidate search, current homo noise)
for both benchmarks across 5 seeds.  At each BO iteration it records
statistics of sigma^2_n(x) over the candidate grid:
    min, mean, max, and the ratio rho_0 / max(sigma^2_n)

If the ratio is near 0 for Branin and non-negligible for Sine, the
hypothesis is confirmed.

Usage:
    cd /scratch2/ih70/dilmia/PROJECT_2/gpucb
    python diagnose_variance.py
"""

import torch
import numpy as np
import gpytorch

from Data.benchmarks import get_benchmark_fn, get_domain
from Data.noise import get_noise_fn
from src.surrogate import fit_gp
from optimization.candidate import candidate_grid

# ── Configuration (matches warm_candidate_current.yaml, homo noise) ────────
RHO_0       = 0.01
N_INIT      = 5
N_ITER      = 50
N_CAND      = 2000
GP_ITER     = 50
SEEDS       = [0, 1, 2, 3, 4]
BENCHMARKS  = ["sine", "branin"]


def run_diagnostic(benchmark: str, seed: int):
    """
    Run GP-UCB on one benchmark/seed and record sigma^2_n statistics
    at every BO iteration.

    Returns
    -------
    records : list of dicts with keys
        n, sig2_min, sig2_mean, sig2_max, ratio_rho_over_sig2max
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    f_true   = get_benchmark_fn(benchmark)
    noise_fn = get_noise_fn(benchmark, "homo", "current")
    dim, lo, hi = get_domain(benchmark)

    cands = candidate_grid(dim, lo, hi, N_CAND)

    # Initial design D_{n0}
    n0 = N_INIT
    init_x = torch.from_numpy(
        rng.uniform(lo, hi, size=(n0, dim)).astype(np.float32)
    )
    f_init   = f_true(init_x)
    rho_init = noise_fn(init_x)
    eps_init = torch.from_numpy(
        rng.standard_normal(n0).astype(np.float32)
    ) * torch.sqrt(rho_init)
    train_x = init_x
    train_y = f_init + eps_init

    records = []
    N = n0 + N_ITER

    for n in range(n0, N):
        # Fit GP on D_n
        if n <= 1:
            nv = 0.01
        else:
            nv = float(train_y.var().clamp(min=1e-4)) * 0.1
        model, likelihood = fit_gp(train_x, train_y,
                                   n_iter=GP_ITER, noise_var=nv)

        # Evaluate sigma^2_n over the full candidate grid
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(cands))
            sig2 = pred.variance.numpy()

        sig2_min  = float(sig2.min())
        sig2_mean = float(sig2.mean())
        sig2_max  = float(sig2.max())
        ratio     = RHO_0 / sig2_max if sig2_max > 0 else float('inf')

        records.append({
            "n":         n,
            "bo_iter":   n - n0 + 1,
            "sig2_min":  sig2_min,
            "sig2_mean": sig2_mean,
            "sig2_max":  sig2_max,
            "ratio":     ratio,
        })

        # Select next point (GP-UCB) and observe
        import math
        from optimization.acquisition import beta_srinivas, acq_value
        from src.surrogate import make_gp_predict_fn

        beta = beta_srinivas(n - n0 + 1, dim=dim)
        predict_fn = make_gp_predict_fn(model, likelihood)

        with torch.no_grad():
            neg_acq = torch.stack([
                acq_value(cands[i:i+1], predict_fn, noise_fn, "gpucb", beta)
                for i in range(cands.shape[0])
            ]).squeeze()
        best_idx = neg_acq.argmin()
        x_next = cands[best_idx].unsqueeze(0)

        f_next   = f_true(x_next)
        rho_next = noise_fn(x_next)
        eps_next = torch.from_numpy(
            rng.standard_normal(1).astype(np.float32)
        ) * torch.sqrt(rho_next)
        y_next = f_next + eps_next

        train_x = torch.cat([train_x, x_next], dim=0)
        train_y = torch.cat([train_y, y_next.reshape(1)], dim=0)

    return records


def print_table(benchmark: str, seed: int, records: list):
    """Print a summary table for one benchmark/seed."""
    print(f"\n{'='*80}")
    print(f"  {benchmark.upper()}  |  seed = {seed}  |  rho_0 = {RHO_0}")
    print(f"{'='*80}")
    print(f"  {'BO iter':>7s}  {'n':>4s}  {'sig2_min':>10s}  "
          f"{'sig2_mean':>10s}  {'sig2_max':>10s}  {'rho0/sig2_max':>13s}")
    print(f"  {'-'*7}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*13}")

    # Print every 5th iteration + first and last
    for r in records:
        i = r["bo_iter"]
        if i == 1 or i % 5 == 0 or i == N_ITER:
            print(f"  {i:7d}  {r['n']:4d}  {r['sig2_min']:10.6f}  "
                  f"{r['sig2_mean']:10.6f}  {r['sig2_max']:10.6f}  "
                  f"{r['ratio']:13.6f}")


def print_summary(benchmark: str, all_records: dict):
    """Print cross-seed summary of sig2_max and the ratio."""
    print(f"\n{'='*80}")
    print(f"  {benchmark.upper()}  |  CROSS-SEED SUMMARY  (sig2_max at selected iterations)")
    print(f"{'='*80}")

    header = f"  {'BO iter':>7s}"
    for seed in SEEDS:
        header += f"  {'seed'+str(seed):>10s}"
    header += f"  {'mean':>10s}"
    print(header)
    print(f"  {'-'*7}" + f"  {'-'*10}" * (len(SEEDS) + 1))

    for i_target in [1, 5, 10, 25, 50]:
        row = f"  {i_target:7d}"
        vals = []
        for seed in SEEDS:
            rec = all_records[seed]
            r = [x for x in rec if x["bo_iter"] == i_target]
            if r:
                v = r[0]["sig2_max"]
                vals.append(v)
                row += f"  {v:10.4f}"
            else:
                row += f"  {'N/A':>10s}"
        if vals:
            row += f"  {np.mean(vals):10.4f}"
        print(row)

    # Print ratio summary
    print(f"\n  rho_0 / sig2_max  (how negligible is the noise variance?):")
    header2 = f"  {'BO iter':>7s}"
    for seed in SEEDS:
        header2 += f"  {'seed'+str(seed):>10s}"
    header2 += f"  {'mean':>10s}"
    print(header2)
    print(f"  {'-'*7}" + f"  {'-'*10}" * (len(SEEDS) + 1))

    for i_target in [1, 5, 10, 25, 50]:
        row = f"  {i_target:7d}"
        vals = []
        for seed in SEEDS:
            rec = all_records[seed]
            r = [x for x in rec if x["bo_iter"] == i_target]
            if r:
                v = r[0]["ratio"]
                vals.append(v)
                row += f"  {v:10.4f}"
            else:
                row += f"  {'N/A':>10s}"
        if vals:
            row += f"  {np.mean(vals):10.4f}"
        print(row)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Diagnose posterior variance: why GP-UCB and GP-UCB-PV "
                    "diverge on Sine but overlap on Branin.")
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=["sine", "branin"],
                        help="Run one benchmark only (default: both)")
    args = parser.parse_args()

    benchmarks = [args.benchmark] if args.benchmark else BENCHMARKS

    print("Diagnosing posterior variance under homo noise")
    print(f"rho_0 = {RHO_0},  n_init = {N_INIT},  n_iter = {N_ITER},  "
          f"seeds = {SEEDS}")
    print(f"benchmarks = {benchmarks}")
    print()
    print("If rho_0/sig2_max is SMALL (< 0.01), rho_0 is negligible")
    print("  => GP-UCB and GP-UCB-PV select the same points (overlap).")
    print("If rho_0/sig2_max is LARGE (> 0.1), rho_0 meaningfully changes")
    print("  the exploration term => algorithms diverge.")

    for benchmark in benchmarks:
        all_records = {}

        for seed in SEEDS:
            print(f"\nRunning {benchmark} seed={seed} ...", flush=True)
            records = run_diagnostic(benchmark, seed)
            all_records[seed] = records
            print_table(benchmark, seed, records)

        print_summary(benchmark, all_records)

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("If Sine shows rho_0/sig2_max >> 0.01 and Branin shows")
    print("rho_0/sig2_max << 0.01, this confirms that the noise")
    print("variance is comparable to the model uncertainty for Sine")
    print("but negligible for Branin, explaining the observed")
    print("divergence vs overlap in the plots.")


if __name__ == "__main__":
    main()