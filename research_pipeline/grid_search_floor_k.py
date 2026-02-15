#!/usr/bin/env python3
"""
Grid search over the floor multiplier k to find the optimal value.

Reuses the pipeline's data loading and parameter fitting (done once),
then sweeps k evaluating:
  - V1 (mean unclamped minutes for label=0 blocks)
  - V2 (% of label=0 blocks under 5 min)
  - Label=1 mean/median estimated minutes
  - % of label=1 blocks with M > 0 and M > 5 (sensitivity check)
  - Semisynthetic MAE and bias

Prints a table of results for each k.
"""
from __future__ import annotations
import sys
import time
import copy
import numpy as np

# Import pipeline internals
from run_full_pipeline import (
    load_config, load_all_data, run_estimation,
    validate_label0_sanity, run_semisynthetic_validation,
    _estimate_blocks, DEFAULT_CONFIG, log,
)
import polars as pl


def evaluate_k(k: float, bundle, fits, base_cfg) -> dict:
    """Run V1/V2 and semisynthetic for a given floor multiplier k."""
    # Create a modified config with the new k
    cfg = copy.copy(base_cfg)
    object.__setattr__(cfg, 'floor_multiplier', k)
    # Also update the bundle's cfg reference
    bundle_copy = copy.copy(bundle)
    object.__setattr__(bundle_copy, 'cfg', cfg)

    # V1/V2: re-estimate with clamp_absent=False
    rng = np.random.default_rng(cfg.seed)
    unclamped_blocks = []
    label1_blocks = []
    for fit in fits:
        be = _estimate_blocks(fit, cfg, rng, clamp_absent=False)
        label0 = be.filter(pl.col("present") == 0)
        label1 = be.filter(pl.col("present") == 1)
        if label0.height > 0:
            unclamped_blocks.append(label0)
        if label1.height > 0:
            label1_blocks.append(label1)

    if unclamped_blocks:
        all_label0 = pl.concat(unclamped_blocks, how="vertical")
        vals = all_label0["minutes_p50"].to_numpy()
        v1 = float(np.mean(vals))
        v2 = float(np.mean(vals < 5) * 100)
    else:
        v1, v2 = 999.0, 0.0

    if label1_blocks:
        all_label1 = pl.concat(label1_blocks, how="vertical")
        l1_vals = all_label1["estimated_occupied_minutes"].to_numpy()
        l1_mean = float(np.mean(l1_vals))
        l1_median = float(np.median(l1_vals))
        l1_pct_gt0 = float(np.mean(l1_vals > 0) * 100)
        l1_pct_gt5 = float(np.mean(l1_vals > 5) * 100)
    else:
        l1_mean, l1_median, l1_pct_gt0, l1_pct_gt5 = 0.0, 0.0, 0.0, 0.0

    # Semisynthetic (full run)
    semi = run_semisynthetic_validation(bundle_copy, fits, cfg)
    if semi.height > 0 and semi["scenario"][0] != "none":
        semi_mae = float(semi.select(pl.col("abs_error").mean()).item())
        semi_bias = float(semi.select(pl.col("error").mean()).item())
        # Continuous 120 only
        cont120 = semi.filter(pl.col("scenario") == "cont_120")
        if cont120.height > 0:
            c120_mae = float(cont120.select(pl.col("abs_error").mean()).item())
            c120_bias = float(cont120.select(pl.col("error").mean()).item())
        else:
            c120_mae, c120_bias = 0.0, 0.0
    else:
        semi_mae, semi_bias = 0.0, 0.0
        c120_mae, c120_bias = 0.0, 0.0

    return {
        "k": k,
        "V1": round(v1, 2),
        "V2": round(v2, 1),
        "L1_mean": round(l1_mean, 1),
        "L1_median": round(l1_median, 1),
        "L1_pct_gt0": round(l1_pct_gt0, 1),
        "L1_pct_gt5": round(l1_pct_gt5, 1),
        "semi_MAE": round(semi_mae, 1),
        "semi_bias": round(semi_bias, 1),
        "c120_MAE": round(c120_mae, 1),
        "c120_bias": round(c120_bias, 1),
    }


def main():
    global _log_file
    # Suppress log output to keep the grid search output clean
    import run_full_pipeline
    run_full_pipeline._log_file = open("/dev/null", "w")

    cfg = load_config(DEFAULT_CONFIG)
    print("Loading data and fitting parameters (one-time)...")
    t0 = time.time()
    bundle = load_all_data(cfg)
    _, _, _, fits = run_estimation(bundle)
    print(f"  Done in {time.time()-t0:.0f}s\n")

    # Fine grid around the transition region
    k_values = [0.67, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00]

    header = (
        f"{'k':>5} | {'V1':>6} | {'V2':>6} | {'L1_mean':>8} | {'L1_med':>7} | "
        f"{'%M>0':>5} | {'%M>5':>5} | {'s_MAE':>6} | {'s_bias':>7} | "
        f"{'c120_MAE':>9} | {'c120_bias':>10}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for k in k_values:
        t1 = time.time()
        r = evaluate_k(k, bundle, fits, cfg)
        elapsed = time.time() - t1
        marker = " <--" if 14.0 < r["V1"] < 15.0 and r["L1_median"] > 0 else ""
        print(
            f"{r['k']:>5.2f} | {r['V1']:>6.2f} | {r['V2']:>5.1f}% | "
            f"{r['L1_mean']:>8.1f} | {r['L1_median']:>7.1f} | "
            f"{r['L1_pct_gt0']:>4.1f}% | {r['L1_pct_gt5']:>4.1f}% | "
            f"{r['semi_MAE']:>6.1f} | {r['semi_bias']:>+7.1f} | "
            f"{r['c120_MAE']:>9.1f} | {r['c120_bias']:>+10.1f}  "
            f"({elapsed:.0f}s){marker}"
        )
        results.append(r)

    print("\n" + "=" * len(header))

    # Find candidates: pass V1<15, V2>60%, L1_median>0
    candidates = [r for r in results if r["V1"] < 15.0 and r["V2"] > 60.0 and r["L1_median"] > 0]
    passing = [r for r in results if r["V1"] < 15.0 and r["V2"] > 60.0]

    print("\nSUMMARY:")
    if candidates:
        best = min(candidates, key=lambda r: r["k"])
        print(f"  Best k (passes V1<15, V2>60%, L1_median>0): k = {best['k']:.2f}")
        print(f"    V1={best['V1']:.2f}, V2={best['V2']:.1f}%, "
              f"L1_mean={best['L1_mean']:.1f}, L1_median={best['L1_median']:.1f}, "
              f"%M>0={best['L1_pct_gt0']:.1f}%, %M>5={best['L1_pct_gt5']:.1f}%")
        print(f"    semi_MAE={best['semi_MAE']:.1f}, semi_bias={best['semi_bias']:+.1f}")
    elif passing:
        best = min(passing, key=lambda r: r["k"])
        print(f"  Smallest k passing V1<15 (but L1_median=0): k = {best['k']:.2f}")
        print(f"    V1={best['V1']:.2f}, V2={best['V2']:.1f}%, L1_mean={best['L1_mean']:.1f}")
        # Also show the last k before L1_median collapses
        with_median = [r for r in results if r["L1_median"] > 0]
        if with_median:
            last = max(with_median, key=lambda r: r["k"])
            print(f"  Last k with L1_median>0: k = {last['k']:.2f} (V1={last['V1']:.2f})")
    else:
        print("  No k passes V1<15 and V2>60%.")
        best = min(results, key=lambda r: r["V1"])
        print(f"  Lowest V1: k={best['k']:.2f}, V1={best['V1']:.2f}")


if __name__ == "__main__":
    main()
