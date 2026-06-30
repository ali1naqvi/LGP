#!/usr/bin/env python3
"""
Prove correlation between fitness and register efficiency (effective_registers / total_registers).

Two modes:

1. From selection CSV (best agent per generation):
   - proportion = best_agent_effective_register_size / best_agent_register_size
   - Correlation(validation_fitness, proportion) across generations.
   - Run per seed, then report mean correlation and p-value across seeds.

2. If your CSV has the new column fitness_register_efficiency_correlation:
   - That is the within-generation Pearson r (fitness vs proportion across all root teams).
   - Plot it over generations: consistently positive => proof.

Usage (existing CSV, best-agent time series):
  python fitness_register_efficiency_correlation.py --dir experiments/5_ant_baseline_27/logs/selection
  python fitness_register_efficiency_correlation.py --dir experiments/5_ant_dynamic_1_fitness_tot-reg/logs/selection

Usage (multiple experiments for comparison):
  python fitness_register_efficiency_correlation.py --dirs experiments/5_ant_baseline_8/logs/selection experiments/5_ant_dynamic_1_fitness_tot-reg/logs/selection
"""

import argparse
import glob
import numpy as np
import os


def load_selection_csv(csv_path):
    """Load selection.*.0.csv; return (generation, validation_fitness, total_reg, eff_reg)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "validation_fitness" not in df.columns or "program_instruction_count" not in df.columns:
        return None
    df = df.dropna(subset=["validation_fitness", "program_instruction_count", "effective_program_instruction_count"])
    df = df[df["program_instruction_count"] > 0]
    df["proportion"] = df["effective_program_instruction_count"] / df["program_instruction_count"]
    return df

# program_instruction_count,effective_program_instruction_count

def correlation_one_seed(df):
    """Pearson and Spearman between validation_fitness and proportion."""
    if df is None or len(df) < 10:
        return None, None, None, None
    from scipy import stats
    r_pearson, p_pearson = stats.pearsonr(df["validation_fitness"], df["proportion"])
    r_spearman, p_spearman = stats.spearmanr(df["validation_fitness"], df["proportion"])
    return r_pearson, p_pearson, r_spearman, p_spearman


def main():
    ap = argparse.ArgumentParser(description="Fitness vs register efficiency correlation")
    ap.add_argument("--dir", type=str, help="Single selection log dir (e.g. experiments/5_ant_baseline_27/logs/selection)")
    ap.add_argument("--dirs", nargs="+", type=str, help="Multiple selection dirs for comparison")
    ap.add_argument("--csv", type=str, help="Single CSV path (alternative to --dir)")
    args = ap.parse_args()

    if args.csv:
        paths = [args.csv]
    elif args.dirs:
        paths = []
        for d in args.dirs:
            pattern = os.path.join(d, "selection.*.0.csv")
            paths.extend(sorted(glob.glob(pattern)))
    elif args.dir:
        pattern = os.path.join(args.dir, "selection.*.0.csv")
        paths = sorted(glob.glob(pattern))
    else:
        print("Use --dir, --dirs, or --csv")
        return

    if not paths:
        print("No selection CSV files found.")
        return

    print("Fitness vs register efficiency (proportion = effective_registers / total_registers)\n")
    print("Using best-agent time series: correlation across generations per seed.\n")

    results = []
    for path in paths:
        df = load_selection_csv(path)
        if df is None:
            continue
        # path like .../experiments/5_ant_baseline_27/logs/selection/selection.2.0.csv
        exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
        seed_name = os.path.basename(exp_dir)
        if "selection." in os.path.basename(path):
            seed_id = os.path.basename(path).replace("selection.", "").replace(".0.csv", "")
            label = f"{seed_name} seed {seed_id}"
        else:
            label = path
        r_p, p_p, r_s, p_s = correlation_one_seed(df)
        if r_p is None:
            print(f"  {label}: not enough data")
            continue
        results.append((label, r_p, p_p, r_s, p_s))
        print(f"  {label}")
        print(f"    Pearson  r = {r_p:.4f}, p = {p_p:.4e}")
        print(f"    Spearman r = {r_s:.4f}, p = {p_s:.4e}")

    if not results:
        print("No valid results.")
        return

    # Summary across seeds
    r_pearson_vals = [x[1] for x in results]
    p_pearson_vals = [x[2] for x in results]
    print("\n--- Summary ---")
    print(f"  Mean Pearson r  = {np.mean(r_pearson_vals):.4f} (std = {np.std(r_pearson_vals):.4f})")
    print(f"  Median p-value  = {np.median(p_pearson_vals):.4e}")
    print(f"  Fraction p<0.05 = {np.mean(np.array(p_pearson_vals) < 0.05):.2f}")
    print("\nInterpretation: Positive r means when the best agent has higher register efficiency")
    print("(more effective regs / total regs), validation fitness tends to be higher.")
    print("If you have the new column 'fitness_register_efficiency_correlation' in your CSV,")
    print("that is the within-generation r (across all root teams) and is a stronger proof.")


if __name__ == "__main__":
    main()
