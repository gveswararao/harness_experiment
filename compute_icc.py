"""
Reads results.csv (task_id, harness, model, trial, passed) and computes
ICC(1,1) per (harness, model) cell, using the between/within variance
decomposition from Mustahsan et al. (2025), with a bootstrap 95% CI.

Usage:
    pip install pandas numpy
    python compute_icc.py
"""
import numpy as np
import pandas as pd

N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)


def icc_1_1(success_matrix):
    """
    success_matrix: 2D array, rows = tasks, cols = trials (0/1 pass values).
    Returns ICC(1,1) = between-task variance / (between + within).
    """
    task_means = success_matrix.mean(axis=1)
    grand_mean = success_matrix.mean()
    between = np.var(task_means, ddof=1) if len(task_means) > 1 else 0.0
    within = np.mean(np.var(success_matrix, axis=1, ddof=1)) if success_matrix.shape[1] > 1 else 0.0
    denom = between + within
    return between / denom if denom > 0 else float("nan")


def bootstrap_ci(success_matrix, n_boot=N_BOOTSTRAP):
    n_tasks = success_matrix.shape[0]
    estimates = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n_tasks, n_tasks)  # resample tasks with replacement
        estimates.append(icc_1_1(success_matrix[idx]))
    estimates = np.array(estimates)
    estimates = estimates[~np.isnan(estimates)]
    if len(estimates) == 0:
        return (float("nan"), float("nan"))
    return (np.percentile(estimates, 2.5), np.percentile(estimates, 97.5))


def main():
    df = pd.read_csv("results.csv")
    print(f"{'Model':<14}{'Harness':<8}{'n_tasks':<9}{'n_trials':<9}{'ICC':<8}{'95% CI'}")
    for (model, harness), group in df.groupby(["model", "harness"]):
        pivot = group.pivot_table(index="task_id", columns="trial", values="passed")
        matrix = pivot.to_numpy()
        icc = icc_1_1(matrix)
        ci_lo, ci_hi = bootstrap_ci(matrix)
        print(f"{model:<14}{harness:<8}{matrix.shape[0]:<9}{matrix.shape[1]:<9}"
              f"{icc:<8.3f}[{ci_lo:.3f}, {ci_hi:.3f}]")


if __name__ == "__main__":
    main()
