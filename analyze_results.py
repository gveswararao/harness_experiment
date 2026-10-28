"""
Full analysis pipeline matching the paper's Section 4.4/4.7 analysis plan.
Reads results.csv and reports:
  1. ICC(1,1) per (harness, model) cell with bootstrap 95% CI  [primary]
  2. Harness-Induced Variance (HV) and Model-Induced Variance (MV) per task [secondary]
  3. Ranking reversals: does the best-performing model change across harnesses? [secondary]

Usage:
    python analyze_results.py
"""
import numpy as np
import pandas as pd

N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)


def icc_1_1(success_matrix):
    task_means = success_matrix.mean(axis=1)
    between = np.var(task_means, ddof=1) if len(task_means) > 1 else 0.0
    within = np.mean(np.var(success_matrix, axis=1, ddof=1)) if success_matrix.shape[1] > 1 else 0.0
    denom = between + within
    return between / denom if denom > 0 else float("nan")


def bootstrap_ci(success_matrix, n_boot=N_BOOTSTRAP):
    n_tasks = success_matrix.shape[0]
    estimates = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n_tasks, n_tasks)
        estimates.append(icc_1_1(success_matrix[idx]))
    estimates = np.array(estimates)
    estimates = estimates[~np.isnan(estimates)]
    if len(estimates) == 0:
        return (float("nan"), float("nan"))
    return (np.percentile(estimates, 2.5), np.percentile(estimates, 97.5))


def reliability_band(icc):
    if np.isnan(icc):
        return "n/a"
    if icc > 0.8:
        return "high"
    if icc >= 0.5:
        return "moderate"
    return "low"


def section_icc(df):
    print("=" * 70)
    print("1. ICC(1,1) per (model, harness) cell  [primary analysis]")
    print("=" * 70)
    print(f"{'Model':<18}{'Harness':<8}{'n_tasks':<9}{'ICC':<8}{'95% CI':<20}{'Band'}")
    rows = []
    for (model, harness), group in df.groupby(["model", "harness"]):
        pivot = group.pivot_table(index="task_id", columns="trial", values="passed")
        matrix = pivot.to_numpy()
        icc = icc_1_1(matrix)
        ci_lo, ci_hi = bootstrap_ci(matrix)
        band = reliability_band(icc)
        # flag inconclusive per the paper's rule: CI spans low/moderate boundary (0.5) or moderate/high (0.8)
        inconclusive = not np.isnan(ci_lo) and (ci_lo < 0.5 < ci_hi or ci_lo < 0.8 < ci_hi)
        flag = " (INCONCLUSIVE - CI spans a band boundary)" if inconclusive else ""
        print(f"{model:<18}{harness:<8}{matrix.shape[0]:<9}{icc:<8.3f}"
              f"[{ci_lo:.3f}, {ci_hi:.3f}]{'':<3}{band}{flag}")
        rows.append({"model": model, "harness": harness, "icc": icc, "ci_lo": ci_lo, "ci_hi": ci_hi})
    print()
    return pd.DataFrame(rows)


def section_hv_mv(df):
    print("=" * 70)
    print("2. Harness-Induced (HV) vs Model-Induced (MV) variance per task")
    print("=" * 70)
    task_rates = df.groupby(["task_id", "harness", "model"])["passed"].mean().reset_index()
    print(f"{'Task':<8}{'HV (harness-driven)':<24}{'MV (model-driven)':<20}")
    for task_id, group in task_rates.groupby("task_id"):
        # HV: variance across harnesses, averaged over models
        hv = group.groupby("harness")["passed"].mean().var(ddof=1) if group["harness"].nunique() > 1 else float("nan")
        # MV: variance across models, averaged over harnesses
        mv = group.groupby("model")["passed"].mean().var(ddof=1) if group["model"].nunique() > 1 else float("nan")
        print(f"{task_id:<8}{hv:<24.4f}{mv:<20.4f}")
    print()


def section_ranking_reversals(df):
    print("=" * 70)
    print("3. Ranking reversals: does the best model change across harnesses?")
    print("=" * 70)
    pivot = df.groupby(["harness", "model"])["passed"].mean().reset_index()
    best_per_harness = pivot.loc[pivot.groupby("harness")["passed"].idxmax()]
    print(best_per_harness[["harness", "model", "passed"]].to_string(index=False))
    n_unique_winners = best_per_harness["model"].nunique()
    if n_unique_winners > 1:
        print(f"\n-> REVERSAL DETECTED: the best-performing model changes depending on harness rigor.")
    else:
        print(f"\n-> No reversal: {best_per_harness['model'].iloc[0]} is best under every harness tested.")
    print()


def main():
    df = pd.read_csv("results.csv")
    print(f"Loaded {len(df)} trial rows: "
          f"{df['task_id'].nunique()} tasks x {df['harness'].nunique()} harnesses x "
          f"{df['model'].nunique()} models x up to {df.groupby(['task_id','harness','model']).size().max()} trials\n")
    section_icc(df)
    section_hv_mv(df)
    section_ranking_reversals(df)


if __name__ == "__main__":
    main()
