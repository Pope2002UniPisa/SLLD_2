from __future__ import annotations

"""PHASE 3.3 — MULTICOLLINEARITY DIAGNOSTICS ON 2210 FEATURES."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IN = Path("./outputs/III/phase3_1_supervised_data_loading_checks")
OUT = Path("./outputs/III/phase3_3_multicollinearity_analysis")
OUT.mkdir(parents=True, exist_ok=True)


def write_json(obj, path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    X = pd.read_csv(IN / "slld_phase3_1_X_train.csv")
    n, p = X.shape

    # Correlation matrix over 2210 features. This is large but still manageable.
    # Compute the 2210 x 2210 correlation matrix in memory.
    # We do not write the full matrix to CSV because it is extremely large;
    # the diagnostics below save threshold counts, top pairs and feature summaries.
    corr = X.corr().fillna(0.0)
    abs_corr = corr.abs()

    upper = np.triu(np.ones(abs_corr.shape, dtype=bool), k=1)
    vals = abs_corr.values[upper]
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
    counts = pd.DataFrame({"threshold": thresholds, "n_pairs_abs_corr_ge_threshold": [int((vals >= t).sum()) for t in thresholds]})
    counts.to_csv(OUT / "correlation_threshold_counts.csv", index=False)

    idx = np.argpartition(vals, -200)[-200:]
    upper_indices = np.argwhere(upper)
    top_rows = []
    for pos in idx:
        i, j = upper_indices[pos]
        top_rows.append({"feature_1": corr.index[i], "feature_2": corr.columns[j], "correlation": corr.iat[i, j], "abs_correlation": abs(corr.iat[i, j])})
    top = pd.DataFrame(top_rows).sort_values("abs_correlation", ascending=False)
    top.to_csv(OUT / "top_200_correlation_pairs.csv", index=False)
    top[top["abs_correlation"] >= 0.90].to_csv(OUT / "high_correlation_pairs_ge_090.csv", index=False)

    summary_by_feature = pd.DataFrame({
        "feature": abs_corr.columns,
        "mean_abs_correlation": abs_corr.where(~np.eye(p, dtype=bool)).mean(axis=0).values,
        "max_abs_correlation": abs_corr.where(~np.eye(p, dtype=bool)).max(axis=0).values,
    }).sort_values("max_abs_correlation", ascending=False)
    summary_by_feature.to_csv(OUT / "correlation_summary_by_feature.csv", index=False)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(vals, bins=60)
    ax.set_xlabel("Absolute pairwise correlation")
    ax.set_ylabel("Number of feature pairs")
    ax.set_title("Correlation distribution across 2210 expanded terms")
    save(fig, "correlation_distribution_train.png")

    fig, ax = plt.subplots(figsize=(8,6))
    top50 = summary_by_feature.head(50).sort_values("max_abs_correlation")
    ax.barh(top50["feature"], top50["max_abs_correlation"])
    ax.set_xlabel("Maximum absolute correlation")
    ax.set_title("Top 50 most collinear expanded terms")
    save(fig, "top_collinear_features.png")

    # Rank diagnostics: p > n implies X'X is singular. Exact full SVD is intentionally
    # avoided here because the diagnostic itself is not needed to prove singularity.
    rank_upper_bound = min(n, p)
    rank_deficiency_lower_bound = p - rank_upper_bound

    summary = {
        "n_observations": int(n),
        "n_features": int(p),
        "p_greater_than_n": bool(p > n),
        "rank_upper_bound": int(rank_upper_bound),
        "rank_deficiency_lower_bound": int(rank_deficiency_lower_bound),
        "classical_vif_status": "not computed because X'X is singular when p > n; rank(X) <= n < p",
        "correlation_threshold_counts": counts.to_dict(orient="records"),
        "max_abs_pairwise_correlation": float(vals.max()),
        "mean_abs_pairwise_correlation": float(vals.mean()),
    }
    write_json(summary, OUT / "phase3_3_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
