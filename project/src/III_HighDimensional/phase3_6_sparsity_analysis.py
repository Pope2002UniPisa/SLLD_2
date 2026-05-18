from __future__ import annotations

"""PHASE 3.6 — SPARSITY ANALYSIS FOR 2210-FEATURE REGULARIZED MODELS."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

IN = Path("./outputs/III/phase3_5_regularized_models")
OUT = Path("./outputs/III/phase3_6_sparsity_analysis")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["lasso", "elastic_net"]


def write_json(obj, path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = []
    active_rows = []
    for model in MODELS:
        coef = pd.read_csv(IN / f"{model}_coefficients.csv", index_col=0)
        active_mask = coef.abs() > 1e-8
        n_total = coef.size
        n_nonzero = int(active_mask.sum().sum())
        active_features = active_mask.any(axis=0)
        rows.append({
            "model": model,
            "n_features_input_after_screening": int(coef.shape[1]),
            "n_classes": int(coef.shape[0]),
            "n_coefficients_total": int(n_total),
            "n_nonzero_coefficients": n_nonzero,
            "coefficient_sparsity_pct": float((1 - n_nonzero / n_total) * 100),
            "n_active_features": int(active_features.sum()),
            "feature_sparsity_pct": float((1 - active_features.sum() / coef.shape[1]) * 100),
        })
        for cls in coef.index:
            cls_mask = active_mask.loc[cls]
            active_rows.append({"model": model, "class": cls, "n_nonzero_coefficients": int(cls_mask.sum())})
        active = coef.loc[:, active_features].T
        active["max_abs_coefficient"] = active.abs().max(axis=1)
        active.sort_values("max_abs_coefficient", ascending=False).to_csv(OUT / f"active_features_{model}.csv")
        coef.to_csv(OUT / f"coefficient_matrix_{model}.csv")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT / "sparsity_summary.csv", index=False)
    pd.DataFrame(active_rows).to_csv(OUT / "sparsity_by_class.csv", index=False)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(summary_df["model"], summary_df["n_active_features"])
    ax.set_ylabel("Number of active features")
    ax.set_title("Active features after L1/Elastic-Net regularization")
    save(fig, "active_features_by_model.png")

    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(summary_df["model"], summary_df["feature_sparsity_pct"])
    ax.set_ylabel("Feature sparsity (%)")
    ax.set_title("Feature sparsity after regularization")
    save(fig, "sparsity_overview.png")

    lasso_features = set(pd.read_csv(OUT / "active_features_lasso.csv", index_col=0).index)
    en_features = set(pd.read_csv(OUT / "active_features_elastic_net.csv", index_col=0).index)
    shared = sorted(lasso_features & en_features)
    specific = pd.DataFrame({
        "category": ["shared", "lasso_only", "elastic_net_only"],
        "n_features": [len(shared), len(lasso_features - en_features), len(en_features - lasso_features)],
    })
    specific.to_csv(OUT / "shared_and_model_specific_features.csv", index=False)
    write_json({"shared_features": shared}, OUT / "shared_features.json")

    summary = {"models": rows, "n_shared_active_features": len(shared)}
    write_json(summary, OUT / "phase3_6_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
