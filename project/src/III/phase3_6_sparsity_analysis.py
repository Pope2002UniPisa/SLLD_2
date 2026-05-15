from __future__ import annotations

"""
============================================================
PHASE 3.6 — SUPERVISED ANALYSIS: SPARSITY ANALYSIS
Statistical Learning and Large Data Project
Scuola Superiore Sant'Anna di Pisa

Goal:
- Analyse sparsity induced by the Phase 3.5 LASSO and Elastic Net models.
- Count zero and non-zero coefficients globally and by class.
- Identify features actively used by each model.
- Distinguish shared features, model-specific features and class-specific effects.
- Produce tables and figures that support the interpretation of the final model.

Input:
  outputs/III/phase3_5_regularized_models/lasso_coefficients.csv
  outputs/III/phase3_5_regularized_models/elastic_net_coefficients.csv
  outputs/III/phase3_5_regularized_models/selected_features_lasso.csv
  outputs/III/phase3_5_regularized_models/selected_features_elastic_net.csv

Outputs (all under outputs/III/phase3_6_sparsity_analysis/):
  - sparsity_summary.csv
  - sparsity_by_class.csv
  - active_features_by_model.csv
  - active_features_by_class.csv
  - shared_and_model_specific_features.csv
  - top_active_coefficients.csv
  - coefficient_matrix_lasso.csv
  - coefficient_matrix_elastic_net.csv
  - sparsity_overview.png
  - nonzero_coefficients_by_class.png
  - active_features_by_model.png
  - coefficient_heatmap_lasso.png
  - coefficient_heatmap_elastic_net.png
  - top_active_coefficients.png
  - phase3_6_summary.json
============================================================
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

OUTPUT_DIR = Path("./outputs/III/phase3_6_sparsity_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = OUTPUT_DIR / "phase3_6_summary.json"

PHASE3_5_CANDIDATES = [
    Path("./outputs/III/phase3_5_regularized_models"),
    Path("./phase3_5_regularized_models"),
]


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

COEF_ZERO_TOL = 1e-8
TOP_N_COEFFICIENTS = 30
CLASS_ORDER_DISPLAY = ["Thing", "Action", "Property"]


# ------------------------------------------------------------
# 3. Helpers
# ------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> Path:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_json(obj: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def resolve_phase3_5_dir() -> Path:
    for candidate in PHASE3_5_CANDIDATES:
        if (candidate / "lasso_coefficients.csv").exists() and (candidate / "elastic_net_coefficients.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find Phase 3.5 regularized model outputs.")


# ------------------------------------------------------------
# 4. Load data
# ------------------------------------------------------------

def load_coefficients() -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    phase3_5_dir = resolve_phase3_5_dir()
    lasso = pd.read_csv(phase3_5_dir / "lasso_coefficients.csv")
    elastic = pd.read_csv(phase3_5_dir / "elastic_net_coefficients.csv")
    for df in [lasso, elastic]:
        df["is_nonzero"] = df["coefficient"].abs() > COEF_ZERO_TOL
        df["abs_coefficient"] = df["coefficient"].abs()
    return lasso, elastic, phase3_5_dir


# ------------------------------------------------------------
# 5. Sparsity summaries
# ------------------------------------------------------------

def summarize_sparsity(coefs: pd.DataFrame, model_name: str) -> dict:
    total = len(coefs)
    nonzero = int(coefs["is_nonzero"].sum())
    zero = int(total - nonzero)
    feature_level = coefs.groupby("feature")["is_nonzero"].any()
    active_features = int(feature_level.sum())
    inactive_features = int((~feature_level).sum())
    return {
        "model": model_name,
        "total_coefficients": total,
        "zero_coefficients": zero,
        "nonzero_coefficients": nonzero,
        "zero_share": zero / total if total else np.nan,
        "nonzero_share": nonzero / total if total else np.nan,
        "total_features": int(feature_level.shape[0]),
        "active_features": active_features,
        "inactive_features": inactive_features,
        "active_feature_share": active_features / feature_level.shape[0] if feature_level.shape[0] else np.nan,
    }


def summarize_by_class(coefs: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for cls, sub in coefs.groupby("class_display"):
        total = len(sub)
        nonzero = int(sub["is_nonzero"].sum())
        rows.append({
            "model": model_name,
            "class_display": cls,
            "total_coefficients": total,
            "zero_coefficients": int(total - nonzero),
            "nonzero_coefficients": nonzero,
            "zero_share": (total - nonzero) / total if total else np.nan,
            "nonzero_share": nonzero / total if total else np.nan,
            "max_abs_coefficient": sub["abs_coefficient"].max(),
        })
    return pd.DataFrame(rows)


def active_features(coefs: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = (
        coefs.groupby("feature", as_index=False)
        .agg(
            selected=("is_nonzero", "any"),
            n_active_classes=("is_nonzero", "sum"),
            max_abs_coefficient=("abs_coefficient", "max"),
            sum_abs_coefficient=("abs_coefficient", "sum"),
        )
    )
    out["model"] = model_name
    return out.sort_values(["selected", "max_abs_coefficient", "feature"], ascending=[False, False, True])


def active_by_class(coefs: pd.DataFrame, model_name: str) -> pd.DataFrame:
    return (
        coefs[coefs["is_nonzero"]]
        .sort_values(["class_display", "abs_coefficient"], ascending=[True, False])
        [["model", "class_display", "feature", "coefficient", "abs_coefficient"]]
        .assign(model=model_name)
    )


def shared_specific(lasso_active: pd.DataFrame, elastic_active: pd.DataFrame) -> pd.DataFrame:
    lasso_set = set(lasso_active.loc[lasso_active["selected"], "feature"])
    elastic_set = set(elastic_active.loc[elastic_active["selected"], "feature"])
    all_features = sorted(set(lasso_active["feature"]) | set(elastic_active["feature"]))
    rows = []
    for f in all_features:
        in_lasso = f in lasso_set
        in_elastic = f in elastic_set
        if in_lasso and in_elastic:
            status = "shared"
        elif in_lasso:
            status = "lasso_only"
        elif in_elastic:
            status = "elastic_net_only"
        else:
            status = "inactive_in_both"
        rows.append({
            "feature": f,
            "selected_lasso": in_lasso,
            "selected_elastic_net": in_elastic,
            "selection_status": status,
        })
    return pd.DataFrame(rows)


def coefficient_matrix(coefs: pd.DataFrame, model_name: str) -> pd.DataFrame:
    mat = coefs.pivot(index="feature", columns="class_display", values="coefficient").fillna(0.0)
    ordered_cols = [c for c in CLASS_ORDER_DISPLAY if c in mat.columns]
    mat = mat[ordered_cols]
    mat["max_abs_coefficient"] = mat.abs().max(axis=1)
    mat = mat.sort_values("max_abs_coefficient", ascending=False)
    mat.drop(columns="max_abs_coefficient").to_csv(OUTPUT_DIR / f"coefficient_matrix_{model_name.lower().replace(' ', '_')}.csv")
    return mat.drop(columns="max_abs_coefficient")


# ------------------------------------------------------------
# 6. Plots
# ------------------------------------------------------------

def plot_sparsity_overview(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(summary))
    width = 0.35
    ax.bar(x - width/2, summary["zero_coefficients"], width, label="Zero")
    ax.bar(x + width/2, summary["nonzero_coefficients"], width, label="Non-zero")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=15, ha="right")
    ax.set_ylabel("Number of coefficients")
    ax.set_title("Coefficient sparsity overview")
    ax.legend()
    _save(fig, "sparsity_overview.png")


def plot_nonzero_by_class(by_class: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    models = by_class["model"].unique().tolist()
    classes = [c for c in CLASS_ORDER_DISPLAY if c in by_class["class_display"].unique()]
    x = np.arange(len(classes))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        vals = by_class[by_class["model"] == model].set_index("class_display").reindex(classes)["nonzero_coefficients"].fillna(0)
        ax.bar(x + (i - (len(models)-1)/2) * width, vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Non-zero coefficients")
    ax.set_title("Non-zero coefficients by class")
    ax.legend()
    _save(fig, "nonzero_coefficients_by_class.png")


def plot_active_features(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["model"], summary["active_features"])
    ax.set_ylabel("Active features")
    ax.set_title("Number of active features by model")
    _save(fig, "active_features_by_model.png")


def plot_coeff_heatmap(mat: pd.DataFrame, filename: str, title: str, top_n: int = 30) -> None:
    show = mat.head(min(top_n, len(mat)))
    fig, ax = plt.subplots(figsize=(7, max(5, 0.25 * len(show))))
    vmax = max(float(np.abs(show.values).max()), 1e-9)
    im = ax.imshow(show.values, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(show.shape[1]))
    ax.set_xticklabels(show.columns)
    ax.set_yticks(range(show.shape[0]))
    ax.set_yticklabels(show.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, filename)


def plot_top_coefficients(all_coefs: pd.DataFrame) -> None:
    top = all_coefs[all_coefs["is_nonzero"]].sort_values("abs_coefficient", ascending=False).head(TOP_N_COEFFICIENTS).copy()
    top["label"] = top["model"] + " | " + top["class_display"] + " | " + top["feature"]
    top = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(top))))
    ax.barh(top["label"], top["coefficient"])
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title("Largest active coefficients across regularized models")
    _save(fig, "top_active_coefficients.png")


# ------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------

def main() -> None:
    lasso, elastic, phase3_5_dir = load_coefficients()

    summary = pd.DataFrame([
        summarize_sparsity(lasso, "LASSO"),
        summarize_sparsity(elastic, "Elastic Net"),
    ])
    summary.to_csv(OUTPUT_DIR / "sparsity_summary.csv", index=False)

    by_class = pd.concat([
        summarize_by_class(lasso, "LASSO"),
        summarize_by_class(elastic, "Elastic Net"),
    ], ignore_index=True)
    by_class.to_csv(OUTPUT_DIR / "sparsity_by_class.csv", index=False)

    lasso_active = active_features(lasso, "LASSO")
    elastic_active = active_features(elastic, "Elastic Net")
    active_all = pd.concat([lasso_active, elastic_active], ignore_index=True)
    active_all.to_csv(OUTPUT_DIR / "active_features_by_model.csv", index=False)

    active_class = pd.concat([
        active_by_class(lasso, "LASSO"),
        active_by_class(elastic, "Elastic Net"),
    ], ignore_index=True)
    active_class.to_csv(OUTPUT_DIR / "active_features_by_class.csv", index=False)

    shared = shared_specific(lasso_active, elastic_active)
    shared.to_csv(OUTPUT_DIR / "shared_and_model_specific_features.csv", index=False)

    lasso_mat = coefficient_matrix(lasso, "lasso")
    elastic_mat = coefficient_matrix(elastic, "elastic_net")
    all_coefs = pd.concat([lasso, elastic], ignore_index=True)
    all_coefs.sort_values("abs_coefficient", ascending=False).head(TOP_N_COEFFICIENTS).to_csv(
        OUTPUT_DIR / "top_active_coefficients.csv", index=False
    )

    plot_sparsity_overview(summary)
    plot_nonzero_by_class(by_class)
    plot_active_features(summary)
    plot_coeff_heatmap(lasso_mat, "coefficient_heatmap_lasso.png", "LASSO coefficient heatmap")
    plot_coeff_heatmap(elastic_mat, "coefficient_heatmap_elastic_net.png", "Elastic Net coefficient heatmap")
    plot_top_coefficients(all_coefs)

    summary_json = {
        "phase": "3.6_sparsity_analysis",
        "input_phase3_5_dir": str(phase3_5_dir),
        "lasso_active_features": int(summary.loc[summary.model == "LASSO", "active_features"].iloc[0]),
        "elastic_net_active_features": int(summary.loc[summary.model == "Elastic Net", "active_features"].iloc[0]),
        "shared_active_features": int((shared["selection_status"] == "shared").sum()),
        "lasso_only_features": int((shared["selection_status"] == "lasso_only").sum()),
        "elastic_net_only_features": int((shared["selection_status"] == "elastic_net_only").sum()),
        "outputs_dir": str(OUTPUT_DIR),
    }
    _write_json(summary_json, SUMMARY_FILE)
    print(json.dumps(summary_json, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
