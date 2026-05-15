from __future__ import annotations

"""
============================================================
PHASE 3.3 — SUPERVISED ANALYSIS: MULTICOLLINEARITY
Statistical Learning and Large Data Project
Scuola Superiore Sant'Anna di Pisa

Goal:
- Continue the supervised analysis from Phase 3.1 / Phase 3.2 outputs.
- Load the supervised feature matrices X_train and X_test and the target
  vectors y_train and y_test.
- Diagnose multicollinearity among semantic predictors before applying
  screening, LASSO and Elastic Net.
- Compute pairwise correlations, highly correlated feature pairs,
  Variance Inflation Factors (VIF), partial R² values and a hierarchical
  clustering of variables based on correlation distance.
- Produce diagnostic tables, figures and a JSON summary in the same
  project style used by the previous phases.

Input:
  Preferred:
    outputs/III/phase3_1_supervised_data_loading_checks/slld_phase3_1_X_train.csv
    outputs/III/phase3_1_supervised_data_loading_checks/slld_phase3_1_X_test.csv
    outputs/III/phase3_1_supervised_data_loading_checks/slld_phase3_1_y_train.csv
    outputs/III/phase3_1_supervised_data_loading_checks/slld_phase3_1_y_test.csv
    outputs/III/phase3_1_supervised_data_loading_checks/feature_names.json
    outputs/III/phase3_1_supervised_data_loading_checks/class_names.json

  Accepted fallback locations:
    phase3_1_supervised_data_loading_checks/
    outputs/phase3_1_supervised_data_loading_checks/
    package_phase3_1/outputs/III/phase3_1_supervised_data_loading_checks/

Outputs (all under outputs/III/phase3_3_multicollinearity_analysis/):
  - correlation_matrix_train.csv
  - absolute_correlation_matrix_train.csv
  - correlation_summary_by_feature.csv
  - high_correlation_pairs.csv
  - correlation_threshold_counts.csv
  - vif_table.csv
  - high_vif_features.csv
  - feature_clusters.csv
  - redundant_feature_candidates.csv
  - correlation_heatmap_train.png
  - correlation_distribution_train.png
  - top_correlation_pairs.png
  - vif_distribution.png
  - top_vif_features.png
  - partial_r2_distribution.png
  - feature_dendrogram.png
  - feature_cluster_sizes.png
  - multicollinearity_summary.json
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
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

OUTPUT_DIR = Path("./outputs/III/phase3_3_multicollinearity_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "phase3_3_summary.json"

PHASE3_1_CANDIDATES = [
    Path("./outputs/III/phase3_1_supervised_data_loading_checks"),
    Path("./phase3_1_supervised_data_loading_checks"),
    Path("./outputs/phase3_1_supervised_data_loading_checks"),
    Path("./package_phase3_1/outputs/III/phase3_1_supervised_data_loading_checks"),
]


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

TARGET_COL = "target_word_class"
RANDOM_STATE = 42

X_TRAIN_FILE_NAME = "slld_phase3_1_X_train.csv"
Y_TRAIN_FILE_NAME = "slld_phase3_1_y_train.csv"
X_TEST_FILE_NAME = "slld_phase3_1_X_test.csv"
Y_TEST_FILE_NAME = "slld_phase3_1_y_test.csv"
FEATURE_NAMES_FILE_NAME = "feature_names.json"
CLASS_NAMES_FILE_NAME = "class_names.json"

# Human-readable class names used in the rest of the project.
LABEL_MAP = {"noun": "Thing", "verb": "Action", "adjective": "Property"}
CLASS_ORDER_DISPLAY = ["Thing", "Action", "Property"]

# Correlation thresholds used for multicollinearity diagnostics.
CORRELATION_THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.95]
HIGH_CORRELATION_THRESHOLD = 0.80
VERY_HIGH_CORRELATION_THRESHOLD = 0.90

# VIF thresholds are only diagnostic. They are not used to automatically
# remove features, because feature deletion must be coordinated with the
# later screening and regularization phases.
VIF_WARNING_THRESHOLD = 5.0
VIF_STRONG_THRESHOLD = 10.0

# For the dendrogram, the distance is 1 - |correlation|. Thus features with
# strong positive or negative correlation are considered close.
CLUSTER_DISTANCE_THRESHOLD = 1.0 - HIGH_CORRELATION_THRESHOLD

# Number of items displayed in compact plots.
TOP_N_PAIRS = 25
TOP_N_VIF = 25


# ------------------------------------------------------------
# 3. Helpers
# ------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> Path:
    """Save a matplotlib figure in the phase output directory."""
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_json(obj: dict | list, path: Path) -> None:
    """Write JSON files with stable indentation and UTF-8 encoding."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _label_display(y: pd.Series | np.ndarray | list[str]) -> pd.Series:
    """Map raw target labels to human-readable project labels."""
    y_series = pd.Series(y)
    return y_series.map(LABEL_MAP).fillna(y_series.astype(str))


def resolve_phase3_1_dir() -> Path:
    """Find the Phase 3.1 directory containing the supervised X/y files."""
    for candidate in PHASE3_1_CANDIDATES:
        required = [
            candidate / X_TRAIN_FILE_NAME,
            candidate / Y_TRAIN_FILE_NAME,
            candidate / X_TEST_FILE_NAME,
            candidate / Y_TEST_FILE_NAME,
            candidate / FEATURE_NAMES_FILE_NAME,
            candidate / CLASS_NAMES_FILE_NAME,
        ]
        if all(p.exists() for p in required):
            return candidate

    searched = "\n".join(str(p) for p in PHASE3_1_CANDIDATES)
    raise FileNotFoundError(
        "Could not find Phase 3.1 supervised X/y outputs. "
        f"Searched in:\n{searched}"
    )


def load_json_list(path: Path) -> list[str]:
    """Load a JSON list from disk."""
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise TypeError(f"Expected a JSON list in {path}, found {type(obj)}")
    return obj


def finite_or_nan(value: float) -> float | None:
    """Convert non-finite floating point values to None for clean JSON."""
    if value is None:
        return None
    if np.isfinite(value):
        return float(value)
    return None


# ------------------------------------------------------------
# 4. Load data
# ------------------------------------------------------------

def load_phase3_1_outputs() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str], Path]:
    """
    Load the supervised matrices and labels produced by Phase 3.1.

    The multicollinearity diagnostics are fitted only on X_train. X_test is
    loaded for structural consistency checks, but it is not used to choose
    variables. This avoids leaking information from the held-out test set.
    """
    phase3_1_dir = resolve_phase3_1_dir()

    X_train = pd.read_csv(phase3_1_dir / X_TRAIN_FILE_NAME)
    X_test = pd.read_csv(phase3_1_dir / X_TEST_FILE_NAME)

    y_train_df = pd.read_csv(phase3_1_dir / Y_TRAIN_FILE_NAME)
    y_test_df = pd.read_csv(phase3_1_dir / Y_TEST_FILE_NAME)

    if TARGET_COL not in y_train_df.columns or TARGET_COL not in y_test_df.columns:
        raise ValueError(f"The y files must contain the column '{TARGET_COL}'.")

    y_train = y_train_df[TARGET_COL].copy()
    y_test = y_test_df[TARGET_COL].copy()

    feature_names = load_json_list(phase3_1_dir / FEATURE_NAMES_FILE_NAME)
    class_names = load_json_list(phase3_1_dir / CLASS_NAMES_FILE_NAME)

    return X_train, y_train, X_test, y_test, feature_names, class_names, phase3_1_dir


# ------------------------------------------------------------
# 5. Structural checks
# ------------------------------------------------------------

def validate_feature_matrices(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Check that train and test feature matrices are aligned and numeric.

    The later models require that columns are identical and ordered in the
    same way across train and test. This function writes a compact report and
    stops execution if a structural error is detected.
    """
    train_cols = list(X_train.columns)
    test_cols = list(X_test.columns)

    report = pd.DataFrame([
        {
            "check": "same_number_of_features_train_vs_test",
            "passed": X_train.shape[1] == X_test.shape[1],
            "train_value": X_train.shape[1],
            "test_value": X_test.shape[1],
        },
        {
            "check": "same_feature_order_train_vs_test",
            "passed": train_cols == test_cols,
            "train_value": "|".join(train_cols[:5]) + ("..." if len(train_cols) > 5 else ""),
            "test_value": "|".join(test_cols[:5]) + ("..." if len(test_cols) > 5 else ""),
        },
        {
            "check": "same_feature_names_as_phase3_1_metadata",
            "passed": train_cols == feature_names,
            "train_value": len(train_cols),
            "test_value": len(feature_names),
        },
        {
            "check": "all_train_features_numeric",
            "passed": bool(all(pd.api.types.is_numeric_dtype(X_train[c]) for c in X_train.columns)),
            "train_value": int(sum(pd.api.types.is_numeric_dtype(X_train[c]) for c in X_train.columns)),
            "test_value": X_train.shape[1],
        },
        {
            "check": "all_test_features_numeric",
            "passed": bool(all(pd.api.types.is_numeric_dtype(X_test[c]) for c in X_test.columns)),
            "train_value": int(sum(pd.api.types.is_numeric_dtype(X_test[c]) for c in X_test.columns)),
            "test_value": X_test.shape[1],
        },
    ])
    report.to_csv(OUTPUT_DIR / "feature_alignment_check.csv", index=False)

    if not bool(report["passed"].all()):
        raise ValueError(
            "Feature matrix validation failed. See feature_alignment_check.csv."
        )

    return report


def summarize_dataset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    phase3_1_dir: Path,
) -> pd.DataFrame:
    """Save a compact shape and class-distribution summary."""
    y_train_display = _label_display(y_train)
    y_test_display = _label_display(y_test)

    rows = []
    for split_name, X, y_display in [
        ("train", X_train, y_train_display),
        ("test", X_test, y_test_display),
        ("combined", pd.concat([X_train, X_test], axis=0), pd.concat([y_train_display, y_test_display], axis=0)),
    ]:
        counts = y_display.value_counts().reindex(CLASS_ORDER_DISPLAY).fillna(0).astype(int)
        row = {
            "split": split_name,
            "n_observations": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_missing_values": int(X.isna().sum().sum()),
        }
        for cls in CLASS_ORDER_DISPLAY:
            row[f"n_{cls}"] = int(counts.loc[cls])
            row[f"pct_{cls}"] = float(counts.loc[cls] / max(len(y_display), 1))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["phase3_1_source_dir"] = str(phase3_1_dir)
    summary.to_csv(OUTPUT_DIR / "dataset_shape_and_class_summary.csv", index=False)
    return summary


# ------------------------------------------------------------
# 6. Correlation analysis
# ------------------------------------------------------------

def compute_correlation_diagnostics(X_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute correlation matrix, absolute correlations and pairwise ranking.

    Pairwise correlation is the first diagnostic for linear dependencies among
    predictors. Strong absolute correlations indicate redundant information and
    possible instability for unregularized or sparsity-based models.
    """
    corr = X_train.corr(method="pearson")
    abs_corr = corr.abs()

    corr.to_csv(OUTPUT_DIR / "correlation_matrix_train.csv")
    abs_corr.to_csv(OUTPUT_DIR / "absolute_correlation_matrix_train.csv")

    # Extract only the upper triangular part to avoid duplicated pairs and the
    # diagonal correlation of each feature with itself.
    upper_mask = np.triu(np.ones(abs_corr.shape, dtype=bool), k=1)
    pair_rows = []
    feature_array = np.array(abs_corr.columns)
    for i, j in zip(*np.where(upper_mask)):
        f1 = feature_array[i]
        f2 = feature_array[j]
        pair_rows.append({
            "feature_1": f1,
            "feature_2": f2,
            "correlation": float(corr.iat[i, j]),
            "abs_correlation": float(abs_corr.iat[i, j]),
        })

    pairs = pd.DataFrame(pair_rows).sort_values(
        "abs_correlation", ascending=False
    ).reset_index(drop=True)
    pairs.to_csv(OUTPUT_DIR / "all_correlation_pairs_ranked.csv", index=False)

    high_pairs = pairs[pairs["abs_correlation"] >= HIGH_CORRELATION_THRESHOLD].copy()
    high_pairs.to_csv(OUTPUT_DIR / "high_correlation_pairs.csv", index=False)

    threshold_counts = []
    for threshold in CORRELATION_THRESHOLDS:
        threshold_counts.append({
            "abs_correlation_threshold": threshold,
            "n_pairs": int((pairs["abs_correlation"] >= threshold).sum()),
            "pct_pairs": float((pairs["abs_correlation"] >= threshold).mean()) if len(pairs) else 0.0,
        })
    threshold_counts_df = pd.DataFrame(threshold_counts)
    threshold_counts_df.to_csv(OUTPUT_DIR / "correlation_threshold_counts.csv", index=False)

    summary_rows = []
    for feature in X_train.columns:
        vals = abs_corr.loc[feature].drop(index=feature)
        summary_rows.append({
            "feature": feature,
            "mean_abs_correlation": float(vals.mean()),
            "median_abs_correlation": float(vals.median()),
            "max_abs_correlation": float(vals.max()),
            "n_corr_ge_0_70": int((vals >= 0.70).sum()),
            "n_corr_ge_0_80": int((vals >= 0.80).sum()),
            "n_corr_ge_0_90": int((vals >= 0.90).sum()),
        })
    feature_summary = pd.DataFrame(summary_rows).sort_values(
        ["max_abs_correlation", "mean_abs_correlation"], ascending=False
    )
    feature_summary.to_csv(OUTPUT_DIR / "correlation_summary_by_feature.csv", index=False)

    return corr, abs_corr, pairs, threshold_counts_df


# ------------------------------------------------------------
# 7. VIF and partial R²
# ------------------------------------------------------------

def compute_vif_table(X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factors for all features.

    For each feature X_j, we regress X_j on all other features and compute:

        VIF_j = 1 / (1 - R²_j)

    The associated partial R² is simply:

        partial_R2_j = 1 - 1 / VIF_j

    If perfect or near-perfect multicollinearity occurs, R² can be 1 and the
    VIF becomes infinite. In that case we keep the infinite value in the CSV and
    convert it to null only inside the JSON summary.
    """
    feature_names = list(X_train.columns)
    X_values = X_train.to_numpy(dtype=float)
    n_features = X_values.shape[1]

    rows = []
    for j, feature in enumerate(feature_names):
        y_j = X_values[:, j]
        X_others = np.delete(X_values, j, axis=1)

        # LinearRegression is the direct diagnostic model for VIF. If the
        # auxiliary regression becomes numerically problematic, Ridge with a tiny
        # alpha is used as a stable fallback while still approximating the same
        # diagnostic idea.
        try:
            model = LinearRegression()
            model.fit(X_others, y_j)
            pred = model.predict(X_others)
        except Exception:
            model = Ridge(alpha=1e-8, random_state=RANDOM_STATE)
            model.fit(X_others, y_j)
            pred = model.predict(X_others)

        r2 = float(r2_score(y_j, pred))
        # Numerical rounding may produce values slightly above 1.
        r2 = min(max(r2, 0.0), 1.0)

        if 1.0 - r2 <= 1e-12:
            vif = np.inf
        else:
            vif = 1.0 / (1.0 - r2)

        partial_r2 = 1.0 - (1.0 / vif) if np.isfinite(vif) and vif != 0 else 1.0

        rows.append({
            "feature": feature,
            "r2_against_other_features": r2,
            "partial_r2": partial_r2,
            "vif": vif,
            "vif_flag": (
                "strong" if vif >= VIF_STRONG_THRESHOLD else
                "warning" if vif >= VIF_WARNING_THRESHOLD else
                "ok"
            ),
        })

    vif_table = pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)
    vif_table.to_csv(OUTPUT_DIR / "vif_table.csv", index=False)

    high_vif = vif_table[vif_table["vif"] >= VIF_WARNING_THRESHOLD].copy()
    high_vif.to_csv(OUTPUT_DIR / "high_vif_features.csv", index=False)

    return vif_table


# ------------------------------------------------------------
# 8. Hierarchical clustering of variables
# ------------------------------------------------------------

def compute_feature_clusters(abs_corr: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Cluster features with hierarchical clustering using correlation distance.

    Distance = 1 - |correlation|.
    Complete linkage is conservative: a cluster is tight only if all its members
    remain close under the selected distance threshold.
    """
    distance = 1.0 - abs_corr.copy()
    np.fill_diagonal(distance.values, 0.0)

    # squareform expects a condensed distance matrix. Checks are disabled
    # because floating point operations may introduce tiny asymmetries.
    condensed = squareform(distance.values, checks=False)
    Z = linkage(condensed, method="complete")

    cluster_ids = fcluster(Z, t=CLUSTER_DISTANCE_THRESHOLD, criterion="distance")
    cluster_df = pd.DataFrame({
        "feature": abs_corr.index.tolist(),
        "cluster_id": cluster_ids,
    })

    cluster_sizes = cluster_df.groupby("cluster_id").size().rename("cluster_size")
    cluster_df = cluster_df.merge(cluster_sizes, on="cluster_id", how="left")
    cluster_df = cluster_df.sort_values(["cluster_size", "cluster_id", "feature"], ascending=[False, True, True])
    cluster_df.to_csv(OUTPUT_DIR / "feature_clusters.csv", index=False)

    return cluster_df, Z


def compute_redundant_feature_candidates(
    feature_summary: pd.DataFrame,
    vif_table: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a conservative list of possible redundant features.

    This is not an automatic deletion list. It simply highlights features that
    combine three signs of redundancy:
    - they belong to a cluster with more than one feature;
    - they have at least one high absolute pairwise correlation;
    - they have elevated VIF.

    The later screening and regularized models decide what is actually retained.
    """
    merged = (
        feature_summary
        .merge(vif_table[["feature", "vif", "partial_r2", "vif_flag"]], on="feature", how="left")
        .merge(cluster_df[["feature", "cluster_id", "cluster_size"]], on="feature", how="left")
    )

    merged["redundancy_score"] = (
        merged["n_corr_ge_0_80"].astype(float)
        + merged["n_corr_ge_0_90"].astype(float) * 2.0
        + (merged["vif"].replace(np.inf, np.nan).fillna(VIF_STRONG_THRESHOLD) >= VIF_WARNING_THRESHOLD).astype(float)
        + (merged["cluster_size"] > 1).astype(float)
    )

    candidates = merged[
        (merged["cluster_size"] > 1)
        | (merged["n_corr_ge_0_80"] > 0)
        | (merged["vif"] >= VIF_WARNING_THRESHOLD)
    ].copy()
    candidates = candidates.sort_values(
        ["redundancy_score", "vif", "max_abs_correlation"], ascending=False
    )
    candidates.to_csv(OUTPUT_DIR / "redundant_feature_candidates.csv", index=False)
    return candidates


# ------------------------------------------------------------
# 9. Plots
# ------------------------------------------------------------

def plot_correlation_heatmap(abs_corr: pd.DataFrame) -> None:
    """Plot a compact heatmap of absolute correlations among features."""
    fig_size = max(8, min(16, 0.18 * abs_corr.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(abs_corr.values, aspect="auto", vmin=0, vmax=1)
    ax.set_title("Absolute Correlation Matrix — Training Features")
    ax.set_xlabel("Features")
    ax.set_ylabel("Features")

    # With many features, labels become unreadable. Show them only when the
    # matrix is small enough; otherwise keep the heatmap as a global diagnostic.
    if abs_corr.shape[0] <= 35:
        ax.set_xticks(range(abs_corr.shape[0]))
        ax.set_yticks(range(abs_corr.shape[0]))
        ax.set_xticklabels(abs_corr.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(abs_corr.index, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Pearson correlation|")
    _save(fig, "correlation_heatmap_train.png")


def plot_correlation_distribution(pairs: pd.DataFrame) -> None:
    """Plot the distribution of absolute pairwise correlations."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(pairs["abs_correlation"], bins=30, alpha=0.85)
    for threshold in [0.70, 0.80, 0.90]:
        ax.axvline(threshold, linestyle="--", linewidth=1, label=f"|r| = {threshold:.2f}")
    ax.set_xlabel("Absolute Pearson correlation")
    ax.set_ylabel("Number of feature pairs")
    ax.set_title("Distribution of Absolute Pairwise Correlations")
    ax.legend()
    _save(fig, "correlation_distribution_train.png")


def plot_top_correlation_pairs(pairs: pd.DataFrame) -> None:
    """Plot the strongest correlated feature pairs."""
    top = pairs.head(TOP_N_PAIRS).copy()
    if top.empty:
        return
    top["pair"] = top["feature_1"] + "  |  " + top["feature_2"]
    top = top.sort_values("abs_correlation")

    fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(top))))
    ax.barh(top["pair"], top["abs_correlation"])
    ax.axvline(HIGH_CORRELATION_THRESHOLD, linestyle="--", linewidth=1, label="High-correlation threshold")
    ax.set_xlabel("Absolute Pearson correlation")
    ax.set_title(f"Top {len(top)} Correlated Feature Pairs")
    ax.legend()
    _save(fig, "top_correlation_pairs.png")


def plot_vif_distribution(vif_table: pd.DataFrame) -> None:
    """Plot VIF distribution, capping infinite values for readability."""
    vif_values = vif_table["vif"].replace(np.inf, np.nan)
    finite_values = vif_values.dropna()
    cap = max(VIF_STRONG_THRESHOLD, finite_values.quantile(0.95) if len(finite_values) else VIF_STRONG_THRESHOLD)
    plotted = vif_table["vif"].replace(np.inf, cap * 1.1).clip(upper=cap * 1.1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(plotted, bins=30, alpha=0.85)
    ax.axvline(VIF_WARNING_THRESHOLD, linestyle="--", linewidth=1, label=f"VIF = {VIF_WARNING_THRESHOLD:.0f}")
    ax.axvline(VIF_STRONG_THRESHOLD, linestyle=":", linewidth=1, label=f"VIF = {VIF_STRONG_THRESHOLD:.0f}")
    ax.set_xlabel("VIF (capped for plotting if necessary)")
    ax.set_ylabel("Number of features")
    ax.set_title("Distribution of Variance Inflation Factors")
    ax.legend()
    _save(fig, "vif_distribution.png")


def plot_top_vif_features(vif_table: pd.DataFrame) -> None:
    """Plot the features with the largest VIF values."""
    top = vif_table.head(TOP_N_VIF).copy()
    if top.empty:
        return
    finite = top["vif"].replace(np.inf, np.nan).dropna()
    cap = max(VIF_STRONG_THRESHOLD, finite.max() if len(finite) else VIF_STRONG_THRESHOLD)
    top["vif_for_plot"] = top["vif"].replace(np.inf, cap * 1.05).clip(upper=cap * 1.05)
    top = top.sort_values("vif_for_plot")

    fig, ax = plt.subplots(figsize=(8, max(5, 0.28 * len(top))))
    ax.barh(top["feature"], top["vif_for_plot"])
    ax.axvline(VIF_WARNING_THRESHOLD, linestyle="--", linewidth=1, label="Warning threshold")
    ax.axvline(VIF_STRONG_THRESHOLD, linestyle=":", linewidth=1, label="Strong threshold")
    ax.set_xlabel("VIF")
    ax.set_title(f"Top {len(top)} Features by VIF")
    ax.legend()
    _save(fig, "top_vif_features.png")


def plot_partial_r2_distribution(vif_table: pd.DataFrame) -> None:
    """Plot partial R² values implied by the VIF diagnostics."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(vif_table["partial_r2"], bins=30, alpha=0.85)
    ax.set_xlabel("Partial R² against all other features")
    ax.set_ylabel("Number of features")
    ax.set_title("Distribution of Partial R² Values")
    _save(fig, "partial_r2_distribution.png")


def plot_feature_dendrogram(Z: np.ndarray, feature_names: list[str]) -> None:
    """Plot the hierarchical dendrogram of variables."""
    fig, ax = plt.subplots(figsize=(16, 6))
    dendrogram(
        Z,
        labels=feature_names,
        leaf_rotation=90,
        leaf_font_size=6,
        color_threshold=CLUSTER_DISTANCE_THRESHOLD,
        ax=ax,
    )
    ax.axhline(CLUSTER_DISTANCE_THRESHOLD, linestyle="--", linewidth=1, label="Cluster cut: |r| ≥ 0.80")
    ax.set_ylabel("Distance = 1 - |correlation|")
    ax.set_title("Feature Dendrogram Based on Correlation Distance")
    ax.legend()
    _save(fig, "feature_dendrogram.png")


def plot_feature_cluster_sizes(cluster_df: pd.DataFrame) -> None:
    """Plot sizes of feature clusters produced from the dendrogram cut."""
    sizes = cluster_df.groupby("cluster_id").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(1, len(sizes) + 1), sizes.values)
    ax.set_xlabel("Cluster rank by size")
    ax.set_ylabel("Number of features")
    ax.set_title("Feature Cluster Sizes at |r| ≥ 0.80 Cut")
    _save(fig, "feature_cluster_sizes.png")


# ------------------------------------------------------------
# 10. Summary
# ------------------------------------------------------------

def build_summary(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    pairs: pd.DataFrame,
    threshold_counts: pd.DataFrame,
    vif_table: pd.DataFrame,
    cluster_df: pd.DataFrame,
    redundant_candidates: pd.DataFrame,
    phase3_1_dir: Path,
) -> dict:
    """Build the JSON summary for this phase."""
    finite_vif = vif_table["vif"].replace(np.inf, np.nan).dropna()
    largest_cluster_size = int(cluster_df["cluster_size"].max()) if not cluster_df.empty else 0
    n_non_singleton_clusters = int((cluster_df.groupby("cluster_id").size() > 1).sum()) if not cluster_df.empty else 0

    summary = {
        "phase": "3.3_multicollinearity_analysis",
        "input_phase3_1_dir": str(phase3_1_dir),
        "n_train_observations": int(X_train.shape[0]),
        "n_test_observations": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "correlation": {
            "n_feature_pairs": int(len(pairs)),
            "max_abs_correlation": finite_or_nan(pairs["abs_correlation"].max() if len(pairs) else np.nan),
            "mean_abs_correlation": finite_or_nan(pairs["abs_correlation"].mean() if len(pairs) else np.nan),
            "median_abs_correlation": finite_or_nan(pairs["abs_correlation"].median() if len(pairs) else np.nan),
            "n_pairs_abs_corr_ge_0_80": int((pairs["abs_correlation"] >= HIGH_CORRELATION_THRESHOLD).sum()) if len(pairs) else 0,
            "n_pairs_abs_corr_ge_0_90": int((pairs["abs_correlation"] >= VERY_HIGH_CORRELATION_THRESHOLD).sum()) if len(pairs) else 0,
            "threshold_counts": threshold_counts.to_dict(orient="records"),
        },
        "vif": {
            "max_finite_vif": finite_or_nan(finite_vif.max() if len(finite_vif) else np.nan),
            "median_finite_vif": finite_or_nan(finite_vif.median() if len(finite_vif) else np.nan),
            "n_infinite_vif": int(np.isinf(vif_table["vif"]).sum()),
            "n_vif_ge_5": int((vif_table["vif"] >= VIF_WARNING_THRESHOLD).sum()),
            "n_vif_ge_10": int((vif_table["vif"] >= VIF_STRONG_THRESHOLD).sum()),
        },
        "clusters": {
            "distance": "1 - abs(Pearson correlation)",
            "linkage": "complete",
            "distance_threshold": float(CLUSTER_DISTANCE_THRESHOLD),
            "equivalent_abs_correlation_threshold": float(HIGH_CORRELATION_THRESHOLD),
            "n_clusters": int(cluster_df["cluster_id"].nunique()) if not cluster_df.empty else 0,
            "n_non_singleton_clusters": n_non_singleton_clusters,
            "largest_cluster_size": largest_cluster_size,
        },
        "redundancy_candidates": {
            "n_candidates": int(len(redundant_candidates)),
            "note": "Diagnostic list only. No feature is removed in Phase 3.3.",
        },
        "outputs": {
            "correlation_matrix": str(OUTPUT_DIR / "correlation_matrix_train.csv"),
            "high_correlation_pairs": str(OUTPUT_DIR / "high_correlation_pairs.csv"),
            "vif_table": str(OUTPUT_DIR / "vif_table.csv"),
            "feature_clusters": str(OUTPUT_DIR / "feature_clusters.csv"),
            "redundant_feature_candidates": str(OUTPUT_DIR / "redundant_feature_candidates.csv"),
        },
    }
    _write_json(summary, SUMMARY_FILE)
    return summary


# ------------------------------------------------------------
# 11. Main execution
# ------------------------------------------------------------

def main() -> None:
    """Run Phase 3.3 multicollinearity diagnostics."""
    X_train, y_train, X_test, y_test, feature_names, class_names, phase3_1_dir = load_phase3_1_outputs()

    alignment_report = validate_feature_matrices(X_train, X_test, feature_names)
    dataset_summary = summarize_dataset(X_train, y_train, X_test, y_test, phase3_1_dir)

    corr, abs_corr, pairs, threshold_counts = compute_correlation_diagnostics(X_train)
    # Reload the feature summary because it has already been persisted inside
    # compute_correlation_diagnostics and is needed for redundancy candidates.
    feature_summary = pd.read_csv(OUTPUT_DIR / "correlation_summary_by_feature.csv")

    vif_table = compute_vif_table(X_train)
    cluster_df, Z = compute_feature_clusters(abs_corr)
    redundant_candidates = compute_redundant_feature_candidates(feature_summary, vif_table, cluster_df)

    plot_correlation_heatmap(abs_corr)
    plot_correlation_distribution(pairs)
    plot_top_correlation_pairs(pairs)
    plot_vif_distribution(vif_table)
    plot_top_vif_features(vif_table)
    plot_partial_r2_distribution(vif_table)
    plot_feature_dendrogram(Z, feature_names)
    plot_feature_cluster_sizes(cluster_df)

    summary = build_summary(
        X_train=X_train,
        X_test=X_test,
        pairs=pairs,
        threshold_counts=threshold_counts,
        vif_table=vif_table,
        cluster_df=cluster_df,
        redundant_candidates=redundant_candidates,
        phase3_1_dir=phase3_1_dir,
    )

    print("PHASE 3.3 — MULTICOLLINEARITY ANALYSIS COMPLETED")
    print(f"Input Phase 3.1 directory: {phase3_1_dir}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Max |correlation|: {summary['correlation']['max_abs_correlation']}")
    print(f"Pairs with |correlation| >= 0.80: {summary['correlation']['n_pairs_abs_corr_ge_0_80']}")
    print(f"Features with VIF >= 5: {summary['vif']['n_vif_ge_5']}")
    print(f"Non-singleton feature clusters: {summary['clusters']['n_non_singleton_clusters']}")


if __name__ == "__main__":
    main()
