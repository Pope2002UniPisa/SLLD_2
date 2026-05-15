from __future__ import annotations

"""
============================================================
PHASE 4 — SUPERVISED ANALYSIS WITH CLUSTER LABELS
Statistical Learning and Large Data Project
Scuola Superiore Sant'Anna di Pisa

Goal:
- Repeat the supervised classification pipeline using clustering labels
  as alternative targets.
- Reconstruct the clustering targets coherently with Phase 2:
  PCA denoising on the standardised semantic feature space followed by
  K-Means clustering.
- Use K-Means labels as target variables for two settings:
    1) k = 3, to make the cluster target directly comparable with the
       three original lexical-semantic labels;
    2) k = best-k from Phase 2 silhouette, when available, usually k = 14.
- Fit the same families of supervised models used in Phase 3:
    - multinomial logistic baseline;
    - marginal feature screening;
    - LASSO multinomial logistic regression;
    - Elastic Net multinomial logistic regression;
    - post-selection logistic regression on selected features.
- Compare predictive results obtained with cluster labels against the
  results obtained with the original labels.

Input:
  outputs/III/phase3_1_supervised_data_loading_checks/
    - slld_phase3_1_X_train.csv
    - slld_phase3_1_y_train.csv
    - slld_phase3_1_X_test.csv
    - slld_phase3_1_y_test.csv
    - slld_phase3_1_train_modeling_dataset.csv
    - slld_phase3_1_test_modeling_dataset.csv
    - feature_names.json

Optional input:
  outputs/II/phase2_unsupervised/phase2_summary.json
  outputs/III/phase3_7_final_model_evaluation/full_model_comparison.csv

Outputs:
  outputs/IV/phase4_cluster_label_supervised_analysis/
    - cluster_target_assignments_train.csv
    - cluster_target_assignments_test.csv
    - cluster_target_distribution.csv
    - phase4_model_metrics.csv
    - phase4_original_vs_cluster_comparison.csv
    - per-target model reports, coefficients, selected features and plots
    - phase4_summary.json
============================================================
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

OUTPUT_DIR = Path("./outputs/IV/phase4_cluster_label_supervised_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = OUTPUT_DIR / "phase4_summary.json"

PHASE3_1_CANDIDATES = [
    Path("./outputs/III/phase3_1_supervised_data_loading_checks"),
    Path("./package_phase3_1/outputs/III/phase3_1_supervised_data_loading_checks"),
    Path("./phase3_1_supervised_data_loading_checks"),
]

PHASE2_SUMMARY_CANDIDATES = [
    Path("./outputs/II/phase2_unsupervised/phase2_summary.json"),
    Path("./phase2_unsupervised/phase2_summary.json"),
]

PHASE3_FINAL_COMPARISON_CANDIDATES = [
    Path("./outputs/III/phase3_7_final_model_evaluation/full_model_comparison.csv"),
    Path("./phase3_7_final_model_evaluation/full_model_comparison.csv"),
]


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

RANDOM_STATE = 42
TARGET_COL = "target_word_class"
ID_COLS = ["entry_id", "word", "target_word_class"]

FEATURE_NAMES_FILE = "feature_names.json"
X_TRAIN_FILE = "slld_phase3_1_X_train.csv"
Y_TRAIN_FILE = "slld_phase3_1_y_train.csv"
X_TEST_FILE = "slld_phase3_1_X_test.csv"
Y_TEST_FILE = "slld_phase3_1_y_test.csv"
TRAIN_MODELING_FILE = "slld_phase3_1_train_modeling_dataset.csv"
TEST_MODELING_FILE = "slld_phase3_1_test_modeling_dataset.csv"

# Phase 2 used the number of PCs needed to retain roughly 75% variance.
DEFAULT_N_PCS_CLUSTER = 10
DEFAULT_CLUSTER_KS = [3]

# Regularized multinomial logistic models.
CS_GRID = np.array([0.01, 0.1, 1.0, 10.0])
ELASTIC_NET_L1_RATIOS = [0.50]
MAX_ITER = 5000
TOL = 1e-4
COEF_ZERO_TOL = 1e-8
TOP_N_FEATURES = 25
TOP_N_SCREENING = 500

LABEL_MAP = {"noun": "Thing", "verb": "Action", "adjective": "Property"}


# ------------------------------------------------------------
# 3. Generic helpers
# ------------------------------------------------------------

def _save(fig: plt.Figure, name: str, target_dir: Path = OUTPUT_DIR) -> Path:
    path = target_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_json(obj: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_existing_dir(candidates: list[Path], required_files: list[str]) -> Path:
    for candidate in candidates:
        if all((candidate / f).exists() for f in required_files):
            return candidate
    checked = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not resolve required directory. Checked:\n{checked}")


def first_existing_file(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def display_original_label(y: pd.Series) -> pd.Series:
    return y.map(LABEL_MAP).fillna(y.astype(str))


def cluster_display(labels: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(labels).astype(int)
    return s.map(lambda x: f"Cluster_{x + 1:02d}")


def safe_cv_splits(y: pd.Series, requested: int = 5) -> int:
    counts = y.value_counts()
    min_count = int(counts.min())
    return max(2, min(requested, min_count))


# ------------------------------------------------------------
# 4. Data loading
# ------------------------------------------------------------

def load_phase3_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, list[str], Path]:
    phase3_dir = resolve_existing_dir(
        PHASE3_1_CANDIDATES,
        [X_TRAIN_FILE, Y_TRAIN_FILE, X_TEST_FILE, Y_TEST_FILE, FEATURE_NAMES_FILE],
    )

    X_train = pd.read_csv(phase3_dir / X_TRAIN_FILE)
    y_train_original = pd.read_csv(phase3_dir / Y_TRAIN_FILE)[TARGET_COL]
    X_test = pd.read_csv(phase3_dir / X_TEST_FILE)
    y_test_original = pd.read_csv(phase3_dir / Y_TEST_FILE)[TARGET_COL]

    # Modeling files are used only to preserve entry_id and word in output assignments.
    train_meta = pd.read_csv(phase3_dir / TRAIN_MODELING_FILE)
    test_meta = pd.read_csv(phase3_dir / TEST_MODELING_FILE)
    feature_names = _read_json(phase3_dir / FEATURE_NAMES_FILE)

    if not isinstance(feature_names, list):
        raise TypeError("feature_names.json must contain a JSON list.")

    X_train = X_train[feature_names]
    X_test = X_test[feature_names]
    return X_train, y_train_original, X_test, y_test_original, train_meta, test_meta, feature_names, phase3_dir


# ------------------------------------------------------------
# 5. Cluster target construction
# ------------------------------------------------------------

def load_phase2_decisions() -> tuple[int, list[int], Path | None]:
    summary_path = first_existing_file(PHASE2_SUMMARY_CANDIDATES)
    if summary_path is None:
        return DEFAULT_N_PCS_CLUSTER, DEFAULT_CLUSTER_KS, None

    summary = _read_json(summary_path)
    n_pcs = int(summary.get("pca", {}).get("n_pcs_used_for_clustering", DEFAULT_N_PCS_CLUSTER))
    # For this supervised rerun we use k=3, so that the alternative
    # cluster target has the same number of classes as the original
    # Thing/Action/Property target and can be compared directly.
    cluster_ks = [3]
    return n_pcs, cluster_ks, summary_path


def build_cluster_targets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_original: pd.Series,
    y_test_original: pd.Series,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    n_pcs_cluster: int,
    cluster_ks: list[int],
) -> tuple[dict[str, dict], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit PCA+KMeans on training data and assign cluster labels to train/test.

    Phase 2 clustered the training split only, because its input was the
    standardised Phase 1 training matrix. For supervised evaluation we need a
    test target as well. K-Means is therefore refitted on the same training
    space and test labels are obtained with the fitted centroids.
    """
    n_pcs_cluster = min(n_pcs_cluster, X_train.shape[1], X_train.shape[0] - 1)
    pca = PCA(n_components=n_pcs_cluster, random_state=RANDOM_STATE)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)

    target_specs: dict[str, dict] = {}
    train_assignments = train_meta[["entry_id", "word", TARGET_COL]].copy()
    test_assignments = test_meta[["entry_id", "word", TARGET_COL]].copy()
    train_assignments["original_label_display"] = display_original_label(y_train_original)
    test_assignments["original_label_display"] = display_original_label(y_test_original)

    distributions = []

    for k in cluster_ks:
        model_name = f"kmeans_k{k}"
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        train_raw = km.fit_predict(Z_train)
        test_raw = km.predict(Z_test)
        train_display = cluster_display(train_raw)
        test_display = cluster_display(test_raw)

        train_assignments[model_name] = train_display.values
        test_assignments[model_name] = test_display.values

        for split_name, y_cluster in [("train", train_display), ("test", test_display)]:
            counts = y_cluster.value_counts().sort_index()
            for label, count in counts.items():
                distributions.append({
                    "target": model_name,
                    "split": split_name,
                    "class": label,
                    "count": int(count),
                    "proportion": float(count / len(y_cluster)),
                })

        target_specs[model_name] = {
            "target_name": model_name,
            "k": k,
            "pca_n_components": n_pcs_cluster,
            "classes": sorted(train_display.unique().tolist()),
            "train_labels": train_display,
            "test_labels": test_display,
            "kmeans_model": km,
            "pca_model": pca,
        }

    distribution_df = pd.DataFrame(distributions)
    train_assignments.to_csv(OUTPUT_DIR / "cluster_target_assignments_train.csv", index=False)
    test_assignments.to_csv(OUTPUT_DIR / "cluster_target_assignments_test.csv", index=False)
    distribution_df.to_csv(OUTPUT_DIR / "cluster_target_distribution.csv", index=False)

    plot_cluster_distributions(distribution_df)
    plot_cluster_vs_original(train_assignments, split="train")
    plot_cluster_vs_original(test_assignments, split="test")

    return target_specs, train_assignments, test_assignments, distribution_df


def plot_cluster_distributions(distribution_df: pd.DataFrame) -> None:
    for target in distribution_df["target"].unique():
        sub = distribution_df[distribution_df["target"] == target]
        pivot = sub.pivot(index="class", columns="split", values="count").fillna(0)
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(pivot)), 4))
        x = np.arange(len(pivot.index))
        width = 0.38
        ax.bar(x - width / 2, pivot.get("train", pd.Series(0, index=pivot.index)), width, label="train")
        ax.bar(x + width / 2, pivot.get("test", pd.Series(0, index=pivot.index)), width, label="test")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(f"Cluster target distribution — {target}")
        ax.legend()
        _save(fig, f"{target}_target_distribution.png")


def plot_cluster_vs_original(assignments: pd.DataFrame, split: str) -> None:
    cluster_cols = [c for c in assignments.columns if c.startswith("kmeans_k")]
    for col in cluster_cols:
        ct = pd.crosstab(assignments[col], assignments["original_label_display"])
        ct.to_csv(OUTPUT_DIR / f"{col}_vs_original_labels_{split}.csv")
        fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * ct.shape[0])))
        im = ax.imshow(ct.values, aspect="auto")
        ax.set_xticks(range(len(ct.columns)))
        ax.set_xticklabels(ct.columns)
        ax.set_yticks(range(len(ct.index)))
        ax.set_yticklabels(ct.index)
        ax.set_xlabel("Original label")
        ax.set_ylabel("Cluster label")
        ax.set_title(f"{col} vs original labels — {split}")
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                ax.text(j, i, int(ct.values[i, j]), ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.75)
        _save(fig, f"{col}_vs_original_labels_{split}.png")


# ------------------------------------------------------------
# 6. Evaluation helpers
# ------------------------------------------------------------

def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    target_name: str,
    target_dir: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    pred_train = pd.Series(model.predict(X_train), name="predicted_label")
    pred_test = pd.Series(model.predict(X_test), name="predicted_label")

    metrics = {
        "target": target_name,
        "model": model_name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "n_classes": int(y_train.nunique()),
        "accuracy_train": float(accuracy_score(y_train, pred_train)),
        "accuracy_test": float(accuracy_score(y_test, pred_test)),
        "balanced_accuracy_train": float(balanced_accuracy_score(y_train, pred_train)),
        "balanced_accuracy_test": float(balanced_accuracy_score(y_test, pred_test)),
        "macro_precision_test": float(precision_score(y_test, pred_test, average="macro", zero_division=0)),
        "macro_recall_test": float(recall_score(y_test, pred_test, average="macro", zero_division=0)),
        "macro_f1_train": float(f1_score(y_train, pred_train, average="macro", zero_division=0)),
        "macro_f1_test": float(f1_score(y_test, pred_test, average="macro", zero_division=0)),
        "weighted_f1_test": float(f1_score(y_test, pred_test, average="weighted", zero_division=0)),
    }

    labels = sorted(pd.Series(y_train).unique().tolist())
    cm = confusion_matrix(y_test, pred_test, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(target_dir / f"{model_name}_confusion_matrix_test.csv")

    cm_norm = confusion_matrix(y_test, pred_test, labels=labels, normalize="true")
    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels)
    cm_norm_df.to_csv(target_dir / f"{model_name}_confusion_matrix_test_normalized.csv")

    report = pd.DataFrame(classification_report(y_test, pred_test, zero_division=0, output_dict=True)).T
    report.to_csv(target_dir / f"{model_name}_classification_report_test.csv")

    pred_df = pd.DataFrame({"true_label": y_test.values, "predicted_label": pred_test.values})
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        pred_df["max_predicted_probability"] = probs.max(axis=1)
    pred_df.to_csv(target_dir / f"{model_name}_test_predictions.csv", index=False)

    plot_confusion_matrix(cm_df, f"{model_name} — confusion matrix — {target_name}", f"{model_name}_confusion_matrix_test.png", target_dir)
    plot_confusion_matrix(cm_norm_df, f"{model_name} — normalized confusion matrix — {target_name}", f"{model_name}_confusion_matrix_test_normalized.png", target_dir, decimals=2)
    return metrics, cm_df, report


def plot_confusion_matrix(cm_df: pd.DataFrame, title: str, filename: str, target_dir: Path, decimals: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * cm_df.shape[1]), max(4, 0.45 * cm_df.shape[0])))
    im = ax.imshow(cm_df.values, aspect="auto")
    ax.set_xticks(range(cm_df.shape[1]))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(cm_df.shape[0]))
    ax.set_yticklabels(cm_df.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fmt = f".{{:.{decimals}f}}" if decimals else "{}"
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            value = cm_df.values[i, j]
            text = f"{value:.{decimals}f}" if decimals else str(int(value))
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.75)
    _save(fig, filename, target_dir)


def coefficient_table(model, feature_names: list[str], model_name: str, target_name: str, target_dir: Path) -> pd.DataFrame:
    classes = [str(c) for c in model.classes_]
    coef = pd.DataFrame(model.coef_, index=classes, columns=feature_names)
    long = (
        coef.reset_index(names="class")
        .melt(id_vars="class", var_name="feature", value_name="coefficient")
    )
    long["abs_coefficient"] = long["coefficient"].abs()
    long["model"] = model_name
    long["target"] = target_name
    long = long.sort_values("abs_coefficient", ascending=False)
    long.to_csv(target_dir / f"{model_name}_coefficients.csv", index=False)

    top = long.head(TOP_N_FEATURES).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(8, max(5, 0.22 * len(top))))
    labels = top["class"].astype(str) + " :: " + top["feature"].astype(str)
    ax.barh(labels, top["coefficient"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title(f"Top coefficients — {model_name} — {target_name}")
    _save(fig, f"{model_name}_top_coefficients.png", target_dir)
    return long


def selected_features_from_model(model, feature_names: list[str], model_name: str, target_name: str, target_dir: Path) -> pd.DataFrame:
    coef = np.asarray(model.coef_)
    active_mask = np.any(np.abs(coef) > COEF_ZERO_TOL, axis=0)
    selected = pd.DataFrame({
        "feature": feature_names,
        "is_selected": active_mask,
        "max_abs_coefficient": np.max(np.abs(coef), axis=0),
        "n_nonzero_class_coefficients": np.sum(np.abs(coef) > COEF_ZERO_TOL, axis=0),
    }).sort_values(["is_selected", "max_abs_coefficient"], ascending=[False, False])
    selected.to_csv(target_dir / f"selected_features_{model_name}.csv", index=False)
    _write_json(selected.loc[selected["is_selected"], "feature"].tolist(), target_dir / f"selected_feature_names_{model_name}.json")
    return selected


# ------------------------------------------------------------
# 7. Feature screening
# ------------------------------------------------------------

def run_screening(X_train: pd.DataFrame, y_train: pd.Series, target_name: str, target_dir: Path) -> list[str]:
    f_scores, f_pvalues = f_classif(X_train, y_train)
    anova = pd.DataFrame({
        "feature": X_train.columns,
        "anova_f_score": f_scores,
        "anova_p_value": f_pvalues,
    }).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    anova["anova_rank"] = anova["anova_f_score"].rank(ascending=False, method="min")
    anova = anova.sort_values("anova_f_score", ascending=False)
    anova.to_csv(target_dir / "anova_fscore_ranking.csv", index=False)

    mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    mi = pd.DataFrame({"feature": X_train.columns, "mutual_information": mi_scores})
    mi["mi_rank"] = mi["mutual_information"].rank(ascending=False, method="min")
    mi = mi.sort_values("mutual_information", ascending=False)
    mi.to_csv(target_dir / "mutual_information_ranking.csv", index=False)

    combined = anova[["feature", "anova_f_score", "anova_p_value", "anova_rank"]].merge(
        mi[["feature", "mutual_information", "mi_rank"]], on="feature", how="inner"
    )
    combined["mean_rank"] = combined[["anova_rank", "mi_rank"]].mean(axis=1)
    combined = combined.sort_values("mean_rank")
    n_selected = min(TOP_N_SCREENING, X_train.shape[1], max(1, X_train.shape[0] - 1))
    selected = combined.head(n_selected)["feature"].tolist()

    combined.to_csv(target_dir / "combined_screening_ranking.csv", index=False)
    pd.DataFrame({"feature": selected}).to_csv(target_dir / "selected_screening_features.csv", index=False)
    _write_json(selected, target_dir / "selected_screening_feature_names.json")

    plot_screening_bar(anova.head(TOP_N_FEATURES), "anova_f_score", "ANOVA F-score", "anova_top_features.png", target_dir)
    plot_screening_bar(mi.head(TOP_N_FEATURES), "mutual_information", "Mutual information", "mutual_information_top_features.png", target_dir)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(combined["anova_rank"], combined["mi_rank"], alpha=0.75)
    ax.set_xlabel("ANOVA rank")
    ax.set_ylabel("Mutual information rank")
    ax.set_title(f"Screening rank agreement — {target_name}")
    _save(fig, "screening_rank_scatter.png", target_dir)

    return selected


def plot_screening_bar(df: pd.DataFrame, score_col: str, xlabel: str, filename: str, target_dir: Path) -> None:
    top = df.sort_values(score_col).tail(TOP_N_FEATURES)
    fig, ax = plt.subplots(figsize=(8, max(5, 0.22 * len(top))))
    ax.barh(top["feature"], top[score_col])
    ax.set_xlabel(xlabel)
    ax.set_title(f"Top {len(top)} features — {xlabel}")
    _save(fig, filename, target_dir)


# ------------------------------------------------------------
# 8. Model fitting
# ------------------------------------------------------------

def fit_baseline(X_train: pd.DataFrame, y_train: pd.Series, cv_splits: int, target_dir: Path) -> tuple[LogisticRegression, pd.DataFrame]:
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
    cv_df = pd.DataFrame({"fold": range(1, len(cv_scores) + 1), "macro_f1": cv_scores})
    cv_df.to_csv(target_dir / "baseline_cv_scores.csv", index=False)
    return model, cv_df


def fit_lasso(X_train: pd.DataFrame, y_train: pd.Series, cv_splits: int) -> LogisticRegression:
    """Fit a fixed-C multinomial LASSO model.

    This keeps Phase 4 lightweight while preserving the same model family
    used in Phase 3. The target here can have many cluster classes, so a
    full cross-validated grid is intentionally avoided.
    """
    model = LogisticRegression(
        C=1.0,
        penalty="l1",
        solver="saga",
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def fit_elastic_net(X_train: pd.DataFrame, y_train: pd.Series, cv_splits: int) -> LogisticRegression:
    """Fit a fixed-C multinomial Elastic Net model."""
    model = LogisticRegression(
        C=1.0,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def save_cv_curve(model, model_name: str, target_dir: Path) -> pd.DataFrame:
    """Save a compact tuning summary.

    Phase 4 uses fixed regularization settings for speed and reproducibility
    across cluster targets. This function preserves the same output interface
    used by the Phase 3 scripts.
    """
    row = {
        "model": model_name,
        "C": float(getattr(model, "C", np.nan)),
        "penalty": getattr(model, "penalty", None),
        "solver": getattr(model, "solver", None),
        "l1_ratio": getattr(model, "l1_ratio", np.nan),
    }
    cv_df = pd.DataFrame([row])
    cv_df.to_csv(target_dir / f"{model_name}_cv_scores.csv", index=False)
    return cv_df


def fit_post_selection(
    X_train_screened: pd.DataFrame,
    y_train: pd.Series,
    selected_feature_names: list[str],
) -> tuple[LogisticRegression, list[str]]:
    if not selected_feature_names:
        selected_feature_names = X_train_screened.columns.tolist()
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_screened[selected_feature_names], y_train)
    return model, selected_feature_names


# ------------------------------------------------------------
# 9. Per-target supervised pipeline
# ------------------------------------------------------------

def run_supervised_for_target(
    target_name: str,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    target_dir = OUTPUT_DIR / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    y_train = pd.Series(y_train, name=target_name).astype(str)
    y_test = pd.Series(y_test, name=target_name).astype(str)
    cv_splits = safe_cv_splits(y_train, requested=5)

    metrics_rows = []

    # Dataset and target diagnostics.
    pd.DataFrame({
        "split": ["train", "test"],
        "n_observations": [len(y_train), len(y_test)],
        "n_features": [X_train.shape[1], X_test.shape[1]],
        "n_classes": [y_train.nunique(), y_test.nunique()],
        "cv_splits_used": [cv_splits, cv_splits],
    }).to_csv(target_dir / "target_dataset_summary.csv", index=False)

    # Feature screening.
    selected_screening = run_screening(X_train, y_train, target_name, target_dir)
    X_train_screened = X_train[selected_screening]
    X_test_screened = X_test[selected_screening]

    # Baseline.
    baseline, _ = fit_baseline(X_train_screened, y_train, cv_splits, target_dir)
    m, _, _ = evaluate_model(baseline, X_train_screened, y_train, X_test_screened, y_test, "baseline", target_name, target_dir)
    metrics_rows.append(m)
    coefficient_table(baseline, selected_screening, "baseline", target_name, target_dir)

    # LASSO.
    lasso = fit_lasso(X_train_screened, y_train, cv_splits)
    save_cv_curve(lasso, "lasso", target_dir)
    m, _, _ = evaluate_model(lasso, X_train_screened, y_train, X_test_screened, y_test, "lasso", target_name, target_dir)
    metrics_rows.append(m)
    coefficient_table(lasso, selected_screening, "lasso", target_name, target_dir)
    lasso_selected = selected_features_from_model(lasso, selected_screening, "lasso", target_name, target_dir)

    # Elastic Net.
    elastic_net = fit_elastic_net(X_train_screened, y_train, cv_splits)
    save_cv_curve(elastic_net, "elastic_net", target_dir)
    m, _, _ = evaluate_model(elastic_net, X_train_screened, y_train, X_test_screened, y_test, "elastic_net", target_name, target_dir)
    metrics_rows.append(m)
    coefficient_table(elastic_net, selected_screening, "elastic_net", target_name, target_dir)
    en_selected = selected_features_from_model(elastic_net, selected_screening, "elastic_net", target_name, target_dir)

    # Post-selection model: use Elastic Net selected features if available, otherwise LASSO selected features.
    en_features = en_selected.loc[en_selected["is_selected"], "feature"].tolist()
    lasso_features = lasso_selected.loc[lasso_selected["is_selected"], "feature"].tolist()
    post_features = en_features if en_features else lasso_features
    post_model, post_features = fit_post_selection(X_train_screened, y_train, post_features)
    m, _, _ = evaluate_model(
        post_model,
        X_train_screened[post_features],
        y_train,
        X_test_screened[post_features],
        y_test,
        "post_selection_logistic",
        target_name,
        target_dir,
    )
    m["n_features"] = int(len(post_features))
    metrics_rows.append(m)
    coefficient_table(post_model, post_features, "post_selection_logistic", target_name, target_dir)
    pd.DataFrame({"feature": post_features}).to_csv(target_dir / "post_selection_features.csv", index=False)
    _write_json(post_features, target_dir / "post_selection_feature_names.json")

    # Save models.
    for name, model in [("baseline", baseline), ("lasso", lasso), ("elastic_net", elastic_net), ("post_selection_logistic", post_model)]:
        with (target_dir / f"{name}_model.pkl").open("wb") as f:
            pickle.dump(model, f)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(target_dir / "target_model_metrics.csv", index=False)
    plot_model_comparison(metrics_df, target_name, target_dir)

    best_idx = metrics_df["macro_f1_test"].idxmax()
    best_row = metrics_df.loc[best_idx].to_dict()
    summary = {
        "target": target_name,
        "classes": sorted(y_train.unique().tolist()),
        "cv_splits_used": cv_splits,
        "screened_features": len(selected_screening),
        "best_model_by_test_macro_f1": best_row,
    }
    _write_json(summary, target_dir / f"{target_name}_summary.json")
    return metrics_df, summary


def plot_model_comparison(metrics_df: pd.DataFrame, target_name: str, target_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(metrics_df.shape[0])
    width = 0.35
    ax.bar(x - width / 2, metrics_df["accuracy_test"], width, label="Accuracy")
    ax.bar(x + width / 2, metrics_df["macro_f1_test"], width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["model"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Model comparison — {target_name}")
    ax.legend()
    _save(fig, "target_model_comparison.png", target_dir)


# ------------------------------------------------------------
# 10. Comparison against original-label results
# ------------------------------------------------------------

def load_original_phase3_comparison() -> pd.DataFrame:
    path = first_existing_file(PHASE3_FINAL_COMPARISON_CANDIDATES)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["target_type"] = "original_labels"
    df["target"] = "Thing_Action_Property"
    return df


def build_final_comparison(phase4_metrics: pd.DataFrame) -> pd.DataFrame:
    original = load_original_phase3_comparison()
    cluster = phase4_metrics.copy()
    cluster["target_type"] = "cluster_labels"

    common_cols = [
        "target_type",
        "target",
        "model",
        "n_features",
        "n_classes",
        "accuracy_test",
        "balanced_accuracy_test",
        "macro_precision_test",
        "macro_recall_test",
        "macro_f1_test",
        "weighted_f1_test",
    ]
    frames = []
    if not original.empty:
        for col in common_cols:
            if col not in original.columns:
                original[col] = np.nan
        frames.append(original[common_cols])
    for col in common_cols:
        if col not in cluster.columns:
            cluster[col] = np.nan
    frames.append(cluster[common_cols])
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUTPUT_DIR / "phase4_original_vs_cluster_comparison.csv", index=False)

    plot_original_vs_cluster(out)
    return out


def plot_original_vs_cluster(comparison: pd.DataFrame) -> None:
    # Keep the best model for each target according to macro F1.
    best = comparison.sort_values("macro_f1_test", ascending=False).groupby(["target_type", "target"], as_index=False).head(1)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(best))))
    labels = best["target_type"].astype(str) + " :: " + best["target"].astype(str) + " :: " + best["model"].astype(str)
    y = np.arange(len(best))
    ax.barh(y, best["macro_f1_test"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Test macro F1")
    ax.set_title("Best supervised performance: original labels vs cluster labels")
    _save(fig, "best_original_vs_cluster_macro_f1.png")


# ------------------------------------------------------------
# 11. Main
# ------------------------------------------------------------

def main() -> None:
    print("=== PHASE 4 — SUPERVISED ANALYSIS WITH CLUSTER LABELS ===")

    X_train, y_train_original, X_test, y_test_original, train_meta, test_meta, feature_names, phase3_dir = load_phase3_data()
    n_pcs_cluster, cluster_ks, phase2_summary_path = load_phase2_decisions()

    print(f"Loaded Phase 3.1 data from: {phase3_dir}")
    print(f"Train shape: {X_train.shape}; Test shape: {X_test.shape}")
    print(f"Cluster targets: K-Means k={cluster_ks}; PCA components={n_pcs_cluster}")

    target_specs, train_assign, test_assign, cluster_distribution = build_cluster_targets(
        X_train=X_train,
        X_test=X_test,
        y_train_original=y_train_original,
        y_test_original=y_test_original,
        train_meta=train_meta,
        test_meta=test_meta,
        n_pcs_cluster=n_pcs_cluster,
        cluster_ks=cluster_ks,
    )

    all_metrics = []
    target_summaries = {}
    for target_name, spec in target_specs.items():
        print(f"\nRunning supervised models for {target_name}...")
        metrics_df, summary = run_supervised_for_target(
            target_name=target_name,
            y_train=spec["train_labels"],
            y_test=spec["test_labels"],
            X_train=X_train,
            X_test=X_test,
        )
        all_metrics.append(metrics_df)
        target_summaries[target_name] = summary

    phase4_metrics = pd.concat(all_metrics, ignore_index=True)
    phase4_metrics.to_csv(OUTPUT_DIR / "phase4_model_metrics.csv", index=False)
    comparison = build_final_comparison(phase4_metrics)

    summary = {
        "phase": "PHASE 4 — SUPERVISED ANALYSIS WITH CLUSTER LABELS",
        "phase3_1_input_dir": str(phase3_dir),
        "phase2_summary_file": str(phase2_summary_path) if phase2_summary_path else None,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_pcs_cluster": int(n_pcs_cluster),
        "cluster_targets": cluster_ks,
        "best_models_by_cluster_target": target_summaries,
        "overall_best_cluster_model": phase4_metrics.sort_values("macro_f1_test", ascending=False).iloc[0].to_dict(),
        "outputs": {
            "phase4_model_metrics": str(OUTPUT_DIR / "phase4_model_metrics.csv"),
            "original_vs_cluster_comparison": str(OUTPUT_DIR / "phase4_original_vs_cluster_comparison.csv"),
            "cluster_target_assignments_train": str(OUTPUT_DIR / "cluster_target_assignments_train.csv"),
            "cluster_target_assignments_test": str(OUTPUT_DIR / "cluster_target_assignments_test.csv"),
        },
    }
    _write_json(summary, SUMMARY_FILE)

    print("\nDone.")
    print(f"Outputs written to: {OUTPUT_DIR}")
    print("Best cluster-label model:")
    print(pd.Series(summary["overall_best_cluster_model"]).to_string())


if __name__ == "__main__":
    main()
