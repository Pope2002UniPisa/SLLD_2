from __future__ import annotations

"""
============================================================
PHASE 3.4 — SUPERVISED ANALYSIS: FEATURE SCREENING
Statistical Learning and Large Data Project
Scuola Superiore Sant'Anna di Pisa

Goal:
- Continue the supervised pipeline after Phase 3.3.
- Rank semantic predictors by marginal supervised utility.
- Use conservative screening diagnostics before LASSO / Elastic Net.
- Produce stable tabular outputs, figures and JSON summaries.

Input:
  outputs/III/phase3_1_supervised_data_loading_checks/slld_phase3_1_X_train.csv
  outputs/III/phase3_1_supervised_data_loading_checks/slld_phase3_1_y_train.csv
  outputs/III/phase3_1_supervised_data_loading_checks/slld_phase3_1_X_test.csv
  outputs/III/phase3_1_supervised_data_loading_checks/slld_phase3_1_y_test.csv
  outputs/III/phase3_1_supervised_data_loading_checks/feature_names.json
  outputs/III/phase3_1_supervised_data_loading_checks/class_names.json

Outputs (all under outputs/III/phase3_4_feature_screening/):
  - screening_dataset_summary.csv
  - anova_fscore_ranking.csv
  - mutual_information_ranking.csv
  - combined_screening_ranking.csv
  - selected_screening_features.csv
  - selected_screening_feature_names.json
  - screening_overlap_summary.csv
  - anova_top_features.png
  - mutual_information_top_features.png
  - combined_screening_top_features.png
  - screening_score_scatter.png
  - screening_rank_correlation.png
  - screening_selection_size.png
  - phase3_4_summary.json
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
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

OUTPUT_DIR = Path("./outputs/III/phase3_4_feature_screening")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = OUTPUT_DIR / "phase3_4_summary.json"

PHASE3_1_CANDIDATES = [
    Path("./outputs/III/phase3_1_supervised_data_loading_checks"),
    Path("./phase3_1_supervised_data_loading_checks"),
    Path("./outputs/phase3_1_supervised_data_loading_checks"),
    Path("./package_phase3_1/outputs/III/phase3_1_supervised_data_loading_checks"),
]
PHASE3_3_DIR = Path("./outputs/III/phase3_3_multicollinearity_analysis")


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

LABEL_MAP = {"noun": "Thing", "verb": "Action", "adjective": "Property"}
CLASS_ORDER_DISPLAY = ["Thing", "Action", "Property"]

# Since the current semantic matrix has p=65, screening is mostly a ranking
# and diagnostic step. If later phases expand the feature space, this same
# code automatically keeps a conservative subset of at most 500 predictors.
MAX_SCREENED_FEATURES = 500
TOP_N_PLOT = 25
LOGISTIC_UNIVARIATE_MAX_ITER = 2000


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


def load_json_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise TypeError(f"Expected a JSON list in {path}, found {type(obj)}")
    return obj


def resolve_phase3_1_dir() -> Path:
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
    raise FileNotFoundError("Could not find Phase 3.1 supervised outputs.")


def label_display(y: pd.Series | np.ndarray | list[str]) -> pd.Series:
    s = pd.Series(y)
    return s.map(LABEL_MAP).fillna(s.astype(str))


# ------------------------------------------------------------
# 4. Load data
# ------------------------------------------------------------

def load_phase3_1_outputs() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str], Path]:
    phase3_1_dir = resolve_phase3_1_dir()
    X_train = pd.read_csv(phase3_1_dir / X_TRAIN_FILE_NAME)
    X_test = pd.read_csv(phase3_1_dir / X_TEST_FILE_NAME)
    y_train_df = pd.read_csv(phase3_1_dir / Y_TRAIN_FILE_NAME)
    y_test_df = pd.read_csv(phase3_1_dir / Y_TEST_FILE_NAME)

    if TARGET_COL not in y_train_df.columns or TARGET_COL not in y_test_df.columns:
        raise ValueError(f"The y files must contain '{TARGET_COL}'.")

    y_train = y_train_df[TARGET_COL].copy()
    y_test = y_test_df[TARGET_COL].copy()
    feature_names = load_json_list(phase3_1_dir / FEATURE_NAMES_FILE_NAME)
    class_names = load_json_list(phase3_1_dir / CLASS_NAMES_FILE_NAME)

    X_train = X_train[feature_names]
    X_test = X_test[feature_names]
    return X_train, y_train, X_test, y_test, feature_names, class_names, phase3_1_dir


# ------------------------------------------------------------
# 5. Screening scores
# ------------------------------------------------------------

def compute_anova_scores(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Compute ANOVA F-score for each feature against the multiclass target."""
    scores, pvalues = f_classif(X_train, y_train)
    out = pd.DataFrame({
        "feature": X_train.columns,
        "anova_f_score": scores,
        "anova_p_value": pvalues,
    })
    out["anova_f_score"] = out["anova_f_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["anova_rank"] = out["anova_f_score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["anova_rank", "feature"])


def compute_mutual_information_scores(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Compute model-free marginal mutual information for each predictor."""
    mi = mutual_info_classif(
        X_train,
        y_train,
        discrete_features=False,
        random_state=RANDOM_STATE,
    )
    out = pd.DataFrame({
        "feature": X_train.columns,
        "mutual_information": mi,
    })
    out["mi_rank"] = out["mutual_information"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["mi_rank", "feature"])


def compute_univariate_logistic_scores(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Fit one multinomial logistic model per feature and rank features by
    training negative log-loss. This is used only as an auxiliary marginal
    diagnostic, not as the final model-selection criterion.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    rows = []

    for feature in X_train.columns:
        x = X_train[[feature]].to_numpy()
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=LOGISTIC_UNIVARIATE_MAX_ITER,
            random_state=RANDOM_STATE,
        )
        try:
            model.fit(x, y_enc)
            proba = model.predict_proba(x)
            score = -log_loss(y_enc, proba, labels=np.arange(len(le.classes_)))
        except Exception:
            score = np.nan
        rows.append({"feature": feature, "univariate_logistic_neg_log_loss": score})

    out = pd.DataFrame(rows)
    out["univariate_logistic_neg_log_loss"] = out["univariate_logistic_neg_log_loss"].fillna(out["univariate_logistic_neg_log_loss"].min())
    out["univariate_logistic_rank"] = out["univariate_logistic_neg_log_loss"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["univariate_logistic_rank", "feature"])


def combine_rankings(anova: pd.DataFrame, mi: pd.DataFrame, logistic: pd.DataFrame) -> pd.DataFrame:
    """Create one conservative aggregate screening ranking."""
    combined = anova.merge(mi, on="feature", how="inner").merge(logistic, on="feature", how="inner")

    # Percentile-like normalized scores. Higher is better.
    def normalized_positive(series: pd.Series) -> pd.Series:
        s = series.astype(float)
        if np.isclose(s.max(), s.min()):
            return pd.Series(np.ones(len(s)), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    combined["anova_score_norm"] = normalized_positive(combined["anova_f_score"])
    combined["mi_score_norm"] = normalized_positive(combined["mutual_information"])
    combined["logistic_score_norm"] = normalized_positive(combined["univariate_logistic_neg_log_loss"])

    # ANOVA and MI are the main scores; univariate logistic is included as
    # a tie-breaking supervised marginal check.
    combined["combined_screening_score"] = (
        0.40 * combined["anova_score_norm"]
        + 0.40 * combined["mi_score_norm"]
        + 0.20 * combined["logistic_score_norm"]
    )
    combined["mean_rank"] = combined[["anova_rank", "mi_rank", "univariate_logistic_rank"]].mean(axis=1)
    combined["combined_rank"] = combined["combined_screening_score"].rank(ascending=False, method="min").astype(int)
    combined = combined.sort_values(["combined_rank", "mean_rank", "feature"])
    return combined


# ------------------------------------------------------------
# 6. Outputs and plots
# ------------------------------------------------------------

def build_dataset_summary(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    train_counts = label_display(y_train).value_counts().reindex(CLASS_ORDER_DISPLAY, fill_value=0)
    test_counts = label_display(y_test).value_counts().reindex(CLASS_ORDER_DISPLAY, fill_value=0)
    rows = [
        {"metric": "n_train", "value": int(X_train.shape[0])},
        {"metric": "n_test", "value": int(X_test.shape[0])},
        {"metric": "n_features", "value": int(X_train.shape[1])},
        {"metric": "screening_target_size", "value": int(min(MAX_SCREENED_FEATURES, X_train.shape[1]))},
    ]
    for cls in CLASS_ORDER_DISPLAY:
        rows.append({"metric": f"train_count_{cls}", "value": int(train_counts.loc[cls])})
        rows.append({"metric": f"test_count_{cls}", "value": int(test_counts.loc[cls])})
    return pd.DataFrame(rows)


def plot_top_bar(df: pd.DataFrame, value_col: str, title: str, filename: str, top_n: int = TOP_N_PLOT) -> None:
    top = df.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(5, 0.27 * len(top))))
    ax.barh(top["feature"], top[value_col])
    ax.set_xlabel(value_col)
    ax.set_title(title)
    _save(fig, filename)


def plot_screening_diagnostics(combined: pd.DataFrame, selected: pd.DataFrame) -> None:
    plot_top_bar(combined, "anova_f_score", "Top features by ANOVA F-score", "anova_top_features.png")
    plot_top_bar(combined, "mutual_information", "Top features by mutual information", "mutual_information_top_features.png")
    plot_top_bar(combined, "combined_screening_score", "Top features by combined screening score", "combined_screening_top_features.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(combined["anova_score_norm"], combined["mi_score_norm"], alpha=0.75)
    ax.set_xlabel("ANOVA score, normalized")
    ax.set_ylabel("Mutual information, normalized")
    ax.set_title("Screening score agreement")
    _save(fig, "screening_score_scatter.png")

    rank_cols = ["anova_rank", "mi_rank", "univariate_logistic_rank", "combined_rank"]
    rank_corr = combined[rank_cols].corr(method="spearman")
    rank_corr.to_csv(OUTPUT_DIR / "screening_rank_spearman_correlation.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rank_corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(rank_cols)))
    ax.set_yticks(range(len(rank_cols)))
    ax.set_xticklabels(rank_cols, rotation=45, ha="right")
    ax.set_yticklabels(rank_cols)
    ax.set_title("Spearman correlation among screening ranks")
    for i in range(len(rank_cols)):
        for j in range(len(rank_cols)):
            ax.text(j, i, f"{rank_corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "screening_rank_correlation.png")

    sizes = [10, 20, 30, 40, 50, min(100, len(combined)), len(selected)]
    sizes = sorted(set(s for s in sizes if s <= len(combined)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(s) for s in sizes], sizes)
    ax.set_xlabel("Candidate subset size")
    ax.set_ylabel("Number of retained features")
    ax.set_title("Screening selection sizes")
    _save(fig, "screening_selection_size.png")


def build_overlap_summary(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in [10, 20, 30, 50, 100, min(MAX_SCREENED_FEATURES, len(combined))]:
        if k > len(combined):
            continue
        a = set(combined.nsmallest(k, "anova_rank")["feature"])
        m = set(combined.nsmallest(k, "mi_rank")["feature"])
        l = set(combined.nsmallest(k, "univariate_logistic_rank")["feature"])
        c = set(combined.nsmallest(k, "combined_rank")["feature"])
        rows.append({
            "top_k": int(k),
            "anova_mi_overlap": len(a & m),
            "anova_logistic_overlap": len(a & l),
            "mi_logistic_overlap": len(m & l),
            "all_three_overlap": len(a & m & l),
            "combined_vs_anova_overlap": len(c & a),
            "combined_vs_mi_overlap": len(c & m),
            "combined_vs_logistic_overlap": len(c & l),
        })
    return pd.DataFrame(rows).drop_duplicates("top_k")


# ------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------

def main() -> None:
    X_train, y_train, X_test, y_test, feature_names, class_names, phase3_1_dir = load_phase3_1_outputs()

    dataset_summary = build_dataset_summary(X_train, y_train, X_test, y_test)
    dataset_summary.to_csv(OUTPUT_DIR / "screening_dataset_summary.csv", index=False)

    anova = compute_anova_scores(X_train, y_train)
    mi = compute_mutual_information_scores(X_train, y_train)
    logistic = compute_univariate_logistic_scores(X_train, y_train)
    combined = combine_rankings(anova, mi, logistic)

    selected_size = int(min(MAX_SCREENED_FEATURES, X_train.shape[1]))
    selected = combined.nsmallest(selected_size, "combined_rank").copy()
    selected["selected_by_screening"] = True

    anova.to_csv(OUTPUT_DIR / "anova_fscore_ranking.csv", index=False)
    mi.to_csv(OUTPUT_DIR / "mutual_information_ranking.csv", index=False)
    logistic.to_csv(OUTPUT_DIR / "univariate_logistic_ranking.csv", index=False)
    combined.to_csv(OUTPUT_DIR / "combined_screening_ranking.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "selected_screening_features.csv", index=False)
    _write_json(selected["feature"].tolist(), OUTPUT_DIR / "selected_screening_feature_names.json")

    overlap = build_overlap_summary(combined)
    overlap.to_csv(OUTPUT_DIR / "screening_overlap_summary.csv", index=False)
    plot_screening_diagnostics(combined, selected)

    summary = {
        "phase": "3.4_feature_screening",
        "input_phase3_1_dir": str(phase3_1_dir),
        "input_phase3_3_dir_exists": PHASE3_3_DIR.exists(),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features_before_screening": int(X_train.shape[1]),
        "n_features_after_screening": int(selected.shape[0]),
        "screening_is_reduction": bool(selected.shape[0] < X_train.shape[1]),
        "top_10_combined_features": selected["feature"].head(10).tolist(),
        "class_names": class_names,
        "outputs_dir": str(OUTPUT_DIR),
    }
    _write_json(summary, SUMMARY_FILE)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
