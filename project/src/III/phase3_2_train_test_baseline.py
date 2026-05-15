from __future__ import annotations

"""
============================================================
PHASE 3.2 — SUPERVISED ANALYSIS: TRAIN/TEST BASELINE
Statistical Learning and Large Data Project
Scuola Superiore Sant'Anna di Pisa

Goal:
- Continue the supervised analysis from Phase 3.1 outputs.
- Load X_train, y_train, X_test and y_test produced in Phase 3.1.
- Verify that the train/test split is usable for multiclass supervised
  learning and that the feature matrices are aligned.
- Fit a first simple baseline classifier: multinomial logistic regression,
  i.e. a generalized linear model for multiclass labels.
- Evaluate the initial performance on the training set and on the held-out
  test set.
- Produce diagnostic tables and figures useful as reference for the later
  multicollinearity, screening, LASSO and Elastic Net phases.

Input:
  Preferred:
    outputs/III/phase3_1_supervised_data_loading_checks/slld_phase3_1_X_train.csv
    outputs/III/phase3_1_supervised_data_loading_checks/slld_phase3_1_y_train.csv
    outputs/III/phase3_1_supervised_data_loading_checks/slld_phase3_1_X_test.csv
    outputs/III/phase3_1_supervised_data_loading_checks/slld_phase3_1_y_test.csv
    outputs/III/phase3_1_supervised_data_loading_checks/feature_names.json
    outputs/III/phase3_1_supervised_data_loading_checks/class_names.json

  Accepted fallback locations:
    phase3_1_supervised_data_loading_checks/
    outputs/III/phase3_1_supervised_data_loading_checks/

Outputs (all under outputs/III/phase3_2_train_test_baseline/):
  - baseline_train_predictions.csv
  - baseline_test_predictions.csv
  - baseline_metrics_train_test.csv
  - baseline_classification_report.csv
  - baseline_confusion_matrix_train.csv
  - baseline_confusion_matrix_test.csv
  - baseline_confusion_matrix_train_normalized.csv
  - baseline_confusion_matrix_test_normalized.csv
  - baseline_cv_scores.csv
  - baseline_coefficients.csv
  - baseline_top_coefficients_by_class.csv
  - baseline_model.pkl
  - train_test_shape_check.csv
  - train_test_class_distribution_check.csv
  - feature_alignment_check.csv
  - feature_standardization_check.csv
  - baseline_confusion_matrix_test.png
  - baseline_confusion_matrix_test_normalized.png
  - baseline_class_metrics_test.png
  - baseline_cv_macro_f1.png
  - baseline_top_coefficients_by_class.png
  - baseline_prediction_confidence_test.png
  - phase3_2_summary.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

OUTPUT_DIR = Path("./outputs/III/phase3_2_train_test_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "phase3_2_summary.json"

PHASE3_1_CANDIDATES = [
    Path("./outputs/III/phase3_1_supervised_data_loading_checks"),
    Path("./phase3_1_supervised_data_loading_checks"),
    Path("./outputs/phase3_1_supervised_data_loading_checks"),
]


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

TARGET_COL = "target_word_class"
TARGET_DISPLAY_COL = "target_display"
RANDOM_STATE = 42

X_TRAIN_FILE_NAME = "slld_phase3_1_X_train.csv"
Y_TRAIN_FILE_NAME = "slld_phase3_1_y_train.csv"
X_TEST_FILE_NAME = "slld_phase3_1_X_test.csv"
Y_TEST_FILE_NAME = "slld_phase3_1_y_test.csv"
FEATURE_NAMES_FILE_NAME = "feature_names.json"
CLASS_NAMES_FILE_NAME = "class_names.json"

# Raw labels are the actual labels stored in the data files.
# Display labels are used only in plots and human-readable reports.
LABEL_MAP = {"noun": "Thing", "verb": "Action", "adjective": "Property"}
DISPLAY_TO_RAW = {v: k for k, v in LABEL_MAP.items()}
CLASS_ORDER_DISPLAY = ["Thing", "Action", "Property"]
CLASS_ORDER_RAW = [DISPLAY_TO_RAW[c] for c in CLASS_ORDER_DISPLAY]

COLOR_MAP = {"Thing": "#e07b39", "Action": "#4c7cba", "Property": "#5aa15a"}

# Cross-validation is used only to obtain a baseline stability diagnostic.
# The held-out test set remains the main out-of-sample evaluation for this phase.
N_SPLITS_CV = 5


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


# ------------------------------------------------------------
# 4. Load data
# ------------------------------------------------------------

def load_phase3_1_outputs() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str], Path]:
    """
    Load the train/test matrices and labels produced by Phase 3.1.

    y files contain both the raw class label and the display label. The model
    is fitted on raw labels to preserve exact compatibility with the dataset;
    display labels are used only when saving reports and figures.
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

def check_feature_alignment(X_train: pd.DataFrame, X_test: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Verify that train/test features match the Phase 3.1 feature list."""
    train_cols = X_train.columns.tolist()
    test_cols = X_test.columns.tolist()

    report = pd.DataFrame([
        {
            "check": "train_columns_match_feature_names",
            "passed": bool(train_cols == feature_names),
            "details": "Train columns exactly match feature_names.json" if train_cols == feature_names else "Mismatch detected",
        },
        {
            "check": "test_columns_match_feature_names",
            "passed": bool(test_cols == feature_names),
            "details": "Test columns exactly match feature_names.json" if test_cols == feature_names else "Mismatch detected",
        },
        {
            "check": "train_test_columns_same_order",
            "passed": bool(train_cols == test_cols),
            "details": "Train and test columns are aligned" if train_cols == test_cols else "Train/test column mismatch detected",
        },
    ])

    report.to_csv(OUTPUT_DIR / "feature_alignment_check.csv", index=False)

    if not bool(report["passed"].all()):
        train_only = sorted(set(train_cols) - set(test_cols))
        test_only = sorted(set(test_cols) - set(train_cols))
        feature_list_only = sorted(set(feature_names) - set(train_cols))
        raise ValueError(
            "Feature alignment check failed. "
            f"Train-only: {train_only}; test-only: {test_only}; "
            f"feature_names not in train: {feature_list_only}"
        )

    return report


def check_shapes_and_classes(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check sample sizes, feature counts and class availability."""
    shape_check = pd.DataFrame([
        {
            "split": "train",
            "n_observations_X": int(X_train.shape[0]),
            "n_observations_y": int(y_train.shape[0]),
            "n_features": int(X_train.shape[1]),
            "X_y_same_length": bool(X_train.shape[0] == y_train.shape[0]),
        },
        {
            "split": "test",
            "n_observations_X": int(X_test.shape[0]),
            "n_observations_y": int(y_test.shape[0]),
            "n_features": int(X_test.shape[1]),
            "X_y_same_length": bool(X_test.shape[0] == y_test.shape[0]),
        },
    ])
    shape_check.to_csv(OUTPUT_DIR / "train_test_shape_check.csv", index=False)

    class_rows = []
    for split_name, y in [("train", y_train), ("test", y_test)]:
        counts = y.value_counts().reindex(CLASS_ORDER_RAW, fill_value=0)
        total = int(counts.sum())
        for raw_label, count in counts.items():
            class_rows.append({
                "split": split_name,
                "class_raw": raw_label,
                "class_display": LABEL_MAP.get(raw_label, raw_label),
                "count": int(count),
                "proportion": float(count / total) if total > 0 else np.nan,
            })
    class_check = pd.DataFrame(class_rows)
    class_check.to_csv(OUTPUT_DIR / "train_test_class_distribution_check.csv", index=False)

    if not bool(shape_check["X_y_same_length"].all()):
        raise ValueError("X and y do not have the same number of observations in at least one split.")

    missing_classes = class_check[class_check["count"] == 0]
    if not missing_classes.empty:
        raise ValueError(
            "At least one class is missing in train or test. "
            f"Missing rows:\n{missing_classes}"
        )

    return shape_check, class_check


def check_feature_standardization(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise feature means and standard deviations.

    Phase 1.3 already performed imputation and scaling. This diagnostic does
    not rescale anything; it only verifies that the matrices entering the
    supervised baseline have the expected numerical structure.
    """
    rows = []
    for split_name, X in [("train", X_train), ("test", X_test)]:
        means = X.mean(axis=0)
        stds = X.std(axis=0, ddof=0)
        rows.append({
            "split": split_name,
            "max_abs_feature_mean": float(means.abs().max()),
            "mean_abs_feature_mean": float(means.abs().mean()),
            "min_feature_std": float(stds.min()),
            "max_feature_std": float(stds.max()),
            "mean_feature_std": float(stds.mean()),
            "n_zero_variance_features": int((stds == 0).sum()),
        })

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT_DIR / "feature_standardization_check.csv", index=False)
    return report


# ------------------------------------------------------------
# 6. Baseline model
# ------------------------------------------------------------

def make_baseline_model() -> LogisticRegression:
    """
    Build the multinomial logistic baseline.

    We first try an unpenalized multinomial fit, because this corresponds most
    directly to a simple GLM baseline. If the installed scikit-learn version
    requires the older syntax for no penalty, the fitting function below handles
    the fallback automatically.
    """
    return LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )


def fit_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[LogisticRegression, str]:
    """Fit the baseline model, with compatibility fallback for older sklearn versions."""
    try:
        model = make_baseline_model()
        model.fit(X_train, y_train)
        return model, "unpenalized_multinomial_logistic_regression"
    except Exception as first_error:
        try:
            # Older scikit-learn versions used penalty="none" instead of penalty=None.
            model = LogisticRegression(
                penalty="none",
                        solver="lbfgs",
                max_iter=5000,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)
            return model, "unpenalized_multinomial_logistic_regression_penalty_none_syntax"
        except Exception:
            # Last-resort fallback: a very weak L2 penalty stabilizes the fit while
            # remaining close to the unpenalized baseline.
            model = LogisticRegression(
                penalty="l2",
                C=1e6,
                        solver="lbfgs",
                max_iter=5000,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)
            return model, f"weak_l2_multinomial_logistic_regression_fallback_after_error: {first_error}"


def predict_with_probabilities(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
) -> pd.DataFrame:
    """Generate predicted labels and class probabilities."""
    pred = model.predict(X)
    proba = model.predict_proba(X)
    classes = model.classes_.tolist()

    out = pd.DataFrame({
        "split": split_name,
        "y_true": y.values,
        "y_true_display": _label_display(y).values,
        "y_pred": pred,
        "y_pred_display": _label_display(pred).values,
        "correct": pred == y.values,
        "max_predicted_probability": proba.max(axis=1),
    })

    for i, cls in enumerate(classes):
        display = LABEL_MAP.get(cls, cls)
        out[f"prob_{display}"] = proba[:, i]

    return out


# ------------------------------------------------------------
# 7. Metrics and reports
# ------------------------------------------------------------

def compute_split_metrics(y_true: pd.Series, y_pred: np.ndarray, split_name: str) -> dict:
    """Compute global metrics for one split."""
    return {
        "split": split_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=CLASS_ORDER_RAW, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=CLASS_ORDER_RAW, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER_RAW, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=CLASS_ORDER_RAW, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=CLASS_ORDER_RAW, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER_RAW, average="weighted", zero_division=0)),
    }


def save_metrics_and_reports(
    y_train: pd.Series,
    train_pred: np.ndarray,
    y_test: pd.Series,
    test_pred: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Save global metrics, per-class reports and confusion matrices."""
    metrics = pd.DataFrame([
        compute_split_metrics(y_train, train_pred, "train"),
        compute_split_metrics(y_test, test_pred, "test"),
    ])
    metrics.to_csv(OUTPUT_DIR / "baseline_metrics_train_test.csv", index=False)

    report_rows = []
    for split_name, y_true, y_pred in [("train", y_train, train_pred), ("test", y_test, test_pred)]:
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=CLASS_ORDER_RAW,
            target_names=CLASS_ORDER_DISPLAY,
            output_dict=True,
            zero_division=0,
        )
        for label_name, values in report_dict.items():
            if isinstance(values, dict):
                row = {"split": split_name, "label": label_name}
                row.update(values)
                report_rows.append(row)
            else:
                report_rows.append({"split": split_name, "label": label_name, "score": values})
    class_report = pd.DataFrame(report_rows)
    class_report.to_csv(OUTPUT_DIR / "baseline_classification_report.csv", index=False)

    cm_train = pd.DataFrame(
        confusion_matrix(y_train, train_pred, labels=CLASS_ORDER_RAW),
        index=CLASS_ORDER_DISPLAY,
        columns=CLASS_ORDER_DISPLAY,
    )
    cm_test = pd.DataFrame(
        confusion_matrix(y_test, test_pred, labels=CLASS_ORDER_RAW),
        index=CLASS_ORDER_DISPLAY,
        columns=CLASS_ORDER_DISPLAY,
    )

    cm_train_norm = cm_train.div(cm_train.sum(axis=1).replace(0, np.nan), axis=0)
    cm_test_norm = cm_test.div(cm_test.sum(axis=1).replace(0, np.nan), axis=0)

    cm_train.to_csv(OUTPUT_DIR / "baseline_confusion_matrix_train.csv")
    cm_test.to_csv(OUTPUT_DIR / "baseline_confusion_matrix_test.csv")
    cm_train_norm.to_csv(OUTPUT_DIR / "baseline_confusion_matrix_train_normalized.csv")
    cm_test_norm.to_csv(OUTPUT_DIR / "baseline_confusion_matrix_test_normalized.csv")

    return metrics, class_report, cm_train, cm_test


def run_cross_validation(X_train: pd.DataFrame, y_train: pd.Series, model_kind: str) -> pd.DataFrame:
    """Run stratified cross-validation on the training set for baseline stability."""
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    # Rebuild a fresh estimator for cross-validation. If the selected baseline
    # relied on the weak-L2 fallback, use the same stabilized specification.
    if model_kind.startswith("weak_l2"):
        estimator = LogisticRegression(
            penalty="l2",
            C=1e6,
                solver="lbfgs",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
    else:
        try:
            estimator = make_baseline_model()
        except Exception:
            estimator = LogisticRegression(
                penalty="none",
                        solver="lbfgs",
                max_iter=5000,
                random_state=RANDOM_STATE,
            )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
    }

    try:
        cv_result = cross_validate(estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=None)
    except Exception:
        # Cross-validation has the same compatibility issue as fitting. Retry
        # with the older penalty="none" syntax if needed.
        estimator = LogisticRegression(
            penalty="none",
                solver="lbfgs",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
        cv_result = cross_validate(estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=None)

    rows = []
    for fold_idx in range(N_SPLITS_CV):
        rows.append({
            "fold": fold_idx + 1,
            "accuracy": float(cv_result["test_accuracy"][fold_idx]),
            "balanced_accuracy": float(cv_result["test_balanced_accuracy"][fold_idx]),
            "macro_f1": float(cv_result["test_macro_f1"][fold_idx]),
            "weighted_f1": float(cv_result["test_weighted_f1"][fold_idx]),
        })

    cv_scores = pd.DataFrame(rows)
    cv_scores.to_csv(OUTPUT_DIR / "baseline_cv_scores.csv", index=False)
    return cv_scores


def save_coefficients(model: LogisticRegression, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save all multinomial coefficients and the strongest coefficients per class."""
    coef = pd.DataFrame(model.coef_, index=model.classes_, columns=feature_names)
    coef.insert(0, "class_display", _label_display(coef.index).values)
    coef.to_csv(OUTPUT_DIR / "baseline_coefficients.csv", index_label="class_raw")

    top_rows = []
    for class_raw in model.classes_:
        display = LABEL_MAP.get(class_raw, class_raw)
        class_coef = coef.loc[class_raw, feature_names].astype(float)

        top_positive = class_coef.sort_values(ascending=False).head(10)
        top_negative = class_coef.sort_values(ascending=True).head(10)

        for rank, (feature, value) in enumerate(top_positive.items(), start=1):
            top_rows.append({
                "class_raw": class_raw,
                "class_display": display,
                "direction": "positive",
                "rank": rank,
                "feature": feature,
                "coefficient": float(value),
                "abs_coefficient": float(abs(value)),
            })
        for rank, (feature, value) in enumerate(top_negative.items(), start=1):
            top_rows.append({
                "class_raw": class_raw,
                "class_display": display,
                "direction": "negative",
                "rank": rank,
                "feature": feature,
                "coefficient": float(value),
                "abs_coefficient": float(abs(value)),
            })

    top_coef = pd.DataFrame(top_rows)
    top_coef.to_csv(OUTPUT_DIR / "baseline_top_coefficients_by_class.csv", index=False)
    return coef, top_coef


# ------------------------------------------------------------
# 8. Plots
# ------------------------------------------------------------

def plot_confusion_matrix(cm: pd.DataFrame, filename: str, title: str, normalized: bool = False) -> None:
    """Plot a confusion matrix as a heatmap without external plotting libraries."""
    values = cm.values.astype(float)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(values, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(CLASS_ORDER_DISPLAY)),
        yticks=np.arange(len(CLASS_ORDER_DISPLAY)),
        xticklabels=CLASS_ORDER_DISPLAY,
        yticklabels=CLASS_ORDER_DISPLAY,
        ylabel="True class",
        xlabel="Predicted class",
        title=title,
    )

    threshold = np.nanmax(values) / 2 if values.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if normalized:
                text = f"{values[i, j]:.2f}"
            else:
                text = str(int(values[i, j]))
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if values[i, j] > threshold else "black",
            )

    fig.tight_layout()
    _save(fig, filename)


def plot_test_class_metrics(class_report: pd.DataFrame) -> None:
    """Plot precision, recall and F1-score for the test split."""
    plot_df = class_report[
        (class_report["split"] == "test") &
        (class_report["label"].isin(CLASS_ORDER_DISPLAY))
    ].copy()

    x = np.arange(len(CLASS_ORDER_DISPLAY))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, plot_df["precision"], width, label="precision", color="#4c7cba", alpha=0.85)
    ax.bar(x, plot_df["recall"], width, label="recall", color="#e07b39", alpha=0.85)
    ax.bar(x + width, plot_df["f1-score"], width, label="F1-score", color="#5aa15a", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER_DISPLAY)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Baseline Multinomial Logistic Regression — Test Class Metrics")
    ax.legend()
    _save(fig, "baseline_class_metrics_test.png")


def plot_cv_macro_f1(cv_scores: pd.DataFrame) -> None:
    """Plot macro F1 across CV folds."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cv_scores["fold"], cv_scores["macro_f1"], marker="o", color="#4c7cba")
    ax.axhline(cv_scores["macro_f1"].mean(), color="#e07b39", linestyle="--", label=f"mean = {cv_scores['macro_f1'].mean():.3f}")
    ax.set_xticks(cv_scores["fold"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("CV fold")
    ax.set_ylabel("Macro F1")
    ax.set_title("Baseline Cross-Validation — Macro F1")
    ax.legend()
    _save(fig, "baseline_cv_macro_f1.png")


def plot_top_coefficients(top_coef: pd.DataFrame) -> None:
    """Plot strongest absolute coefficients for each class."""
    # Keep the most influential coefficients by absolute value per class.
    plot_df = (
        top_coef.sort_values(["class_display", "abs_coefficient"], ascending=[True, False])
        .groupby("class_display", group_keys=False)
        .head(8)
        .copy()
    )
    plot_df["label"] = plot_df["class_display"] + " — " + plot_df["feature"]
    plot_df = plot_df.sort_values("coefficient")

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = [COLOR_MAP.get(cls, "#4c7cba") for cls in plot_df["class_display"]]
    ax.barh(plot_df["label"], plot_df["coefficient"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title("Baseline — Strongest Multinomial Coefficients by Class")
    _save(fig, "baseline_top_coefficients_by_class.png")


def plot_prediction_confidence(test_predictions: pd.DataFrame) -> None:
    """Plot the distribution of the maximum predicted probability on the test set."""
    fig, ax = plt.subplots(figsize=(7, 4))
    correct = test_predictions[test_predictions["correct"]]
    wrong = test_predictions[~test_predictions["correct"]]

    bins = np.linspace(0, 1, 16)
    ax.hist(correct["max_predicted_probability"], bins=bins, alpha=0.65, label="correct", color="#5aa15a")
    ax.hist(wrong["max_predicted_probability"], bins=bins, alpha=0.65, label="wrong", color="#e07b39")
    ax.set_xlabel("Maximum predicted class probability")
    ax.set_ylabel("Number of observations")
    ax.set_title("Baseline — Prediction Confidence on Test Set")
    ax.legend()
    _save(fig, "baseline_prediction_confidence_test.png")


# ------------------------------------------------------------
# 9. Save model
# ------------------------------------------------------------

def save_model(model: LogisticRegression) -> None:
    """Persist the fitted baseline model with pickle."""
    with (OUTPUT_DIR / "baseline_model.pkl").open("wb") as f:
        pickle.dump(model, f)


# ------------------------------------------------------------
# 10. Main
# ------------------------------------------------------------

def main() -> None:
    X_train, y_train, X_test, y_test, feature_names, class_names, phase3_1_dir = load_phase3_1_outputs()

    feature_alignment = check_feature_alignment(X_train, X_test, feature_names)
    shape_check, class_check = check_shapes_and_classes(X_train, y_train, X_test, y_test)
    standardization_check = check_feature_standardization(X_train, X_test)

    model, model_kind = fit_baseline_model(X_train, y_train)

    train_predictions = predict_with_probabilities(model, X_train, y_train, "train")
    test_predictions = predict_with_probabilities(model, X_test, y_test, "test")
    train_predictions.to_csv(OUTPUT_DIR / "baseline_train_predictions.csv", index=False)
    test_predictions.to_csv(OUTPUT_DIR / "baseline_test_predictions.csv", index=False)

    metrics, class_report, cm_train, cm_test = save_metrics_and_reports(
        y_train=y_train,
        train_pred=train_predictions["y_pred"].values,
        y_test=y_test,
        test_pred=test_predictions["y_pred"].values,
    )

    cv_scores = run_cross_validation(X_train, y_train, model_kind)
    coefficients, top_coefficients = save_coefficients(model, feature_names)
    save_model(model)

    cm_test_norm = cm_test.div(cm_test.sum(axis=1).replace(0, np.nan), axis=0)
    plot_confusion_matrix(cm_test, "baseline_confusion_matrix_test.png", "Baseline — Test Confusion Matrix", normalized=False)
    plot_confusion_matrix(cm_test_norm, "baseline_confusion_matrix_test_normalized.png", "Baseline — Test Confusion Matrix Normalized", normalized=True)
    plot_test_class_metrics(class_report)
    plot_cv_macro_f1(cv_scores)
    plot_top_coefficients(top_coefficients)
    plot_prediction_confidence(test_predictions)

    test_metrics = metrics[metrics["split"] == "test"].iloc[0].to_dict()
    train_metrics = metrics[metrics["split"] == "train"].iloc[0].to_dict()

    summary = {
        "phase": "PHASE 3.2 — SUPERVISED ANALYSIS: TRAIN/TEST BASELINE",
        "input_dir": str(phase3_1_dir),
        "output_dir": str(OUTPUT_DIR),
        "model_kind": model_kind,
        "model": "multinomial_logistic_regression",
        "target_col": TARGET_COL,
        "class_names": class_names,
        "class_order_raw": CLASS_ORDER_RAW,
        "class_order_display": CLASS_ORDER_DISPLAY,
        "n_train_observations": int(X_train.shape[0]),
        "n_test_observations": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "train_accuracy": float(train_metrics["accuracy"]),
        "train_macro_f1": float(train_metrics["macro_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "cv_macro_f1_mean": float(cv_scores["macro_f1"].mean()),
        "cv_macro_f1_std": float(cv_scores["macro_f1"].std(ddof=1)),
        "feature_alignment_passed": bool(feature_alignment["passed"].all()),
        "n_zero_variance_features_train": int(standardization_check.loc[standardization_check["split"] == "train", "n_zero_variance_features"].iloc[0]),
        "outputs": {
            "baseline_metrics_train_test": str(OUTPUT_DIR / "baseline_metrics_train_test.csv"),
            "baseline_classification_report": str(OUTPUT_DIR / "baseline_classification_report.csv"),
            "baseline_confusion_matrix_test": str(OUTPUT_DIR / "baseline_confusion_matrix_test.csv"),
            "baseline_cv_scores": str(OUTPUT_DIR / "baseline_cv_scores.csv"),
            "baseline_coefficients": str(OUTPUT_DIR / "baseline_coefficients.csv"),
            "baseline_model": str(OUTPUT_DIR / "baseline_model.pkl"),
        },
    }
    _write_json(summary, SUMMARY_FILE)

    print("PHASE 3.2 completed successfully.")
    print(f"Input directory: {phase3_1_dir}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model kind: {model_kind}")
    print(f"Train accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
