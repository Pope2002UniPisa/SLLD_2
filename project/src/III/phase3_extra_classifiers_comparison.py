#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3-extra — Alternative classifiers comparison on selected semantic features
================================================================================

This script extends the supervised analysis pipeline by testing additional
classifiers on the same selected feature space used by the final post-selection
logistic model.

Goal
----
The original Phase 3 final run used a post-selection multinomial logistic
regression: LASSO / Elastic Net selected the relevant semantic features, then a
final logistic classifier was fitted on the selected terms. This extra phase asks
whether non-linear or non-parametric classifiers improve over that final linear
classifier.

Models tested
-------------
1. Post-selection multinomial Logistic Regression
2. Linear SVM
3. RBF-kernel SVM
4. Random Forest
5. k-Nearest Neighbors

Input files
-----------
The script expects the scaled train/test files produced in Phase 1.3:

    slld_phase1_3_train_scaled.csv
    slld_phase1_3_test_scaled.csv

It also tries to reuse the final selected features from Phase 3.7:

    outputs/III/phase3_7_final_model_evaluation/final_selected_feature_names.json

If that file is not found, the script falls back to all numeric semantic features.

Main outputs
------------
- alternative_classifiers_metrics.csv
- alternative_classifiers_cv_results.csv
- alternative_classifiers_best_params.csv
- alternative_classifiers_classification_report_test.csv
- one confusion matrix per classifier, CSV and PNG
- accuracy / macro-F1 comparison plots
- prediction files for each classifier
- fitted model pickle files
- phase3_extra_summary.json

Notes on preprocessing
----------------------
No new scaling is fitted here. The script uses the already scaled train/test
sets, so the comparison is focused on the classifier and not on preprocessing.
For Random Forest, scaling is not required, but using the same input matrix keeps
the comparison controlled and consistent with the rest of the pipeline.
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# Configuration
# =============================================================================

RANDOM_STATE = 42
N_JOBS = 1
CV_FOLDS = 3
SCORING = "f1_macro"

ID_COLUMNS = ["entry_id", "word", "target_word_class"]
TARGET_COLUMN = "target_word_class"

BASE_DIR = Path(__file__).resolve().parents[1] if "generated" in Path(__file__).parts else Path.cwd()
DATA_DIR = BASE_DIR

TRAIN_FILE = DATA_DIR / "slld_phase1_3_train_scaled.csv"
TEST_FILE = DATA_DIR / "slld_phase1_3_test_scaled.csv"

# Preferred location if the previous phase output is available in the project tree.
SELECTED_FEATURES_CANDIDATES = [
    BASE_DIR / "outputs" / "III" / "phase3_7_final_model_evaluation" / "final_selected_feature_names.json",
    BASE_DIR / "inspect_phase3" / "outputs" / "III" / "phase3_7_final_model_evaluation" / "final_selected_feature_names.json",
    BASE_DIR / "phasework" / "outputs" / "III" / "phase3_7_final_model_evaluation" / "final_selected_feature_names.json",
]

OUTPUT_DIR = BASE_DIR / "outputs" / "III" / "phase3_extra_classifiers_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Utility functions
# =============================================================================


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with explicit error reporting."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return pd.read_csv(path)



def load_selected_features(train_df: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Load the final selected feature list from Phase 3.7.

    If the file is unavailable, fall back to all numeric predictors that are not
    identifier or target columns. This fallback makes the script executable even
    when it is moved outside the original output folder.
    """
    for candidate in SELECTED_FEATURES_CANDIDATES:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                feature_names = json.load(f)
            feature_names = [f for f in feature_names if f in train_df.columns]
            if len(feature_names) == 0:
                raise ValueError(
                    f"Selected feature file was found at {candidate}, but none "
                    "of the listed features exist in the training data."
                )
            return feature_names, str(candidate)

    numeric_features = [
        col
        for col in train_df.columns
        if col not in ID_COLUMNS and pd.api.types.is_numeric_dtype(train_df[col])
    ]
    return numeric_features, "fallback_all_numeric_features"



def make_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract X/y matrices from train and test dataframes."""
    missing_train = sorted(set(feature_names) - set(train_df.columns))
    missing_test = sorted(set(feature_names) - set(test_df.columns))
    if missing_train or missing_test:
        raise ValueError(
            "Some selected features are missing from the data. "
            f"Missing train: {missing_train}; missing test: {missing_test}"
        )

    X_train = train_df[feature_names].to_numpy(dtype=float)
    X_test = test_df[feature_names].to_numpy(dtype=float)
    y_train = train_df[TARGET_COLUMN].to_numpy()
    y_test = test_df[TARGET_COLUMN].to_numpy()
    return X_train, X_test, y_train, y_test



def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str,
) -> Dict[str, Any]:
    """Compute standard classification metrics for a prediction vector."""
    return {
        "model": model_name,
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }



def save_confusion_matrix(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> None:
    """Save confusion matrix as CSV and PNG."""
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(OUTPUT_DIR / f"{safe_name}_confusion_matrix_test.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm)
    ax.set_title(f"Test confusion matrix — {model_name}")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{safe_name}_confusion_matrix_test.png", dpi=300)
    plt.close(fig)



def save_classification_report(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Return a classification report in long dataframe format."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"model": model_name, "label": label}
            row.update(values)
            rows.append(row)
        else:
            rows.append({"model": model_name, "label": label, "score": values})
    return pd.DataFrame(rows)



def save_predictions(
    model_name: str,
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    estimator: Any,
) -> None:
    """Save test-set predictions and, when available, class probabilities."""
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    pred_df = test_df[["entry_id", "word", TARGET_COLUMN]].copy()
    pred_df["predicted_class"] = y_pred
    pred_df["correct"] = pred_df[TARGET_COLUMN] == pred_df["predicted_class"]

    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(test_df[feature_names].to_numpy(dtype=float))
            classes = list(estimator.classes_)
            for idx, cls in enumerate(classes):
                pred_df[f"prob_{cls}"] = proba[:, idx]
        except Exception:
            pass

    pred_df.to_csv(OUTPUT_DIR / f"{safe_name}_test_predictions.csv", index=False)



def build_model_grid() -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
    """
    Define all classifiers and hyperparameter grids.

    The grids are intentionally compact. The dataset is small enough for proper
    cross-validation, but not large enough to justify an oversized search space.
    """
    return {
        "Dummy majority": (
            DummyClassifier(strategy="most_frequent"),
            {},
        ),
        "Post-selection Logistic": (
            LogisticRegression(
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            {
                "C": [1.0],
            },
        ),
        "SVM linear": (
            SVC(
                kernel="linear",
                class_weight="balanced",
                probability=False,
                random_state=RANDOM_STATE,
            ),
            {
                "C": [1.0],
            },
        ),
        "SVM RBF": (
            SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=False,
                random_state=RANDOM_STATE,
            ),
            {
                "C": [1.0],
                "gamma": ["scale"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            ),
            {
                "n_estimators": [100],
                "max_depth": [None],
                "min_samples_split": [2],
            },
        ),
        "k-NN": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [5],
                "weights": ["distance"],
                "metric": ["euclidean"],
            },
        ),
    }



def save_metric_barplot(metrics_df: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    """Save a simple barplot for one test metric."""
    plot_df = metrics_df[metrics_df["split"] == "test"].sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(plot_df["model"], plot_df[metric])
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel("Classifier")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45)
    for idx, value in enumerate(plot_df[metric].values):
        ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


# =============================================================================
# Main analysis
# =============================================================================


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. Load train/test data and selected features
    # -------------------------------------------------------------------------
    train_df = load_csv(TRAIN_FILE)
    test_df = load_csv(TEST_FILE)

    feature_names, selected_features_source = load_selected_features(train_df)
    X_train, X_test, y_train, y_test = make_xy(train_df, test_df, feature_names)
    class_names = sorted(pd.Series(y_train).unique().tolist())

    # Save the exact feature list used in this extra analysis.
    with open(OUTPUT_DIR / "extra_selected_feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2, ensure_ascii=False)
    pd.DataFrame({"feature": feature_names}).to_csv(
        OUTPUT_DIR / "extra_selected_features.csv", index=False
    )

    dataset_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "n_rows": X_train.shape[0],
                "n_features": X_train.shape[1],
                "target_column": TARGET_COLUMN,
            },
            {
                "split": "test",
                "n_rows": X_test.shape[0],
                "n_features": X_test.shape[1],
                "target_column": TARGET_COLUMN,
            },
        ]
    )
    dataset_summary.to_csv(OUTPUT_DIR / "alternative_classifiers_dataset_summary.csv", index=False)

    class_distribution = []
    for split_name, y_values in [("train", y_train), ("test", y_test)]:
        counts = pd.Series(y_values).value_counts().sort_index()
        for cls, count in counts.items():
            class_distribution.append(
                {
                    "split": split_name,
                    "class": cls,
                    "count": int(count),
                    "percent": float(count / len(y_values) * 100),
                }
            )
    pd.DataFrame(class_distribution).to_csv(
        OUTPUT_DIR / "alternative_classifiers_class_distribution.csv", index=False
    )

    # -------------------------------------------------------------------------
    # 2. Fit all classifiers with cross-validated hyperparameter tuning
    # -------------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model_grid = build_model_grid()

    metrics_rows: List[Dict[str, Any]] = []
    best_params_rows: List[Dict[str, Any]] = []
    cv_results_tables: List[pd.DataFrame] = []
    reports: List[pd.DataFrame] = []

    fitted_models: Dict[str, Any] = {}

    for model_name, (estimator, param_grid) in model_grid.items():
        print(f"[INFO] Fitting {model_name}...", flush=True)

        if param_grid:
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring=SCORING,
                cv=cv,
                n_jobs=N_JOBS,
                refit=True,
                return_train_score=True,
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            best_params = search.best_params_
            best_cv_score = float(search.best_score_)

            cv_result = pd.DataFrame(search.cv_results_)
            cv_result.insert(0, "model", model_name)
            cv_results_tables.append(cv_result)
        else:
            best_estimator = estimator
            best_estimator.fit(X_train, y_train)
            best_params = {}
            best_cv_score = np.nan

        fitted_models[model_name] = best_estimator

        y_train_pred = best_estimator.predict(X_train)
        y_test_pred = best_estimator.predict(X_test)

        metrics_rows.append(evaluate_predictions(model_name, y_train, y_train_pred, "train"))
        metrics_rows.append(evaluate_predictions(model_name, y_test, y_test_pred, "test"))

        best_params_rows.append(
            {
                "model": model_name,
                "best_cv_macro_f1": best_cv_score,
                "best_params": json.dumps(best_params, ensure_ascii=False),
            }
        )

        reports.append(save_classification_report(model_name, y_test, y_test_pred))
        save_confusion_matrix(model_name, y_test, y_test_pred, class_names)
        save_predictions(model_name, test_df, y_test_pred, best_estimator)

        safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
        with open(MODEL_DIR / f"{safe_name}_model.pkl", "wb") as f:
            pickle.dump(best_estimator, f)

    # -------------------------------------------------------------------------
    # 3. Save tabular comparisons
    # -------------------------------------------------------------------------
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUTPUT_DIR / "alternative_classifiers_metrics.csv", index=False)

    best_params_df = pd.DataFrame(best_params_rows)
    best_params_df.to_csv(OUTPUT_DIR / "alternative_classifiers_best_params.csv", index=False)

    if cv_results_tables:
        cv_results_df = pd.concat(cv_results_tables, ignore_index=True)
        cv_results_df.to_csv(OUTPUT_DIR / "alternative_classifiers_cv_results.csv", index=False)

    report_df = pd.concat(reports, ignore_index=True)
    report_df.to_csv(OUTPUT_DIR / "alternative_classifiers_classification_report_test.csv", index=False)

    # Compact comparison table focused on test performance.
    test_comparison = metrics_df[metrics_df["split"] == "test"].copy()
    test_comparison = test_comparison.merge(best_params_df, on="model", how="left")
    test_comparison = test_comparison.sort_values("macro_f1", ascending=False)
    test_comparison.to_csv(OUTPUT_DIR / "alternative_classifiers_test_comparison.csv", index=False)

    # -------------------------------------------------------------------------
    # 4. Save comparison plots
    # -------------------------------------------------------------------------
    save_metric_barplot(
        metrics_df,
        metric="accuracy",
        filename="alternative_classifiers_accuracy_comparison.png",
        title="Alternative classifiers — test accuracy",
    )
    save_metric_barplot(
        metrics_df,
        metric="macro_f1",
        filename="alternative_classifiers_macro_f1_comparison.png",
        title="Alternative classifiers — test macro F1",
    )
    save_metric_barplot(
        metrics_df,
        metric="balanced_accuracy",
        filename="alternative_classifiers_balanced_accuracy_comparison.png",
        title="Alternative classifiers — test balanced accuracy",
    )

    # Two-metric grouped line-style comparison, useful for reporting.
    plot_df = test_comparison.set_index("model")[["accuracy", "macro_f1", "weighted_f1"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Alternative classifiers — main test metrics")
    ax.set_ylabel("Score")
    ax.set_xlabel("Classifier")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "alternative_classifiers_main_metrics_comparison.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 5. JSON summary
    # -------------------------------------------------------------------------
    best_row = test_comparison.iloc[0].to_dict()
    summary = {
        "phase": "3_extra_alternative_classifiers_comparison",
        "goal": "Compare alternative classifiers on the same selected feature space used by the final post-selection logistic model.",
        "train_file": str(TRAIN_FILE),
        "test_file": str(TEST_FILE),
        "target_column": TARGET_COLUMN,
        "selected_features_source": selected_features_source,
        "n_train_rows": int(X_train.shape[0]),
        "n_test_rows": int(X_test.shape[0]),
        "n_selected_features": int(X_train.shape[1]),
        "class_names": class_names,
        "models_tested": list(model_grid.keys()),
        "cv_folds": CV_FOLDS,
        "cv_scoring": SCORING,
        "best_test_model_by_macro_f1": {
            "model": best_row["model"],
            "accuracy": float(best_row["accuracy"]),
            "balanced_accuracy": float(best_row["balanced_accuracy"]),
            "macro_f1": float(best_row["macro_f1"]),
            "weighted_f1": float(best_row["weighted_f1"]),
            "best_params": best_row.get("best_params", "{}"),
        },
        "methodological_note": (
            "The same train/test split, target labels and selected semantic features are used for all classifiers. "
            "Therefore, differences in test performance reflect classifier behavior rather than changes in preprocessing."
        ),
        "outputs": {
            "metrics": str(OUTPUT_DIR / "alternative_classifiers_metrics.csv"),
            "test_comparison": str(OUTPUT_DIR / "alternative_classifiers_test_comparison.csv"),
            "best_params": str(OUTPUT_DIR / "alternative_classifiers_best_params.csv"),
            "classification_report": str(OUTPUT_DIR / "alternative_classifiers_classification_report_test.csv"),
            "output_dir": str(OUTPUT_DIR),
        },
    }

    with open(OUTPUT_DIR / "phase3_extra_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[DONE] Phase 3-extra completed.")
    print(f"[DONE] Outputs written to: {OUTPUT_DIR}")
    print(
        "[DONE] Best test model by macro-F1: "
        f"{summary['best_test_model_by_macro_f1']['model']} "
        f"(macro-F1={summary['best_test_model_by_macro_f1']['macro_f1']:.4f})"
    )
