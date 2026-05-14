from __future__ import annotations

"""
============================================================
PHASE 3.1 — SUPERVISED ANALYSIS: DATA LOADING AND CHECKS
Statistical Learning and Large Data Project
Scuola Superiore Sant'Anna di Pisa

Goal:
- Start the supervised analysis from the Phase 1 outputs.
- Load the semantic feature matrix and the multiclass target.
- Separate identifier columns from numerical predictor columns.
- Produce X, y, feature_names and class_names for the next phases.
- Run preliminary quality checks before fitting supervised models:
  number of observations, number of features, missing values,
  exact duplicates, duplicated word forms and class distribution.
- Save tabular reports and diagnostic figures in the same project style
  used by the previous phases.

Input:
  Preferred:
    outputs/I/phase1_3_imputation_scaling/slld_phase1_3_train_scaled.csv
    outputs/I/phase1_3_imputation_scaling/slld_phase1_3_test_scaled.csv

  Accepted fallback locations:
    phase1_3_imputation_scaling/slld_phase1_3_train_scaled.csv
    phase1_3_imputation_scaling/slld_phase1_3_test_scaled.csv

Outputs (all under outputs/III/phase3_1_supervised_data_loading_checks/):
  - slld_phase3_1_X_train.csv
  - slld_phase3_1_y_train.csv
  - slld_phase3_1_X_test.csv
  - slld_phase3_1_y_test.csv
  - slld_phase3_1_train_modeling_dataset.csv
  - slld_phase3_1_test_modeling_dataset.csv
  - feature_names.json
  - class_names.json
  - dataset_shape_summary.csv
  - class_distribution.csv
  - missing_values_report.csv
  - duplicate_rows.csv
  - duplicated_word_forms.csv
  - non_numeric_features_report.csv
  - class_distribution_train_test.png
  - missing_values_by_split.png
  - feature_missingness_top20.png
  - duplicates_summary.png
  - phase3_1_summary.json
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

OUTPUT_DIR = Path("./outputs/III/phase3_1_supervised_data_loading_checks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "phase3_1_summary.json"

# The code first looks for the standard project layout used by Phase 2.
# If the Phase 1 folder has been extracted directly in the working directory,
# the fallback paths make the script work without manual path edits.
PHASE1_3_CANDIDATES = [
    Path("./outputs/I/phase1_3_imputation_scaling"),
    Path("./phase1_3_imputation_scaling"),
    Path("./Fase_1_output/phase1_3_imputation_scaling"),
]


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

ID_COLS = ["entry_id", "word", "target_word_class"]
TARGET_COL = "target_word_class"
WORD_COL = "word"
ENTRY_ID_COL = "entry_id"

TRAIN_FILE_NAME = "slld_phase1_3_train_scaled.csv"
TEST_FILE_NAME = "slld_phase1_3_test_scaled.csv"

# Original labels are kept for compatibility with the input files.
# Display labels are used in reports and figures because they match the
# lexical-semantic interpretation used in the project.
LABEL_MAP = {"noun": "Thing", "verb": "Action", "adjective": "Property"}
CLASS_NAMES = ["Thing", "Action", "Property"]
CLASS_ORDER_RAW = ["noun", "verb", "adjective"]

COLOR_MAP = {"Thing": "#e07b39", "Action": "#4c7cba", "Property": "#5aa15a"}


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


def _label_display(y: pd.Series) -> pd.Series:
    """Map raw target labels to human-readable project labels."""
    return y.map(LABEL_MAP).fillna(y.astype(str))


def resolve_phase1_3_dir() -> Path:
    """Find the Phase 1.3 directory containing the scaled train/test files."""
    for candidate in PHASE1_3_CANDIDATES:
        train_path = candidate / TRAIN_FILE_NAME
        test_path = candidate / TEST_FILE_NAME
        if train_path.exists() and test_path.exists():
            return candidate

    searched = "\n".join(str(p) for p in PHASE1_3_CANDIDATES)
    raise FileNotFoundError(
        "Could not find Phase 1.3 scaled train/test files. "
        "Expected slld_phase1_3_train_scaled.csv and "
        f"slld_phase1_3_test_scaled.csv in one of:\n{searched}"
    )


# ------------------------------------------------------------
# 4. Load data
# ------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """
    Load the scaled train and test datasets produced by Phase 1.3.

    These files already contain:
    - identifier columns: entry_id, word, target_word_class;
    - numerical semantic features, already imputed and scaled.
    """
    phase1_3_dir = resolve_phase1_3_dir()
    train_df = pd.read_csv(phase1_3_dir / TRAIN_FILE_NAME)
    test_df = pd.read_csv(phase1_3_dir / TEST_FILE_NAME)
    return train_df, test_df, phase1_3_dir


# ------------------------------------------------------------
# 5. Structural validation and feature extraction
# ------------------------------------------------------------

def validate_required_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Ensure that all identifier and target columns are present."""
    for split_name, df in {"train": train_df, "test": test_df}.items():
        missing = [c for c in ID_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {split_name}: {missing}")


def extract_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Separate the numerical feature matrix X from the target vector y.

    The feature set is defined as all columns except ID_COLS. The function
    checks that all resulting predictors are numeric, because the following
    supervised models require a numerical design matrix.
    """
    feature_names = [c for c in df.columns if c not in ID_COLS]
    X = df[feature_names].copy()
    y = df[TARGET_COL].copy()

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        report = pd.DataFrame({"feature": non_numeric})
        report.to_csv(OUTPUT_DIR / "non_numeric_features_report.csv", index=False)
        raise TypeError(
            "Some predictor columns are not numeric. They must be encoded "
            f"before supervised modeling: {non_numeric}"
        )

    return X, y, feature_names


def check_feature_alignment(train_features: list[str], test_features: list[str]) -> None:
    """Check that train and test contain the same feature columns in the same order."""
    if train_features != test_features:
        train_only = sorted(set(train_features) - set(test_features))
        test_only = sorted(set(test_features) - set(train_features))
        raise ValueError(
            "Train/test feature mismatch. "
            f"Train-only: {train_only}; test-only: {test_only}"
        )


# ------------------------------------------------------------
# 6. Preliminary checks
# ------------------------------------------------------------

def build_shape_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact table with dimensions and basic quality indicators."""
    rows = []
    for split_name, df, X in [
        ("train", train_df, X_train),
        ("test", test_df, X_test),
        ("total", pd.concat([train_df, test_df], ignore_index=True),
         pd.concat([X_train, X_test], ignore_index=True)),
    ]:
        rows.append({
            "split": split_name,
            "n_observations": int(df.shape[0]),
            "n_total_columns": int(df.shape[1]),
            "n_identifier_columns": int(len(ID_COLS)),
            "n_features": int(X.shape[1]),
            "target_col": TARGET_COL,
            "n_missing_cells": int(df.isna().sum().sum()),
            "n_exact_duplicate_rows": int(df.duplicated().sum()),
            "n_duplicated_entry_ids": int(df.duplicated(subset=[ENTRY_ID_COL]).sum())
                if ENTRY_ID_COL in df.columns else 0,
            "n_duplicated_word_forms": int(df.duplicated(subset=[WORD_COL]).sum())
                if WORD_COL in df.columns else 0,
        })
    return pd.DataFrame(rows)


def build_class_distribution(train_y: pd.Series, test_y: pd.Series) -> pd.DataFrame:
    """Count target classes in train, test and total data."""
    records = []
    for split_name, y in [
        ("train", train_y),
        ("test", test_y),
        ("total", pd.concat([train_y, test_y], ignore_index=True)),
    ]:
        counts = y.value_counts().reindex(CLASS_ORDER_RAW, fill_value=0)
        total = int(counts.sum())
        for raw_label, count in counts.items():
            records.append({
                "split": split_name,
                "class_raw": raw_label,
                "class_display": LABEL_MAP.get(raw_label, raw_label),
                "count": int(count),
                "proportion": float(count / total) if total > 0 else np.nan,
            })
    return pd.DataFrame(records)


def build_missing_values_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Report missing values by split and column."""
    records = []
    for split_name, df in [("train", train_df), ("test", test_df)]:
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                records.append({
                    "split": split_name,
                    "column": col,
                    "n_missing": n_missing,
                    "missing_rate": float(n_missing / len(df)),
                })

    if not records:
        return pd.DataFrame(columns=["split", "column", "n_missing", "missing_rate"])
    return pd.DataFrame(records).sort_values(["split", "n_missing"], ascending=[True, False])


def build_duplicate_reports(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect exact duplicate rows and duplicated word forms."""
    combined = pd.concat(
        [
            train_df.assign(split="train"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )

    # Exact duplicates are checked on the original columns, excluding the helper split column.
    original_cols = [c for c in combined.columns if c != "split"]
    exact_mask = combined.duplicated(subset=original_cols, keep=False)
    duplicate_rows = combined.loc[exact_mask].sort_values(ID_COLS).copy()

    # Duplicated word forms are not necessarily errors: the same word may appear
    # with different lexical-semantic roles. The report preserves them for inspection.
    word_mask = combined.duplicated(subset=[WORD_COL], keep=False)
    duplicated_words = combined.loc[word_mask, ["split", ENTRY_ID_COL, WORD_COL, TARGET_COL]].copy()
    if not duplicated_words.empty:
        duplicated_words["target_display"] = _label_display(duplicated_words[TARGET_COL])
        duplicated_words = duplicated_words.sort_values([WORD_COL, TARGET_COL, "split"])

    return duplicate_rows, duplicated_words


# ------------------------------------------------------------
# 7. Plots
# ------------------------------------------------------------

def plot_class_distribution(class_distribution: pd.DataFrame) -> None:
    """Bar chart of class counts for train and test splits."""
    plot_df = class_distribution[class_distribution["split"].isin(["train", "test"])]
    pivot = plot_df.pivot(index="class_display", columns="split", values="count").reindex(CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    ax.bar(x - width / 2, pivot["train"], width, label="train", color="#4c7cba", alpha=0.85)
    ax.bar(x + width / 2, pivot["test"], width, label="test", color="#e07b39", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Number of observations")
    ax.set_title("Supervised Target Distribution — Train/Test")
    ax.legend(title="Split")
    _save(fig, "class_distribution_train_test.png")


def plot_missing_values(missing_report: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Create missing-value diagnostic figures."""
    total_missing_train = int(train_df.isna().sum().sum())
    total_missing_test = int(test_df.isna().sum().sum())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["train", "test"], [total_missing_train, total_missing_test], color=["#4c7cba", "#e07b39"])
    ax.set_ylabel("Missing cells")
    ax.set_title("Missing Values by Split")
    _save(fig, "missing_values_by_split.png")

    if missing_report.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No missing values detected after Phase 1.3", ha="center", va="center", fontsize=12)
        ax.axis("off")
        _save(fig, "feature_missingness_top20.png")
        return

    top = missing_report.copy()
    top["split_column"] = top["split"] + " — " + top["column"]
    top = top.sort_values("n_missing", ascending=False).head(20).sort_values("n_missing")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["split_column"], top["n_missing"], color="#4c7cba", alpha=0.85)
    ax.set_xlabel("Missing values")
    ax.set_title("Top 20 Columns by Missing Values")
    _save(fig, "feature_missingness_top20.png")


def plot_duplicates(shape_summary: pd.DataFrame) -> None:
    """Plot exact duplicates and duplicated word-form counts."""
    plot_df = shape_summary[shape_summary["split"].isin(["train", "test", "total"])]
    x = np.arange(plot_df.shape[0])
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        x - width / 2,
        plot_df["n_exact_duplicate_rows"],
        width,
        label="exact duplicate rows",
        color="#4c7cba",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        plot_df["n_duplicated_word_forms"],
        width,
        label="duplicated word forms",
        color="#e07b39",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["split"])
    ax.set_ylabel("Count")
    ax.set_title("Duplicate Diagnostics")
    ax.legend()
    _save(fig, "duplicates_summary.png")


# ------------------------------------------------------------
# 8. Save modelling objects and reports
# ------------------------------------------------------------

def save_xy_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> None:
    """Save X/y matrices and metadata for the following supervised phases."""
    X_train.to_csv(OUTPUT_DIR / "slld_phase3_1_X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "slld_phase3_1_X_test.csv", index=False)

    pd.DataFrame({TARGET_COL: y_train, "target_display": _label_display(y_train)}).to_csv(
        OUTPUT_DIR / "slld_phase3_1_y_train.csv", index=False
    )
    pd.DataFrame({TARGET_COL: y_test, "target_display": _label_display(y_test)}).to_csv(
        OUTPUT_DIR / "slld_phase3_1_y_test.csv", index=False
    )

    # Complete modelling datasets: identifiers + target + numerical predictors.
    train_df[ID_COLS + feature_names].to_csv(
        OUTPUT_DIR / "slld_phase3_1_train_modeling_dataset.csv", index=False
    )
    test_df[ID_COLS + feature_names].to_csv(
        OUTPUT_DIR / "slld_phase3_1_test_modeling_dataset.csv", index=False
    )

    _write_json(feature_names, OUTPUT_DIR / "feature_names.json")
    _write_json(CLASS_NAMES, OUTPUT_DIR / "class_names.json")


def save_reports(
    shape_summary: pd.DataFrame,
    class_distribution: pd.DataFrame,
    missing_report: pd.DataFrame,
    duplicate_rows: pd.DataFrame,
    duplicated_words: pd.DataFrame,
) -> None:
    """Save all preliminary diagnostic reports."""
    shape_summary.to_csv(OUTPUT_DIR / "dataset_shape_summary.csv", index=False)
    class_distribution.to_csv(OUTPUT_DIR / "class_distribution.csv", index=False)
    missing_report.to_csv(OUTPUT_DIR / "missing_values_report.csv", index=False)
    duplicate_rows.to_csv(OUTPUT_DIR / "duplicate_rows.csv", index=False)
    duplicated_words.to_csv(OUTPUT_DIR / "duplicated_word_forms.csv", index=False)

    # Empty report means everything is already numeric.
    pd.DataFrame(columns=["feature"]).to_csv(
        OUTPUT_DIR / "non_numeric_features_report.csv", index=False
    )


# ------------------------------------------------------------
# 9. Main
# ------------------------------------------------------------

def main() -> None:
    train_df, test_df, phase1_3_dir = load_data()
    validate_required_columns(train_df, test_df)

    X_train, y_train, train_features = extract_xy(train_df)
    X_test, y_test, test_features = extract_xy(test_df)
    check_feature_alignment(train_features, test_features)

    feature_names = train_features

    shape_summary = build_shape_summary(train_df, test_df, X_train, X_test)
    class_distribution = build_class_distribution(y_train, y_test)
    missing_report = build_missing_values_report(train_df, test_df)
    duplicate_rows, duplicated_words = build_duplicate_reports(train_df, test_df)

    save_xy_outputs(
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    )
    save_reports(
        shape_summary=shape_summary,
        class_distribution=class_distribution,
        missing_report=missing_report,
        duplicate_rows=duplicate_rows,
        duplicated_words=duplicated_words,
    )

    plot_class_distribution(class_distribution)
    plot_missing_values(missing_report, train_df, test_df)
    plot_duplicates(shape_summary)

    summary = {
        "phase": "PHASE 3.1 — SUPERVISED ANALYSIS: DATA LOADING AND CHECKS",
        "input_dir": str(phase1_3_dir),
        "train_file": str(phase1_3_dir / TRAIN_FILE_NAME),
        "test_file": str(phase1_3_dir / TEST_FILE_NAME),
        "output_dir": str(OUTPUT_DIR),
        "target_col": TARGET_COL,
        "id_cols": ID_COLS,
        "class_names": CLASS_NAMES,
        "label_map": LABEL_MAP,
        "n_train_observations": int(train_df.shape[0]),
        "n_test_observations": int(test_df.shape[0]),
        "n_total_observations": int(train_df.shape[0] + test_df.shape[0]),
        "n_features": int(len(feature_names)),
        "feature_names_file": str(OUTPUT_DIR / "feature_names.json"),
        "class_names_file": str(OUTPUT_DIR / "class_names.json"),
        "n_missing_cells_train": int(train_df.isna().sum().sum()),
        "n_missing_cells_test": int(test_df.isna().sum().sum()),
        "n_exact_duplicate_rows_total": int(shape_summary.loc[shape_summary["split"] == "total", "n_exact_duplicate_rows"].iloc[0]),
        "n_duplicated_word_forms_total": int(shape_summary.loc[shape_summary["split"] == "total", "n_duplicated_word_forms"].iloc[0]),
        "outputs": {
            "X_train": str(OUTPUT_DIR / "slld_phase3_1_X_train.csv"),
            "y_train": str(OUTPUT_DIR / "slld_phase3_1_y_train.csv"),
            "X_test": str(OUTPUT_DIR / "slld_phase3_1_X_test.csv"),
            "y_test": str(OUTPUT_DIR / "slld_phase3_1_y_test.csv"),
            "dataset_shape_summary": str(OUTPUT_DIR / "dataset_shape_summary.csv"),
            "class_distribution": str(OUTPUT_DIR / "class_distribution.csv"),
            "missing_values_report": str(OUTPUT_DIR / "missing_values_report.csv"),
            "duplicate_rows": str(OUTPUT_DIR / "duplicate_rows.csv"),
            "duplicated_word_forms": str(OUTPUT_DIR / "duplicated_word_forms.csv"),
        },
    }

    _write_json(summary, SUMMARY_FILE)

    print("PHASE 3.1 completed successfully.")
    print(f"Input directory: {phase1_3_dir}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train observations: {train_df.shape[0]}")
    print(f"Test observations: {test_df.shape[0]}")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")


if __name__ == "__main__":
    main()
