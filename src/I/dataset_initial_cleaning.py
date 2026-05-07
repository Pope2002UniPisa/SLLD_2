from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# PHASE 1.1 — INITIAL DATASET CLEANING
# Statistical Learning and Large Data Project
# Scuola Superiore Sant'Anna di Pisa
#
# Goal:
# - Load the 65-feature semantic dataset.
# - Check the basic structure of the data.
# - Detect exact duplicates.
# - Detect duplicated word forms.
# - Check missing values.
# - Preserve all valid lexical entries.
# - Save a structurally cleaned dataset.
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

DATASET_DIR = Path("./dataset")
OUTPUT_DIR = Path("./outputs/phase1_1_initial_cleaning")

INPUT_FILE = DATASET_DIR / "slld_binder_Xy_65_features.csv"

CLEAN_OUTPUT_FILE = OUTPUT_DIR / "slld_phase1_1_clean_raw.csv"
MISSING_REPORT_FILE = OUTPUT_DIR / "missing_values_report.csv"
WORD_DUPLICATES_FILE = OUTPUT_DIR / "duplicated_word_forms.csv"
CLASS_DISTRIBUTION_FILE = OUTPUT_DIR / "target_class_distribution.csv"
SUMMARY_FILE = OUTPUT_DIR / "phase1_1_summary.json"


# ------------------------------------------------------------
# 2. Expected structural columns
# ------------------------------------------------------------

ID_COLS = ["entry_id", "word", "target_word_class"]

VALID_CLASSES = {"noun", "verb", "adjective"}


def main() -> None:
    # --------------------------------------------------------
    # Create output directory if it does not exist
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            f"Expected file path: ./dataset/slld_binder_Xy_65_features.csv"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\n=== PHASE 1.1 — INITIAL DATASET CLEANING ===\n")
    print(f"Dataset loaded from: {INPUT_FILE}")
    print(f"Initial shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_required_cols = [col for col in ID_COLS if col not in df.columns]

    if missing_required_cols:
        raise ValueError(
            "The dataset does not contain the required columns: "
            f"{missing_required_cols}"
        )

    # --------------------------------------------------------
    # Identify feature columns
    # --------------------------------------------------------

    feature_cols = [col for col in df.columns if col not in ID_COLS]

    print(f"Number of feature columns detected: {len(feature_cols)}")

    if len(feature_cols) != 65:
        print(
            "\nWARNING: The number of detected feature columns is not 65. "
            "Check whether the input file is the correct one.\n"
        )

    # --------------------------------------------------------
    # Basic cleaning of textual columns
    # --------------------------------------------------------
    # We only strip accidental blank spaces.
    # We do NOT alter the lexical content of the words.

    df["word"] = df["word"].astype(str).str.strip()
    df["target_word_class"] = df["target_word_class"].astype(str).str.strip()

    # --------------------------------------------------------
    # Check target classes
    # --------------------------------------------------------

    invalid_target_rows = df[~df["target_word_class"].isin(VALID_CLASSES)]

    if not invalid_target_rows.empty:
        print("\nWARNING: Invalid target classes found.")
        print(invalid_target_rows[["entry_id", "word", "target_word_class"]])

        # We remove only structurally invalid target rows.
        # In the prepared dataset this should normally remove nothing.
        df = df[df["target_word_class"].isin(VALID_CLASSES)].copy()

    # --------------------------------------------------------
    # Check exact duplicate rows
    # --------------------------------------------------------

    n_exact_duplicates = int(df.duplicated().sum())

    print(f"\nExact duplicate rows found: {n_exact_duplicates}")

    # Remove only exact duplicates.
    # This is safe because exact duplicates do not add information.
    df_clean = df.drop_duplicates().copy()

    # --------------------------------------------------------
    # Check duplicated word forms
    # --------------------------------------------------------
    # A duplicated word form is not necessarily an error.
    # Example: "used" may appear as both adjective and verb.
    #
    # Since the statistical unit is lexical entry, not only word form,
    # duplicated word forms are reported but not removed.

    duplicated_word_forms = (
        df_clean[df_clean.duplicated("word", keep=False)]
        .sort_values(["word", "target_word_class"])
        .loc[:, ["entry_id", "word", "target_word_class"]]
    )

    duplicated_word_forms.to_csv(WORD_DUPLICATES_FILE, index=False)

    print(f"Duplicated word forms found: {duplicated_word_forms.shape[0]} rows")
    print(f"Duplicated word forms report saved to: {WORD_DUPLICATES_FILE}")

    # --------------------------------------------------------
    # Missing values analysis
    # --------------------------------------------------------

    missing_count = df_clean[feature_cols].isna().sum()
    missing_percent = df_clean[feature_cols].isna().mean() * 100

    missing_report = pd.DataFrame(
        {
            "feature": feature_cols,
            "missing_count": missing_count.values,
            "missing_percent": missing_percent.values,
        }
    )

    missing_report = missing_report.sort_values(
        by=["missing_count", "missing_percent"],
        ascending=False,
    )

    missing_report.to_csv(MISSING_REPORT_FILE, index=False)

    n_rows_with_missing = int(df_clean[feature_cols].isna().any(axis=1).sum())
    n_complete_rows = int((~df_clean[feature_cols].isna().any(axis=1)).sum())

    print(f"\nRows with at least one missing value: {n_rows_with_missing}")
    print(f"Complete rows: {n_complete_rows}")
    print(f"Missing values report saved to: {MISSING_REPORT_FILE}")

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_distribution = (
        df_clean["target_word_class"]
        .value_counts()
        .rename_axis("target_word_class")
        .reset_index(name="count")
    )

    class_distribution["percent"] = (
        class_distribution["count"] / class_distribution["count"].sum() * 100
    )

    class_distribution.to_csv(CLASS_DISTRIBUTION_FILE, index=False)

    print(f"\nClass distribution saved to: {CLASS_DISTRIBUTION_FILE}")
    print(class_distribution)

    # --------------------------------------------------------
    # Save cleaned raw dataset
    # --------------------------------------------------------
    # Important methodological decision:
    # - We do NOT remove rows with missing values.
    # - We do NOT remove columns with missing values.
    # - We do NOT impute missing values here.
    # - We do NOT standardize here.
    #
    # These operations must be done later, preferably inside a
    # train/test pipeline, to avoid data leakage.

    df_clean.to_csv(CLEAN_OUTPUT_FILE, index=False)

    print(f"\nClean raw dataset saved to: {CLEAN_OUTPUT_FILE}")

    # --------------------------------------------------------
    # Save summary JSON
    # --------------------------------------------------------

    columns_with_missing = missing_report[missing_report["missing_count"] > 0]

    summary = {
        "phase": "1.1_initial_dataset_cleaning",
        "input_file": str(INPUT_FILE),
        "output_clean_file": str(CLEAN_OUTPUT_FILE),
        "initial_n_rows": int(df.shape[0]),
        "initial_n_columns": int(df.shape[1]),
        "clean_n_rows": int(df_clean.shape[0]),
        "clean_n_columns": int(df_clean.shape[1]),
        "n_id_columns": len(ID_COLS),
        "n_feature_columns": len(feature_cols),
        "id_columns": ID_COLS,
        "target_column": "target_word_class",
        "valid_target_classes": sorted(list(VALID_CLASSES)),
        "n_exact_duplicate_rows_removed": n_exact_duplicates,
        "n_duplicated_word_form_rows": int(duplicated_word_forms.shape[0]),
        "n_rows_with_at_least_one_missing_value": n_rows_with_missing,
        "n_complete_rows": n_complete_rows,
        "columns_with_missing_values": columns_with_missing.to_dict(orient="records"),
        "methodological_decisions": {
            "exact_duplicates": "removed if present",
            "duplicated_word_forms": (
                "reported but retained, because the statistical unit is the lexical entry"
            ),
            "rows_with_missing_values": (
                "retained, because the dataset is small and class-imbalanced"
            ),
            "columns_with_missing_values": (
                "retained, because missingness is limited and the 65 semantic features "
                "define the original representation space"
            ),
            "imputation": "postponed to a later preprocessing step",
            "standardization": "postponed to a later preprocessing step",
            "train_test_split": "not performed in this phase",
        },
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"Summary saved to: {SUMMARY_FILE}")

    print("\n=== PHASE 1.1 COMPLETED ===\n")


if __name__ == "__main__":
    main()