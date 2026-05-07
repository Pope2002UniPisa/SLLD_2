from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PHASE 1.2 — STRATIFIED TRAIN/TEST SPLIT
# Statistical Learning and Large Data Project
# Scuola Superiore Sant'Anna di Pisa
#
# Goal:
# - Load the structurally cleaned raw dataset from Phase 1.1.
# - Create a reproducible stratified train/test split.
# - Preserve class proportions in train and test sets.
# - Do NOT impute missing values.
# - Do NOT standardize features.
# - Save train/test datasets and diagnostic reports.
#
# Methodological reason:
# - All later preprocessing steps, such as imputation and scaling,
#   must be fitted only on the training set to avoid data leakage.
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PHASE1_1_DIR = Path("./outputs/phase1_1_initial_cleaning")
OUTPUT_DIR = Path("./outputs/phase1_2_train_test_split")

INPUT_FILE = PHASE1_1_DIR / "slld_phase1_1_clean_raw.csv"

TRAIN_OUTPUT_FILE = OUTPUT_DIR / "slld_phase1_2_train_raw.csv"
TEST_OUTPUT_FILE = OUTPUT_DIR / "slld_phase1_2_test_raw.csv"

CLASS_DISTRIBUTION_FILE = OUTPUT_DIR / "split_class_distribution.csv"
MISSING_BY_SPLIT_FILE = OUTPUT_DIR / "missing_values_by_split.csv"
SUMMARY_FILE = OUTPUT_DIR / "phase1_2_summary.json"


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

ID_COLS = ["entry_id", "word", "target_word_class"]

TARGET_COL = "target_word_class"

VALID_CLASSES = {"noun", "verb", "adjective"}

TEST_SIZE = 0.20
RANDOM_STATE = 42


def compute_class_distribution(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    class_distribution = (
        df[TARGET_COL]
        .value_counts()
        .rename_axis(TARGET_COL)
        .reset_index(name="count")
    )

    class_distribution["percent"] = (
        class_distribution["count"] / class_distribution["count"].sum() * 100
    )

    class_distribution.insert(0, "split", split_name)

    return class_distribution


def compute_missing_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_name: str,
) -> pd.DataFrame:
    missing_count = df[feature_cols].isna().sum()
    missing_percent = df[feature_cols].isna().mean() * 100

    missing_summary = pd.DataFrame(
        {
            "feature": feature_cols,
            "missing_count": missing_count.values,
            "missing_percent": missing_percent.values,
        }
    )

    missing_summary.insert(0, "split", split_name)

    return missing_summary.sort_values(
        by=["missing_count", "missing_percent"],
        ascending=False,
    )


def main() -> None:
    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load Phase 1.1 cleaned raw dataset
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run Phase 1.1 first with:\n"
            "python dataset_initial_cleaning.py"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\n=== PHASE 1.2 — STRATIFIED TRAIN/TEST SPLIT ===\n")
    print(f"Dataset loaded from: {INPUT_FILE}")
    print(f"Input shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # --------------------------------------------------------
    # Structural checks
    # --------------------------------------------------------

    missing_required_cols = [col for col in ID_COLS if col not in df.columns]

    if missing_required_cols:
        raise ValueError(
            "The dataset does not contain the required columns: "
            f"{missing_required_cols}"
        )

    invalid_target_rows = df[~df[TARGET_COL].isin(VALID_CLASSES)]

    if not invalid_target_rows.empty:
        raise ValueError(
            "Invalid target classes found. Run Phase 1.1 again or check the input file."
        )

    feature_cols = [col for col in df.columns if col not in ID_COLS]

    print(f"Number of feature columns detected: {len(feature_cols)}")

    if len(feature_cols) != 65:
        print(
            "\nWARNING: The number of detected feature columns is not 65. "
            "Check whether the input file is correct.\n"
        )

    # --------------------------------------------------------
    # Check class counts before splitting
    # --------------------------------------------------------

    full_class_distribution = compute_class_distribution(df, "full_dataset")

    print("\nFull dataset class distribution:")
    print(full_class_distribution)

    min_class_count = int(df[TARGET_COL].value_counts().min())

    if min_class_count < 2:
        raise ValueError(
            "At least one class has fewer than 2 observations. "
            "A stratified train/test split is not possible."
        )

    # --------------------------------------------------------
    # Stratified train/test split
    # --------------------------------------------------------
    # We stratify by target_word_class because the dataset is
    # strongly class-imbalanced.

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
        shuffle=True,
    )

    # Sort by entry_id only for readability and reproducibility in saved files.
    train_df = train_df.sort_values("entry_id").reset_index(drop=True)
    test_df = test_df.sort_values("entry_id").reset_index(drop=True)

    print(f"\nTrain shape: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Test shape: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    # --------------------------------------------------------
    # Save train/test raw datasets
    # --------------------------------------------------------

    train_df.to_csv(TRAIN_OUTPUT_FILE, index=False)
    test_df.to_csv(TEST_OUTPUT_FILE, index=False)

    print(f"\nTrain dataset saved to: {TRAIN_OUTPUT_FILE}")
    print(f"Test dataset saved to: {TEST_OUTPUT_FILE}")

    # --------------------------------------------------------
    # Class distribution reports
    # --------------------------------------------------------

    train_class_distribution = compute_class_distribution(train_df, "train")
    test_class_distribution = compute_class_distribution(test_df, "test")

    split_class_distribution = pd.concat(
        [
            full_class_distribution,
            train_class_distribution,
            test_class_distribution,
        ],
        ignore_index=True,
    )

    split_class_distribution.to_csv(CLASS_DISTRIBUTION_FILE, index=False)

    print(f"\nClass distribution report saved to: {CLASS_DISTRIBUTION_FILE}")
    print(split_class_distribution)

    # --------------------------------------------------------
    # Majority baseline accuracy
    # --------------------------------------------------------

    majority_class = (
        full_class_distribution.sort_values("count", ascending=False)
        .iloc[0][TARGET_COL]
    )

    majority_count = int(
        full_class_distribution.sort_values("count", ascending=False)
        .iloc[0]["count"]
    )

    majority_accuracy = majority_count / int(df.shape[0])

    print(f"\nMajority class baseline: {majority_class}")
    print(f"Majority baseline accuracy: {majority_accuracy:.4f}")

    # --------------------------------------------------------
    # Missing values by split
    # --------------------------------------------------------

    full_missing_summary = compute_missing_summary(
        df,
        feature_cols,
        "full_dataset",
    )

    train_missing_summary = compute_missing_summary(
        train_df,
        feature_cols,
        "train",
    )

    test_missing_summary = compute_missing_summary(
        test_df,
        feature_cols,
        "test",
    )

    missing_by_split = pd.concat(
        [
            full_missing_summary,
            train_missing_summary,
            test_missing_summary,
        ],
        ignore_index=True,
    )

    missing_by_split.to_csv(MISSING_BY_SPLIT_FILE, index=False)

    print(f"\nMissing values by split saved to: {MISSING_BY_SPLIT_FILE}")

    # --------------------------------------------------------
    # Save summary JSON
    # --------------------------------------------------------

    summary = {
        "phase": "1.2_stratified_train_test_split",
        "input_file": str(INPUT_FILE),
        "train_output_file": str(TRAIN_OUTPUT_FILE),
        "test_output_file": str(TEST_OUTPUT_FILE),
        "n_rows_full_dataset": int(df.shape[0]),
        "n_rows_train": int(train_df.shape[0]),
        "n_rows_test": int(test_df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_id_columns": len(ID_COLS),
        "n_feature_columns": len(feature_cols),
        "id_columns": ID_COLS,
        "target_column": TARGET_COL,
        "valid_target_classes": sorted(list(VALID_CLASSES)),
        "test_size": TEST_SIZE,
        "train_size": 1 - TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratification_column": TARGET_COL,
        "majority_class_baseline": {
            "majority_class": str(majority_class),
            "majority_accuracy": float(majority_accuracy),
        },
        "class_distribution": split_class_distribution.to_dict(orient="records"),
        "missing_values": {
            "full_dataset_rows_with_missing": int(
                df[feature_cols].isna().any(axis=1).sum()
            ),
            "train_rows_with_missing": int(
                train_df[feature_cols].isna().any(axis=1).sum()
            ),
            "test_rows_with_missing": int(
                test_df[feature_cols].isna().any(axis=1).sum()
            ),
        },
        "methodological_decisions": {
            "split_type": "stratified train/test split",
            "stratification_reason": (
                "the target classes are strongly imbalanced, so class proportions "
                "must be preserved in both train and test sets"
            ),
            "imputation": "not performed in this phase",
            "standardization": "not performed in this phase",
            "feature_selection": "not performed in this phase",
            "reason_for_no_preprocessing_after_split": (
                "later preprocessing steps must be fitted on the training set only "
                "to avoid data leakage"
            ),
        },
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"Summary saved to: {SUMMARY_FILE}")

    print("\n=== PHASE 1.2 COMPLETED ===\n")


if __name__ == "__main__":
    main()