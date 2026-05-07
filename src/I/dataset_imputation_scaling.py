from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ============================================================
# PHASE 1.3 — IMPUTATION AND STANDARDIZATION
# Statistical Learning and Large Data Project
# Scuola Superiore Sant'Anna di Pisa
#
# Goal:
# - Load raw train/test datasets from Phase 1.2.
# - Impute missing feature values.
# - Standardize semantic features.
# - Fit imputation and scaling ONLY on the training set.
# - Apply the fitted transformations to both train and test.
# - Save imputed and standardized datasets.
#
# Methodological reason:
# - Imputation and standardization must not use information from
#   the test set, otherwise data leakage is introduced.
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PHASE1_2_DIR = Path("./outputs/I//phase1_2_train_test_split")
OUTPUT_DIR = Path("./outputs/I//phase1_3_imputation_scaling")

TRAIN_INPUT_FILE = PHASE1_2_DIR / "slld_phase1_2_train_raw.csv"
TEST_INPUT_FILE = PHASE1_2_DIR / "slld_phase1_2_test_raw.csv"

TRAIN_IMPUTED_FILE = OUTPUT_DIR / "slld_phase1_3_train_imputed.csv"
TEST_IMPUTED_FILE = OUTPUT_DIR / "slld_phase1_3_test_imputed.csv"

TRAIN_SCALED_FILE = OUTPUT_DIR / "slld_phase1_3_train_scaled.csv"
TEST_SCALED_FILE = OUTPUT_DIR / "slld_phase1_3_test_scaled.csv"

IMPUTATION_VALUES_FILE = OUTPUT_DIR / "imputation_values.csv"
SCALING_PARAMETERS_FILE = OUTPUT_DIR / "scaling_parameters.csv"
MISSING_BEFORE_AFTER_FILE = OUTPUT_DIR / "missing_values_before_after.csv"
SUMMARY_FILE = OUTPUT_DIR / "phase1_3_summary.json"


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

ID_COLS = ["entry_id", "word", "target_word_class"]
TARGET_COL = "target_word_class"

IMPUTATION_STRATEGY = "median"


def compute_missing_report(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_name: str,
    stage: str,
) -> pd.DataFrame:
    missing_count = df[feature_cols].isna().sum()
    missing_percent = df[feature_cols].isna().mean() * 100

    report = pd.DataFrame(
        {
            "feature": feature_cols,
            "missing_count": missing_count.values,
            "missing_percent": missing_percent.values,
        }
    )

    report.insert(0, "split", split_name)
    report.insert(1, "stage", stage)

    return report.sort_values(
        by=["missing_count", "missing_percent"],
        ascending=False,
    )


def main() -> None:
    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load train/test datasets from Phase 1.2
    # --------------------------------------------------------

    if not TRAIN_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Train file not found: {TRAIN_INPUT_FILE}\n"
            "Run Phase 1.2 first with:\n"
            "python dataset_train_test_split.py"
        )

    if not TEST_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Test file not found: {TEST_INPUT_FILE}\n"
            "Run Phase 1.2 first with:\n"
            "python dataset_train_test_split.py"
        )

    train_df = pd.read_csv(TRAIN_INPUT_FILE)
    test_df = pd.read_csv(TEST_INPUT_FILE)

    print("\n=== PHASE 1.3 — IMPUTATION AND STANDARDIZATION ===\n")
    print(f"Train dataset loaded from: {TRAIN_INPUT_FILE}")
    print(f"Test dataset loaded from: {TEST_INPUT_FILE}")

    print(f"\nTrain shape: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Test shape: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    # --------------------------------------------------------
    # Structural checks
    # --------------------------------------------------------

    missing_train_cols = [col for col in ID_COLS if col not in train_df.columns]
    missing_test_cols = [col for col in ID_COLS if col not in test_df.columns]

    if missing_train_cols:
        raise ValueError(f"Missing required columns in train set: {missing_train_cols}")

    if missing_test_cols:
        raise ValueError(f"Missing required columns in test set: {missing_test_cols}")

    train_feature_cols = [col for col in train_df.columns if col not in ID_COLS]
    test_feature_cols = [col for col in test_df.columns if col not in ID_COLS]

    if train_feature_cols != test_feature_cols:
        raise ValueError(
            "Train and test feature columns do not match. "
            "Check the Phase 1.2 outputs."
        )

    feature_cols = train_feature_cols

    print(f"\nNumber of feature columns detected: {len(feature_cols)}")

    if len(feature_cols) != 65:
        print(
            "\nWARNING: The number of detected feature columns is not 65. "
            "Check whether the input files are correct.\n"
        )

    # --------------------------------------------------------
    # Ensure feature columns are numeric
    # --------------------------------------------------------

    for col in feature_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    # --------------------------------------------------------
    # Missing values before imputation
    # --------------------------------------------------------

    train_missing_before = compute_missing_report(
        train_df,
        feature_cols,
        split_name="train",
        stage="before_imputation",
    )

    test_missing_before = compute_missing_report(
        test_df,
        feature_cols,
        split_name="test",
        stage="before_imputation",
    )

    n_train_rows_with_missing_before = int(
        train_df[feature_cols].isna().any(axis=1).sum()
    )

    n_test_rows_with_missing_before = int(
        test_df[feature_cols].isna().any(axis=1).sum()
    )

    print(f"\nTrain rows with missing values before imputation: {n_train_rows_with_missing_before}")
    print(f"Test rows with missing values before imputation: {n_test_rows_with_missing_before}")

    # --------------------------------------------------------
    # Fit imputer on train only
    # --------------------------------------------------------

    imputer = SimpleImputer(strategy=IMPUTATION_STRATEGY)

    train_imputed_array = imputer.fit_transform(train_df[feature_cols])
    test_imputed_array = imputer.transform(test_df[feature_cols])

    train_imputed_features = pd.DataFrame(
        train_imputed_array,
        columns=feature_cols,
        index=train_df.index,
    )

    test_imputed_features = pd.DataFrame(
        test_imputed_array,
        columns=feature_cols,
        index=test_df.index,
    )

    train_imputed_df = pd.concat(
        [
            train_df[ID_COLS].reset_index(drop=True),
            train_imputed_features.reset_index(drop=True),
        ],
        axis=1,
    )

    test_imputed_df = pd.concat(
        [
            test_df[ID_COLS].reset_index(drop=True),
            test_imputed_features.reset_index(drop=True),
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Save imputation values
    # --------------------------------------------------------

    imputation_values = pd.DataFrame(
        {
            "feature": feature_cols,
            "imputation_strategy": IMPUTATION_STRATEGY,
            "imputation_value_fitted_on_train": imputer.statistics_,
            "train_missing_count_before": train_df[feature_cols].isna().sum().values,
            "test_missing_count_before": test_df[feature_cols].isna().sum().values,
        }
    )

    imputation_values.to_csv(IMPUTATION_VALUES_FILE, index=False)

    print(f"\nImputation values saved to: {IMPUTATION_VALUES_FILE}")

    # --------------------------------------------------------
    # Missing values after imputation
    # --------------------------------------------------------

    train_missing_after = compute_missing_report(
        train_imputed_df,
        feature_cols,
        split_name="train",
        stage="after_imputation",
    )

    test_missing_after = compute_missing_report(
        test_imputed_df,
        feature_cols,
        split_name="test",
        stage="after_imputation",
    )

    missing_before_after = pd.concat(
        [
            train_missing_before,
            test_missing_before,
            train_missing_after,
            test_missing_after,
        ],
        ignore_index=True,
    )

    missing_before_after.to_csv(MISSING_BEFORE_AFTER_FILE, index=False)

    n_train_rows_with_missing_after = int(
        train_imputed_df[feature_cols].isna().any(axis=1).sum()
    )

    n_test_rows_with_missing_after = int(
        test_imputed_df[feature_cols].isna().any(axis=1).sum()
    )

    print(f"\nTrain rows with missing values after imputation: {n_train_rows_with_missing_after}")
    print(f"Test rows with missing values after imputation: {n_test_rows_with_missing_after}")
    print(f"Missing values before/after report saved to: {MISSING_BEFORE_AFTER_FILE}")

    # --------------------------------------------------------
    # Save imputed but unscaled datasets
    # --------------------------------------------------------

    train_imputed_df.to_csv(TRAIN_IMPUTED_FILE, index=False)
    test_imputed_df.to_csv(TEST_IMPUTED_FILE, index=False)

    print(f"\nTrain imputed dataset saved to: {TRAIN_IMPUTED_FILE}")
    print(f"Test imputed dataset saved to: {TEST_IMPUTED_FILE}")

    # --------------------------------------------------------
    # Fit scaler on train only
    # --------------------------------------------------------

    scaler = StandardScaler()

    train_scaled_array = scaler.fit_transform(train_imputed_df[feature_cols])
    test_scaled_array = scaler.transform(test_imputed_df[feature_cols])

    train_scaled_features = pd.DataFrame(
        train_scaled_array,
        columns=feature_cols,
        index=train_imputed_df.index,
    )

    test_scaled_features = pd.DataFrame(
        test_scaled_array,
        columns=feature_cols,
        index=test_imputed_df.index,
    )

    train_scaled_df = pd.concat(
        [
            train_imputed_df[ID_COLS].reset_index(drop=True),
            train_scaled_features.reset_index(drop=True),
        ],
        axis=1,
    )

    test_scaled_df = pd.concat(
        [
            test_imputed_df[ID_COLS].reset_index(drop=True),
            test_scaled_features.reset_index(drop=True),
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Save scaling parameters
    # --------------------------------------------------------

    scaling_parameters = pd.DataFrame(
        {
            "feature": feature_cols,
            "train_mean_used_for_scaling": scaler.mean_,
            "train_std_used_for_scaling": scaler.scale_,
        }
    )

    scaling_parameters.to_csv(SCALING_PARAMETERS_FILE, index=False)

    print(f"\nScaling parameters saved to: {SCALING_PARAMETERS_FILE}")

    # --------------------------------------------------------
    # Save standardized datasets
    # --------------------------------------------------------

    train_scaled_df.to_csv(TRAIN_SCALED_FILE, index=False)
    test_scaled_df.to_csv(TEST_SCALED_FILE, index=False)

    print(f"\nTrain standardized dataset saved to: {TRAIN_SCALED_FILE}")
    print(f"Test standardized dataset saved to: {TEST_SCALED_FILE}")

    # --------------------------------------------------------
    # Diagnostic check on standardized train set
    # --------------------------------------------------------

    train_scaled_means = train_scaled_df[feature_cols].mean()
    train_scaled_stds = train_scaled_df[feature_cols].std(ddof=0)

    max_abs_train_mean_after_scaling = float(train_scaled_means.abs().max())
    min_train_std_after_scaling = float(train_scaled_stds.min())
    max_train_std_after_scaling = float(train_scaled_stds.max())

    print("\nStandardization diagnostic on train set:")
    print(f"Maximum absolute feature mean: {max_abs_train_mean_after_scaling:.8f}")
    print(f"Minimum feature std: {min_train_std_after_scaling:.8f}")
    print(f"Maximum feature std: {max_train_std_after_scaling:.8f}")

    # --------------------------------------------------------
    # Save summary JSON
    # --------------------------------------------------------

    summary = {
        "phase": "1.3_imputation_and_standardization",
        "train_input_file": str(TRAIN_INPUT_FILE),
        "test_input_file": str(TEST_INPUT_FILE),
        "train_imputed_file": str(TRAIN_IMPUTED_FILE),
        "test_imputed_file": str(TEST_IMPUTED_FILE),
        "train_scaled_file": str(TRAIN_SCALED_FILE),
        "test_scaled_file": str(TEST_SCALED_FILE),
        "n_train_rows": int(train_df.shape[0]),
        "n_test_rows": int(test_df.shape[0]),
        "n_columns": int(train_df.shape[1]),
        "n_id_columns": len(ID_COLS),
        "n_feature_columns": len(feature_cols),
        "id_columns": ID_COLS,
        "target_column": TARGET_COL,
        "imputation": {
            "strategy": IMPUTATION_STRATEGY,
            "fitted_on": "train_set_only",
            "applied_to": ["train_set", "test_set"],
            "imputation_values_file": str(IMPUTATION_VALUES_FILE),
        },
        "standardization": {
            "method": "StandardScaler",
            "fitted_on": "train_set_only_after_imputation",
            "applied_to": ["train_set", "test_set"],
            "scaling_parameters_file": str(SCALING_PARAMETERS_FILE),
        },
        "missing_values": {
            "train_rows_with_missing_before_imputation": n_train_rows_with_missing_before,
            "test_rows_with_missing_before_imputation": n_test_rows_with_missing_before,
            "train_rows_with_missing_after_imputation": n_train_rows_with_missing_after,
            "test_rows_with_missing_after_imputation": n_test_rows_with_missing_after,
        },
        "standardization_diagnostics_train": {
            "max_abs_feature_mean_after_scaling": max_abs_train_mean_after_scaling,
            "min_feature_std_after_scaling": min_train_std_after_scaling,
            "max_feature_std_after_scaling": max_train_std_after_scaling,
        },
        "methodological_decisions": {
            "data_leakage_control": (
                "imputer and scaler are fitted only on the training set; "
                "the test set is only transformed"
            ),
            "median_imputation_reason": (
                "median imputation is simple, robust, reproducible and appropriate "
                "as a first preprocessing strategy for numeric semantic ratings"
            ),
            "standardization_reason": (
                "standardization is required before PCA, clustering, distance-based methods "
                "and regularized models such as Ridge, LASSO and Elastic Net"
            ),
            "pca": "not performed in this phase",
            "clustering": "not performed in this phase",
            "feature_selection": "not performed in this phase",
            "model_training": "not performed in this phase",
        },
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"\nSummary saved to: {SUMMARY_FILE}")

    print("\n=== PHASE 1.3 COMPLETED ===\n")


if __name__ == "__main__":
    main()