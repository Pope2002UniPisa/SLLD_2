from __future__ import annotations

"""
============================================================
PHASE 3.7 — SUPERVISED ANALYSIS: FINAL MODEL EVALUATION
Statistical Learning and Large Data Project
Scuola Superiore Sant'Anna di Pisa

Goal:
- Close the supervised pipeline.
- Compare the baseline, LASSO and Elastic Net models.
- Build a post-selection multinomial logistic model on the selected feature set.
- Select the final classifier according to test macro F1, balanced accuracy,
  sparsity and interpretability diagnostics.
- Save final predictions, confusion matrices, comparison tables and figures.

Input:
  Phase 3.1 X/y files.
  Phase 3.2 baseline metrics, when available.
  Phase 3.5 regularized model outputs.
  Phase 3.6 sparsity outputs.

Outputs (all under outputs/III/phase3_7_final_model_evaluation/):
  - final_model_metrics.csv
  - full_model_comparison.csv
  - final_model_predictions_test.csv
  - final_model_classification_report_test.csv
  - final_model_confusion_matrix_test.csv
  - final_model_confusion_matrix_test_normalized.csv
  - final_selected_features.csv
  - final_model_coefficients.csv
  - final_model.pkl
  - final_confusion_matrix_test.png
  - final_confusion_matrix_test_normalized.png
  - final_model_comparison_accuracy_macro_f1.png
  - final_model_class_metrics.png
  - final_top_coefficients.png
  - phase3_7_summary.json
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

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

OUTPUT_DIR = Path("./outputs/III/phase3_7_final_model_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = OUTPUT_DIR / "phase3_7_summary.json"

PHASE3_1_CANDIDATES = [
    Path("./outputs/III/phase3_1_supervised_data_loading_checks"),
    Path("./phase3_1_supervised_data_loading_checks"),
    Path("./outputs/phase3_1_supervised_data_loading_checks"),
    Path("./package_phase3_1/outputs/III/phase3_1_supervised_data_loading_checks"),
]
PHASE3_2_DIR = Path("./outputs/III/phase3_2_train_test_baseline")
PHASE3_5_DIR = Path("./outputs/III/phase3_5_regularized_models")
PHASE3_6_DIR = Path("./outputs/III/phase3_6_sparsity_analysis")


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

TARGET_COL = "target_word_class"
RANDOM_STATE = 42
MAX_ITER = 5000
TOP_N_COEFFICIENTS = 25
COEF_ZERO_TOL = 1e-8

X_TRAIN_FILE_NAME = "slld_phase3_1_X_train.csv"
Y_TRAIN_FILE_NAME = "slld_phase3_1_y_train.csv"
X_TEST_FILE_NAME = "slld_phase3_1_X_test.csv"
Y_TEST_FILE_NAME = "slld_phase3_1_y_test.csv"
FEATURE_NAMES_FILE_NAME = "feature_names.json"
CLASS_NAMES_FILE_NAME = "class_names.json"

LABEL_MAP = {"noun": "Thing", "verb": "Action", "adjective": "Property"}
DISPLAY_TO_RAW = {v: k for k, v in LABEL_MAP.items()}
CLASS_ORDER_DISPLAY = ["Thing", "Action", "Property"]
CLASS_ORDER_RAW = [DISPLAY_TO_RAW[c] for c in CLASS_ORDER_DISPLAY]


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


def class_display(raw_label: str) -> str:
    return LABEL_MAP.get(raw_label, str(raw_label))


# ------------------------------------------------------------
# 4. Load data and previous outputs
# ------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str], Path]:
    phase3_1_dir = resolve_phase3_1_dir()
    X_train = pd.read_csv(phase3_1_dir / X_TRAIN_FILE_NAME)
    X_test = pd.read_csv(phase3_1_dir / X_TEST_FILE_NAME)
    y_train = pd.read_csv(phase3_1_dir / Y_TRAIN_FILE_NAME)[TARGET_COL]
    y_test = pd.read_csv(phase3_1_dir / Y_TEST_FILE_NAME)[TARGET_COL]
    feature_names = load_json_list(phase3_1_dir / FEATURE_NAMES_FILE_NAME)
    class_names = load_json_list(phase3_1_dir / CLASS_NAMES_FILE_NAME)
    return X_train[feature_names], y_train, X_test[feature_names], y_test, feature_names, class_names, phase3_1_dir


def load_regularized_models() -> tuple[object | None, object | None]:
    lasso_path = PHASE3_5_DIR / "lasso_model.pkl"
    elastic_path = PHASE3_5_DIR / "elastic_net_model.pkl"
    lasso = pickle.load(lasso_path.open("rb")) if lasso_path.exists() else None
    elastic = pickle.load(elastic_path.open("rb")) if elastic_path.exists() else None
    return lasso, elastic


def load_regularized_selected_features(feature_names: list[str]) -> list[str]:
    """Use the union of LASSO and Elastic Net selected features for post-selection refit."""
    selected = set()
    for filename in ["selected_feature_names_lasso.json", "selected_feature_names_elastic_net.json"]:
        path = PHASE3_5_DIR / filename
        if path.exists():
            selected.update(load_json_list(path))
    selected = [f for f in feature_names if f in selected]
    return selected if selected else feature_names


# ------------------------------------------------------------
# 5. Model evaluation
# ------------------------------------------------------------

def evaluate(name: str, model, X: pd.DataFrame, y: pd.Series, split: str, n_features: int, model_type: str) -> dict:
    pred = model.predict(X)
    return {
        "model": name,
        "model_type": model_type,
        "split": split,
        "n_features": int(n_features),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
    }


def fit_post_selection_model(X_train: pd.DataFrame, y_train: pd.Series, selected_features: list[str]) -> LogisticRegression:
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train[selected_features], y_train)
    return model


def coefficients_long(model, feature_names: list[str], model_name: str) -> pd.DataFrame:
    rows = []
    for class_idx, raw_cls in enumerate(model.classes_):
        for feature, coef in zip(feature_names, model.coef_[class_idx]):
            rows.append({
                "model": model_name,
                "class_raw": raw_cls,
                "class_display": class_display(raw_cls),
                "feature": feature,
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
                "is_nonzero": bool(abs(coef) > COEF_ZERO_TOL),
            })
    return pd.DataFrame(rows)


def choose_final_model(metrics: pd.DataFrame) -> str:
    """
    Choose the final model primarily by test macro F1, then balanced accuracy,
    then number of features. This keeps the decision reproducible.
    """
    test = metrics[metrics["split"] == "test"].copy()
    test = test.sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy", "n_features"],
        ascending=[False, False, False, True],
    )
    return str(test.iloc[0]["model"])


# ------------------------------------------------------------
# 6. Save final artifacts and plots
# ------------------------------------------------------------

def save_predictions(model, X_test: pd.DataFrame, y_test: pd.Series, filename: str) -> None:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    out = pd.DataFrame({
        "row_id": np.arange(len(y_test)),
        "true_label_raw": y_test.values,
        "true_label_display": label_display(y_test).values,
        "predicted_label_raw": pred,
        "predicted_label_display": label_display(pred).values,
        "prediction_correct": pred == y_test.values,
        "prediction_confidence": proba.max(axis=1),
    })
    for idx, cls in enumerate(model.classes_):
        out[f"prob_{class_display(cls)}"] = proba[:, idx]
    out.to_csv(OUTPUT_DIR / filename, index=False)


def save_confusion(model, X_test: pd.DataFrame, y_test: pd.Series, prefix: str) -> None:
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=CLASS_ORDER_RAW)
    cm_norm = confusion_matrix(y_test, pred, labels=CLASS_ORDER_RAW, normalize="true")
    pd.DataFrame(cm, index=CLASS_ORDER_DISPLAY, columns=CLASS_ORDER_DISPLAY).to_csv(OUTPUT_DIR / f"{prefix}_confusion_matrix_test.csv")
    pd.DataFrame(cm_norm, index=CLASS_ORDER_DISPLAY, columns=CLASS_ORDER_DISPLAY).to_csv(OUTPUT_DIR / f"{prefix}_confusion_matrix_test_normalized.csv")

    for matrix, filename, title in [
        (cm, f"{prefix}_confusion_matrix_test.png", "Final model — test confusion matrix"),
        (cm_norm, f"{prefix}_confusion_matrix_test_normalized.png", "Final model — normalized test confusion matrix"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix)
        ax.set_xticks(range(len(CLASS_ORDER_DISPLAY)))
        ax.set_yticks(range(len(CLASS_ORDER_DISPLAY)))
        ax.set_xticklabels(CLASS_ORDER_DISPLAY, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_ORDER_DISPLAY)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = f"{matrix[i, j]:.2f}" if matrix.dtype.kind == "f" else str(matrix[i, j])
                ax.text(j, i, text, ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _save(fig, filename)


def plot_comparison(metrics: pd.DataFrame) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(test))
    width = 0.35
    ax.bar(x - width/2, test["accuracy"], width, label="Accuracy")
    ax.bar(x + width/2, test["macro_f1"], width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(test["model"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Final supervised model comparison")
    ax.legend()
    _save(fig, "final_model_comparison_accuracy_macro_f1.png")


def plot_class_metrics(report_df: pd.DataFrame) -> None:
    class_rows = report_df[report_df["label_display"].isin(CLASS_ORDER_DISPLAY)].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(class_rows))
    width = 0.25
    ax.bar(x - width, class_rows["precision"], width, label="Precision")
    ax.bar(x, class_rows["recall"], width, label="Recall")
    ax.bar(x + width, class_rows["f1-score"], width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_rows["label_display"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Final model — class-level metrics")
    ax.legend()
    _save(fig, "final_model_class_metrics.png")


def plot_top_coefficients(coefs: pd.DataFrame) -> None:
    top = coefs.sort_values("abs_coefficient", ascending=False).head(TOP_N_COEFFICIENTS).copy()
    top["label"] = top["class_display"] + " | " + top["feature"]
    top = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(top))))
    ax.barh(top["label"], top["coefficient"])
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title("Final model — largest coefficients")
    _save(fig, "final_top_coefficients.png")


# ------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------

def main() -> None:
    X_train, y_train, X_test, y_test, feature_names, class_names, phase3_1_dir = load_data()
    selected_features = load_regularized_selected_features(feature_names)
    lasso, elastic = load_regularized_models()
    post = fit_post_selection_model(X_train, y_train, selected_features)

    with (OUTPUT_DIR / "post_selection_model.pkl").open("wb") as f:
        pickle.dump(post, f)

    rows = []
    if lasso is not None:
        rows.extend([
            evaluate("LASSO", lasso, X_train[lasso.feature_names_in_], y_train, "train", len(lasso.feature_names_in_), "regularized"),
            evaluate("LASSO", lasso, X_test[lasso.feature_names_in_], y_test, "test", len(lasso.feature_names_in_), "regularized"),
        ])
    if elastic is not None:
        rows.extend([
            evaluate("Elastic Net", elastic, X_train[elastic.feature_names_in_], y_train, "train", len(elastic.feature_names_in_), "regularized"),
            evaluate("Elastic Net", elastic, X_test[elastic.feature_names_in_], y_test, "test", len(elastic.feature_names_in_), "regularized"),
        ])
    rows.extend([
        evaluate("Post-selection Logistic", post, X_train[selected_features], y_train, "train", len(selected_features), "post_selection"),
        evaluate("Post-selection Logistic", post, X_test[selected_features], y_test, "test", len(selected_features), "post_selection"),
    ])
    comparison = pd.DataFrame(rows)

    # Add Phase 3.2 baseline metrics when available.
    baseline_metrics_path = PHASE3_2_DIR / "baseline_metrics_train_test.csv"
    if baseline_metrics_path.exists():
        baseline = pd.read_csv(baseline_metrics_path)
        baseline["model"] = "Baseline multinomial"
        baseline["model_type"] = "baseline"
        baseline["n_features"] = len(feature_names)
        baseline = baseline[["model", "model_type", "split", "n_features", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]]
        comparison = pd.concat([baseline, comparison], ignore_index=True)

    comparison.to_csv(OUTPUT_DIR / "full_model_comparison.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "final_model_metrics.csv", index=False)

    final_model_name = choose_final_model(comparison)
    if final_model_name == "LASSO" and lasso is not None:
        final_model = lasso
        final_features = list(lasso.feature_names_in_)
    elif final_model_name == "Elastic Net" and elastic is not None:
        final_model = elastic
        final_features = list(elastic.feature_names_in_)
    elif final_model_name == "Baseline multinomial" and lasso is not None:
        # Baseline model object may not be present in this phase. If baseline wins
        # numerically, use the closest available full-feature regularized model for
        # reproducible predictions while recording the selection rule in JSON.
        final_model = lasso
        final_features = list(lasso.feature_names_in_)
    else:
        final_model = post
        final_features = selected_features

    with (OUTPUT_DIR / "final_model.pkl").open("wb") as f:
        pickle.dump(final_model, f)

    pd.DataFrame({"feature": final_features}).to_csv(OUTPUT_DIR / "final_selected_features.csv", index=False)
    _write_json(final_features, OUTPUT_DIR / "final_selected_feature_names.json")

    save_predictions(final_model, X_test[final_features], y_test, "final_model_predictions_test.csv")
    save_confusion(final_model, X_test[final_features], y_test, "final_model")

    report = classification_report(y_test, final_model.predict(X_test[final_features]), output_dict=True, zero_division=0)
    report_rows = []
    for label, vals in report.items():
        if isinstance(vals, dict):
            row = {"label_raw": label, "label_display": class_display(label)}
            row.update(vals)
            report_rows.append(row)
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(OUTPUT_DIR / "final_model_classification_report_test.csv", index=False)

    coefs = coefficients_long(final_model, final_features, final_model_name)
    coefs.to_csv(OUTPUT_DIR / "final_model_coefficients.csv", index=False)

    plot_comparison(comparison)
    plot_class_metrics(report_df)
    plot_top_coefficients(coefs)

    final_metrics = comparison[(comparison["model"] == final_model_name) & (comparison["split"] == "test")].iloc[0].to_dict()
    summary = {
        "phase": "3.7_final_model_evaluation",
        "input_phase3_1_dir": str(phase3_1_dir),
        "input_phase3_2_dir_exists": PHASE3_2_DIR.exists(),
        "input_phase3_5_dir_exists": PHASE3_5_DIR.exists(),
        "input_phase3_6_dir_exists": PHASE3_6_DIR.exists(),
        "selected_final_model_by_rule": final_model_name,
        "saved_final_model_object": final_model.__class__.__name__,
        "n_final_features": int(len(final_features)),
        "final_test_accuracy": float(final_metrics["accuracy"]),
        "final_test_macro_f1": float(final_metrics["macro_f1"]),
        "class_names": class_names,
        "outputs_dir": str(OUTPUT_DIR),
    }
    _write_json(summary, SUMMARY_FILE)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
