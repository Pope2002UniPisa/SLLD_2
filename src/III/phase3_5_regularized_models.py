from __future__ import annotations

"""
============================================================
PHASE 3.5 — SUPERVISED ANALYSIS: REGULARIZED MODELS
Statistical Learning and Large Data Project
Scuola Superiore Sant'Anna di Pisa

Goal:
- Load the supervised train/test matrices and the screened feature subset.
- Fit multinomial LASSO logistic regression using an L1 penalty.
- Fit multinomial Elastic Net logistic regression using mixed L1/L2 penalties.
- Tune regularization by stratified cross-validation on the training set.
- Evaluate both models on train and held-out test data.
- Save metrics, predictions, coefficients, selected features and plots.

Input:
  Phase 3.1 X/y files.
  Phase 3.4 selected_screening_feature_names.json, when available.

Outputs (all under outputs/III/phase3_5_regularized_models/):
  - regularized_model_metrics.csv
  - lasso_test_predictions.csv
  - elastic_net_test_predictions.csv
  - lasso_coefficients.csv
  - elastic_net_coefficients.csv
  - selected_features_lasso.csv
  - selected_features_elastic_net.csv
  - lasso_cv_scores.csv
  - elastic_net_cv_scores.csv
  - lasso_confusion_matrix_test.png
  - elastic_net_confusion_matrix_test.png
  - lasso_cv_macro_f1.png
  - elastic_net_cv_macro_f1.png
  - lasso_top_coefficients.png
  - elastic_net_top_coefficients.png
  - regularized_model_comparison.png
  - lasso_model.pkl
  - elastic_net_model.pkl
  - phase3_5_summary.json
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
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

OUTPUT_DIR = Path("./outputs/III/phase3_5_regularized_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = OUTPUT_DIR / "phase3_5_summary.json"

PHASE3_1_CANDIDATES = [
    Path("./outputs/III/phase3_1_supervised_data_loading_checks"),
    Path("./phase3_1_supervised_data_loading_checks"),
    Path("./outputs/phase3_1_supervised_data_loading_checks"),
    Path("./package_phase3_1/outputs/III/phase3_1_supervised_data_loading_checks"),
]
PHASE3_4_CANDIDATES = [
    Path("./outputs/III/phase3_4_feature_screening"),
    Path("./phase3_4_feature_screening"),
]


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

TARGET_COL = "target_word_class"
RANDOM_STATE = 42
N_SPLITS_CV = 5
MAX_ITER = 10000
TOL = 1e-4

X_TRAIN_FILE_NAME = "slld_phase3_1_X_train.csv"
Y_TRAIN_FILE_NAME = "slld_phase3_1_y_train.csv"
X_TEST_FILE_NAME = "slld_phase3_1_X_test.csv"
Y_TEST_FILE_NAME = "slld_phase3_1_y_test.csv"
FEATURE_NAMES_FILE_NAME = "feature_names.json"
CLASS_NAMES_FILE_NAME = "class_names.json"
SCREENED_FEATURES_FILE_NAME = "selected_screening_feature_names.json"

LABEL_MAP = {"noun": "Thing", "verb": "Action", "adjective": "Property"}
DISPLAY_TO_RAW = {v: k for k, v in LABEL_MAP.items()}
CLASS_ORDER_DISPLAY = ["Thing", "Action", "Property"]
CLASS_ORDER_RAW = [DISPLAY_TO_RAW[c] for c in CLASS_ORDER_DISPLAY]

# C is the inverse of regularization strength. Small C = stronger shrinkage.
CS_GRID = np.logspace(-3, 2, 10)
ELASTIC_NET_L1_RATIOS = [0.10, 0.30, 0.50, 0.70, 0.90]
TOP_N_COEFFICIENTS = 25
COEF_ZERO_TOL = 1e-8


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


def resolve_screened_features(feature_names: list[str]) -> tuple[list[str], Path | None]:
    for candidate in PHASE3_4_CANDIDATES:
        path = candidate / SCREENED_FEATURES_FILE_NAME
        if path.exists():
            selected = load_json_list(path)
            selected = [f for f in selected if f in feature_names]
            if selected:
                return selected, path
    return feature_names, None


def label_display(y: pd.Series | np.ndarray | list[str]) -> pd.Series:
    s = pd.Series(y)
    return s.map(LABEL_MAP).fillna(s.astype(str))


def class_display(raw_label: str) -> str:
    return LABEL_MAP.get(raw_label, str(raw_label))


# ------------------------------------------------------------
# 4. Load data
# ------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str], Path, Path | None]:
    phase3_1_dir = resolve_phase3_1_dir()
    X_train = pd.read_csv(phase3_1_dir / X_TRAIN_FILE_NAME)
    X_test = pd.read_csv(phase3_1_dir / X_TEST_FILE_NAME)
    y_train = pd.read_csv(phase3_1_dir / Y_TRAIN_FILE_NAME)[TARGET_COL]
    y_test = pd.read_csv(phase3_1_dir / Y_TEST_FILE_NAME)[TARGET_COL]
    feature_names = load_json_list(phase3_1_dir / FEATURE_NAMES_FILE_NAME)
    class_names = load_json_list(phase3_1_dir / CLASS_NAMES_FILE_NAME)

    screened_features, screened_path = resolve_screened_features(feature_names)
    X_train = X_train[screened_features]
    X_test = X_test[screened_features]
    return X_train, y_train, X_test, y_test, screened_features, class_names, phase3_1_dir, screened_path


# ------------------------------------------------------------
# 5. Model fitting and evaluation
# ------------------------------------------------------------

def fit_lasso(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegressionCV:
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    model = LogisticRegressionCV(
        Cs=CS_GRID,
        cv=cv,
        penalty="l1",
        solver="saga",
        scoring="f1_macro",
        max_iter=MAX_ITER,
        tol=TOL,
        n_jobs=-1,
        refit=True,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def fit_elastic_net(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegressionCV:
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    model = LogisticRegressionCV(
        Cs=CS_GRID,
        cv=cv,
        penalty="elasticnet",
        solver="saga",
        l1_ratios=ELASTIC_NET_L1_RATIOS,
        scoring="f1_macro",
        max_iter=MAX_ITER,
        tol=TOL,
        n_jobs=-1,
        refit=True,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(name: str, model, X: pd.DataFrame, y: pd.Series, split: str) -> dict:
    pred = model.predict(X)
    return {
        "model": name,
        "split": split,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
    }


def save_predictions(name: str, model, X: pd.DataFrame, y: pd.Series, filename: str) -> None:
    pred = model.predict(X)
    proba = model.predict_proba(X)
    out = pd.DataFrame({
        "row_id": np.arange(len(y)),
        "true_label_raw": y.values,
        "true_label_display": label_display(y).values,
        "predicted_label_raw": pred,
        "predicted_label_display": label_display(pred).values,
        "prediction_correct": pred == y.values,
        "prediction_confidence": proba.max(axis=1),
    })
    for idx, cls in enumerate(model.classes_):
        out[f"prob_{class_display(cls)}"] = proba[:, idx]
    out.to_csv(OUTPUT_DIR / filename, index=False)


def coefficients_long(model, feature_names: list[str], model_name: str) -> pd.DataFrame:
    rows = []
    for class_idx, raw_cls in enumerate(model.classes_):
        display_cls = class_display(raw_cls)
        for feature, coef in zip(feature_names, model.coef_[class_idx]):
            rows.append({
                "model": model_name,
                "class_raw": raw_cls,
                "class_display": display_cls,
                "feature": feature,
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
                "is_nonzero": bool(abs(coef) > COEF_ZERO_TOL),
            })
    return pd.DataFrame(rows)


def selected_features_from_coefficients(coefs: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = (
        coefs.groupby("feature", as_index=False)
        .agg(
            max_abs_coefficient=("abs_coefficient", "max"),
            n_active_classes=("is_nonzero", "sum"),
        )
    )
    out["model"] = model_name
    out["selected"] = out["max_abs_coefficient"] > COEF_ZERO_TOL
    return out.sort_values(["selected", "max_abs_coefficient", "feature"], ascending=[False, False, True])


# ------------------------------------------------------------
# 6. CV score extraction and plots
# ------------------------------------------------------------

def cv_scores_table(model: LogisticRegressionCV, model_name: str) -> pd.DataFrame:
    rows = []
    for raw_cls, arr in model.scores_.items():
        # LASSO: folds x Cs. Elastic Net: folds x Cs x l1_ratios.
        arr = np.asarray(arr)
        if arr.ndim == 2:
            for fold_idx in range(arr.shape[0]):
                for c_idx, C in enumerate(model.Cs_):
                    rows.append({
                        "model": model_name,
                        "class_raw": raw_cls,
                        "class_display": class_display(raw_cls),
                        "fold": fold_idx + 1,
                        "C": float(C),
                        "l1_ratio": np.nan,
                        "macro_f1_cv": float(arr[fold_idx, c_idx]),
                    })
        elif arr.ndim == 3:
            for fold_idx in range(arr.shape[0]):
                for c_idx, C in enumerate(model.Cs_):
                    for r_idx, ratio in enumerate(model.l1_ratios_):
                        rows.append({
                            "model": model_name,
                            "class_raw": raw_cls,
                            "class_display": class_display(raw_cls),
                            "fold": fold_idx + 1,
                            "C": float(C),
                            "l1_ratio": float(ratio),
                            "macro_f1_cv": float(arr[fold_idx, c_idx, r_idx]),
                        })
    return pd.DataFrame(rows)


def plot_cv_scores(cv_table: pd.DataFrame, filename: str, title: str) -> None:
    grouped = (
        cv_table.groupby(["C", "l1_ratio"], dropna=False)["macro_f1_cv"]
        .mean()
        .reset_index()
        .sort_values("C")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    if grouped["l1_ratio"].isna().all():
        ax.plot(grouped["C"], grouped["macro_f1_cv"], marker="o")
    else:
        for ratio, sub in grouped.groupby("l1_ratio"):
            ax.plot(sub["C"], sub["macro_f1_cv"], marker="o", label=f"l1_ratio={ratio:.2f}")
        ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("C, inverse regularization strength")
    ax.set_ylabel("Mean CV macro F1")
    ax.set_title(title)
    _save(fig, filename)


def plot_confusion(model, X_test: pd.DataFrame, y_test: pd.Series, filename: str, title: str) -> None:
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=CLASS_ORDER_RAW)
    pd.DataFrame(cm, index=CLASS_ORDER_DISPLAY, columns=CLASS_ORDER_DISPLAY).to_csv(OUTPUT_DIR / filename.replace(".png", ".csv"))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(CLASS_ORDER_DISPLAY)))
    ax.set_yticks(range(len(CLASS_ORDER_DISPLAY)))
    ax.set_xticklabels(CLASS_ORDER_DISPLAY, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_ORDER_DISPLAY)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, filename)


def plot_top_coefficients(coefs: pd.DataFrame, filename: str, title: str) -> None:
    top = coefs.sort_values("abs_coefficient", ascending=False).head(TOP_N_COEFFICIENTS).copy()
    top["label"] = top["class_display"] + " | " + top["feature"]
    top = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(top))))
    ax.barh(top["label"], top["coefficient"])
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title(title)
    _save(fig, filename)


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(test))
    width = 0.35
    ax.bar(x - width/2, test["accuracy"], width, label="Accuracy")
    ax.bar(x + width/2, test["macro_f1"], width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(test["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Regularized model comparison on test set")
    ax.legend()
    _save(fig, "regularized_model_comparison.png")


# ------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------

def main() -> None:
    X_train, y_train, X_test, y_test, features, class_names, phase3_1_dir, screened_path = load_data()

    lasso = fit_lasso(X_train, y_train)
    elastic = fit_elastic_net(X_train, y_train)

    with (OUTPUT_DIR / "lasso_model.pkl").open("wb") as f:
        pickle.dump(lasso, f)
    with (OUTPUT_DIR / "elastic_net_model.pkl").open("wb") as f:
        pickle.dump(elastic, f)

    metrics = pd.DataFrame([
        evaluate_model("LASSO", lasso, X_train, y_train, "train"),
        evaluate_model("LASSO", lasso, X_test, y_test, "test"),
        evaluate_model("Elastic Net", elastic, X_train, y_train, "train"),
        evaluate_model("Elastic Net", elastic, X_test, y_test, "test"),
    ])
    metrics.to_csv(OUTPUT_DIR / "regularized_model_metrics.csv", index=False)

    save_predictions("LASSO", lasso, X_test, y_test, "lasso_test_predictions.csv")
    save_predictions("Elastic Net", elastic, X_test, y_test, "elastic_net_test_predictions.csv")

    report_rows = []
    for model_name, model in [("LASSO", lasso), ("Elastic Net", elastic)]:
        report = classification_report(y_test, model.predict(X_test), output_dict=True, zero_division=0)
        for label, vals in report.items():
            if isinstance(vals, dict):
                row = {"model": model_name, "label_raw": label, "label_display": class_display(label)}
                row.update(vals)
                report_rows.append(row)
    pd.DataFrame(report_rows).to_csv(OUTPUT_DIR / "regularized_classification_report_test.csv", index=False)

    lasso_coefs = coefficients_long(lasso, features, "LASSO")
    elastic_coefs = coefficients_long(elastic, features, "Elastic Net")
    lasso_coefs.to_csv(OUTPUT_DIR / "lasso_coefficients.csv", index=False)
    elastic_coefs.to_csv(OUTPUT_DIR / "elastic_net_coefficients.csv", index=False)

    lasso_selected = selected_features_from_coefficients(lasso_coefs, "LASSO")
    elastic_selected = selected_features_from_coefficients(elastic_coefs, "Elastic Net")
    lasso_selected.to_csv(OUTPUT_DIR / "selected_features_lasso.csv", index=False)
    elastic_selected.to_csv(OUTPUT_DIR / "selected_features_elastic_net.csv", index=False)
    _write_json(lasso_selected.loc[lasso_selected["selected"], "feature"].tolist(), OUTPUT_DIR / "selected_feature_names_lasso.json")
    _write_json(elastic_selected.loc[elastic_selected["selected"], "feature"].tolist(), OUTPUT_DIR / "selected_feature_names_elastic_net.json")

    lasso_cv = cv_scores_table(lasso, "LASSO")
    elastic_cv = cv_scores_table(elastic, "Elastic Net")
    lasso_cv.to_csv(OUTPUT_DIR / "lasso_cv_scores.csv", index=False)
    elastic_cv.to_csv(OUTPUT_DIR / "elastic_net_cv_scores.csv", index=False)

    plot_cv_scores(lasso_cv, "lasso_cv_macro_f1.png", "LASSO — CV macro F1 by C")
    plot_cv_scores(elastic_cv, "elastic_net_cv_macro_f1.png", "Elastic Net — CV macro F1 by C and l1_ratio")
    plot_confusion(lasso, X_test, y_test, "lasso_confusion_matrix_test.png", "LASSO — Test confusion matrix")
    plot_confusion(elastic, X_test, y_test, "elastic_net_confusion_matrix_test.png", "Elastic Net — Test confusion matrix")
    plot_top_coefficients(lasso_coefs, "lasso_top_coefficients.png", "LASSO — largest coefficients")
    plot_top_coefficients(elastic_coefs, "elastic_net_top_coefficients.png", "Elastic Net — largest coefficients")
    plot_model_comparison(metrics)

    summary = {
        "phase": "3.5_regularized_models",
        "input_phase3_1_dir": str(phase3_1_dir),
        "screened_features_file": str(screened_path) if screened_path else None,
        "n_features_used": int(len(features)),
        "lasso_best_C": float(np.ravel(lasso.C_)[0]),
        "elastic_net_best_C": float(np.ravel(elastic.C_)[0]),
        "elastic_net_best_l1_ratio": float(np.ravel(elastic.l1_ratio_)[0]),
        "lasso_selected_features": int(lasso_selected["selected"].sum()),
        "elastic_net_selected_features": int(elastic_selected["selected"].sum()),
        "lasso_test_macro_f1": float(metrics[(metrics.model == "LASSO") & (metrics.split == "test")]["macro_f1"].iloc[0]),
        "elastic_net_test_macro_f1": float(metrics[(metrics.model == "Elastic Net") & (metrics.split == "test")]["macro_f1"].iloc[0]),
        "outputs_dir": str(OUTPUT_DIR),
    }
    _write_json(summary, SUMMARY_FILE)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
