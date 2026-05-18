from __future__ import annotations

"""PHASE 3.2 — BASELINE MODELS ON THE 2210-FEATURE SPACE."""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

IN = Path("./outputs/III/phase3_1_supervised_data_loading_checks")
OUT = Path("./outputs/III/phase3_2_train_test_baseline")
OUT.mkdir(parents=True, exist_ok=True)
CLASS_RAW = ["noun", "verb", "adjective"]
CLASS_DISPLAY = {"noun": "Thing", "verb": "Action", "adjective": "Property"}
CLASS_NAMES = ["Thing", "Action", "Property"]
RANDOM_STATE = 42


def write_json(obj, path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def save_confusion(y_true, y_pred, prefix):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_RAW)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(OUT / f"{prefix}_confusion_matrix_test.csv")
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm)
    ax.set_xticks(range(3), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(3), CLASS_NAMES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(prefix.replace("_", " "))
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, f"{prefix}_confusion_matrix_test.png")


def main():
    X_train = pd.read_csv(IN / "slld_phase3_1_X_train.csv")
    X_test = pd.read_csv(IN / "slld_phase3_1_X_test.csv")
    y_train = pd.read_csv(IN / "slld_phase3_1_y_train.csv")["target_word_class"]
    y_test = pd.read_csv(IN / "slld_phase3_1_y_test.csv")["target_word_class"]

    rows = []
    models = {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "ridge_logistic_all_2210": LogisticRegression(
            penalty="l2", C=0.1, solver="lbfgs",
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
        ),
    }
    for name, model in models.items():
        # No cross-validation here: this phase is only a fast baseline on the full 2210-feature matrix.
        # Model selection is performed later after feature screening.
        cv_scores = np.array([np.nan])
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        m_train = metrics(y_train, pred_train)
        m_test = metrics(y_test, pred_test)
        row = {"model": name, **{f"train_{k}": v for k, v in m_train.items()}, **{f"test_{k}": v for k, v in m_test.items()}, "cv_macro_f1_mean": cv_scores.mean(), "cv_macro_f1_sd": cv_scores.std()}
        rows.append(row)
        save_confusion(y_test, pred_test, name)
        pd.DataFrame(classification_report(y_test, pred_test, labels=CLASS_RAW, target_names=CLASS_NAMES, output_dict=True, zero_division=0)).T.to_csv(OUT / f"{name}_classification_report_test.csv")
        pd.DataFrame({"y_true": y_test, "y_pred": pred_test}).to_csv(OUT / f"{name}_test_predictions.csv", index=False)
        joblib.dump(model, OUT / f"{name}_model.pkl")

        if hasattr(model, "coef_"):
            coef = pd.DataFrame(model.coef_, index=[CLASS_DISPLAY.get(c, c) for c in model.classes_], columns=X_train.columns)
            coef.to_csv(OUT / f"{name}_coefficients.csv")
            top = []
            for cls in coef.index:
                s = coef.loc[cls].sort_values(key=np.abs, ascending=False).head(30)
                top.extend([{"class": cls, "feature": f, "coefficient": v, "abs_coefficient": abs(v)} for f, v in s.items()])
            pd.DataFrame(top).to_csv(OUT / f"{name}_top_coefficients_by_class.csv", index=False)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT / "baseline_model_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(metrics_df["model"], metrics_df["test_macro_f1"])
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Baseline comparison on 2210 features")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "baseline_model_comparison.png")

    summary = {"n_train": int(X_train.shape[0]), "n_test": int(X_test.shape[0]), "n_features_used": int(X_train.shape[1]), "models": rows}
    write_json(summary, OUT / "phase3_2_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
