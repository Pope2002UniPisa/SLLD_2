from __future__ import annotations

"""PHASE 3.5 — LASSO AND ELASTIC NET AFTER 2210-FEATURE SCREENING."""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

IN = Path("./outputs/III/phase3_1_supervised_data_loading_checks")
SCREEN = Path("./outputs/III/phase3_4_feature_screening/selected_screening_feature_names.json")
OUT = Path("./outputs/III/phase3_5_regularized_models")
OUT.mkdir(parents=True, exist_ok=True)
CLASS_RAW = ["noun", "verb", "adjective"]
CLASS_NAMES = ["Thing", "Action", "Property"]
CLASS_DISPLAY = {"noun": "Thing", "verb": "Action", "adjective": "Property"}
RANDOM_STATE = 42


def write_json(obj, path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def metric_row(y_train, pred_train, y_test, pred_test, name):
    return {
        "model": name,
        "train_accuracy": accuracy_score(y_train, pred_train),
        "train_balanced_accuracy": balanced_accuracy_score(y_train, pred_train),
        "train_macro_f1": f1_score(y_train, pred_train, average="macro", zero_division=0),
        "test_accuracy": accuracy_score(y_test, pred_test),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, pred_test),
        "test_macro_f1": f1_score(y_test, pred_test, average="macro", zero_division=0),
        "test_weighted_f1": f1_score(y_test, pred_test, average="weighted", zero_division=0),
    }


def save_model_outputs(model, name, X_train, X_test, y_train, y_test):
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    row = metric_row(y_train, pred_train, y_test, pred_test, name)

    cm = confusion_matrix(y_test, pred_test, labels=CLASS_RAW)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(OUT / f"{name}_confusion_matrix_test.csv")
    pd.DataFrame(classification_report(y_test, pred_test, labels=CLASS_RAW, target_names=CLASS_NAMES, output_dict=True, zero_division=0)).T.to_csv(OUT / f"{name}_classification_report_test.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": pred_test}).to_csv(OUT / f"{name}_test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm)
    ax.set_xticks(range(3), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(3), CLASS_NAMES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name}: test confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, f"{name}_confusion_matrix_test.png")

    coef = pd.DataFrame(model.coef_, index=[CLASS_DISPLAY.get(c, c) for c in model.classes_], columns=X_train.columns)
    coef.to_csv(OUT / f"{name}_coefficients.csv")
    nz = (coef.abs() > 1e-8).any(axis=0)
    selected = coef.loc[:, nz]
    selected_features = selected.columns.tolist()
    pd.DataFrame({"feature": selected_features}).to_csv(OUT / f"selected_features_{name}.csv", index=False)
    write_json(selected_features, OUT / f"selected_feature_names_{name}.json")

    top_rows = []
    for cls in coef.index:
        top = coef.loc[cls].sort_values(key=np.abs, ascending=False).head(30)
        top_rows.extend([{"model": name, "class": cls, "feature": f, "coefficient": v, "abs_coefficient": abs(v)} for f, v in top.items()])
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(OUT / f"{name}_top_coefficients.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_df = top_df.sort_values("abs_coefficient", ascending=False).head(30).sort_values("abs_coefficient")
    ax.barh(plot_df["class"] + " | " + plot_df["feature"], plot_df["abs_coefficient"])
    ax.set_title(f"Top coefficients: {name}")
    save(fig, f"{name}_top_coefficients.png")

    joblib.dump(model, OUT / f"{name}_model.pkl")
    row["n_screened_features_used"] = int(X_train.shape[1])
    row["n_nonzero_features"] = int(len(selected_features))
    row["best_C_values"] = str(model.C_.tolist())
    return row


def main():
    X_train_all = pd.read_csv(IN / "slld_phase3_1_X_train.csv")
    X_test_all = pd.read_csv(IN / "slld_phase3_1_X_test.csv")
    y_train = pd.read_csv(IN / "slld_phase3_1_y_train.csv")["target_word_class"]
    y_test = pd.read_csv(IN / "slld_phase3_1_y_test.csv")["target_word_class"]
    selected_features = read_json(SCREEN)
    X_train = X_train_all[selected_features]
    X_test = X_test_all[selected_features]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    Cs = np.logspace(-2, 1, 5)
    models = {
        "lasso": LogisticRegressionCV(
            Cs=Cs, cv=cv, penalty="l1", solver="saga", scoring="f1_macro",
            class_weight="balanced", max_iter=6000,
            n_jobs=-1, random_state=RANDOM_STATE, refit=True,
        ),
        "elastic_net": LogisticRegressionCV(
            Cs=Cs, cv=cv, penalty="elasticnet", solver="saga", l1_ratios=[0.25, 0.5, 0.75],
            scoring="f1_macro", class_weight="balanced",
            max_iter=6000, n_jobs=-1, random_state=RANDOM_STATE, refit=True,
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        rows.append(save_model_outputs(model, name, X_train, X_test, y_train, y_test))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "regularized_model_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(metrics["model"], metrics["test_macro_f1"])
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Regularized models after screening from 2210 features")
    save(fig, "regularized_model_comparison.png")

    summary = {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features_before_screening": int(X_train_all.shape[1]),
        "n_features_used_after_screening": int(X_train.shape[1]),
        "models": rows,
    }
    write_json(summary, OUT / "phase3_5_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
