from __future__ import annotations

"""PHASE 3.7 — FINAL MODEL EVALUATION FOR THE 2210-FEATURE PIPELINE."""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

DATA = Path("./outputs/III/phase3_1_supervised_data_loading_checks")
REG = Path("./outputs/III/phase3_5_regularized_models")
SP = Path("./outputs/III/phase3_6_sparsity_analysis")
OUT = Path("./outputs/III/phase3_7_final_model_evaluation")
OUT.mkdir(parents=True, exist_ok=True)
CLASS_RAW = ["noun", "verb", "adjective"]
CLASS_NAMES = ["Thing", "Action", "Property"]
CLASS_DISPLAY = {"noun": "Thing", "verb": "Action", "adjective": "Property"}


def write_json(obj, path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def main():
    comparison = pd.read_csv(REG / "regularized_model_metrics.csv")
    best_name = comparison.sort_values("test_macro_f1", ascending=False).iloc[0]["model"]
    selected_features = pd.read_csv(REG / f"selected_features_{best_name}.csv")["feature"].tolist()
    if not selected_features:
        selected_features = pd.read_json(REG / "../phase3_4_feature_screening/selected_screening_feature_names.json").iloc[:,0].tolist()

    X_train_all = pd.read_csv(DATA / "slld_phase3_1_X_train.csv")
    X_test_all = pd.read_csv(DATA / "slld_phase3_1_X_test.csv")
    y_train = pd.read_csv(DATA / "slld_phase3_1_y_train.csv")["target_word_class"]
    y_test = pd.read_csv(DATA / "slld_phase3_1_y_test.csv")["target_word_class"]
    X_train = X_train_all[selected_features]
    X_test = X_test_all[selected_features]

    final = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", class_weight="balanced",
        max_iter=3000,
    )
    final.fit(X_train, y_train)
    pred_train = final.predict(X_train)
    pred_test = final.predict(X_test)

    rows = []
    rows.append({"model": f"post_selection_logistic_from_{best_name}", "split": "train", **compute_metrics(y_train, pred_train)})
    rows.append({"model": f"post_selection_logistic_from_{best_name}", "split": "test", **compute_metrics(y_test, pred_test)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "final_model_metrics.csv", index=False)
    comparison.to_csv(OUT / "full_model_comparison.csv", index=False)

    cm = confusion_matrix(y_test, pred_test, labels=CLASS_RAW)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(OUT / "final_model_confusion_matrix_test.csv")
    pd.DataFrame(classification_report(y_test, pred_test, labels=CLASS_RAW, target_names=CLASS_NAMES, output_dict=True, zero_division=0)).T.to_csv(OUT / "final_model_classification_report_test.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": pred_test}).to_csv(OUT / "final_model_predictions_test.csv", index=False)
    pd.DataFrame({"feature": selected_features}).to_csv(OUT / "final_selected_features.csv", index=False)
    write_json(selected_features, OUT / "final_selected_feature_names.json")

    coef = pd.DataFrame(final.coef_, index=[CLASS_DISPLAY.get(c, c) for c in final.classes_], columns=selected_features)
    coef.to_csv(OUT / "final_model_coefficients.csv")

    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm)
    ax.set_xticks(range(3), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(3), CLASS_NAMES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Final post-selection model")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "final_model_confusion_matrix_test.png")

    fig, ax = plt.subplots(figsize=(7,4))
    plot_df = pd.concat([
        comparison[["model", "test_accuracy", "test_macro_f1"]].rename(columns={"test_accuracy":"accuracy", "test_macro_f1":"macro_f1"}),
        pd.DataFrame([{"model": "final_post_selection", "accuracy": rows[1]["accuracy"], "macro_f1": rows[1]["macro_f1"]}])
    ])
    x = range(len(plot_df))
    ax.bar([i - 0.2 for i in x], plot_df["accuracy"], width=0.4, label="accuracy")
    ax.bar([i + 0.2 for i in x], plot_df["macro_f1"], width=0.4, label="macro-F1")
    ax.set_xticks(list(x), plot_df["model"], rotation=25, ha="right")
    ax.legend()
    ax.set_title("Final comparison")
    save(fig, "final_model_comparison_accuracy_macro_f1.png")

    joblib.dump(final, OUT / "final_model.pkl")
    summary = {
        "best_regularized_model_by_test_macro_f1": best_name,
        "n_features_before_screening": int(X_train_all.shape[1]),
        "n_features_in_final_post_selection_model": int(len(selected_features)),
        "test_metrics": rows[1],
    }
    write_json(summary, OUT / "phase3_7_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
