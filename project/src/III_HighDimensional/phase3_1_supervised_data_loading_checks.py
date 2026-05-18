from __future__ import annotations

"""PHASE 3.1 — SUPERVISED DATA LOADING CHECKS ON 2210 FEATURES."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

IN_DIR = Path("./outputs/I/phase1_4_high_dimensional_expansion")
TRAIN = IN_DIR / "slld_phase1_4_train_2210_scaled.csv"
TEST = IN_DIR / "slld_phase1_4_test_2210_scaled.csv"
OUT = Path("./outputs/III/phase3_1_supervised_data_loading_checks")
OUT.mkdir(parents=True, exist_ok=True)
ID_COLS = ["entry_id", "word", "target_word_class"]
TARGET_COL = "target_word_class"
LABEL_MAP = {"noun": "Thing", "verb": "Action", "adjective": "Property"}
CLASS_NAMES = ["Thing", "Action", "Property"]


def write_json(obj, path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    for c in ID_COLS:
        if c not in train.columns or c not in test.columns:
            raise ValueError(f"Missing required column: {c}")
    features = [c for c in train.columns if c not in ID_COLS]
    if len(features) != 2210:
        raise ValueError(f"Expected 2210 features, found {len(features)}")
    if features != [c for c in test.columns if c not in ID_COLS]:
        raise ValueError("Train/test features are not aligned.")

    X_train, X_test = train[features], test[features]
    y_train, y_test = train[TARGET_COL], test[TARGET_COL]

    X_train.to_csv(OUT / "slld_phase3_1_X_train.csv", index=False)
    X_test.to_csv(OUT / "slld_phase3_1_X_test.csv", index=False)
    y_train.to_csv(OUT / "slld_phase3_1_y_train.csv", index=False)
    y_test.to_csv(OUT / "slld_phase3_1_y_test.csv", index=False)
    train.to_csv(OUT / "slld_phase3_1_train_modeling_dataset.csv", index=False)
    test.to_csv(OUT / "slld_phase3_1_test_modeling_dataset.csv", index=False)
    write_json(features, OUT / "feature_names.json")
    write_json(CLASS_NAMES, OUT / "class_names.json")

    shape = pd.DataFrame([
        {"split": "train", "n_observations": train.shape[0], "n_features": len(features), "n_columns_with_ids": train.shape[1]},
        {"split": "test", "n_observations": test.shape[0], "n_features": len(features), "n_columns_with_ids": test.shape[1]},
    ])
    shape.to_csv(OUT / "dataset_shape_summary.csv", index=False)

    class_dist = []
    for split, y in [("train", y_train), ("test", y_test)]:
        vc = y.map(LABEL_MAP).value_counts().reindex(CLASS_NAMES).fillna(0).astype(int)
        for cls, n in vc.items():
            class_dist.append({"split": split, "class": cls, "n": int(n), "percentage": float(n / len(y) * 100)})
    class_dist = pd.DataFrame(class_dist)
    class_dist.to_csv(OUT / "class_distribution.csv", index=False)

    missing = pd.DataFrame([
        {"split": "train", "missing_values": int(train.isna().sum().sum())},
        {"split": "test", "missing_values": int(test.isna().sum().sum())},
    ])
    missing.to_csv(OUT / "missing_values_report.csv", index=False)

    dup_rows = pd.concat([train.assign(split="train"), test.assign(split="test")]).duplicated(subset=features + [TARGET_COL], keep=False)
    pd.concat([train.assign(split="train"), test.assign(split="test")])[dup_rows].to_csv(OUT / "duplicate_rows.csv", index=False)
    pd.concat([train[["word", TARGET_COL]].assign(split="train"), test[["word", TARGET_COL]].assign(split="test")]).loc[lambda d: d.duplicated("word", keep=False)].to_csv(OUT / "duplicated_word_forms.csv", index=False)

    fig, ax = plt.subplots(figsize=(7,4))
    for split, g in class_dist.groupby("split"):
        ax.bar(g["class"] + "\n" + split, g["n"])
    ax.set_ylabel("Count")
    ax.set_title("Class distribution: train/test, 2210-feature dataset")
    save(fig, "class_distribution_train_test.png")

    summary = {
        "input_train_file": str(TRAIN),
        "input_test_file": str(TEST),
        "n_train": int(train.shape[0]),
        "n_test": int(test.shape[0]),
        "n_features": int(len(features)),
        "feature_space": "2210 second-degree semantic terms",
        "class_distribution_train": {r["class"]: int(r["n"]) for _, r in class_dist[class_dist.split == "train"].iterrows()},
        "class_distribution_test": {r["class"]: int(r["n"]) for _, r in class_dist[class_dist.split == "test"].iterrows()},
        "missing_values_train": int(train.isna().sum().sum()),
        "missing_values_test": int(test.isna().sum().sum()),
    }
    write_json(summary, OUT / "phase3_1_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
