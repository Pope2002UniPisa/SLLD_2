from __future__ import annotations

"""PHASE 3.4 — MARGINAL FEATURE SCREENING ON ALL 2210 FEATURES."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

IN = Path("./outputs/III/phase3_1_supervised_data_loading_checks")
OUT = Path("./outputs/III/phase3_4_feature_screening")
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
SCREEN_TOP_N = 300


def write_json(obj, path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def percentile_rank_desc(s: pd.Series) -> pd.Series:
    return s.rank(ascending=False, method="average") / len(s)


def main():
    X = pd.read_csv(IN / "slld_phase3_1_X_train.csv")
    y = pd.read_csv(IN / "slld_phase3_1_y_train.csv")["target_word_class"]
    n, p = X.shape
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    f_scores, f_p = f_classif(X, y_enc)
    anova = pd.DataFrame({"feature": X.columns, "anova_f": f_scores, "anova_pvalue": f_p}).replace([np.inf, -np.inf], np.nan).fillna(0)
    anova = anova.sort_values("anova_f", ascending=False)
    anova.to_csv(OUT / "anova_fscore_ranking.csv", index=False)

    mi_scores = mutual_info_classif(X, y_enc, discrete_features=False, random_state=RANDOM_STATE, n_neighbors=3)
    mi = pd.DataFrame({"feature": X.columns, "mutual_information": mi_scores}).sort_values("mutual_information", ascending=False)
    mi.to_csv(OUT / "mutual_information_ranking.csv", index=False)

    # Fast marginal supervised association score: maximum absolute standardized class-mean contrast.
    # This keeps the screening step on all 2210 features without fitting 2210 separate models.
    contrast_rows = []
    global_mean = X.mean(axis=0)
    for cls in sorted(y.unique()):
        cls_mean = X.loc[y == cls].mean(axis=0)
        contrast_rows.append((cls_mean - global_mean).abs())
    max_contrast = pd.concat(contrast_rows, axis=1).max(axis=1)
    uni = pd.DataFrame({"feature": X.columns, "max_abs_class_mean_contrast": max_contrast.values}).sort_values("max_abs_class_mean_contrast", ascending=False)
    uni.to_csv(OUT / "marginal_class_contrast_ranking.csv", index=False)

    merged = anova.merge(mi, on="feature").merge(uni, on="feature")
    merged["rank_anova"] = merged["anova_f"].rank(ascending=False, method="average")
    merged["rank_mi"] = merged["mutual_information"].rank(ascending=False, method="average")
    merged["rank_class_contrast"] = merged["max_abs_class_mean_contrast"].rank(ascending=False, method="average")
    merged["combined_rank"] = merged[["rank_anova", "rank_mi", "rank_class_contrast"]].mean(axis=1)
    merged = merged.sort_values("combined_rank")
    merged.to_csv(OUT / "combined_screening_ranking.csv", index=False)

    selected = merged.head(SCREEN_TOP_N).copy()
    selected.to_csv(OUT / "selected_screening_features.csv", index=False)
    write_json(selected["feature"].tolist(), OUT / "selected_screening_feature_names.json")

    pd.DataFrame({
        "n_observations_train": [n],
        "n_features_before_screening": [p],
        "n_features_after_screening": [SCREEN_TOP_N],
        "n_over_log_n": [n / np.log(n)],
        "n_minus_1": [n - 1],
    }).to_csv(OUT / "screening_dataset_summary.csv", index=False)

    for metric, fname, title in [
        ("anova_f", "anova_top_features.png", "Top ANOVA F-scores"),
        ("mutual_information", "mutual_information_top_features.png", "Top mutual information scores"),
        ("max_abs_class_mean_contrast", "marginal_class_contrast_top_features.png", "Top marginal class-mean contrasts"),
    ]:
        top = merged.sort_values(metric, ascending=False).head(30).sort_values(metric)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.barh(top["feature"], top[metric])
        ax.set_title(title + " — 2210 features")
        save(fig, fname)

    top = selected.sort_values("combined_rank", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(top["feature"].tail(30), top["combined_rank"].tail(30))
    ax.set_title("Best selected features by combined screening rank")
    save(fig, "combined_screening_top_features.png")

    summary = {
        "n_observations_train": int(n),
        "n_features_before_screening": int(p),
        "screen_top_n": int(SCREEN_TOP_N),
        "n_features_after_screening": int(selected.shape[0]),
        "fan_lv_n_over_log_n": float(n / np.log(n)),
        "selected_feature_examples": selected["feature"].head(20).tolist(),
    }
    write_json(summary, OUT / "phase3_4_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
