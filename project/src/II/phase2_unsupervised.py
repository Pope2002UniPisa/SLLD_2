from __future__ import annotations

"""
============================================================
PHASE 2 — UNSUPERVISED ANALYSIS
Statistical Learning and Large Data Project
Scuola Superiore Sant'Anna di Pisa

Goal:
- Apply PCA to the standardised feature space.
- Study the variance explained by the principal components.
- Visualise observations along the first two PCs.
- Apply K-Means and Agglomerative Hierarchical Clustering.
- Select the optimal number of clusters via the elbow method
  (inertia + Hartigan index) and average silhouette score.
- Compare the obtained clusters with the three known class
  labels (noun / verb / adjective).
- Produce a summary JSON and all diagnostic figures.

Input:
  outputs/I/phase1_3_imputation_scaling/slld_phase1_3_train_scaled.csv

Outputs (all under outputs/II/phase2_unsupervised/):
  - pca_scree_plot.png
  - pca_cumulative_variance.png
  - pca_biplot_pc1_pc2.png
  - pca_loadings_pc1.png
  - pca_loadings_pc2.png
  - kmeans_elbow_inertia.png
  - kmeans_silhouette_by_k.png
  - kmeans_clusters_pc12_k2.png
  - kmeans_clusters_pc12_k10.png
  - ahc_dendrogram_k10.png
  - ahc_clusters_pc12_k2.png
  - ahc_clusters_pc12_k10.png
  - cluster_vs_label_kmeans_k3.png
  - cluster_vs_label_ahc_k3.png
  - phase2_summary.json
============================================================
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PHASE1_3_DIR = Path("./outputs/I/phase1_3_imputation_scaling")
OUTPUT_DIR = Path("./outputs/II/phase2_unsupervised")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SCALED_FILE = PHASE1_3_DIR / "slld_phase1_3_train_scaled.csv"
SUMMARY_FILE = OUTPUT_DIR / "phase2_summary.json"


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

ID_COLS = ["entry_id", "word", "target_word_class"]
TARGET_COL = "target_word_class"
RANDOM_STATE = 42

# PCA
N_COMPONENTS_FULL = 65          # run full PCA first
VAR_THRESHOLD = 0.75            # number of PCs retaining 75% variance

# Clustering
K_RANGE = range(2, 16)          # search range for k
K_SEMANTIC = 3                  # compare against the 3 known classes


# ------------------------------------------------------------
# 3. Helpers
# ------------------------------------------------------------

LABEL_MAP = {"noun": "Thing", "verb": "Action", "adjective": "Property"}
COLOR_MAP = {"Thing": "#e07b39", "Action": "#4c7cba", "Property": "#5aa15a"}
MARKER_MAP = {"Thing": "o", "Action": "^", "Property": "s"}


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _label_display(y: pd.Series) -> pd.Series:
    return y.map(LABEL_MAP)


# ------------------------------------------------------------
# 4. Load data
# ------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(TRAIN_SCALED_FILE)
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]
    return df, X, y


# ------------------------------------------------------------
# 5. PCA
# ------------------------------------------------------------

def run_pca(X: pd.DataFrame) -> dict:
    """Fit full PCA, return results dict."""
    pca_full = PCA(n_components=N_COMPONENTS_FULL, random_state=RANDOM_STATE)
    pca_full.fit(X)

    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Number of PCs to reach VAR_THRESHOLD
    n_pcs_75 = int(np.searchsorted(cumulative, VAR_THRESHOLD)) + 1
    # First 2 PCs cumulative variance
    cpve_2 = float(cumulative[1])

    # Scree plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(1, 21), explained[:20] * 100, color="#4c7cba", alpha=0.8)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Scree Plot — Variance Explained by Each PC")
    ax.set_xticks(range(1, 21))
    _save(fig, "pca_scree_plot.png")

    # Cumulative variance
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, N_COMPONENTS_FULL + 1), cumulative * 100,
            color="#e07b39", linewidth=2)
    ax.axhline(75, color="grey", linestyle="--", linewidth=1,
               label="75% threshold")
    ax.axvline(n_pcs_75, color="grey", linestyle=":", linewidth=1,
               label=f"{n_pcs_75} PCs → 75%")
    ax.axhline(cumulative[1] * 100, color="#5aa15a", linestyle="--",
               linewidth=1, label=f"PC1+PC2 = {cpve_2*100:.1f}%")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("Cumulative Proportion of Variance Explained (cPVE)")
    ax.legend()
    _save(fig, "pca_cumulative_variance.png")

    # Biplot on PC1 and PC2
    scores = pca_full.transform(X)
    pca_2 = PCA(n_components=2, random_state=RANDOM_STATE)
    scores_2 = pca_2.fit_transform(X)

    return {
        "pca_full": pca_full,
        "pca_2": pca_2,
        "scores_2": scores_2,
        "scores_full": scores,
        "explained": explained,
        "cumulative": cumulative,
        "n_pcs_75": n_pcs_75,
        "cpve_2": cpve_2,
        "feature_cols": X.columns.tolist(),
    }


def plot_pca_biplot(scores_2: np.ndarray, y: pd.Series,
                   title: str, filename: str) -> None:
    y_display = _label_display(y)
    fig, ax = plt.subplots(figsize=(10, 7))
    for label in ["Thing", "Action", "Property"]:
        mask = y_display == label
        ax.scatter(
            scores_2[mask, 0], scores_2[mask, 1],
            label=label,
            alpha=0.6, s=20,
            color=COLOR_MAP[label],
            marker=MARKER_MAP[label],
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(title="Word class")
    _save(fig, filename)


def plot_pca_loadings(pca_full: PCA, feature_cols: list[str],
                      top_n: int = 12) -> None:
    """Bar chart of top contributing features for PC1 and PC2."""
    loadings = pd.DataFrame(
        pca_full.components_[:2, :].T,
        index=feature_cols,
        columns=["PC1", "PC2"],
    )
    for pc in ["PC1", "PC2"]:
        top = loadings[pc].abs().nlargest(top_n).index
        vals = loadings.loc[top, pc].sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ["#e07b39" if v > 0 else "#4c7cba" for v in vals]
        ax.barh(vals.index, vals.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Top {top_n} feature loadings — {pc}")
        ax.set_xlabel("Loading")
        _save(fig, f"pca_loadings_{pc.lower()}.png")


# ------------------------------------------------------------
# 6. Clustering — search for optimal k
# ------------------------------------------------------------

def search_k_kmeans(X_pca: np.ndarray) -> dict:
    """Elbow (inertia + Hartigan) and silhouette over K_RANGE."""
    inertias, silhouettes, labels_by_k = [], [], {}

    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        lbl = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_pca, lbl))
        labels_by_k[k] = lbl

    # Hartigan index: (inertia_k / inertia_{k+1} - 1) * (n - k - 1)
    n = X_pca.shape[0]
    hartigan = []
    for i in range(len(inertias) - 1):
        k = list(K_RANGE)[i]
        h = (inertias[i] / inertias[i + 1] - 1) * (n - k - 1)
        hartigan.append(h)

    # Elbow / inertia plot
    ks = list(K_RANGE)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ks, inertias, marker="o", color="#4c7cba")
    ax.set_xlabel("k")
    ax.set_ylabel("Within-cluster inertia")
    ax.set_title("K-Means — Elbow Method (Inertia)")
    ax.set_xticks(ks)
    _save(fig, "kmeans_elbow_inertia.png")

    # Silhouette plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ks, silhouettes, marker="o", color="#e07b39")
    ax.set_xlabel("k")
    ax.set_ylabel("Average Silhouette Score")
    ax.set_title("K-Means — Average Silhouette Score by k")
    ax.set_xticks(ks)
    best_k_sil = ks[int(np.argmax(silhouettes))]
    ax.axvline(best_k_sil, color="grey", linestyle="--",
               label=f"Best k = {best_k_sil}")
    ax.legend()
    _save(fig, "kmeans_silhouette_by_k.png")

    # Hartigan plot
    ks_h = list(K_RANGE)[:-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ks_h, hartigan, marker="o", color="#5aa15a")
    ax.set_xlabel("k")
    ax.set_ylabel("Hartigan Index")
    ax.set_title("K-Means — Hartigan Index")
    ax.set_xticks(ks_h)
    _save(fig, "kmeans_elbow_hartigan.png")

    best_sil = float(max(silhouettes))
    return {
        "inertias": inertias,
        "silhouettes": silhouettes,
        "hartigan": hartigan,
        "labels_by_k": labels_by_k,
        "best_k_silhouette": best_k_sil,
        "best_silhouette_score": best_sil,
    }


def plot_kmeans_clusters(scores_2: np.ndarray, labels: np.ndarray,
                         k: int, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = matplotlib.colormaps.get_cmap("tab10").resampled(k)
    for c in range(k):
        mask = labels == c
        ax.scatter(
            scores_2[mask, 0], scores_2[mask, 1],
            label=f"Cluster {c + 1}",
            alpha=0.6, s=20, color=palette(c),
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(title="Cluster", fontsize=7, ncol=2)
    _save(fig, filename)


# ------------------------------------------------------------
# 7. Agglomerative Hierarchical Clustering
# ------------------------------------------------------------

def run_ahc(X_pca: np.ndarray) -> dict:
    """AHC with complete linkage; dendrogram + silhouettes."""
    Z = linkage(X_pca, method="complete", metric="euclidean")

    # Full dendrogram (truncated for readability)
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, ax=ax, truncate_mode="level", p=6,
               no_labels=True, color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title("AHC — Dendrogram (Euclidean, Complete Linkage)")
    ax.set_ylabel("Height")
    _save(fig, "ahc_dendrogram_full.png")

    # Silhouettes for AHC
    ahc_silhouettes, ahc_labels_by_k = [], {}
    for k in K_RANGE:
        lbl = fcluster(Z, t=k, criterion="maxclust") - 1
        s = silhouette_score(X_pca, lbl)
        ahc_silhouettes.append(s)
        ahc_labels_by_k[k] = lbl

    ks = list(K_RANGE)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ks, ahc_silhouettes, marker="o", color="#4c7cba")
    best_k_ahc = ks[int(np.argmax(ahc_silhouettes))]
    ax.axvline(best_k_ahc, color="grey", linestyle="--",
               label=f"Best k = {best_k_ahc}")
    ax.set_xlabel("k")
    ax.set_ylabel("Average Silhouette Score")
    ax.set_title("AHC — Average Silhouette Score by k")
    ax.set_xticks(ks)
    ax.legend()
    _save(fig, "ahc_silhouette_by_k.png")

    # Dendrogram for K=10 (coloured)
    labels_k10 = ahc_labels_by_k[10]
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, ax=ax, truncate_mode="level", p=6, no_labels=True,
               color_threshold=Z[-9, 2])
    ax.set_title("AHC — Dendrogram cut at k = 10")
    ax.set_ylabel("Height")
    _save(fig, "ahc_dendrogram_k10.png")

    return {
        "linkage_matrix": Z,
        "silhouettes": ahc_silhouettes,
        "labels_by_k": ahc_labels_by_k,
        "best_k_silhouette": best_k_ahc,
        "best_silhouette_score": float(max(ahc_silhouettes)),
    }


def plot_ahc_clusters(scores_2: np.ndarray, labels: np.ndarray,
                      k: int, title: str, filename: str) -> None:
    plot_kmeans_clusters(scores_2, labels, k, title, filename)


# ------------------------------------------------------------
# 8. Compare clusters with known labels
# ------------------------------------------------------------

def compare_with_labels(labels: np.ndarray, y: pd.Series,
                        method: str, k: int) -> dict:
    """ARI and NMI between cluster assignments and true labels."""
    y_num = y.map({"noun": 0, "verb": 1, "adjective": 2}).values
    ari = float(adjusted_rand_score(y_num, labels))
    nmi = float(normalized_mutual_info_score(y_num, labels))

    # Cross-tabulation
    df_ct = pd.crosstab(
        pd.Series(labels, name="Cluster"),
        pd.Series(y.map(LABEL_MAP).values, name="Word class"),
    )
    ct_path = OUTPUT_DIR / f"crosstab_{method}_k{k}.csv"
    df_ct.to_csv(ct_path)

    # Heatmap
    fig, ax = plt.subplots(figsize=(6, max(3, k // 2)))
    im = ax.imshow(df_ct.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(df_ct.columns)))
    ax.set_xticklabels(df_ct.columns)
    ax.set_yticks(range(len(df_ct.index)))
    ax.set_yticklabels([f"C{i+1}" for i in df_ct.index])
    ax.set_xlabel("Word class")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Cluster vs label — {method} k={k}\nARI={ari:.3f}  NMI={nmi:.3f}")
    for i in range(df_ct.shape[0]):
        for j in range(df_ct.shape[1]):
            ax.text(j, i, df_ct.values[i, j], ha="center", va="center",
                    fontsize=9, color="black")
    plt.colorbar(im, ax=ax, shrink=0.7)
    _save(fig, f"cluster_vs_label_{method}_k{k}.png")

    return {"ari": ari, "nmi": nmi}


# ------------------------------------------------------------
# 9. Main
# ------------------------------------------------------------

def main() -> None:
    print("=== PHASE 2 — UNSUPERVISED ANALYSIS ===\n")

    # Load
    df, X, y = load_data()
    print(f"Loaded {X.shape[0]} observations, {X.shape[1]} features.\n")

    # --- PCA ---
    print("Running PCA...")
    pca_res = run_pca(X)
    plot_pca_biplot(
        pca_res["scores_2"], y,
        "PCA — Observations on PC1 and PC2 (coloured by word class)",
        "pca_biplot_pc1_pc2.png",
    )
    plot_pca_loadings(pca_res["pca_full"], pca_res["feature_cols"])
    print(f"  PCs needed for 75% variance: {pca_res['n_pcs_75']}")
    print(f"  cPVE of PC1+PC2: {pca_res['cpve_2']*100:.1f}%\n")

    # Use the first 10 PCs as the reduced space for clustering
    # (retaining ~75% variance, consistent with the presentation)
    n_pcs_cluster = pca_res["n_pcs_75"]
    X_pca = pca_res["scores_full"][:, :n_pcs_cluster]
    scores_2 = pca_res["scores_2"]

    # --- K-Means ---
    print("Running K-Means search...")
    km_res = search_k_kmeans(X_pca)
    best_km_k = km_res["best_k_silhouette"]
    print(f"  Best k by silhouette: {best_km_k} "
          f"(sil={km_res['best_silhouette_score']:.3f})")

    for k in [2, 3, 10]:
        plot_kmeans_clusters(
            scores_2, km_res["labels_by_k"][k], k,
            f"K-Means Clustering — k = {k}",
            f"kmeans_clusters_pc12_k{k}.png",
        )
    print()

    # --- AHC ---
    print("Running Agglomerative Hierarchical Clustering...")
    ahc_res = run_ahc(X_pca)
    best_ahc_k = ahc_res["best_k_silhouette"]
    print(f"  Best k by silhouette: {best_ahc_k} "
          f"(sil={ahc_res['best_silhouette_score']:.3f})")

    for k in [2, 3, 10]:
        plot_ahc_clusters(
            scores_2, ahc_res["labels_by_k"][k], k,
            f"AHC Clustering (Complete Linkage) — k = {k}",
            f"ahc_clusters_pc12_k{k}.png",
        )
    print()

    # --- Compare with labels (k = 3, the known number of classes) ---
    print("Comparing clusters with known labels (k=3)...")
    km_comp = compare_with_labels(
        km_res["labels_by_k"][K_SEMANTIC], y, "kmeans", K_SEMANTIC)
    ahc_comp = compare_with_labels(
        ahc_res["labels_by_k"][K_SEMANTIC], y, "ahc", K_SEMANTIC)
    print(f"  K-Means  k=3: ARI={km_comp['ari']:.3f}, NMI={km_comp['nmi']:.3f}")
    print(f"  AHC      k=3: ARI={ahc_comp['ari']:.3f}, NMI={ahc_comp['nmi']:.3f}")
    print()

    # Also compare at best k
    km_comp_best = compare_with_labels(
        km_res["labels_by_k"][best_km_k], y, "kmeans", best_km_k)
    ahc_comp_best = compare_with_labels(
        ahc_res["labels_by_k"][best_ahc_k], y, "ahc", best_ahc_k)

    # --- Summary JSON ---
    summary = {
        "phase": "2_unsupervised_analysis",
        "input_file": str(TRAIN_SCALED_FILE),
        "n_observations": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "pca": {
            "n_components_full": N_COMPONENTS_FULL,
            "n_pcs_75pct_variance": int(pca_res["n_pcs_75"]),
            "cpve_pc1": float(pca_res["explained"][0]),
            "cpve_pc2": float(pca_res["explained"][1]),
            "cpve_pc1_pc2": float(pca_res["cpve_2"]),
            "n_pcs_used_for_clustering": int(n_pcs_cluster),
        },
        "kmeans": {
            "k_range": [int(k) for k in K_RANGE],
            "best_k_by_silhouette": int(best_km_k),
            "best_silhouette_score": round(km_res["best_silhouette_score"], 4),
            "silhouette_k2": round(km_res["silhouettes"][0], 4),
            "silhouette_k3": round(km_res["silhouettes"][1], 4),
            "silhouette_k10": round(km_res["silhouettes"][8], 4),
        },
        "ahc": {
            "linkage": "complete",
            "metric": "euclidean",
            "k_range": [int(k) for k in K_RANGE],
            "best_k_by_silhouette": int(best_ahc_k),
            "best_silhouette_score": round(ahc_res["best_silhouette_score"], 4),
            "silhouette_k2": round(ahc_res["silhouettes"][0], 4),
            "silhouette_k3": round(ahc_res["silhouettes"][1], 4),
            "silhouette_k10": round(ahc_res["silhouettes"][8], 4),
        },
        "comparison_with_labels_k3": {
            "kmeans": {"ari": round(km_comp["ari"], 4), "nmi": round(km_comp["nmi"], 4)},
            "ahc":    {"ari": round(ahc_comp["ari"], 4), "nmi": round(ahc_comp["nmi"], 4)},
        },
        f"comparison_with_labels_best_k_kmeans_{best_km_k}": {
            "ari": round(km_comp_best["ari"], 4),
            "nmi": round(km_comp_best["nmi"], 4),
        },
        f"comparison_with_labels_best_k_ahc_{best_ahc_k}": {
            "ari": round(ahc_comp_best["ari"], 4),
            "nmi": round(ahc_comp_best["nmi"], 4),
        },
        "methodological_decisions": {
            "pca_role": (
                "PCA used as a denoising step before clustering; "
                "the first n_pcs_75pct_variance components are retained"
            ),
            "clustering_space": "PCA-reduced space (first ~10 PCs)",
            "kmeans_n_init": 20,
            "ahc_linkage": "complete",
            "k_selection_criterion": (
                "elbow/Hartigan index for K-Means + "
                "average silhouette score for both methods"
            ),
            "label_comparison_k": K_SEMANTIC,
            "label_comparison_metrics": ["adjusted_rand_score", "normalized_mutual_info"],
        },
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"Summary saved to: {SUMMARY_FILE}")
    print("\n=== PHASE 2 COMPLETED ===\n")


if __name__ == "__main__":
    main()