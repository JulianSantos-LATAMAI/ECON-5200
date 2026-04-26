# Unsupervised Learning — Clustering & Dimensionality Reduction

**Course:** ECON 5200: Causal Machine Learning & Applied Analytics — Lab 22

## Objective
Diagnose and repair a broken K-Means pipeline applied to World Development Indicators data, then extend the analysis to customer segmentation using synthetic behavioral data, comparing PCA and UMAP for dimensionality reduction.

## Methodology
- Identified four planted errors in a broken K-Means pipeline: missing feature standardization, incorrect scikit-learn parameter name (`k` vs `n_clusters`), PCA applied before `StandardScaler`, and omitted `random_state`
- Built a corrected pipeline: `StandardScaler` → `KMeans(n_clusters=4, random_state=42)` → PCA visualization
- Applied the corrected pipeline to 9 World Bank WDI indicators across 265 countries
- Generated synthetic customer behavioral data (n=2,000) with 4 latent segments across 6 features
- Compared PCA and UMAP projections of K-Means cluster assignments on the customer data
- Packaged all logic into a reusable `clustering_utils.py` module with `run_kmeans_pipeline()`, `evaluate_k_range()`, and `plot_pca_clusters()`

## Key Findings
- On the WDI country data, the corrected pipeline yielded a silhouette score of **0.18** with cluster sizes of 49, 56, 63, and 97 countries, and PC1 explained **44%** of variance — confirming that standardization successfully distributed explanatory power across all 9 features rather than concentrating it in GDP per capita
- On the customer segmentation data, K-Means achieved a silhouette score of **0.24** with K=4
\
