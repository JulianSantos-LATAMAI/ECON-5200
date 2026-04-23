# Tree-Based Models — Random Forests
## ECON 5200: Causal Machine Learning & Applied Analytics

## Objective
Benchmark nonparametric tree-based models against regularized OLS on the California Housing dataset, diagnose common evaluation and interpretability pitfalls, and deliver production-grade SHAP explanations via a reusable Python module.

## Methodology
- Loaded California Housing data (20,640 observations, 8 features) and established a Ridge regression baseline
- Identified and corrected a train/test leakage bug in model evaluation that inflated Random Forest R² above 0.97
- Diagnosed causal overclaiming from MDI feature importance and connected the flaw to DAG-based confounding (Ch. 10) and prediction vs. explanation (Ch. 15)
- Tuned Random Forest hyperparameters (`n_estimators`, `max_depth`, `max_features`) via GridSearchCV with 3-fold cross-validation
- Fit a GradientBoostingRegressor baseline and compared Test RMSE and R² across Ridge, RF (default), RF (tuned), and GBR
- Computed and contrasted MDI vs. permutation feature importance to surface MDI's high-cardinality bias
- Generated SHAP waterfall plots for three individual predictions and a beeswarm plot for global feature attribution using `shap.TreeExplainer`
- Built a reusable `src/shap_utils.py` module with full type hints, docstrings, and error handling
- Developed an interactive Plotly + ipywidgets dashboard with live sliders for `n_estimators` and `max_features`

## Key Findings
- RF (tuned) and GBR both substantially outperformed Ridge, confirming the value of nonparametric flexibility on this dataset
- RF achieved R² = **[YOUR VALUE]** vs. Ridge R² = **[YOUR VALUE]** on the held-out test set
- MDI and SHAP rankings largely agreed on `MedInc` as the dominant predictor, but diverged on mid-tier features — consistent with MDI's known bias toward high-cardinality continuous variables
- Marginal gains from additional trees plateau beyond ~200 estimators, suggesting diminishing returns to further tuning

## Repository Structure
