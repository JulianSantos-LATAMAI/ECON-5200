# High-Dimensional GDP Growth Forecasting with Regularized Regression

## Objective
Forecast five-year average GDP per capita growth across 120+ countries using a high-dimensional set of World Development Indicators, leveraging Ridge and Lasso regularization to diagnose and correct OLS overfitting in a cross-sectional macroeconomic setting.

---

## Data
- **Source:** World Bank World Development Indicators (WDI), retrieved via the `wbgapi` Python API
- **Coverage:** 120+ countries, 35+ indicators, averaging the 2013–2019 period to smooth business-cycle noise
- **Indicator domains:** Trade openness, macroeconomic balances, education attainment, infrastructure access, public health, financial depth, natural resource dependence, agricultural productivity, and institutional governance

---

## Methodology
- **Feature engineering:** Constructed a cross-sectional design matrix by computing country-level period averages across all WDI indicators; applied `StandardScaler` to ensure unit-invariant penalization across heterogeneous economic variables
- **Baseline (OLS):** Estimated an unregularized OLS model to establish an overfitting benchmark — high in-sample fit with poor out-of-sample generalization
- **Regularization:** Applied `RidgeCV` and `LassoCV` from scikit-learn with 5-fold cross-validation over a logarithmic grid of penalty parameters (λ), selecting the optimal λ by held-out mean squared error
- **Coefficient path analysis:** Traced the full Lasso regularization path using `lasso_path()` to visualize sequential predictor shrinkage and identify the λ region that balances sparsity with predictive accuracy
- **Evaluation:** Assessed all models on a held-out test set using R², directly comparing in-sample and out-of-sample performance to quantify the bias-variance tradeoff

---

## Key Findings
- **OLS severely overfits:** Training R² approached 1.0 while test R² was low or negative, confirming that with 35+ correlated predictors and ~100 observations, OLS absorbs noise rather than signal
- **Regularization restores generalization:** Both Ridge and Lasso yielded substantially improved out-of-sample R², demonstrating that penalized regression is essential in high-dimensional macroeconomic forecasting
- **Lasso achieves comparable accuracy with far greater parsimony:** Lasso matched Ridge's test R² while retaining only a sparse subset of predictors — distinguishing variables that are *economically irrelevant* from those that are merely *redundant* given correlated peers
- **Bias-variance tradeoff is empirically visible:** The coefficient path and cross-validated λ selection jointly illustrate the textbook tradeoff — aggressive penalization eliminates variance at the cost of systematic bias, and the optimal λ sits precisely at the inflection point between these competing forces
