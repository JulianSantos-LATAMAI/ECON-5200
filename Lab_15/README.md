# Forecasting Architecture and the Bias-Variance Tradeoff

## Objective
Empirically diagnose the bias-variance tradeoff by engineering a deliberately overfit polynomial regression model on hyper-volatile NVIDIA quarterly revenue data, then deploy K-Fold Cross-Validation to quantify the model's true out-of-sample operational risk and validate the necessity of L2 regularization for predictive stability.

---

## Methodology

- **Data Ingestion & EDA:** Loaded eight quarters of NVIDIA total corporate revenue (FY2025–FY2026), reflecting the AI infrastructure capital expenditure supercycle. Conducted visual exploratory data analysis to confirm non-linear, exponential curvature in the growth trajectory.

- **High-Bias Baseline (Underfitting):** Fit a standard OLS linear regression to the time index. Computed training MSE to establish a performance floor and visually confirmed systematic underfitting against the exponential revenue curve.

- **High-Variance Model (Overfitting):** Expanded the single time-index feature into a 7-dimensional polynomial feature matrix using `PolynomialFeatures(degree=7)`. Fit a new OLS regression to this space, driving training MSE toward zero — exposing the model's capacity to memorize stochastic noise rather than learn the underlying economic signal.

- **Extrapolation Stress Test:** Forced the overfit model to forecast the immediately subsequent quarter (Q1 FY2027), observing catastrophic hallucination as the polynomial's terminal trajectory diverged violently from any economically plausible revenue estimate.

- **K-Fold Cross-Validation:** Deployed 4-Fold Cross-Validation via `cross_val_score` to rotate holdout sets and evaluate out-of-sample performance rigorously. Compared the near-zero training MSE against the true cross-validated MSE to quantify the variance gap.

- **Ridge Regularization (L2):** Applied `RidgeCV` with automated hyperparameter tuning across a logarithmic alpha grid using 4-Fold CV. The L2 penalty explicitly shrinks polynomial coefficient magnitudes, suppressing the chaotic oscillations responsible for extrapolation failure and recovering predictive stability.

---

## Key Findings

A degree-7 polynomial regression achieved a near-zero training MSE, creating a superficial illusion of model excellence. However, K-Fold Cross-Validation exposed a catastrophically elevated true operational error, with the variance gap confirming that the model had memorized noise rather than internalized the revenue growth dynamic. Forced extrapolation to the next unseen quarter produced a hallucinated forecast with no economic grounding — a direct simulation of the real-world financial risk embedded in overfit algorithmic systems.

Ridge Regularization (L2), with an optimally tuned alpha selected via cross-validation, successfully constrained coefficient magnitude, dramatically narrowing the gap between training and cross-validated error. This demonstrates that model complexity must be governed not by training performance alone, but by regularized, cross-validated generalization — the only credible measure of true predictive capability in volatile financial markets.

---

## Tech Stack
`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib`

**Key scikit-learn modules:** `PolynomialFeatures`, `LinearRegression`, `RidgeCV`, `cross_val_score`, `mean_squared_error`

---

## Data Source
NVIDIA Quarterly Financial Reports — Total Corporate Revenue, FY2025–FY2026 (AI Infrastructure Supercycle Period)
