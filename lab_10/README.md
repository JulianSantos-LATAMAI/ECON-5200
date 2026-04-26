 Lab 10: Correlation, Causality, and Spurious Regression

**Computational Macro Forensics with FRED Data, VIF Diagnostics, and DAG Reasoning**

---

## Overview

This project investigates one of the most dangerous failure modes in applied econometrics: a model that looks statistically valid but rests on non-causal relationships. Using real U.S. macroeconomic data from the Federal Reserve Economic Data (FRED) database, the analysis demonstrates how shared trends, policy reactions, and common shocks can generate strong correlations between variables that have no direct causal connection.

---

## Objective

To prove that macroeconomic variables can appear strongly related in raw data due to secular trend, policy endogeneity, and common aggregate shocks — and to systematically diagnose this problem using correlation heatmaps, Variance Inflation Factor (VIF) analysis, and Directed Acyclic Graph (DAG)-based causal reasoning.

---

## Dataset

- **Source:** Federal Reserve Economic Data (FRED), accessed via `pandas_datareader`
- **Frequency:** Monthly
- **Sample Period:** January 2010 – December 2024
- **Variables:**

| FRED Code | Variable | Role |
|-----------|----------|------|
| CPIAUCSL | Consumer Price Index | Dependent variable (inflation proxy) |
| UNRATE | Unemployment Rate | Labor market condition |
| FEDFUNDS | Federal Funds Rate | Monetary policy instrument |
| INDPRO | Industrial Production Index | Real activity measure |
| RSAFS | Retail Sales | Consumer demand proxy |
| DGS10 | 10-Year Treasury Yield | Financial conditions |
| PAYEMS | Nonfarm Payrolls | Employment level |
| M2SL | M2 Money Supply | Monetary aggregate |

---

## Tech Stack

- **Language:** Python 3.10+
- **Data:** `pandas-datareader` (FRED API)
- **Analysis:** `pandas`, `numpy`, `statsmodels`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Causal Reasoning:** `networkx` (DAG construction)

---

## Methodology

### Phase 1: Manual Forensics

#### 1. Raw Correlation Heatmap
Constructed a correlation matrix from raw variable levels. Strong correlations (r > 0.95) were observed between CPI, payrolls, retail sales, and M2 — not because these variables are causally linked, but because all share a common upward trend driven by secular growth and inflation over the sample period.

#### 2. Naive OLS Regression
Estimated a multiple regression of CPI levels on all macroeconomic predictors. Despite an extremely high R², the model is misleading: the fit largely captures the shared time trend, not any genuine structural relationship. Standard errors are also unreliable under multicollinearity.

#### 3. VIF Diagnostics (Multicollinearity Audit)
Computed Variance Inflation Factors for all predictors. Variables such as M2, payrolls, and retail sales exhibited VIF values in excess of 100, indicating near-perfect collinearity. Predictors were iteratively dropped in order of severity until the remaining VIFs fell within an acceptable range. The regression was re-estimated on the reduced predictor set.

#### 4. YoY Transformation (Stationarity & Trend Removal)
Level variables were converted to year-over-year (YoY) percentage growth rates. This transformation removes the shared long-run trend that drives spurious correlations in levels. The correlation structure changed substantially after transformation — the very high correlations collapsed, and the residual correlations are more likely to reflect genuine cyclical co-movement rather than common drift.

#### 5. DAG-Based Causal Critique
Constructed a Directed Acyclic Graph (DAG) to formalize the causal structure underlying the observed inflation–interest rate correlation. Identified three sources of confounding: reverse causality (policy reaction), common cause (aggregate demand shock), and supply-side confounding.
