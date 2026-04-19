# Time Series Diagnostics & Advanced Decomposition

> A robust, reusable time series analysis pipeline that diagnoses common decomposition misspecifications, applies appropriate transformations, and extracts statistically rigorous structural insights from macroeconomic and energy demand data.

---

## Table of Contents

- [Objective](#objective)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Module Reference](#module-reference)
- [Tech Stack](#tech-stack)

---

## Objective

Develop a robust, reusable time series analysis pipeline that diagnoses common decomposition misspecifications, applies appropriate transformations, and extracts statistically rigorous structural insights from macroeconomic and energy demand data.

---

## Methodology

### 1. STL Decomposition Correction
Identified an additive model misspecification on multiplicative data; applied log-transformation to stabilize variance prior to decomposition, then back-transformed components for interpretability.

### 2. ADF Test Recalibration
Corrected a misspecified Augmented Dickey-Fuller test by selecting the appropriate regression parameter (`ct` for data with drift and trend), eliminating false stationarity conclusions.

### 3. Multi-Seasonal Decomposition (MSTL)
Applied MSTL to hourly electricity demand data to simultaneously extract daily and weekly seasonal cycles, isolating the underlying trend component from overlapping periodicities.

### 4. Moving Block Bootstrap
Implemented block bootstrap resampling on GDP trend residuals to construct empirically grounded uncertainty bands, preserving autocorrelation structure absent in naive i.i.d. resampling.

### 5. Structural Break Detection
Applied the PELT (Pruned Exact Linear Time) algorithm to detect changepoints in the GDP series; conducted regime-specific ADF stationarity tests to assess integration order within each sub-period.

### 6. Module Engineering
Packaged all methods into a production-ready `decompose.py` module exposing three public interfaces with consistent return signatures for pipeline composability.

---

## Key Findings

- **GDP Integration Order:** GDP exhibits integration of order one — I(1) — confirmed by failure to reject the unit root null on levels and rejection on first differences across all regression specifications.
- **Structural Breaks:** Breaks identified near **[YOUR DATES]** partition the series into regimes with meaningfully distinct drift and volatility characteristics. Any forecasting model treating the full sample as homogeneous would be misspecified.
- **Electricity Demand:** Weekly seasonality accounts for the dominant share of predictable variation beyond the daily cycle, with trend innovations concentrated in early morning hours.

---

## Module Reference

```python
from decompose import run_stl, test_stationarity, detect_breaks

# Run STL decomposition with automatic log-transform detection
trend, seasonal, resid = run_stl(series, period=12, robust=True)

# Test stationarity with correct regression specification
result = test_stationarity(series, regression="ct", autolag="AIC")

# Detect structural breaks using PELT
breakpoints = detect_breaks(series, model="rbf", penalty="bic")
```

| Function | Description |
|---|---|
| `run_stl()` | STL decomposition with optional log-transform |
| `test_stationarity()` | ADF test with configurable regression parameter |
| `detect_breaks()` | PELT-based structural break detection |

---

## Tech Stack

| Library | Purpose |
|---|---|
| `statsmodels` | STL / MSTL decomposition, ADF testing |
| `ruptures` | PELT structural break detection |
| `sklearn` | Preprocessing utilities |
| `pandas` | Data wrangling and time series indexing |
| `numpy` | Numerical computation and bootstrap resampling |
