
## Time Series Forecasting — ARIMA, GARCH & Bootstrap

**Objective:** Diagnose and correct a broken time series pipeline, then extend it with
volatility modeling and distribution-free forecast uncertainty quantification using
S&P 500 and CPI data.

**Methodology:**
- Identified three deliberate errors in a broken ARIMA pipeline: incorrect differencing
  order (d=0 on I(2) CPI series), omitted seasonal structure, and missing Ljung-Box
  residual diagnostic
- Corrected pipeline to SARIMA(2,2,1)(0,0,2,12), verified stationarity via ADF on
  second-differenced CPI and confirmed clean residuals via Ljung-Box
- Fit GARCH(1,1) to S&P 500 daily log returns (2000–2024) to model time-varying
  conditional volatility
- Built a reusable forecast_evaluation.py module implementing compute_mase() and
  backtest_expanding_window() for production-grade model evaluation
- Implemented moving block bootstrap to generate distribution-free forecast intervals
  robust to volatility clustering and heavy tails

**Key Findings:**
- CPI required two differences (d=2) to achieve stationarity — consistent with
  inflation itself being non-stationary over the post-2020 period
- GARCH(1,1) estimated alpha+beta = 0.983, confirming highly persistent but
  stationary volatility in S&P 500 returns
- Volatility half-life of 39.5 days implies shocks like the COVID crash (peak
  conditional volatility 6.85% on 2020-03-17) take roughly six weeks to decay
  to half their peak magnitude
- GARCH long-run volatility (1.16%) closely matches sample volatility (1.22%),
  confirming model calibration
