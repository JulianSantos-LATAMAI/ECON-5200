"""
decompose.py — Reusable time-series decomposition module
=========================================================
Provides classical, STL, and MSTL decomposition, ADF stationarity
testing, PELT structural-break detection, and block-bootstrap
trend confidence intervals.

Author : AI Co-Pilot (P.R.I.M.E. lab extension)
Requires: statsmodels >= 0.14, ruptures, pandas, numpy
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL, MSTL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# ── Optional dependency ──────────────────────────────────────────────────────
try:
    import ruptures as rpt
    _RUPTURES_AVAILABLE = True
except ImportError:
    _RUPTURES_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# 1. CLASSICAL DECOMPOSITION
# ════════════════════════════════════════════════════════════════════════════

def run_classical(
    series: pd.Series,
    period: int,
    model: str = "additive",
) -> seasonal_decompose:
    """
    Classical (moving-average) decomposition.

    Parameters
    ----------
    series  : pd.Series  — univariate time series (datetime index preferred)
    period  : int        — dominant seasonal period (e.g. 12 for monthly)
    model   : str        — "additive" or "multiplicative"
                          Use additive when amplitude is roughly constant;
                          multiplicative when variance grows with level.

    Returns
    -------
    statsmodels DecomposeResult
    """
    _validate_series(series, min_len=2 * period)
    if model not in ("additive", "multiplicative"):
        raise ValueError(f"model must be 'additive' or 'multiplicative', got {model!r}")
    if model == "multiplicative" and (series <= 0).any():
        raise ValueError("Multiplicative decomposition requires strictly positive values.")
    return seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")


# ════════════════════════════════════════════════════════════════════════════
# 2. STL DECOMPOSITION
# ════════════════════════════════════════════════════════════════════════════

def run_stl(
    series: pd.Series,
    period: int,
    robust: bool = True,
    seasonal_deg: int = 1,
) -> STL:
    """
    STL (Seasonal and Trend decomposition using Loess).

    Why STL over classical?
    - Handles any seasonal period (not just 2 or 4 or 12).
    - `robust=True` downweights outliers via iterative re-weighting, so
      a single anomalous spike doesn't bleed into the trend.
    - Seasonal component is allowed to change over time (loess window).

    Parameters
    ----------
    series      : pd.Series
    period      : int          — seasonal period
    robust      : bool         — use outlier-robust loess fitting
    seasonal_deg: int          — 0 = constant seasonal, 1 = locally linear

    Returns
    -------
    statsmodels STLFit (call .plot() or access .trend, .seasonal, .resid)
    """
    _validate_series(series, min_len=2 * period)
    stl = STL(series, period=period, robust=robust, seasonal_deg=seasonal_deg)
    return stl.fit()


# ════════════════════════════════════════════════════════════════════════════
# 3. MSTL DECOMPOSITION
# ════════════════════════════════════════════════════════════════════════════

def run_mstl(
    series: pd.Series,
    periods: Sequence[int],
    windows: Sequence[int] | None = None,
    iterate: int = 2,
    stl_kwargs: dict | None = None,
) -> object:
    """
    MSTL (Multiple STL) — iterative multi-seasonal decomposition.

    How MSTL works
    --------------
    MSTL generalises STL to K seasonal components:
      1. Initialise the seasonally-adjusted series as the raw data.
      2. For each seasonal period s_k (in ascending order):
           a. Apply STL to extract S_k from the current adjusted series.
           b. Remove S_k: adjusted ← adjusted − S_k.
      3. Repeat steps 2–3 for `iterate` full passes until convergence.
      4. Trend T is estimated from the fully adjusted (de-seasonalised)
         series using the final STL loess smoother.
      5. Residual R = Y − T − ΣS_k.

    The ascending-period ordering matters: shorter cycles are extracted
    first so they don't confound the longer-cycle estimation.

    Typical use-cases
    -----------------
    - Hourly electricity load:  periods=(24, 168)       [daily + weekly]
    - Daily retail sales:       periods=(7, 365)         [weekly + annual]
    - Monthly industrial data:  periods=(12, 60)         [annual + 5-yr]

    Parameters
    ----------
    series     : pd.Series           — univariate time series
    periods    : sequence of int     — seasonal periods, e.g. (24, 168)
    windows    : sequence of int     — STL seasonal window for each period;
                                       defaults to (period // 2 * 2 + 1) per period
    iterate    : int                 — number of full MSTL passes (default 2)
    stl_kwargs : dict                — extra kwargs forwarded to each inner STL

    Returns
    -------
    statsmodels MSTLResult
        Attributes: .trend, .seasonal (DataFrame, one col per period),
                    .resid, .weights
    """
    periods = list(periods)
    if len(periods) < 2:
        raise ValueError("MSTL requires at least 2 seasonal periods. Use run_stl for a single period.")
    _validate_series(series, min_len=2 * max(periods))

    if windows is None:
        # Rule-of-thumb: odd window slightly larger than the period
        windows = [_default_stl_window(p) for p in periods]

    stl_kwargs = stl_kwargs or {}
    model = MSTL(
        series,
        periods=periods,
        windows=windows,
        iterate=iterate,
        stl_kwargs=stl_kwargs,
    )
    return model.fit()


# ════════════════════════════════════════════════════════════════════════════
# 4. ADF STATIONARITY TEST
# ════════════════════════════════════════════════════════════════════════════

def run_adf(
    series: pd.Series,
    regression: str = "c",
    autolag: str = "AIC",
) -> dict:
    """
    Augmented Dickey–Fuller unit-root test.

    Common mistake: using regression='nc' (no constant) on data with a
    non-zero mean, or regression='ct' (constant + trend) on data that is
    genuinely trend-stationary — both bias the test statistic.

    Guideline
    ---------
    - Purely mean-stationary data   → regression='c'   (default, most common)
    - Data with deterministic trend → regression='ct'
    - Zero-mean symmetric process   → regression='nc'

    Parameters
    ----------
    series     : pd.Series
    regression : str   — 'c' | 'ct' | 'ctt' | 'nc'
    autolag    : str   — lag selection criterion ('AIC', 'BIC', 't-stat', None)

    Returns
    -------
    dict with keys: stat, pvalue, usedlag, nobs, critical_values, conclusion
    """
    _validate_series(series, min_len=20)
    result = adfuller(series.dropna(), regression=regression, autolag=autolag)
    adf_stat, pvalue, usedlag, nobs, crit_vals, _ = result
    conclusion = "Stationary (reject H₀)" if pvalue < 0.05 else "Non-stationary (fail to reject H₀)"
    return {
        "stat": adf_stat,
        "pvalue": pvalue,
        "usedlag": usedlag,
        "nobs": nobs,
        "critical_values": crit_vals,
        "conclusion": conclusion,
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. STRUCTURAL BREAK DETECTION (PELT)
# ════════════════════════════════════════════════════════════════════════════

def detect_breaks(
    series: pd.Series,
    penalty: float = 10.0,
    model: str = "rbf",
    min_size: int = 10,
) -> list[int]:
    """
    Detect structural break-points using the PELT algorithm (ruptures).

    Why PELT's penalty controls bias-variance tradeoff
    ---------------------------------------------------
    PELT solves:  minimise Σ_k cost(segment_k)  +  λ × (number of breaks)

    - λ (penalty) is a regularisation term penalising model complexity.
    - Low  λ → many break-points (low bias, high variance → overfitting noise).
    - High λ → few break-points  (high bias, low variance → may miss real breaks).
    - BIC-consistent penalty: λ = log(n) × σ²  (the `"bic"` preset in ruptures).
    - In practice, inspect the elbow in the cost-vs-n_breaks curve to choose λ.

    Parameters
    ----------
    series   : pd.Series
    penalty  : float   — regularisation penalty λ (higher = fewer breaks)
    model    : str     — cost function: 'rbf', 'l2', 'l1', 'normal', 'ar'
    min_size : int     — minimum segment length (prevents spurious micro-breaks)

    Returns
    -------
    list of integer positions of detected change-points (excluding endpoint)
    """
    if not _RUPTURES_AVAILABLE:
        raise ImportError("Install ruptures: pip install ruptures")
    _validate_series(series, min_len=2 * min_size)

    signal = series.dropna().values.reshape(-1, 1)
    algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
    breakpoints = algo.predict(pen=penalty)
    # ruptures includes the series end-point; exclude it
    return [bp for bp in breakpoints if bp < len(signal)]


# ════════════════════════════════════════════════════════════════════════════
# 6. BLOCK BOOTSTRAP FOR TREND UNCERTAINTY
# ════════════════════════════════════════════════════════════════════════════

def block_bootstrap_trend(
    series: pd.Series,
    n_bootstrap: int = 500,
    block_size: int = 12,
    stl_period: int | None = None,
    ci_level: float = 0.95,
    random_state: int | None = None,
) -> dict:
    """
    Block bootstrap confidence intervals for the STL trend component.

    Why block bootstrap (not i.i.d. bootstrap)?
    --------------------------------------------
    Classical i.i.d. bootstrap resamples individual observations at random.
    For time series this is wrong: it destroys autocorrelation structure.

    Example: if residuals follow AR(1) with ρ=0.8, consecutive observations
    are highly correlated. Randomly shuffling them produces a white-noise
    surrogate that is statistically quite different from the original process,
    leading to under-coverage of confidence intervals and inflated test power.

    Block bootstrap fixes this by resampling *contiguous blocks* of length L:
    - Within each block, the autocorrelation structure is preserved exactly.
    - Between blocks, dependence is severed — a controlled approximation.
    - As n→∞ and L→∞ (slowly), the estimator is consistent under weak
      stationarity conditions (Künsch 1989; Liu & Singh 1992).

    Choosing block size L:
    - Too small → autocorrelation not captured (approaches i.i.d. bootstrap).
    - Too large → few distinct blocks, high Monte-Carlo variance.
    - Rule-of-thumb: L ≈ n^(1/3) for general dependence; use the dominant
      seasonal period for seasonal data.

    Algorithm
    ---------
    For b in 1..n_bootstrap:
      1. Draw ⌈n/L⌉ blocks with replacement from the series.
      2. Concatenate to length ≥ n, trim to n.
      3. Re-fit STL on the bootstrap replicate.
      4. Store the trend estimate.
    Compute α/2 and 1−α/2 quantiles across bootstrap trends.

    Parameters
    ----------
    series       : pd.Series  — original time series
    n_bootstrap  : int        — number of bootstrap replicates (≥200 recommended)
    block_size   : int        — length L of each contiguous block
    stl_period   : int        — seasonal period for inner STL (auto-detected if None)
    ci_level     : float      — confidence level, e.g. 0.95 for 95% CI
    random_state : int        — seed for reproducibility

    Returns
    -------
    dict with keys:
        trend_mean   : np.ndarray  — mean bootstrap trend
        lower        : np.ndarray  — lower CI bound
        upper        : np.ndarray  — upper CI bound
        ci_level     : float
        n_bootstrap  : int
        block_size   : int
    """
    rng = np.random.default_rng(random_state)
    clean = series.dropna()
    n = len(clean)
    values = clean.values

    if stl_period is None:
        stl_period = _infer_period(clean)
    _validate_series(clean, min_len=2 * stl_period)

    if block_size >= n:
        raise ValueError(f"block_size ({block_size}) must be less than series length ({n}).")

    n_blocks_needed = int(np.ceil(n / block_size))
    trend_matrix = np.empty((n_bootstrap, n))

    for b in range(n_bootstrap):
        # Draw random block start positions (with replacement)
        starts = rng.integers(0, n - block_size + 1, size=n_blocks_needed)
        # Build bootstrap sample by concatenating blocks
        boot_vals = np.concatenate([values[s : s + block_size] for s in starts])[:n]
        boot_series = pd.Series(boot_vals, index=clean.index)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stl_fit = STL(boot_series, period=stl_period, robust=True).fit()
                trend_matrix[b] = stl_fit.trend
            except Exception:
                # If STL fails on a particular replicate, skip it gracefully
                trend_matrix[b] = np.nan

    alpha = 1.0 - ci_level
    lower = np.nanpercentile(trend_matrix, 100 * alpha / 2, axis=0)
    upper = np.nanpercentile(trend_matrix, 100 * (1 - alpha / 2), axis=0)
    mean_trend = np.nanmean(trend_matrix, axis=0)

    return {
        "trend_mean": mean_trend,
        "lower": lower,
        "upper": upper,
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
    }


# ════════════════════════════════════════════════════════════════════════════
# HELPERS (private)
# ════════════════════════════════════════════════════════════════════════════

def _validate_series(series: pd.Series, min_len: int = 4) -> None:
    """Basic sanity checks shared by all public functions."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"Expected pd.Series, got {type(series).__name__}")
    if len(series) < min_len:
        raise ValueError(f"Series too short: need ≥ {min_len} observations, got {len(series)}.")
    if series.isnull().all():
        raise ValueError("Series is entirely NaN.")


def _default_stl_window(period: int) -> int:
    """Odd window slightly larger than period (statsmodels convention)."""
    w = period + (period % 2 == 0)   # make odd
    return max(w + 2, 7)             # at least 7


def _infer_period(series: pd.Series) -> int:
    """
    Heuristically infer dominant seasonal period from DatetimeIndex frequency.
    Falls back to 12 (monthly) if inference fails.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return 12
    freq = pd.infer_freq(series.index)
    if freq is None:
        return 12
    freq_upper = freq.upper() if isinstance(freq, str) else ""
    mapping = {
        "H": 24, "T": 60, "MIN": 60,
        "D": 7,  "B": 5,
        "W": 52, "M": 12, "MS": 12,
        "Q": 4,  "QS": 4,
        "A": 1,  "AS": 1,
    }
    for key, val in mapping.items():
        if freq_upper.startswith(key):
            return val
    return 12
