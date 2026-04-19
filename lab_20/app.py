"""
app.py — Interactive FRED Time-Series Decomposition Explorer
=============================================================
Run with:  streamlit run app.py

Dependencies:
    pip install streamlit plotly fredapi statsmodels ruptures pandas numpy

Set your key:
    export FRED_API_KEY="your_key_here"
    # or create a .env file with FRED_API_KEY=...

What this app reveals about parameter sensitivity
--------------------------------------------------
Decomposition results are NOT objective facts — they are model outputs that
depend heavily on the analyst's choices:

1. Classical vs STL vs MSTL
   Switching from additive classical to STL on multiplicative data (e.g.,
   industrial production) dramatically changes the residual variance; MSTL
   can further isolate a weekly sub-cycle invisible to single-period methods.

2. Seasonal period
   A wrong period (e.g., 6 instead of 12 for monthly data) creates aliased
   residuals and a trend that absorbs the missing seasonality.

3. STL robustness flag
   robust=True produces a smoother trend in the presence of spikes; robust=False
   lets outliers pull the trend toward them, inflating the residual elsewhere.

4. PELT penalty
   Low penalty fragments the trend into many break-point segments, each
   appearing "structural." High penalty may miss genuine regime changes.

5. Bootstrap block size
   Too-small blocks understate autocorrelation → CIs too narrow (overconfident).
   Too-large blocks → very few distinct blocks → high Monte-Carlo variance.

The app makes these sensitivities *interactive* so the analyst sees them
empirically rather than taking model outputs at face value.
"""

from __future__ import annotations

import os
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Local module (must be in same directory or on PYTHONPATH) ────────────────
try:
    from decompose import (
        block_bootstrap_trend,
        detect_breaks,
        run_adf,
        run_classical,
        run_mstl,
        run_stl,
        _infer_period,
    )
except ImportError as exc:
    st.error(
        f"Could not import decompose.py — make sure it lives in the same directory.\n{exc}"
    )
    st.stop()

# ── Optional FRED API ────────────────────────────────────────────────────────
try:
    from fredapi import Fred
    _FRED_AVAILABLE = True
except ImportError:
    _FRED_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLE
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FRED Decomposition Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* Dark theme accent colours */
        :root {
            --accent: #00d4aa;
            --accent2: #ff6b6b;
            --bg-card: rgba(255,255,255,0.04);
        }
        section[data-testid="stSidebar"] {background: #0e1117;}
        .stMetric {background: var(--bg-card); border-radius: 8px; padding: 12px;}
        h1, h2, h3 {letter-spacing: -0.5px;}
        .caption-box {
            background: var(--bg-card);
            border-left: 3px solid var(--accent);
            padding: 10px 14px;
            border-radius: 0 8px 8px 0;
            font-size: 0.85rem;
            color: #aaa;
            margin: 8px 0 16px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred(series_id: str, api_key: str) -> pd.Series:
    """Download and cache a FRED series (1-hr TTL)."""
    fred = Fred(api_key=api_key)
    raw = fred.get_series(series_id)
    return raw.dropna().sort_index()


def _plotly_theme() -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="monospace", size=12),
    )


def _add_breaks_to_fig(
    fig: go.Figure,
    series: pd.Series,
    breakpoints: list[int],
    row: int = 1,
    col: int = 1,
) -> None:
    """Overlay vertical lines at detected structural break positions."""
    idx = series.index
    for bp in breakpoints:
        if bp < len(idx):
            fig.add_vline(
                x=idx[bp],
                line_dash="dash",
                line_color="#ff6b6b",
                opacity=0.7,
                row=row,
                col=col,
                annotation_text="break",
                annotation_font_color="#ff6b6b",
            )


def _decomp_subplots(
    series: pd.Series,
    trend: np.ndarray | pd.Series,
    seasonal: np.ndarray | pd.Series,
    resid: np.ndarray | pd.Series,
    title: str,
    breakpoints: list[int] | None = None,
) -> go.Figure:
    """4-panel decomposition figure: observed / trend / seasonal / residual."""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
    )
    idx = series.index
    panels = [
        (series.values, "#a8d8ea", "Observed"),
        (np.asarray(trend), "#00d4aa", "Trend"),
        (np.asarray(seasonal), "#f7c59f", "Seasonal"),
        (np.asarray(resid), "#ff6b6b", "Residual"),
    ]
    for row, (y, colour, name) in enumerate(panels, start=1):
        fig.add_trace(
            go.Scatter(x=idx, y=y, mode="lines", line=dict(color=colour, width=1.2), name=name),
            row=row,
            col=1,
        )
    if breakpoints:
        _add_breaks_to_fig(fig, series, breakpoints, row=2, col=1)

    fig.update_layout(height=700, title=title, showlegend=False, **_plotly_theme())
    return fig


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — user inputs
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙️ Configuration")

    # ── FRED credentials ─────────────────────────────────────────────────
    st.subheader("FRED API")
    api_key = st.text_input(
        "API Key",
        value=os.getenv("FRED_API_KEY", ""),
        type="password",
        help="Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html",
    )
    series_id = st.text_input(
        "Series ID",
        value="INDPRO",
        help="Examples: INDPRO (industrial production), UNRATE (unemployment), "
             "CPIAUCSL (CPI), HOUST (housing starts)",
    ).strip().upper()

    st.divider()

    # ── Decomposition method ─────────────────────────────────────────────
    st.subheader("Decomposition")
    method = st.radio(
        "Method",
        ["Classical", "STL", "MSTL"],
        horizontal=True,
    )

    period = st.slider("Primary period", min_value=2, max_value=365, value=12, step=1,
                       help="Dominant seasonal cycle (e.g. 12 for monthly annual seasonality)")

    if method == "Classical":
        cl_model = st.selectbox("Model", ["additive", "multiplicative"])

    if method == "STL":
        robust = st.checkbox("Robust STL", value=True,
                             help="Downweights outliers via iterative re-weighting")

    if method == "MSTL":
        period2 = st.slider("Secondary period", min_value=2, max_value=365, value=52,
                            help="Second seasonal cycle (e.g. 52 for weekly in daily data)")
        mstl_iterate = st.slider("MSTL iterations", 1, 5, 2)

    st.divider()

    # ── Structural breaks ─────────────────────────────────────────────────
    st.subheader("Structural Breaks (PELT)")
    show_breaks = st.checkbox("Detect breaks", value=True)
    pelt_penalty = st.slider(
        "PELT penalty λ",
        min_value=1.0,
        max_value=100.0,
        value=15.0,
        step=1.0,
        help="Higher λ → fewer breaks. Controls bias-variance tradeoff.",
    )
    pelt_model = st.selectbox("Cost function", ["rbf", "l2", "l1", "normal"])

    st.divider()

    # ── Block bootstrap ───────────────────────────────────────────────────
    st.subheader("Block Bootstrap CI")
    run_boot = st.checkbox("Compute bootstrap CI", value=False,
                           help="Computationally intensive — runs 100–500 STL fits")
    n_boot = st.slider("Bootstrap replicates", 100, 500, 200, step=50)
    block_size = st.slider(
        "Block size L",
        min_value=2,
        max_value=60,
        value=max(2, period // 2),
        help="Larger blocks preserve more autocorrelation but reduce block diversity",
    )
    ci_level = st.select_slider("CI level", options=[0.80, 0.90, 0.95, 0.99], value=0.95)
    boot_seed = st.number_input("Random seed", value=42, step=1)


# ════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ════════════════════════════════════════════════════════════════════════════

st.title("📈 FRED Decomposition Explorer")
st.markdown(
    '<div class="caption-box">Fetch any FRED series · Compare Classical / STL / MSTL '
    '· Detect structural breaks · Bootstrap trend uncertainty</div>',
    unsafe_allow_html=True,
)

if not _FRED_AVAILABLE:
    st.error("fredapi not installed.  Run:  pip install fredapi")
    st.stop()

if not api_key:
    st.info("👈 Enter your FRED API key in the sidebar to begin.")
    st.stop()

# ── Fetch data ──────────────────────────────────────────────────────────────
with st.spinner(f"Fetching {series_id} from FRED…"):
    try:
        raw = fetch_fred(series_id, api_key)
    except Exception as exc:
        st.error(f"FRED fetch failed: {exc}")
        st.stop()

if raw.empty:
    st.warning("Series returned no data. Check the series ID.")
    st.stop()

# Infer frequency label
inferred_freq = pd.infer_freq(raw.index) or "?"

# ── Summary metrics ─────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Observations", f"{len(raw):,}")
col2.metric("Start", str(raw.index[0].date()))
col3.metric("End", str(raw.index[-1].date()))
col4.metric("Frequency", inferred_freq)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# ADF STATIONARITY TEST
# ════════════════════════════════════════════════════════════════════════════

st.subheader("🧪 Stationarity Tests")
adf_col, diff_col = st.columns(2)

with adf_col:
    st.markdown("**ADF on original series** (`regression='c'`)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_raw = run_adf(raw, regression="c")
    st.metric("ADF Statistic", f"{adf_raw['stat']:.4f}")
    st.metric("p-value", f"{adf_raw['pvalue']:.4f}")
    verdict_colour = "🟢" if "Stationary" in adf_raw["conclusion"] else "🔴"
    st.markdown(f"{verdict_colour} **{adf_raw['conclusion']}**")

with diff_col:
    st.markdown("**ADF on first difference**")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_diff = run_adf(raw.diff().dropna(), regression="c")
    st.metric("ADF Statistic", f"{adf_diff['stat']:.4f}")
    st.metric("p-value", f"{adf_diff['pvalue']:.4f}")
    verdict_colour2 = "🟢" if "Stationary" in adf_diff["conclusion"] else "🔴"
    st.markdown(f"{verdict_colour2} **{adf_diff['conclusion']}**")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# STRUCTURAL BREAK DETECTION
# ════════════════════════════════════════════════════════════════════════════

breakpoints: list[int] = []
if show_breaks:
    st.subheader("🔴 Structural Break Detection (PELT)")
    try:
        with st.spinner("Running PELT…"):
            breakpoints = detect_breaks(raw, penalty=pelt_penalty, model=pelt_model, min_size=period)
        if breakpoints:
            bp_dates = [str(raw.index[bp].date()) for bp in breakpoints]
            st.success(f"Detected **{len(breakpoints)}** break(s) at: {', '.join(bp_dates)}")
            st.markdown(
                '<div class="caption-box">'
                "PELT penalty λ={:.0f} — increase λ to merge nearby breaks; "
                "decrease to surface subtler regime shifts."
                "</div>".format(pelt_penalty),
                unsafe_allow_html=True,
            )
        else:
            st.info("No structural breaks detected at this penalty level.")
    except Exception as exc:
        st.warning(f"Break detection skipped: {exc}")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# DECOMPOSITION
# ════════════════════════════════════════════════════════════════════════════

st.subheader(f"🔬 {method} Decomposition")

try:
    with st.spinner(f"Running {method} decomposition…"):
        if method == "Classical":
            result = run_classical(raw, period=period, model=cl_model)
            trend_arr   = result.trend
            seasonal_arr = result.seasonal
            resid_arr   = result.resid
            seasonal_label = f"Seasonal (period={period})"

        elif method == "STL":
            result = run_stl(raw, period=period, robust=robust)
            trend_arr   = result.trend
            seasonal_arr = result.seasonal
            resid_arr   = result.resid
            seasonal_label = f"Seasonal (period={period})"

        else:  # MSTL
            periods_list = sorted([period, period2])
            result = run_mstl(raw, periods=periods_list, iterate=mstl_iterate)
            trend_arr   = result.trend
            # MSTL returns seasonal as a DataFrame; sum all seasonal components for plot
            if hasattr(result.seasonal, "sum"):
                seasonal_arr = result.seasonal.sum(axis=1)
            else:
                seasonal_arr = result.seasonal
            resid_arr   = result.resid
            seasonal_label = f"Seasonal (periods={periods_list})"

    fig_decomp = _decomp_subplots(
        raw,
        trend_arr,
        seasonal_arr,
        resid_arr,
        title=f"{series_id} — {method} Decomposition",
        breakpoints=breakpoints if show_breaks else None,
    )
    st.plotly_chart(fig_decomp, use_container_width=True)

    # MSTL: also show individual seasonal components
    if method == "MSTL" and hasattr(result.seasonal, "columns"):
        st.markdown("**Individual MSTL seasonal components**")
        fig_seas = go.Figure(layout=dict(height=300, **_plotly_theme()))
        colours = ["#f7c59f", "#a8d8ea", "#c7f2a4"]
        for i, col in enumerate(result.seasonal.columns):
            fig_seas.add_trace(
                go.Scatter(
                    x=raw.index,
                    y=result.seasonal[col],
                    mode="lines",
                    name=str(col),
                    line=dict(color=colours[i % len(colours)], width=1.1),
                )
            )
        st.plotly_chart(fig_seas, use_container_width=True)
        st.markdown(
            '<div class="caption-box">'
            "MSTL iteratively removes each seasonal component from shortest to longest "
            "period, ensuring they don't confound one another."
            "</div>",
            unsafe_allow_html=True,
        )

except Exception as exc:
    st.error(f"Decomposition failed: {exc}")
    st.stop()

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# BLOCK BOOTSTRAP TREND CI
# ════════════════════════════════════════════════════════════════════════════

if run_boot:
    st.subheader(f"🎲 Block Bootstrap Trend CI  ({int(ci_level*100)}%)")
    with st.spinner(f"Running {n_boot} bootstrap replicates (block_size={block_size})…"):
        try:
            boot = block_bootstrap_trend(
                raw,
                n_bootstrap=n_boot,
                block_size=block_size,
                stl_period=period,
                ci_level=ci_level,
                random_state=int(boot_seed),
            )

            fig_boot = go.Figure(layout=dict(height=400, **_plotly_theme()))
            idx = raw.dropna().index

            # Shaded CI band
            fig_boot.add_trace(
                go.Scatter(
                    x=np.concatenate([idx, idx[::-1]]),
                    y=np.concatenate([boot["upper"], boot["lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(0,212,170,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{int(ci_level*100)}% CI",
                )
            )
            # Mean bootstrap trend
            fig_boot.add_trace(
                go.Scatter(
                    x=idx,
                    y=boot["trend_mean"],
                    mode="lines",
                    line=dict(color="#00d4aa", width=2),
                    name="Bootstrap mean trend",
                )
            )
            # Original STL trend (for comparison)
            if method in ("STL", "Classical", "MSTL"):
                fig_boot.add_trace(
                    go.Scatter(
                        x=idx,
                        y=np.asarray(trend_arr)[: len(idx)],
                        mode="lines",
                        line=dict(color="#f7c59f", width=1.5, dash="dot"),
                        name="Original trend",
                    )
                )

            fig_boot.update_layout(
                title=(
                    f"{series_id} — Block Bootstrap Trend CI  "
                    f"(n={n_boot}, block={block_size}, seed={boot_seed})"
                ),
                yaxis_title="Value",
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_boot, use_container_width=True)

            st.markdown(
                f'<div class="caption-box">'
                f"Block size = {block_size} preserves autocorrelation within blocks. "
                f"Try reducing it to ~2 to see how CIs narrow artificially when "
                f"autocorrelation is destroyed (i.i.d. bootstrap)."
                f"</div>",
                unsafe_allow_html=True,
            )

        except Exception as exc:
            st.error(f"Bootstrap failed: {exc}")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# RAW DATA DOWNLOAD
# ════════════════════════════════════════════════════════════════════════════

with st.expander("📥 Download raw series"):
    csv = raw.reset_index().rename(columns={"index": "date", 0: series_id}).to_csv(index=False)
    st.download_button("Download CSV", csv, file_name=f"{series_id}.csv", mime="text/csv")

st.caption(
    "Decompose Explorer · Built with Streamlit, Plotly, statsmodels, ruptures · "
    "Data via FRED (St. Louis Fed)"
)
