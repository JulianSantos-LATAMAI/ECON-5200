# Lab 5: The Architecture of Bias

> *"Bad data fed into a perfect model produces a perfect prediction of the wrong thing."*

## Overview

An applied investigation into the **Data Generating Process (DGP)** — the upstream mechanisms that determine *which* observations enter a dataset before any model ever sees them. This lab treats data collection not as a solved problem but as the primary source of model failure, demonstrating that bias is architectural, not accidental.

**Tech Stack:** Python · pandas · NumPy · SciPy · scikit-learn

---

## Methodology

### 1. Simple Random Sampling — Demonstrating Variance
Manually simulated random sampling on the Titanic dataset (`n=891`) by generating a permuted index array with `np.random.permutation`, then slicing an 80/20 train/test split. The exercise quantified **sampling error** — the survival rate delta between train and test sets — exposing how a naive split produces non-deterministic covariate distributions across runs.

### 2. Stratified Sampling — Eliminating Covariate Shift
Replaced the manual split with `sklearn.model_selection.train_test_split(stratify=df['pclass'])`. By sampling each passenger class independently, the procedure mathematically enforces identical class distributions in both partitions. This eliminates **covariate shift** — the condition where `P(X)` differs between train and test — which would otherwise cause a model's learned decision boundary to be evaluated on a different population than it was trained on.

### 3. Sample Ratio Mismatch (SRM) Forensic Audit
Simulated a corrupted A/B test (450 Control / 550 Treatment against a planned 500/500 split) and applied `scipy.stats.chisquare` to diagnose the failure. The resulting Chi-Square statistic of **10.0** (`p = 0.0016`) decisively rejected the null hypothesis of a fair assignment mechanism at α = 0.01 — confirming an **engineering failure**, not random variance. SRM invalidates downstream lift measurements regardless of metric significance, making this check a prerequisite to any experiment analysis.

---

## Theoretical Extension: Survivorship Bias & the Heckman Correction

### Why Analyzing TechCrunch Unicorns Produces Survivorship Bias

A dataset scraped from TechCrunch coverage of unicorn startups (`valuation ≥ $1B`) contains only companies that **survived long enough to be covered**. This is a canonical case of **survivorship bias**: the sample is conditioned on an outcome (survival + visibility) that is itself correlated with the features under study (funding strategy, founding team composition, market timing).

The population you *observe* is:

```
P(X | Company survived AND was covered by TechCrunch)
```

The population you *want to model* is:

```
P(X | Company was founded)
```

Any regression trained on the observed sample will systematically overestimate the effect of features common to survivors (e.g., "raised a Series A") because the counterfactual — companies that raised a Series A and *still failed* — is structurally absent from the data.

### The Ghost Data Required: Selection Mechanism Variables

To apply a **Heckman Selection Correction**, you need two distinct datasets:

| Data Type | Description | Example Variables |
|---|---|---|
| **Outcome Equation Data** | The unicorn sample (already have this) | Valuation, growth rate, team size |
| **Selection Equation Data** (Ghost Data) | Records of *all founded companies*, including failures | Founding date, initial funding, industry, dissolution date |

The Heckman procedure works in two stages:

**Stage 1 — Model the Selection Mechanism:**
Using the full population (survivors + failures), fit a Probit model predicting the probability that a company enters your observed sample:

```python
# P(observed = 1 | Z) — Z includes instruments not in outcome equation
probit_model = sm.Probit(observed_flag, selection_features).fit()
df['imr'] = probit_model.predict()  # Inverse Mills Ratio (λ)
```

**Stage 2 — Correct the Outcome Equation:**
Include the **Inverse Mills Ratio (λ)** as a regressor in your main model. This term absorbs the portion of the error term correlated with selection, rendering the OLS estimates consistent:

```python
outcome_model = sm.OLS(valuation, outcome_features + ['imr']).fit()
```

The critical requirement is at least one **instrument** — a variable that predicts *selection into the sample* (TechCrunch coverage) but has no direct effect on the *outcome* (valuation). Without ghost data on failed companies, Stage 1 cannot be estimated and the correction is impossible. The bias is therefore not fixable with better modeling — it requires better data collection upstream, at the DGP level.

---

## Key Concepts

| Concept | Definition |
|---|---|
| **Sampling Bias** | Systematic error from non-representative data collection |
| **Covariate Shift** | `P_train(X) ≠ P_test(X)` — different feature distributions across splits |
| **Sample Ratio Mismatch** | Assignment mechanism failure in A/B tests; detected via Chi-Square |
| **Survivorship Bias** | Conditioning analysis on a post-hoc outcome filter |
| **Ghost Data** | Unobserved records excluded by the selection mechanism |
| **Heckman Correction** | Two-stage estimator that adjusts for non-random sample selection |
