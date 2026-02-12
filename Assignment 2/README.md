# Audit 02: Deconstructing Statistical Lies

## Overview
This audit investigates three common statistical deceptions that plague real-world data analysis: Latency Skew, False Positives, and Survivorship Bias. Through hands-on Python analysis, we expose how these biases distort conclusions and lead to catastrophic business decisions.

---

## 🚨 Finding 1: Latency Skew - Why Standard Deviation Lies

### The Data Generation Process
Simulated 1,000 server latency measurements:
- **980 normal requests:** 20-50ms (typical traffic)
- **20 spike requests:** 1,000-5,000ms (system failures/DDoS)

### The Deceptive Metric: Standard Deviation
```python
std_dev = np.std(latency_logs)
mad_value = calculate_mad(latency_logs)

print(f"Standard Deviation: {std_dev:.2f}")  # 427.36 ms
print(f"MAD: {mad_value:.2f}")                # 8.00 ms
```

**Results:**
- **Standard Deviation:** 427.36 ms
- **Median Absolute Deviation (MAD):** 8.00 ms

### The Problem
Standard deviation suggests the "typical" latency is 427ms, when 98% of requests are actually **20-50ms**. The 20 outliers (2%) completely distort the metric because standard deviation squares deviations, amplifying extreme values exponentially.

### Why MAD is Superior
MAD uses the **median** of absolute deviations, making it robust to outliers:
1. Calculate median of data
2. Find absolute deviations from median
3. Take median of those deviations

**MAD = 8.00ms** accurately reflects that most requests deviate by only ~8ms from the median, revealing the true system performance.

### Key Lesson
**Standard deviation is not "wrong"—it's measuring something different.** Use MAD for skewed data with outliers (latency, income, prices). Use standard deviation only for normal distributions.

---

## 🎯 Finding 2: False Positives - The IntegrityAI Paradox

### The Scenario
IntegrityAI claims 98% sensitivity and 98% specificity for detecting academic cheating. Sounds impressive, but **base rates determine everything**.

### Three Test Cases

#### Scenario A: Bootcamp (50% Base Rate)
```python
bayesian_audit(prior=0.50, sensitivity=0.98, specificity=0.98)
# Result: 98.0% probability of cheating if flagged
```
✅ **High confidence** - Half of students cheat, so flags are reliable.

#### Scenario B: Econ Class (5% Base Rate)
```python
bayesian_audit(prior=0.05, sensitivity=0.98, specificity=0.98)
# Result: 72.1% probability of cheating if flagged
```
⚠️ **Moderate confidence** - 28% of flags are innocent students.

#### Scenario C: Honors Seminar (0.1% Base Rate)
```python
bayesian_audit(prior=0.001, sensitivity=0.98, specificity=0.98)
# Result: 4.7% probability of cheating if flagged
```
❌ **CATASTROPHIC** - 95% of flagged students are innocent!

### The Mathematics (Bayes' Theorem)
```
P(Cheater | Flagged) = [Sensitivity × Prior] / 
                        [Sensitivity × Prior + (1-Specificity) × (1-Prior)]
```

### Key Lesson
A "98% accurate" test becomes **95% wrong** when the base rate is low. Before trusting any detection system, always ask: **"How common is the thing we're detecting?"**

**Real-world applications:**
- Medical screening (rare diseases)
- Fraud detection (most transactions are legitimate)
- Security screening (most people are not threats)

---

## 🔍 Finding 3: A/B Test Validity - The FinFlash Investigation

### The Claim
FinFlash ran an A/B test with 100,000 users claiming a 50/50 split and "huge win."

### The Suspicious Data
- **Control Users:** 50,250
- **Treatment Users:** 49,750
- **Difference:** 500 users missing from Treatment

### Chi-Square Goodness of Fit Test
```python
observed = np.array([50250, 49750])
expected = np.array([50000, 50000])
chi_square = np.sum((observed - expected)**2 / expected)
# Result: χ² = 2.5
```

### The Verdict
**χ² = 2.5 < 3.84 (critical value at p < 0.05)**

✅ **EXPERIMENT IS VALID** - The 500-user difference falls within normal random variation. No systematic randomization failure detected.

### Important Nuance
While the **randomization is statistically valid**, FinFlash's claim of a "perfect 50/50 split" is **technically inaccurate**. The actual split is 50.25% / 49.75%. However, this deviation is expected and acceptable in random assignment.

### Key Lesson
Not every imbalance indicates engineering failure. Chi-Square tests distinguish **random noise** from **systematic bias**. Always verify randomization quality before trusting A/B test results.

---

## 💀 Finding 4: Survivorship Bias - The Crypto Graveyard

### The Simulation
Modeled 10,000 cryptocurrency token launches using a **Pareto Distribution (Power Law)** where 99% of tokens fail or remain near-zero value.

### The Shocking Results
```
Total Tokens Launched: 10,000
Survivors (Top 1%): 100
Failed/Obscure (Bottom 99%): 9,900

Mean Market Cap (ALL tokens): $47,650.34M
Mean Market Cap (SURVIVORS only): $1,490,810.57M

BIAS MULTIPLIER: 31.29x
```

### The Deception Visualized
- **The Graveyard (All 10,000 tokens):** Mean = $47,650M
- **The Survivors (Top 1%):** Mean = $1,490,811M

**If you only study successful tokens, you overestimate average success by 31.3x!**

### Why This Happens (Pareto Distribution)
```python
market_caps = (np.random.pareto(shape=1.16, size=10000) + 1) * 10000
```
Power law distribution where:
- 99% of tokens cluster near zero
- 1% capture exponential value
- Winner-takes-all dynamics

### Real-World Implications
This bias affects **every success story you hear:**

| Domain | What You See | What You Don't See |
|--------|-------------|-------------------|
| Startups | Unicorns in TechCrunch | 90% that failed |
| Investments | Fund's winning picks | Losing positions sold off |
| Careers | LinkedIn success posts | Millions of average outcomes |
| Music | Spotify Top 50 | 40M+ songs with <1,000 plays |

### The WWII Example
Engineers wanted to armor planes **where they saw bullet holes** on returning aircraft. Statistician Abraham Wald realized: **Armor where there are NO holes—those planes didn't return.**

### Key Lesson
**Dead tokens tell no tales.** Media, influencers, and case studies focus exclusively on survivors, creating a systematically false picture of reality. Always ask: **"What am I NOT seeing in this data?"**

---

## 🎓 Summary of Findings

| Statistical Lie | The Trap | The Truth | The Fix |
|----------------|----------|-----------|---------|
| **Latency Skew** | Std Dev = 427ms | 98% of requests are 20-50ms | Use MAD (8ms) |
| **False Positives** | "98% accurate" test | 95% of flags are wrong (low base rate) | Apply Bayes' Theorem |
| **A/B Test Validity** | "500 users missing!" | Random variation (χ²=2.5 < 3.84) | Chi-Square test |
| **Survivorship Bias** | "Average token is worth millions" | 99% failed, mean is 31x inflated | Include the graveyard |

---

## 🛡️ Defense Strategy Against Statistical Lies

### 1. For Skewed Distributions
- ✅ Use **MAD** instead of standard deviation
- ✅ Visualize distributions (histograms, box plots)
- ✅ Report median alongside mean

### 2. For Classification/Detection Systems
- ✅ Always calculate **base rates** first
- ✅ Use **Bayes' Theorem** for true probability
- ✅ Question "accuracy" claims without context

### 3. For A/B Tests
- ✅ Run **Chi-Square tests** on randomization
- ✅ Check for systematic dropoff patterns
- ✅ Investigate missing data mechanisms

### 4. For Success Analysis
- ✅ Actively seek **failed/hidden cases**
- ✅ Calculate metrics on **complete populations**
- ✅ Visualize both winners AND losers
- ✅ Question sample selection bias

---

## 📁 Code Files

### Core Analyses
- `latency_analysis.py` - MAD vs Standard Deviation comparison
- `bayesian_false_positives.py` - IntegrityAI paradox calculation
- `chi_square_ab_test.py` - FinFlash randomization validation
- `survivorship_bias_crypto.py` - 10,000 token simulation

### Outputs
- `survivorship_bias_crypto.png` - Dual histogram visualization

---

## 🛠️ Technical Stack
- **NumPy** - Random distributions, statistical calculations
- **Pandas** - DataFrame manipulation
- **Matplotlib** - Data visualization
- **Statistical Methods** - Chi-Square, Bayes' Theorem, MAD

---

## 💡 Final Takeaway

> **"Data doesn't lie, but incomplete data tells half-truths."**

The most dangerous statistical lies aren't fabrications—they're **true statements about unrepresentative samples**. Standard deviation truly is 427ms. The test truly is 98% accurate. Surviving tokens truly average $1.49B. 

But without context—without seeing the **full distribution**, the **base rate**, and the **graveyard of failures**—these numbers become weapons of mass deception.

**Always ask:**
1. What's the distribution shape?
2. What's the base rate?
3. What am I NOT seeing?
4. Who benefits from this narrative?

---

*"In God we trust. All others must bring data... and explain what's missing from it."* - Modified W. Edwards Deming
