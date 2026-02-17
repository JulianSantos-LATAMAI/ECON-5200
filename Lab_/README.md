Hypothesis Testing & Causal Evidence Architecture
The Epistemology of Falsification: Causal Inference on the Lalonde Dataset

Objective
Most applied data science workflows stop at estimation — fitting a model, reading a coefficient, and calling it a day. This project takes a deliberate step further, reframing the analytical task as one of falsification: can we construct a statistically defensible case for causality, or are we simply pattern-matching noise?
Using the canonical Lalonde (1986) dataset — a landmark study on the effect of subsidized job training on post-intervention earnings — I operationalized the scientific method to adjudicate between competing causal narratives. The central question isn't "what is the effect?" but rather "can we reject the world in which there is no effect?"

Technical Approach

Parametric Inference via Welch's T-Test (SciPy): Computed the Average Treatment Effect (ATE) by modeling the signal-to-noise ratio between treated and control earnings distributions. Welch's formulation was selected over Student's T-Test to account for heterogeneous within-group variances — a common and consequential violation in observational economic data.
Non-Parametric Validation via Permutation Testing (SciPy / NumPy): To guard against assumptions of normality in the earnings distributions — which are characteristically right-skewed — I implemented a Monte Carlo Permutation Test across 10,000 resamples. By repeatedly shuffling treatment labels and reconstructing the null distribution empirically, this approach validates the parametric result without relying on distributional approximations.
Type I Error Control: Both testing frameworks were evaluated against a pre-specified significance threshold (α = 0.05), enforcing a disciplined hypothesis-testing protocol that protects against false discovery. The convergence of both the parametric and non-parametric p-values beneath this threshold provides compounding evidence against the null.


Key Findings
The analysis identified a statistically significant Average Treatment Effect of approximately +$1,795 in real post-intervention earnings among participants who received job training. The null hypothesis — that the program produced no measurable lift — was rejected via Proof by Statistical Contradiction, with corroborating evidence from both the parametric and permutation-based tests.

Business Insight: Hypothesis Testing as the Safety Valve of the Algorithmic Economy
In production data environments, the pressure to surface actionable insights creates a structural incentive toward data dredging — running analyses until something looks significant, and shipping that result upstream. The consequences are well-documented: spurious correlations get encoded into decision pipelines, A/B tests are called early, and model performance metrics are optimized against the training set rather than reality.
Rigorous hypothesis testing is the safety valve that prevents this failure mode from compounding. By committing to a falsifiable null hypothesis before analysis begins and controlling the Type I error rate, we create an audit trail that distinguishes genuine signal from statistical artifact. In the algorithmic economy — where model outputs increasingly drive hiring, lending, pricing, and resource allocation — the cost of a false positive isn't just a bad dashboard; it's a systematically biased decision at scale.
The Lalonde dataset is a useful proving ground precisely because the ground truth is contested. Deploying both parametric and non-parametric frameworks isn't methodological redundancy — it's epistemic due diligence.

Tools & Libraries: Python · SciPy · NumPy · Pandas
Dataset: Lalonde, R.J. (1986). "Evaluating the Econometric Evaluations of Training Programs." AER.
