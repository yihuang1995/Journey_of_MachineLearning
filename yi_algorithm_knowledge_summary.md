# Yi Huang — Algorithm Knowledge Summary

Based on our conversations, your algorithm knowledge can be summarized as **applied ML + decision systems + experimentation/causal inference + production ML**, with a growing layer in **LLMs, agents, and post-training**.

---

## 1. Core ML / Predictive Modeling

You have strong working knowledge of practical supervised ML, especially for business decision systems.

Main areas:

```text
classification / regression
logistic regression
tree-based models
LightGBM / XGBoost / CatBoost
multi-task learning
MMOE deep learning
calibration
feature importance
model evaluation
offline vs online validation
```

You often think beyond simply training a model and focus on:

```text
What is the prediction target?
How will the prediction be used?
What business decision does it drive?
How do we evaluate downstream impact?
```

This is very Staff-level applied ML thinking.

---

## 2. LTV / ROI / Long-Term Value Modeling

This is probably your strongest domain-specific algorithm area.

You have discussed and worked on:

```text
LTV prediction
ROI forecasting
long-horizon user value
early signal modeling
cold-start forecasting
cohort-based forecasting
retention / monetization prediction
budget allocation decision systems
```

Important methods include:

```text
LightGBM
MMOE multi-task models
hierarchical Bayesian modeling
curve fitting / statistical extrapolation
time-series features
cohort-normalized rolling statistics
drift-resistant features
long-horizon target construction
offline + online evaluation
```

Your strength is not just the model, but the full system:

```text
early user signals
→ long-term value forecast
→ ROI decision
→ budget allocation
→ business impact measurement
```

---

## 3. Time-Series / Forecasting / Change Detection

You have a solid applied forecasting toolkit.

Topics we discussed include:

```text
rolling averages
EMA smoothing
Kalman filtering
time-series decomposition
forecast residual monitoring
change point detection
CUSUM
PELT
Bayesian change point detection
Hidden Markov Models
Z-score / Bollinger Bands
seasonality / regime shift
drift monitoring
```

You also understand how forecasting connects to production decisions:

```text
detecting data drift
choosing retraining windows
monitoring feature shifts
distinguishing temporary noise vs structural drift
handling delayed labels
```

This is especially relevant to underwriting, fraud, LTV, and ROI systems.

---

## 4. Experimentation and Statistical Inference

You have strong experimentation knowledge, especially for product/data science interviews.

Topics covered:

```text
A/B testing
two-sample t-test
Welch’s t-test
z-test for proportions
confidence intervals
p-values
standard error
sample size / power
one-sided vs two-sided tests
bootstrap
jackknife
novelty effect
launch-and-holdout
long-term holdout
switchback experiments
geo experiments
multiple testing
sequential testing
```

You also understand metric design:

```text
primary metric
secondary metrics
guardrails
North Star metric
leading vs lagging metrics
counter metrics
business metric vs model metric
```

This is one of your strongest interview areas.

---

## 5. Causal Inference / Incrementality

You have discussed causal inference extensively, especially around marketing, cannibalization, and marketplace/product effects.

Topics include:

```text
DiD
synthetic control
propensity score matching
CEM
double machine learning
instrumental variables
uplift modeling
CATE
incrementality
cannibalization
holdout design
interference / SUTVA violation
network effects
cluster randomization
```

Your strongest framing here is:

> The hard part is estimating the counterfactual.

Examples:

```text
What would organic installs have been without paid acquisition?
What would merchant default have been under a different underwriting policy?
What would recruiter retention have been without an onboarding intervention?
```

That is a very strong causal/product DS framing.

---

## 6. ML System Design

You have been building strong MLE/system design knowledge.

Systems we discussed:

```text
fraud detection system
credit underwriting system
cash loan decisioning system
retraining system
feature drift monitoring system
ML CI/CD pipeline
agent-based code editing system
feature store / ETL / model serving
```

Key concepts you know:

```text
online serving path
offline training path
feature store
point-in-time feature snapshots
training-serving skew
data leakage
model registry
shadow deployment
canary deployment
blue-green deployment
monitoring
rollback
scheduled batch retraining
trigger-based retraining
```

You are especially strong at connecting ML models to decision engines:

```text
model score → decision rule → business action → monitoring → retraining
```

That is exactly what interviewers want in ML system design.

---

## 7. Credit / Underwriting / Fraud Algorithms

Recently you have built a good framework around risk systems.

Topics include:

```text
PD: probability of default
LGD: loss given default
EAD: exposure at default
expected loss
risk-adjusted profit
credit decisioning
loan amount optimization
risk-based pricing
fraud score thresholding
precision-recall tradeoff
PR-AUC for imbalanced fraud data
manual review bands
false positive cost
label delay
reject inference
```

You understand that underwriting is not just:

```text
Will this user default?
```

but:

```text
Should we approve?
How much should we offer?
At what price?
What is the expected risk-adjusted value?
What constraints do we need?
```

That is a much more senior view.

---

## 8. Recommender / Ranking / Marketplace Algorithms

You have ranking/recommendation knowledge mostly from a product and decision-system perspective.

Topics include:

```text
engagement prediction
retention-aware ranking
long-term value optimization
exploration vs exploitation
marketplace health
two-sided marketplace dynamics
creator / content ecosystem
candidate matching
search / recommendation surfaces
ranking-adjacent experimentation
```

For Roblox, TikTok, Paraform, and similar roles, your strongest angle is:

> Ranking is not only about short-term CTR. It is an ecosystem optimization problem involving retention, supply quality, fairness, creator/merchant/recruiter health, and long-term value.

---

## 9. LLM / Agent / Post-Training Knowledge

You have been actively learning and experimenting with LLM systems.

Topics discussed:

```text
pretraining → SFT → RLHF pipeline
LoRA / QLoRA
DPO
RLHF
GRPO
reward modeling
LLM-as-judge
best-of-k / majority voting
prompt engineering
vLLM / SGLang
KV cache
prompt cache
agentic workflows
MCP
tool use
Claude Code-style repo agents
```

You also explored practical LLM fine-tuning:

```text
math reasoning SFT
structured reasoning data
positive/negative pairs
format accuracy
pass@k
exact match
LoRA rank / alpha / dropout
training loss debugging
evaluation during SFT
```

Your LLM knowledge is not yet framed as industry-level LLM infrastructure expertise, but it is clearly more advanced than casual usage and shows strong hands-on learning.

---

## 10. Python / SQL / Algorithm Coding

You have practiced practical coding and interview-style problems.

Python / pandas topics:

```text
groupby
idxmax
sorting
bootstrap simulation
array manipulation
dynamic programming basics
recursion
countBits
climbStairs
repeated substring pattern
F1 / precision / recall calculation
```

SQL topics:

```text
window functions
cumulative sum
groupby aggregation
median from frequency table
ranking / filtering
experiment metric queries
```

This is more data science coding and analytics coding than pure LeetCode-heavy algorithm specialization, but it fits DS/MLE interview needs.

---

## 11. Statistics and Probability Fundamentals

You have repeatedly reviewed core statistical concepts:

```text
mean / variance / standard deviation
standard error
confidence interval
t-statistic
z-test
t-test
Welch’s t-test
two-proportion test
binomial distribution
normal approximation
p-value
bootstrap
sample size
power
precision / recall / F1
AUC / PR-AUC
calibration
log loss
```

You often connect these to real experiments and model evaluation, which is stronger than memorizing formulas alone.

---

## 12. Strongest Algorithm Identity

Your algorithm profile can be summarized as:

> You are strongest in applied machine learning for large-scale decision systems: LTV/ROI forecasting, experimentation, causal measurement, feature engineering, long-term prediction, and production model monitoring. You are also building strong MLE system design knowledge around feature stores, retraining, drift detection, CI/CD, fraud, and underwriting systems. Your LLM/agent knowledge is growing quickly, especially around LoRA, SFT, RLHF/DPO, evaluation, and agent workflows.

---

## 13. Best Interview Positioning

For interviews, your algorithm knowledge can be positioned as:

```text
Applied ML / Decision Systems:
Very strong

Experimentation / Causal Inference:
Very strong

Forecasting / LTV / ROI:
Very strong

ML System Design:
Strong and improving

Underwriting / Fraud / Risk Modeling:
Strong transferable knowledge

LLM / Agents / Post-training:
Growing, hands-on, high motivation

Pure LeetCode Algorithms:
Functional, but not your main brand

Low-level distributed systems / infra:
Some understanding, not your core strength

Production RL:
Exposure and learning, but not industry-level core experience
```

---

## One-Line Summary

> Your algorithm strength is not just knowing models; it is knowing how to turn models into production decision systems that optimize long-term business outcomes under uncertainty.
