# SENTINEL-AERPN
## Risk-Aware Evidence Arbitration for Reliable Vision-Language Assistance

<p align="center">
  <img src="https://img.shields.io/badge/Research-Assistive%20AI-6f42c1.svg" alt="Research"/>
  <img src="https://img.shields.io/badge/VLM-Qwen2.5--VL--3B-0b7285.svg" alt="VLM"/>
  <img src="https://img.shields.io/badge/Grounding-OWLv2-495057.svg" alt="Grounding"/>
  <img src="https://img.shields.io/badge/Policy-PPO-f08c46.svg" alt="PPO"/>
  <img src="https://img.shields.io/badge/Dataset-VizWiz--VQA-2b8a3e.svg" alt="Dataset"/>
  <img src="https://img.shields.io/badge/Hardware-Tesla%20T4-212529.svg" alt="Hardware"/>
</p>

> **Reliable visual assistance should not always answer. It should decide when to answer, investigate, alert, or abstain.**

---

## Abstract

Vision-language models (VLMs) can provide useful visual assistance, but an incorrect answer in an assistive setting can be more harmful than refusing to answer. This work introduces **SENTINEL-AERPN**, a risk-aware decision layer that treats visual assistance as an **evidence-risk arbitration problem** rather than a conventional image-to-answer task.

Given an image and a user query, the system constructs an evidence state from visual grounding confidence, answerability, generation uncertainty, predicted human agreement, estimated risk, memory, and investigation cost. A policy then chooses among four actions: **ANSWER, INVESTIGATE, ALERT, and ABSTAIN**. When investigation is selected, the system acquires additional visual evidence through crop-and-reperception and subsequently re-evaluates the decision.

AERPN-v2 additionally introduces a learned **Expected Investigation Gain (EIG)** estimator. EIG predicts whether acquiring additional evidence is likely to improve the answer, while realized investigation gain is used only as training-time privileged feedback. This separates deployable evidence from oracle information available during training.

The system is evaluated on VizWiz-VQA using VQA accuracy, selective accuracy, coverage, unsafe-answer rate, investigation behavior, and reward-based decision metrics. The study emphasizes safety–coverage trade-offs and explicitly reports the limitations of active investigation rather than assuming that additional visual evidence is always beneficial.

> **Research status:** The current cached experiments are a reduced prototype evaluation. Final numerical claims should be replaced with the verified full held-out evaluation and multi-seed results before publication.

---

## 1. Introduction

Assistive visual question answering is fundamentally different from ordinary visual question answering. For a conventional VQA system, the objective is often summarized as:

```text
Image + Question
       ↓
      VLM
       ↓
    Answer
```

For an assistive system, this formulation is incomplete. An incorrect answer about an everyday object, surrounding environment, text, obstacle, or potentially hazardous situation can lead to an inappropriate user decision. Consequently, a useful assistive system should recognize when its current evidence is insufficient.

We therefore formulate the problem as:

```text
Image + Question
       ↓
 Evidence State
       ↓
 Risk-aware Policy
       ↓
 ┌───────────┬─────────────┬────────┬───────────┐
 │  ANSWER   │ INVESTIGATE │ ALERT  │  ABSTAIN  │
 └───────────┴─────────────┴────────┴───────────┘
```

The central research question is:

> **Given the current visual evidence and potential harm, what should the assistive system do next?**

This shifts the objective from improving answer generation alone to learning an **action-selection policy over evidence and risk**.

### Contributions

1. **Evidence-risk arbitration formulation.** Reliable visual assistance is formulated as a decision problem in which the system chooses whether to answer, acquire additional evidence, alert, or abstain.

2. **AERPN policy architecture.** VLM uncertainty, visual grounding, answerability, risk, predicted human agreement, memory, investigation cost, and expected investigation gain are combined into a compact policy state.

3. **Closed-loop evidence acquisition.** Investigation is implemented as crop-and-reperception rather than a simulated confidence adjustment, allowing the policy to observe updated evidence before a final decision.

---

# 2. Problem Formulation

Let \(I\) denote an input image, \(q\) the user question, \(E_t\) the evidence state at decision step \(t\), and \(a_t\) the selected action.

The action space is:

\[
\mathcal{A} =
\{\text{ANSWER},\text{INVESTIGATE},\text{ALERT},\text{ABSTAIN}\}.
\]

The policy is:

\[
\pi_\theta(a_t \mid E_t).
\]

The system does not simply maximize answer accuracy. It balances answer correctness, estimated risk, uncertainty, evidence quality, investigation benefit, investigation cost, and safe refusal.

---

# 3. System Architecture

```text
                         ┌──────────────────────┐
                         │    IMAGE + QUESTION  │
                         └──────────┬───────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
      ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
      │ Qwen2.5-VL  │       │    OWLv2    │       │ Risk Prompt │
      │     3B      │       │  Grounding  │       │ / Risk Est. │
      └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
             └─────────────────────┼──────────────────────┘
                                   ▼
                     ┌─────────────────────────┐
                     │    Evidence Features    │
                     │ G  U  A  R  H-hat       │
                     └────────────┬────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │  EIG + Memory + Cost    │
                     └────────────┬────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │     Evidence State      │
                     │ [G,A,U,H-hat,R,EIG,C]   │
                     └────────────┬────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │     AERPN Policy        │
                     │    PPO Actor-Critic     │
                     └────────────┬────────────┘
                                  │
            ┌─────────────┬───────┼────────┬─────────────┐
            ▼             ▼       ▼        ▼             │
         ANSWER      INVESTIGATE ALERT   ABSTAIN          │
                          │                              │
                          ▼                              │
                    Crop + Re-perception                │
                          │                              │
                          └────────► New Evidence ───────┘
```

---

# 4. Evidence State

The core evidence representation is:

\[
E_t=[G_t,A_t,U_t,\hat H_t,R_t,M_t,C_t].
\]

| Symbol | Meaning | Source |
|:---:|---|---|
| \(G\) | Grounding confidence | OWLv2 |
| \(A\) | Answerability | Learned MLP |
| \(U\) | VLM uncertainty | Generation entropy / confidence |
| \(\hat H\) | Predicted human agreement | Learned regressor |
| \(R\) | Risk estimate | Learned risk surrogate |
| \(M\) | Memory | EMA of grounding |
| \(C\) | Investigation cost | Accumulated cost |

AERPN-v2 introduces:

\[
\widehat{EIG}_t=f_\phi(G_t,U_t,A_t).
\]

The practical v2 state is:

\[
E_t^{v2}=[G_t,A_t,U_t,\hat H_t,R_t,\widehat{EIG}_t,C_t].
\]

Memory can be retained as an auxiliary temporal feature when required by the experiment.

---

# 5. Evidence Extraction

## 5.1 Qwen2.5-VL

Qwen2.5-VL-3B is used as the main vision-language perception backbone. It produces a candidate answer together with uncertainty and risk-related signals.

Expensive visual features are cached so that repeated policy experiments can reuse the same perception outputs.

## 5.2 OWLv2 Grounding

OWLv2 provides grounding confidence and a bounding box used to identify a region for investigation.

## 5.3 Uncertainty

A normalized uncertainty signal is derived from VLM generation behavior. Higher uncertainty indicates weaker confidence in the current answer, but uncertainty alone is not treated as a complete safety signal.

---

# 6. Learned Evidence Models

## 6.1 Answerability

Answerability is predicted from deployable evidence:

\[
A=f_A(G,U).
\]

The answerability model is a small multilayer perceptron.

## 6.2 Risk

Risk is estimated using:

\[
R=f_R(G,U,A).
\]

In the current implementation, supervision originates from VLM-derived risk estimates. Therefore \(R\) should be described as a **learned risk surrogate**, not an independently validated human safety probability.

## 6.3 Human Agreement

A regression model predicts human agreement:

\[
\hat H=f_H(G,A,U,R).
\]

Human agreement is used as training supervision and is not intended as an inference-time input.

---

# 7. Expected Investigation Gain

AERPN-v2 introduces an explicit prediction of whether additional evidence is likely to help.

The realized investigation gain is:

\[
\Delta_{inv}=Acc_{inv}-Acc_{base}.
\]

The EIG model learns:

\[
\widehat{\Delta}_{inv}=f_\phi(G,U,A).
\]

### Privileged training signal

The realized investigation gain can be calculated during training because benchmark answers are available. However:

> **The future investigated accuracy is never provided as a deployment input.**

Deployment uses only the predicted EIG and other deployable evidence features.

This creates a clean boundary:

```text
Training:
[G, U, A] + realized investigation gain
                 ↓
              EIG model

Deployment:
[G, U, A]
     ↓
 predicted EIG
     ↓
 policy decision
```

---

# 8. Closed-Loop Investigation

Investigation is an actual evidence-acquisition operation:

```text
Initial Image
     ↓
Grounding
     ↓
Bounding Box
     ↓
Crop / Zoom
     ↓
Qwen Re-perception
     ↓
Updated Evidence
     ↓
AERPN Re-decision
```

Formally:

\[
E_t
\xrightarrow{INVESTIGATE}
E_{t+1}.
\]

This allows the policy to learn whether additional visual evidence is worth its cost.

---

# 9. Action Space

| Action | Meaning | Behavior |
|:---|---|---|
| **ANSWER** | Evidence is sufficient | Return VLM answer |
| **INVESTIGATE** | More evidence may help | Crop, re-perceive, re-decide |
| **ALERT** | Risk is high | Warn the user |
| **ABSTAIN** | Evidence is insufficient or unsafe | Decline to answer |

The central idea is:

> **The policy chooses an action, not merely an answer.**

---

# 10. Risk-Aware Reward

A conceptual reward is:

\[
r_t=
r_{answer}+r_{safety}+r_{investigation}-\lambda C_t.
\]

For investigation:

\[
r_{investigate}
=
-c_{inv}
+
\alpha(Acc_{inv}-Acc_{base}).
\]

The cost discourages unnecessary investigation while the gain term rewards investigation when it actually improves the outcome during training.

The final reward values should be reported exactly as used in the final experiment rather than mixing prototype and final configurations.

---

# 11. PPO Policy

The AERPN policy is a compact actor-critic network:

```text
Evidence State
      ↓
Linear(64)
      ↓
ReLU
      ↓
Linear(64)
      ↓
ReLU
      │
   ┌──┴───┐
   ▼      ▼
 Actor   Critic
   │      │
   ▼      ▼
 4 actions V(E)
```

The actor models:

\[
\pi_\theta(a|E)
\]

and the critic estimates:

\[
V_\psi(E).
\]

PPO is used with clipped policy updates and generalized advantage estimation.

---

# 12. Experimental Protocol

## Dataset

The primary benchmark is **VizWiz-VQA**, with VizWiz answer-grounding data used for visual grounding analysis.

### Prototype cache

Current development cache:

```text
Train : 1786
Val   : 197
Test  : 489
```

These reduced splits are intended for development and debugging.

### Final protocol

The final evaluation should use:

```text
Official VizWiz train
        │
        ├── 90% training
        └── 10% tuning

Official VizWiz validation
        │
        ▼
Final held-out evaluation
```

The withheld official test answers should not be used as supervised accuracy targets.

---

# 13. Hardware and Efficiency

The project is designed for constrained GPU environments.

- GPU: NVIDIA Tesla T4
- VRAM: approximately 16 GB per GPU
- Python: 3.12+
- PyTorch
- Transformers
- PEFT / LoRA
- Accelerate
- Kaggle Notebook

The expensive perception stage is cached:

```text
Qwen + OWLv2
      ↓
Feature Extraction
      ↓
Persistent Cache
      ↓
Repeated PPO / EIG / Ablation Runs
```

This avoids repeatedly executing expensive VLM inference.

---

# 14. Baselines

The final evaluation should compare AERPN against:

### B0 — Always Answer

Always return the VLM answer.

### B1 — Uncertainty Threshold

```text
if U < τ:
    ANSWER
else:
    ABSTAIN
```

### B2 — Answerability Threshold

Answer only when predicted answerability exceeds a threshold.

### B3 — Risk Threshold

Answer only when predicted risk is below a threshold.

### B4 — Random Sequential Policy

Use the same environment and investigation mechanism with random actions.

### B5 — PPO without EIG

Use PPO but remove EIG from the state/reward.

### B6 — PPO without Investigation

Remove the investigation action.

### B7 — Full AERPN-v2

Use the complete evidence-risk arbitration system.

---

# 15. Evaluation Metrics

## VQA Accuracy

\[
Acc=\frac{1}{N}\sum_i Acc_i.
\]

## Coverage

\[
Coverage=\frac{N_{answer}}{N}.
\]

## Selective Accuracy

\[
Acc_{sel}
=
\frac{
\sum_{i:a_i=ANSWER}Acc_i
}{
N_{answer}
}.
\]

Selective accuracy must always be reported together with coverage.

## Unsafe Answer Rate

\[
UAR=
\frac{N_{unsafe\ answers}}
{N_{answered}}.
\]

The safety criterion and denominator must remain fixed across baselines.

## Investigation Rate

For a sequential system:

\[
IR=
\frac{N_{episodes\ containing\ investigation}}{N}.
\]

An episode such as:

```text
INVESTIGATE → ANSWER
```

must count as investigated even though its terminal action is ANSWER.

## Investigation Gain

\[
IG=Acc_{after}-Acc_{before}.
\]

Report mean gain and the fraction of positive-gain examples.

## Alert Precision

\[
Precision_{alert}
=
\frac{TP}{TP+FP}.
\]

## Calibration

Report ECE and reliability diagrams where appropriate.

---

# 16. Results

> **Important:** The values in this section must be replaced by the final verified results. Do not report prototype subset numbers as final benchmark claims.

| Method | Coverage ↑ | Selective Acc. ↑ | Overall Acc. ↑ | Unsafe Rate ↓ | Investigation Rate |
|:---|---:|---:|---:|---:|---:|
| Always Answer | 1.000 | — | `FINAL` | `FINAL` | 0.000 |
| Uncertainty Threshold | `FINAL` | `FINAL` | `FINAL` | `FINAL` | 0.000 |
| Answerability Threshold | `FINAL` | `FINAL` | `FINAL` | `FINAL` | 0.000 |
| Risk Threshold | `FINAL` | `FINAL` | `FINAL` | `FINAL` | 0.000 |
| Random Policy | `FINAL` | `FINAL` | `FINAL` | `FINAL` | `FINAL` |
| PPO w/o EIG | `FINAL` | `FINAL` | `FINAL` | `FINAL` | `FINAL` |
| PPO + EIG | `FINAL` | `FINAL` | `FINAL` | `FINAL` | `FINAL` |
| **AERPN-v2 Mean ± Std** | **`FINAL`** | **`FINAL`** | **`FINAL`** | **`FINAL`** | **`FINAL`** |

### Required result questions

1. Does AERPN reduce unsafe answers?
2. Does it maintain useful coverage?
3. Does EIG increase useful investigation?
4. Does investigation improve answers enough to justify its cost?
5. Does the full model outperform simpler threshold policies?

---

# 17. Ablation Study

| Variant | G | A | U | H | R | EIG | M | C |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Full AERPN-v2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| – Grounding | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| – Answerability | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| – Uncertainty | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| – Human Agreement | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| – Risk | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| – EIG | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| – Memory | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| – Cost | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| No Investigation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

The most important comparison is:

```text
PPO without EIG
       vs
PPO + EIG
```

because this directly tests whether EIG improves investigation selection.

---

# 18. Statistical Analysis

Policy training is stochastic. The final experiment should use at least three seeds:

```text
42
123
456
```

Report:

\[
mean\pm std.
\]

For paired predictions, suitable tests include:

- McNemar's test for paired correctness,
- Wilcoxon signed-rank test for paired continuous outcomes,
- bootstrap confidence intervals.

Statistical significance should be reported together with effect size where possible.

---

# 19. Safety Analysis

A safety-oriented system should evaluate more than average accuracy.

The most important failure category is:

```text
High Risk + Incorrect ANSWER
```

AERPN is designed to reduce this category by allowing the system to:

- ALERT,
- ABSTAIN,
- or INVESTIGATE

instead of immediately returning an uncertain answer.

---

# 20. Failure Modes

### 20.1 Over-Abstention

```text
Risk sensitivity ↑
       ↓
Coverage ↓
       ↓
User utility may decrease
```

A system that refuses nearly everything may be safe but not useful.

### 20.2 Investigation Collapse

The policy may learn:

```text
Investigation Cost > Expected Benefit
```

and avoid investigation.

EIG is introduced to explicitly model expected investigation benefit.

### 20.3 Miscalibrated Risk

VLM self-rated risk may be poorly calibrated and should not automatically be interpreted as a true probability of harm.

### 20.4 Grounding Failure

```text
Incorrect grounding
       ↓
Incorrect crop
       ↓
Incorrect re-perception
       ↓
Incorrect decision
```

### 20.5 Selective Accuracy Illusion

High selective accuracy can be achieved by answering very few samples. Coverage must therefore always be reported.

---

# 21. Limitations

1. Current development experiments use a reduced cached subset and should not be treated as the final benchmark.
2. The current risk model is based on VLM-derived risk supervision rather than independently validated human safety labels.
3. Investigation gain uses benchmark correctness during training as privileged feedback.
4. Investigation depends on grounding quality.
5. Short-horizon investigation does not represent unlimited real-world active perception.
6. Benchmark performance does not establish real-world assistive safety. Human-subject, accessibility, latency, and domain-specific safety validation would be required before deployment.

---

# 22. Reproducibility

Recommended artifact structure:

```text
artifacts/
├── cache/
│   ├── train_raw_cache.pkl
│   ├── val_raw_cache.pkl
│   └── test_raw_cache.pkl
│
├── models/
│   ├── answerability.pt
│   ├── risk.pt
│   ├── reliability.pt
│   └── eig.pt
│
├── policies/
│   ├── policy_seed42.pt
│   ├── policy_seed123.pt
│   └── policy_seed456.pt
│
└── results/
    ├── metrics.json
    ├── ablations.json
    └── plots/
```

Each experiment should record:

- random seed,
- dataset split,
- checkpoint,
- feature version,
- reward configuration,
- PPO configuration,
- threshold configuration,
- evaluation metrics.

---

# 23. Why AERPN Is Different

The contribution is not simply another VLM, uncertainty estimator, or abstention rule.

The central formulation is:

> **Reliable visual assistance is an evidence-risk arbitration problem in which a learned policy decides what action should happen next.**

This produces:

```text
Current Evidence
      ↓
AERPN Policy
      │
 ┌────┼───────────────┐
 ▼    ▼               ▼
ANSWER INVESTIGATE ALERT/ABSTAIN
       │
       ▼
 New Evidence
       │
       └──────► Re-decision
```

The novelty therefore lies in the **decision layer and action arbitration**, rather than in replacing the underlying VLM.

---

# 24. Discussion

A useful assistive system should not optimize a single metric.

The desired operating point is a balance:

```text
                Safety
                  ▲
                  │
                  │       ● Desired AERPN region
                  │     /
                  │   /
                  │ /
                  └──────────────────► Coverage
```

AERPN should therefore be interpreted as a policy for navigating the safety–coverage trade-off.

If investigation improves answers while remaining cost-effective, it strengthens the active-perception component. If investigation remains rarely selected, the result should be reported honestly as evidence that active evidence acquisition remains difficult under the current state, reward, and grounding design.

---

# 25. Conclusion

SENTINEL-AERPN reframes assistive visual question answering as a **risk-aware evidence arbitration problem**.

Instead of forcing a VLM to answer every question, the system learns to choose among:

\[
\boxed{
ANSWER,\ INVESTIGATE,\ ALERT,\ ABSTAIN
}
\]

The architecture combines visual grounding, uncertainty, answerability, risk, predicted human agreement, memory, investigation cost, and expected investigation gain.

AERPN-v2 introduces EIG to address a central challenge observed during development: an RL policy may avoid investigation even when additional evidence can sometimes improve the answer. By predicting investigation benefit from deployable evidence and using realized accuracy gain only as training-time privileged feedback, the method provides a principled connection between active perception and safe selective assistance.

The final evaluation should prioritize:

1. unsafe-answer reduction,
2. useful coverage,
3. selective accuracy,
4. investigation frequency and gain,
5. risk calibration,
6. multi-seed reproducibility,
7. fair baseline and ablation comparisons.

The broader goal is not simply to make VLMs answer more questions.

> **The goal is to make assistive VLMs know what they should do next when evidence is uncertain and the cost of being wrong matters.**

---

# Appendix A — Core Algorithm

```text
Algorithm: SENTINEL-AERPN

Input:
    image I
    question q

1. Generate candidate answer using Qwen2.5-VL.
2. Estimate uncertainty U.
3. Obtain grounding score G and region using OWLv2.
4. Predict answerability A.
5. Predict risk R.
6. Predict human agreement H-hat.
7. Estimate memory M and investigation cost C.
8. Predict expected investigation gain EIG.
9. Construct evidence state E_t.
10. Select action using PPO.

11. If ANSWER:
        return candidate answer

12. If ALERT:
        return safety warning

13. If ABSTAIN:
        return safe refusal

14. If INVESTIGATE:
        crop grounded region
        re-run visual perception
        recompute evidence
        update state
        repeat decision

Output:
    answer / alert / abstention
    + action and evidence diagnostics
```

---

# Appendix B — Final Research Checklist

- [ ] Full held-out evaluation completed
- [ ] No test-label tuning
- [ ] Three random seeds completed
- [ ] Investigation tracked independently of terminal action
- [ ] Unsafe rate uses the actual answer after investigation
- [ ] Selective accuracy reported with coverage
- [ ] Random baseline uses the same sequential environment
- [ ] PPO-without-EIG ablation completed
- [ ] EIG ablation completed
- [ ] Risk calibration analyzed
- [ ] Grounding metric reported appropriately
- [ ] Mean ± standard deviation reported
- [ ] No placeholder numerical results remain
- [ ] No unsupported state-of-the-art claim
- [ ] No unsupported Pareto-optimal claim
- [ ] Privileged training signals clearly separated from deployment inputs
- [ ] Complete bibliography verified before submission

---

# Appendix C — Repository Structure

```text
SENTINEL-AERPN/
│
├── README.md
├── PAPER.md
├── requirements.txt
├── LICENSE
│
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_evidence_models.ipynb
│   └── 04_aerpn_v2_ppo.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   ├── evidence/
│   ├── policy/
│   ├── environment/
│   └── evaluation/
│
├── artifacts/
│   ├── cache/
│   ├── models/
│   ├── policies/
│   └── results/
│
└── assets/
    ├── architecture.svg
    └── banner.png
```

---

# References / Related Work

The current project materials identify these related directions:

1. **ReCoVERR** — active evidence collection through follow-up interaction.
2. **SIEVES** — visual evidence scoring for selective prediction.
3. **Selectively Answering Visual Questions** — confidence-based selective VQA.
4. **Variational VQA** — probabilistic / Bayesian uncertainty estimation.
5. **VizWiz-VQA** — assistive visual question answering benchmark.
6. **VizWiz Answer Grounding** — grounding annotations and evaluation resource.

---

<p align="center">
  <b>🛡️ SENTINEL-AERPN</b><br>
  <i>Evidence first. Risk aware. Answer only when appropriate.</i>
</p>
