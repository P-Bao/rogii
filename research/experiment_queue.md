# Experiment Queue

## Template

Each experiment must fill ALL fields before running.

```
## EXP-[ID]: [Short Name]

- Objective: [what you want to test]
- Hypothesis: [why you expect improvement]
- Expected gain: [e.g., +0.002 AUC]
- Risk: Low / Medium / High — [reason]
- Compute estimate: [e.g., 20 min on T4 x2]
- Status: Queued / Running / Done / Cancelled
- Result: [fill after run]
- Notes: [fill after run]
```

---

## EXP-001: LightGBM Baseline

- Objective: Establish reproducible baseline CV score
- Hypothesis: Standard LGB with default params gives usable starting point
- Expected gain: N/A (baseline)
- Risk: Low
- Compute estimate: 5 min CPU (tiny sample), 15 min Kaggle T4
- Status: Queued
- Result: [FILL]
- Notes: [FILL]

---

## EXP-002: Feature Engineering v1

- Objective: Add domain-specific features
- Hypothesis: Raw features miss interaction terms visible to domain experts
- Expected gain: +0.003–0.005 AUC
- Risk: Medium — might introduce leakage
- Compute estimate: 10 min CPU
- Status: Queued
- Result: [FILL]
- Notes: [FILL]

---

## EXP-003: Cross-validation Strategy

- Objective: Test GroupKFold vs StratifiedKFold
- Hypothesis: GroupKFold reduces leakage if groups exist in data
- Expected gain: Better CV-LB correlation
- Risk: Low
- Compute estimate: 5 min CPU
- Status: Queued
- Result: [FILL]
- Notes: [FILL]

---

## Cancelled / Deprioritized

| ID | Name | Reason |
|----|------|--------|
| [FILL] | [FILL] | [FILL] |
