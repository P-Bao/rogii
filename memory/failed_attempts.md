# Failed Attempts

Critical memory. Read before starting any new experiment to avoid repeating failures.

---

## Template

```
## FAIL-[ID]: [Short title]

- Date: [FILL]
- What was tried: [FILL]
- Why it seemed like a good idea: [FILL]
- What happened: [FILL — error or score drop]
- Root cause: [FILL]
- Do NOT repeat because: [FILL]
- Alternative to try: [FILL]
```

---

## FAIL-001: Example — Target Leakage via Post-event Feature

- Date: [FILL]
- What was tried: Used `next_day_price` as a feature (available in train, not test)
- Why it seemed like a good idea: Correlated with target in EDA
- What happened: CV 0.98 → LB 0.51 (near random)
- Root cause: Target leakage — feature uses future information
- Do NOT repeat because: Any feature with AUC > 0.90 alone should be suspected
- Alternative to try: Use only features available at prediction time

---

## FAIL-002: Example — Memory OOM on Full Dataset

- Date: [FILL]
- What was tried: Load all data into pandas DataFrame without chunking
- Why it seemed like a good idea: Simplest approach
- What happened: OOM kill on 16GB RAM machine
- Root cause: Dataset is 40GB uncompressed
- Do NOT repeat because: Always profile memory before loading large files
- Alternative to try: Use `pd.read_csv(chunksize=...)` or Polars lazy API

---

## FAIL-003: Ridge stack blended with weak PF feature column

- Date: 2026-06-10
- What was tried: EXP-007 — Ridge meta-learner over 6 base-model OOF preds, blended with
  `pf_ancc`/`pf_z` columns from train_df. Used tuned α=0.95 (nearly pure stack) + EXP-009
  polynomial projection postprocess. Submitted to Kaggle.
- Why it seemed like a good idea: Top-score notebooks use `0.30*ridge + 0.70*PF_heuristic`
  and score 7.748. We had pf_ancc columns available in train_df.
- What happened: Standalone PF heuristic RMSE = 13.74 (catastrophically weak). Ridge-only
  OOF = 10.562 vs baseline 10.452. 0.30/0.70 blend = 12.143. Tuned α=0.95 = 10.550.
  Submission LB = **10.208** — WORSE than baseline 9.964 by **0.244 RMSE**.
- Root cause: `train_df['pf_ancc']` is the single-seed feature-extraction particle filter
  output, NOT the reference notebooks' 128-seed × 4-scale likelihood-weighted PF ensemble.
  The heuristic branch is effectively absent. Secondary: Ridge α=1.0 too weak for 6
  correlated models (coef distribution uninspected).
- Do NOT repeat because: Never blend Ridge with the pf_ancc feature column as if it were
  a quality PF heuristic. The inference-mode PF ensemble must be built explicitly (EXP-014)
  before any 0.30/0.70 blend is meaningful.
- Alternative to try: (1) EXP-013 — projection postprocess on top of BASELINE_V1 (no stacking);
  (2) EXP-012 — confirm PF quality gap; (3) EXP-014 — build 128-seed inference PF ensemble;
  (4) retry EXP-007 ONLY after EXP-014 delivers a quality heuristic.

---

## Quick Reference: Do Not Repeat

| ID | Short Rule |
|----|-----------|
| FAIL-001 | Never use post-event features |
| FAIL-002 | Always chunk large CSV loads |
| FAIL-003 | Never use pf_ancc feature column as a quality PF heuristic in a 0.30/0.70 blend — it is a single-seed feature artifact (RMSE 13.74), not the inference-mode 128-seed PF ensemble |
