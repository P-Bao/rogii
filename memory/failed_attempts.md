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

## Quick Reference: Do Not Repeat

| ID | Short Rule |
|----|-----------|
| FAIL-001 | Never use post-event features |
| FAIL-002 | Always chunk large CSV loads |
| [FILL] | [FILL] |
