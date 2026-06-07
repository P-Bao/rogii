# Previous Runs

Full history of all training runs. Append only. Never delete rows.

---

## Run Log

| Run ID | Date | Model | Features | CV Score | Public LB | Private LB | Config | Notes |
|--------|------|-------|----------|----------|-----------|------------|--------|-------|
| R-001 | [FILL] | LGB baseline | raw | [FILL] | [FILL] | — | [link/path] | First run |
| R-002 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | — | [FILL] | [FILL] |

---

## Best Configurations

### Best CV
- Run ID: [FILL]
- Score: [FILL]
- Config: [FILL path or params snapshot]

### Best Public LB
- Run ID: [FILL]
- Score: [FILL]
- Config: [FILL]

### Selected for Final Submission
- Run ID: [FILL]
- Reason: [FILL]

---

## Config Snapshots

### R-001 Config
```python
# Paste key config here after each run
params = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 1000,
}
```

---

## Ensemble Log

| Ensemble ID | Components | Weights | CV | LB | Notes |
|-------------|-----------|---------|----|----|-------|
| [FILL] | [Run IDs] | [FILL] | [FILL] | [FILL] | [FILL] |
