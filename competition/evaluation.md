# Evaluation Metric

## Primary Metric

**Metric**: Root Mean Square Error (RMSE)

**Formula**:
```
RMSE = sqrt( mean( (y_true - y_pred)^2 ) )
```

**Direction**: Lower is better

**Optimized on**: Public LB / Private LB (final ranking)

## Baseline Scores

| Model | CV Score | Public LB | Private LB | Notes |
|-------|----------|-----------|------------|-------|
| Constant prediction (mean/mode) | [FILL] | [FILL] | — | Dumb baseline |
| Simple LGB default | [FILL] | [FILL] | — | First real baseline |

## Current Best

| Date | Model | CV Score | Public LB | Notes |
|------|-------|----------|-----------|-------|
| [FILL] | [FILL] | [FILL] | [FILL] | |

## Score Interpretation

- Public LB split: [FILL]% of test data
- Private LB split: [FILL]% of test data
- Shake-up risk: Low / Medium / High — [FILL reason]

## Metric Implementation

```python
# Example: compute metric locally
import numpy as np
from sklearn.metrics import mean_squared_error

def compute_metric(y_true, y_pred):
    """
    Compute competition metric (RMSE).
    Returns: float score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Cross-validation usage
# score = compute_metric(val_targets, val_preds)
# print(f"CV Score: {score:.5f}")
```

## Overfitting Detection

CV - LB gap threshold: [FILL, e.g., > 0.005 = suspect]

Rules:
- If CV improves but LB drops → likely overfit to CV folds
- If LB improves but CV drops → data leakage or fold distribution mismatch
