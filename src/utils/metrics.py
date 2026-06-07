"""
Metric utilities for cross-validation and ensemble evaluation.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    log_loss,
    average_precision_score,
)
from typing import Callable


METRIC_REGISTRY: dict[str, Callable] = {}


def register_metric(name: str):
    def decorator(fn):
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator


@register_metric('auc')
def metric_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return roc_auc_score(y_true, y_pred)


@register_metric('logloss')
def metric_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return log_loss(y_true, y_pred)


@register_metric('rmse')
def metric_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_squared_error(y_true, y_pred, squared=False)


@register_metric('mae')
def metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)


@register_metric('f1')
def metric_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, (y_pred > 0.5).astype(int))


@register_metric('map')
def metric_map(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return average_precision_score(y_true, y_pred)


def get_metric(name: str) -> Callable:
    if name not in METRIC_REGISTRY:
        raise ValueError(f'Unknown metric: {name}. Available: {list(METRIC_REGISTRY)}')
    return METRIC_REGISTRY[name]


def evaluate_oof(y_true: np.ndarray, oof_preds: np.ndarray, metric: str = 'auc') -> dict:
    fn    = get_metric(metric)
    score = fn(y_true, oof_preds)
    return {'metric': metric, 'score': round(float(score), 6)}
