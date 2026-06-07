"""
Feature engineering utilities.
Add functions here; import them in notebooks.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def add_ratio_features(df: pd.DataFrame, num_cols: List[str], epsilon: float = 1e-6) -> pd.DataFrame:
    """Add pairwise ratio features for numeric columns."""
    df = df.copy()
    for i, col_a in enumerate(num_cols):
        for col_b in num_cols[i + 1:]:
            df[f'ratio_{col_a}_div_{col_b}'] = df[col_a] / (df[col_b].abs() + epsilon)
    return df


def add_aggregation_features(
    df: pd.DataFrame,
    group_col: str,
    agg_cols: List[str],
    aggs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add group aggregation features (mean, std, min, max per group)."""
    if aggs is None:
        aggs = ['mean', 'std', 'min', 'max']
    df = df.copy()
    grouped = df.groupby(group_col)[agg_cols].agg(aggs)
    grouped.columns = [f'{col}_{agg}_{group_col}' for col, agg in grouped.columns]
    grouped = grouped.reset_index()
    df = df.merge(grouped, on=group_col, how='left')
    return df


def add_lag_features(
    df: pd.DataFrame,
    sort_col: str,
    group_col: str,
    value_col: str,
    lags: List[int],
) -> pd.DataFrame:
    """Add lag features for time-series style data."""
    df = df.copy().sort_values([group_col, sort_col])
    for lag in lags:
        df[f'{value_col}_lag_{lag}'] = (
            df.groupby(group_col)[value_col].shift(lag)
        )
    return df


def label_encode_categoricals(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode categoricals using train-fit mapping. Unknown test values → -1."""
    train, test = train.copy(), test.copy()
    for col in cat_cols:
        mapping = {v: i for i, v in enumerate(train[col].dropna().unique())}
        train[col] = train[col].map(mapping).fillna(-1).astype(int)
        test[col]  = test[col].map(mapping).fillna(-1).astype(int)
    return train, test
