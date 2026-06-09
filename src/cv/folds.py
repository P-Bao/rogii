"""
CV fold utilities for well-level cross-validation.

Rogii competition: wells are spatially interpolated, not extrapolated.
Use GroupKFold (prevent leakage) as default; StratifiedGroupKFold for
stratum-balanced splits (signed azimuth × median-TVT × spatial bins).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, KFold
from typing import List, Optional, Tuple


def make_group_folds(
    df: pd.DataFrame,
    well_col: str = "well_code",
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Standard GroupKFold — prevents same well in train and val.
    Returns list of (train_idx, val_idx) index pairs.
    """
    groups = df[well_col].values
    kf = GroupKFold(n_splits=n_splits)
    return list(kf.split(df, groups=groups))


def make_stratified_group_folds(
    df: pd.DataFrame,
    well_col: str = "well_code",
    azimuth_col: Optional[str] = "azimuth",
    tvt_col: Optional[str] = "TVT",
    n_splits: int = 5,
    seed: int = 42,
    n_spatial_bins: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    StratifiedGroupKFold with stratum = signed_azimuth_bin × median_tvt_bin × spatial_bin.
    Prevents same well appearing in train and val, while balancing spatial/depth strata.
    Matches the CV scheme validated in research/current_plan.md Decision Log 2026-06-08.
    """
    well_df = df.groupby(well_col).first().reset_index()

    stratum_parts = []

    if azimuth_col and azimuth_col in df.columns:
        well_az = df.groupby(well_col)[azimuth_col].mean()
        az_bin = pd.cut(well_az, bins=4, labels=False).fillna(0).astype(int)
        stratum_parts.append(az_bin.values)

    if tvt_col and tvt_col in df.columns:
        well_tvt = df.groupby(well_col)[tvt_col].median()
        tvt_bin = pd.cut(well_tvt, bins=n_spatial_bins, labels=False).fillna(0).astype(int)
        stratum_parts.append(tvt_bin.values)

    if stratum_parts:
        strata_matrix = np.column_stack(stratum_parts)
        strata = np.zeros(len(strata_matrix), dtype=int)
        multiplier = 1
        for col in range(strata_matrix.shape[1] - 1, -1, -1):
            strata += strata_matrix[:, col] * multiplier
            multiplier *= (strata_matrix[:, col].max() + 1)
    else:
        strata = np.zeros(len(well_df), dtype=int)

    well_groups = well_df[well_col].values

    # Map well-level splits back to row-level
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    well_splits = list(kf.split(well_df, strata, groups=well_groups))

    # Convert well-level indices → row-level indices
    well_to_rows = df.reset_index().groupby(well_col)['index'].apply(list).to_dict()
    row_splits = []
    for tr_well_idx, val_well_idx in well_splits:
        tr_wells = well_df[well_col].iloc[tr_well_idx].values
        val_wells = well_df[well_col].iloc[val_well_idx].values
        tr_rows = np.concatenate([well_to_rows[w] for w in tr_wells if w in well_to_rows])
        val_rows = np.concatenate([well_to_rows[w] for w in val_wells if w in well_to_rows])
        row_splits.append((tr_rows, val_rows))

    return row_splits


def log_fold_summary(splits: list, df: pd.DataFrame, well_col: str = "well_code") -> None:
    """Print fold sizes and well counts for inspection."""
    print(f"{'Fold':>4} {'Train rows':>12} {'Val rows':>10} {'Train wells':>13} {'Val wells':>11}")
    for i, (tr, val) in enumerate(splits):
        tr_wells = df[well_col].iloc[tr].nunique()
        val_wells = df[well_col].iloc[val].nunique()
        print(f"{i+1:>4} {len(tr):>12} {len(val):>10} {tr_wells:>13} {val_wells:>11}")
