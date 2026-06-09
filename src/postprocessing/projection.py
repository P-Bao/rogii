"""
Robust per-well polynomial projection postprocess — EXP-009.

Reproduces the top-score notebooks' final postprocess step:
  U = TVT + Z - anchor     (anchor = last_known_TVT + last_known_Z)
  s = (MD - MD_last) / (MD_end - MD_last)   normalized position ∈ [0, 1]
  fit robust degree-5 poly U ~ f(s) via Tukey iterative reweighting (4 iters, c=2.0)
  corrected_TVT = U_fit - Z + anchor

Top-score notebook claims "CV-validated: raw PF -0.54, deployed components -0.33 RMSE".
This is a pure postprocess — no retraining, works on any prediction array.
"""

import numpy as np
import pandas as pd
from typing import Optional


def _robfit(
    s: np.ndarray,
    u: np.ndarray,
    deg: int = 5,
    n_iter: int = 4,
    c: float = 2.0,
) -> np.ndarray:
    """
    Iterative Tukey-style robust polynomial fit.
    Returns fitted U values at positions s (same shape).
    Weights: 1 if |residual|/scale <= c else 0 (hard threshold, Tukey-style).
    """
    w = np.ones(len(s))
    u_fit = u.copy()
    for _ in range(n_iter):
        coeffs = np.polyfit(s, u, deg=deg, w=w)
        u_fit = np.polyval(coeffs, s)
        resid = u - u_fit
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = 1.4826 * mad if mad > 1e-12 else 1.0
        w = (np.abs(resid) / scale <= c).astype(float)
        w = np.maximum(w, 1e-6)  # avoid singular polyfit
    return u_fit


def project_well(
    md: np.ndarray,
    z: np.ndarray,
    tvt_pred: np.ndarray,
    md_last_known: float,
    tvt_last_known: float,
    z_last_known: float,
    md_end: Optional[float] = None,
    deg: int = 5,
    n_iter: int = 4,
    c: float = 2.0,
) -> np.ndarray:
    """
    Apply robust polynomial projection to eval zone of a single well.

    Args:
        md:             MD values of eval rows
        z:              Z values of eval rows
        tvt_pred:       model TVT predictions for eval rows
        md_last_known:  MD of last known (train) row for this well
        tvt_last_known: TVT of last known (train) row
        z_last_known:   Z of last known (train) row
        md_end:         MD of last eval row (defaults to max(md))
        deg:            polynomial degree (default 5)
        n_iter:         reweighting iterations (default 4)
        c:              Tukey scale threshold (default 2.0)

    Returns:
        corrected TVT predictions (same shape as tvt_pred)
    """
    if md_end is None:
        md_end = md.max()

    span = md_end - md_last_known
    if span <= 0 or len(md) < deg + 1:
        return tvt_pred.copy()

    anchor = tvt_last_known + z_last_known
    s = (md - md_last_known) / span
    u_pred = tvt_pred + z - anchor

    u_fit = _robfit(s, u_pred, deg=deg, n_iter=n_iter, c=c)
    return u_fit - z + anchor


def apply_projection_postprocess(
    df: pd.DataFrame,
    pred_col: str = "tvt_pred",
    well_col: str = "well_code",
    md_col: str = "MD",
    z_col: str = "Z",
    tvt_col: str = "TVT",
    is_eval_col: str = "is_eval",
    deg: int = 5,
    n_iter: int = 4,
    c: float = 2.0,
    verbose: bool = False,
) -> pd.Series:
    """
    Apply per-well robust projection postprocess to a full DataFrame.

    df must contain both known (is_eval=False) and eval (is_eval=True) rows per well,
    sorted by MD within each well (or sortable by md_col).

    Returns a Series of corrected predictions, same index as df.
    Only eval rows are modified; known rows retain their original pred values.
    """
    corrected = df[pred_col].copy().astype(float)
    wells_processed = 0

    for well_id, wdf in df.groupby(well_col):
        wdf_sorted = wdf.sort_values(md_col)
        known = wdf_sorted[wdf_sorted[is_eval_col].astype(bool) == False]
        eval_ = wdf_sorted[wdf_sorted[is_eval_col].astype(bool)]

        if len(known) == 0 or len(eval_) < deg + 1:
            if verbose and len(eval_) > 0:
                print(f"  Skip {well_id}: known={len(known)}, eval={len(eval_)} < deg+1")
            continue

        last_known = known.iloc[-1]
        md_end = eval_[md_col].iloc[-1]

        corrected_tvt = project_well(
            md=eval_[md_col].values,
            z=eval_[z_col].values,
            tvt_pred=eval_[pred_col].values,
            md_last_known=float(last_known[md_col]),
            tvt_last_known=float(last_known[tvt_col]),
            z_last_known=float(last_known[z_col]),
            md_end=float(md_end),
            deg=deg,
            n_iter=n_iter,
            c=c,
        )
        corrected.loc[eval_.index] = corrected_tvt
        wells_processed += 1

    if verbose:
        print(f"Projection applied to {wells_processed} wells")

    return corrected
