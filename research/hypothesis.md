# Hypothesis Log

All hypotheses — confirmed, rejected, and untested. Never delete.

---

## Active Hypotheses

| ID | Hypothesis | Basis | Status | Tested Via |
|----|-----------|-------|--------|-----------|
| H-001 | Q-3D wellbore tortuosity (Jing et al. 2022) is the single highest-value domain feature group, improving RMSE by roughly the −0.107 the toolkit measured, because high-tortuosity sections mark active steering that correlates with formation deviation | mycarta toolkit cumulative ablation table (external, documented) | Untested locally | EXP-002 |
| H-002 | StratifiedGroupKFold (strata = signed azimuth quadrant, median-TVT bin, XY spatial bin) tracks public LB more closely than plain KFold or pure spatial-block CV, because it prevents well-level leakage without being more pessimistic than the true (interpolation-style) test split | mycarta toolkit `rogii_cv.py` design rationale | Untested locally | EXP-001, EXP-005 |
| H-003 | A per-well linear TVT ≈ a·Z + b structural-baseline feature is only useful if fit on **lateral-only** known points; fitting on the full known section (incl. build) recovers the global cross-well slope (≈ −1, r = −0.96) and biases toe-end extrapolation by hundreds of ft, while the true within-lateral slope is ≈ −0.14 | `methodology/within_well_tvt_z_decoupling.md` (empirical: ΔZ ~70-125ft, ΔTVT ~5-13ft in eval zone) | Untested locally | EXP-003 |
| H-004 | Multi-scale normalized cross-correlation (NCC) of lateral GR against the typewell GR, plus self-correlation against the lateral's own known section, is the dominant signal source — most achievable RMSE reduction comes from getting "where does this GR shape match in the formation column" right, not from model architecture exotica | mycarta toolkit README + competition slide 6 framing ("the geology of an offset well can help predict the geology of the current well") | Untested locally | EXP-002 |
| H-005 | MiniROCKET / Multi-ROCKET convolutional features add non-redundant signal on top of NCC (different inductive bias: local shape vs. alignment-based correlation), unlike MassSNN which is mathematically identical to scipy NCC up to a monotonic transform | `methodology/AEON_evaluation.md` — proof that z-normalized Euclidean distance d satisfies d² = 2n(1−ρ) vs. Pearson ρ | Untested locally | EXP-006 (Phase 2) |
| H-006 | ~~Original framing~~ — **REVISED and moved to Confirmed Hypotheses below** (2026-06-08): "Ridge" is a stacking meta-learner, not a standalone low-variance model. See revised entry. | — | Superseded — see Confirmed table | — |
| H-007 | Well-level time-series summary features (e.g., per-well aggregate stats fit across the whole known section) overfit badly under group-based CV, because they leak structural/build-section information that does not generalize to the masked toe | mycarta toolkit README: "well-level time-series features substantially overfit under group-based validation schemes" | Untested locally | EXP-002 (ablation should reproduce this as a *negative* result) |
| H-008 | Blending a learned stack (Ridge-over-LGB/CB) with an *independent* model-free PF/beam heuristic at a fixed ratio (0.30 / 0.70) beats either alone, because the two branches make uncorrelated errors — one is a data-fit regression, the other a physics/sequential-tracking heuristic that never sees `target` | `rogii-ridge-sp45-proj.ipynb` §"6.3 Blending": `tvt = 0.3*ridge + 0.7*heuristic`; their results plot colors "ridge" entries cyan vs. grey baselines, heuristic scores separately tracked | Untested locally | EXP-007 |
| H-009 | A per-well adaptive "selector" — classifying each well by `n_eval` count (threshold 4840) and `Z`-span (thresholds 136.73 / 185.51) into 6 bins, each mapped to a distinct PF-scale/beam-hold parameter variant — outperforms one global PF/beam configuration, because well geometry (lateral length, vertical excursion) varies enough that no single parameterization is optimal everywhere | `SELECTOR_N_EVAL_THRESHOLD`, `SELECTOR_Z_SPAN_THRESHOLDS`, `SELECTOR_BIN_VARIANTS` (6 variants), `SELECTOR_GLOBAL_VARIANT='pf_scale_8_hold_0.2'`, `SELECTOR_SCALES=(3,5,8,12)` — present in ALL THREE top-score notebooks verbatim (shared lineage) | Untested locally | EXP-008 |
| H-010 | A geometry-aware postprocess — robust per-well degree-5 polynomial projection in `U = TVT + Z - anchor` space, parameterized by normalized along-lateral position `s = (MD-MD_last)/(MD_end-MD_last)`, with iterative reweighting (Tukey-style, 4 iters) — removes residual jitter and "wrong-branch" outliers that direct TVT-space smoothing (our `sg_smooth`/Savitzky-Golay) cannot, because `U` is the natural coordinate where the trajectory is smooth (formation-relative) while raw `TVT` jumps with `Z` | Top notebook's own inline comment: "CV-validated: raw PF -0.54, deployed components -0.33" RMSE; code block titled "robust low-order PROJECTION post-processing ... OVERWRITES submission.csv" runs as the FINAL step after the 0.3/0.7 blend | Untested locally | EXP-009 |
| H-011 | A heavily-regularized, low-learning-rate LightGBM config (Optuna-tuned: `lr≈0.0093, num_leaves=64, reg_alpha≈10.8, reg_lambda≈95.8, colsample_bytree≈0.39, min_child_samples=40`) generalizes better as a stack input on this thin-eval-zone regression than our hand-set high-capacity defaults (`num_leaves=255, reg_lambda=3.0, lr=0.02-0.03`) — lower per-model variance feeds a tighter Ridge stack | `rogii-sel15-forced-selector.ipynb` §"3. Training" `lgb_params[1]`/`[2]` — both Optuna-tuned variants used as stack inputs alongside a default-regime variant (`lr=0.03, num_leaves=255`-style) | Untested locally | EXP-010 |

---

## Confirmed Hypotheses

| ID | Hypothesis | Evidence | Date |
|----|-----------|----------|------|
| H-006 (revised) | "Ridge" in top-score notebooks is a stacking meta-learner over 6 base-model OOF predictions (LGB×3 + CB×3), and the score win comes from blending that stack 0.30/0.70 with an independent model-free PF/beam heuristic — NOT from Ridge being a strong standalone competitor to GBMs (the original H-006 framing was wrong) | Direct read of `rogii-ridge-sp45-proj.ipynb` source: `ridge_trainer.fit(oof_preds, y)` then `0.3*ridge + 0.7*heuristic` blend in §6.3 | 2026-06-08 |

---

## Rejected Hypotheses

| ID | Hypothesis | Evidence | Date | Lesson |
|----|-----------|----------|------|--------|
| (none yet) | | | | |

---

## Adding a Hypothesis

```
| H-[ID] | [clear falsifiable statement] | [basis for belief] | Untested | EXP-[ID] |
```

Good hypothesis: "Adding lag features for the last 7 days will improve RMSE by >0.5% because the target shows weekly seasonality in EDA."

Bad hypothesis: "More features = better" (not falsifiable, no mechanism)

**Note**: H-001 through H-007 are currently *externally sourced* (from the mycarta toolkit's own write-up, which we have not yet run ourselves). They stay in "Untested" status until we reproduce the corresponding experiment locally and log CV evidence — do not promote to "Confirmed" on the strength of someone else's ablation table alone.
