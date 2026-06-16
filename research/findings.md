# Research Findings

Append new findings. Never delete old ones. Date every entry.

---

## 2026-06-08 — Finding: We already have a scored pipeline (LB 9.964); the gap to top scores is architectural, not feature-related

**Context**: User uploaded 5 of "our own" notebooks (`rogii-eda-features`, `rogii-train-lightgbm`,
`rogii-train-catboost`, `rogii-postprocess-research`, `rogii-submission`) plus 3 top-scoring
public notebooks (`rogii-ridge-sp45-proj` 7.748, `rogii-ridge-artifact-parameter-experiments`
7.881, `rogii-sel15-forced-selector` 7.807) to `notebooks/reference/`. Read all 8 in full.

**Finding — Part 1: our REAL baseline** (this supersedes the entire original "build from
mycarta toolkit from scratch" plan):

We already have a working "Another-Work" (`aw`) pipeline:
- **Feature core** (`rogii-eda-features.ipynb`, `build_well`): ~150 features/well — multi-scale
  NCC, DTW (Sakoe-Chiba band, multiscale radii 20/50/100/200, stochastic Gumbel-noise
  realizations), particle filters (`_pf_ancc` tracks via ANCC+GR likelihood, `_pf_z` via
  Z-trajectory regression), beam search (7 configs: cons/loose/vcons/sm5/vloose/mid/stiff),
  spatial-KNN imputation (`FormationPlaneKNN`, `DenseANCCImputer`), formation-contact features.
  Target = **residual** `TVT - last_known_tvt`.
- **Models**: LightGBM×3 (seeds 42/7/123, `num_leaves=255, reg_lambda=3.0, lr=0.02-0.03,
  n_estimators=8000`) + CatBoost×3 (seeds 42/7/123, `depth=7, l2_leaf_reg=2.0, iterations=8000`),
  `GroupKFold(5)`.
- **Ensembling**: fixed-weight blend of 3 selected models —
  `{catboost-2: 0.2718483, catboost-3: 0.3763824, lightgbm-1: 0.3517693}`.
- **Postprocess**: `apply_pp` (blend model-residual with PF-residual via `w_pf=0.07`, apply
  exponential ramp `1-exp(-md_since/tau)` with `tau=65`, scale by `alpha=1.04`) +
  `sg_smooth` (Savitzky-Golay, `sg_w=27, sg_p=3`).
- **Real measured score: OOF RMSE = 10.4521227, Public LB = 9.964** (constants
  `BASELINE_V1_OOF_RMSE`/`BASELINE_V1_LB` in `rogii-postprocess-research.ipynb`).

**Finding — Part 2: the architectural delta to top scores (~2.0-2.2 RMSE gap)**

The top-score notebooks' feature core (`BEAMS`, `_pf_ancc`, `_beam_jit`, `FORMATIONS`,
`PF_N=600`) is near-IDENTICAL to ours — strongly suggesting shared lineage (likely both derive
from the same public reference notebook chain; `rogii-sel15-forced-selector.ipynb` even lists
12 public-notebook URLs it drew from, including `aiwody/physical-model-less-overfitting-noise`
and `romantamrazov/rogii-better-solution`). **The gap is NOT feature richness.** It is five
specific architectural choices:

1. **Stacking, not blending**: they train a `Ridge` meta-learner on OOF predictions of ALL 6
   base models (`ridge_trainer.fit(oof_preds, y)` in `rogii-ridge-sp45-proj.ipynb` §4) — we use
   a fixed-weight blend of 3 hand-picked models. (Corrects original H-006, which assumed Ridge
   was a standalone competitor model — it's a stacker.)
2. **Blend the learned stack with an INDEPENDENT model-free heuristic**: final =
   `0.30 * ridge_stack_pred + 0.70 * PF/beam_selector_heuristic` (§6.3 "Blending"). This
   diversifies across inductive bias — a data-fit regression vs. a sequential physics/PF
   tracker that never sees `target` — not just across model families within one paradigm.
3. **Per-well adaptive selector**: `selector_well_code` classifies each well by `n_eval` count
   (threshold `4840.0`) and `Z`-span (`(136.73, 185.51)`) into 6 bins, each mapped to a distinct
   PF-scale/beam-hold parameter variant (`SELECTOR_BIN_VARIANTS`,
   `SELECTOR_GLOBAL_VARIANT='pf_scale_8_hold_0.2'`, `SELECTOR_SCALES=(3,5,8,12)`) — present
   verbatim in all 3 notebooks (shared lineage). We use one global PF/beam config for every well.
4. **Geometry-aware projection postprocess**: their FINAL step (overwrites the blended
   `submission.csv`) is a robust per-well degree-5 polynomial fit (`_robfit`, 4 iterative-
   reweighting passes, Tukey-style scale `1.4826*MAD`) of `U = TVT + Z - anchor` vs. normalized
   along-lateral position `s = (MD-MD_last)/(MD_end-MD_last)`. Their inline comment claims
   **"CV-validated: raw PF -0.54, deployed components -0.33"** RMSE — i.e. potentially worth
   ~0.3-0.5 RMSE alone, in a totally different coordinate system than our `sg_smooth` (which
   smooths raw TVT directly, not the formation-relative `U`).
5. **Heavily-regularized, low-LR LightGBM regime**: one of their 3 LGB stack inputs is an
   Optuna-tuned config (`lr≈0.0093, num_leaves=64, reg_alpha≈10.79, reg_lambda≈95.75,
   colsample_bytree≈0.39, min_child_samples=40`) — a dramatically different bias/variance
   trade-off than our hand-set `num_leaves=255, reg_lambda=3.0`.

Score progression visible inside their own notebooks: previous champion 7.910 → reference
target 7.839 → best variant 7.748. They iterate on this SAME bundled architecture, tuning
individual knobs (`w_r`, PF particle/seed counts, `sigma_0`, projection degree).

**Evidence**: CV=10.4521227 (ours), LB=9.964 (ours) vs. LB=7.748-7.910 (reference notebooks)

**Implication**: Stop planning a from-scratch feature-engineering build (the original
`current_plan.md` Phase 1 assumed we'd start from the external mycarta toolkit). Instead,
reproduce the 5 architectural deltas above in priority order — cheapest/highest-claimed-value
first (projection postprocess, EXP-009, no retraining needed) then the big lever (Ridge-stack +
heuristic blend, EXP-007). Ablate each individually; do not assume all 5 contribute equally, and
do not cargo-cult their literal tuned constants (selector thresholds, blend ratio 0.30/0.70)
without re-deriving against our own data/CV split.

**Action taken**: Rewrote `research/current_plan.md` (new "Reality Check" + Phase 1 = close-the-
gap framing), added H-008/H-009/H-010/H-011 and corrected H-006 in `research/hypothesis.md`,
replaced EXP-004 with EXP-007..011 in `research/experiment_queue.md`, added the train/test
well-overlap question to `research/open_questions.md` Critical table, recorded
`BASELINE_V1_OOF_RMSE`/`BASELINE_V1_LB`/winning weights/pp params in
`memory/leaderboard_progress.md` and `memory/previous_runs.md`.

---

## 2026-06-10 — Finding: EXP-007 (Ridge stack) regressed; pf_ancc is NOT a quality PF heuristic

**Context**: EXP-007 ran on Kaggle 2×T4 (2026-06-10, ~2.5 hrs). Full ablation results logged
in `exp007_ridge_stack__GPU.ipynb`. Submission built via `rogii-submission-ridge-stack-proj.ipynb`
with blend α=0.95 + EXP-009 polynomial projection postprocess.

**Finding — Part 1: Ridge meta-learner alone does NOT beat baseline**

| Variant | OOF RMSE (residual) |
|---------|---------------------|
| A — PF heuristic only (0.5×pf_ancc + 0.5×pf_z) | 13.744 |
| B — Ridge stack only | 10.562 |
| C — 0.30 Ridge / 0.70 PF (reference ratio) | 12.143 |
| D — Tuned blend (α=0.95) | 10.550 |
| Baseline aw pipeline | **10.452** |

Ridge stack alone: OOF 10.562 vs baseline 10.452 — **regression of 0.11 RMSE**.

**Finding — Part 2: The PF heuristic branch is missing — pf_ancc is a single-seed feature artifact**

`train_df['pf_ancc']` has standalone RMSE = 13.744, compared to our 6 base models which all
score 10.6–10.7. This means `pf_ancc` contributes no useful signal as a final predictor. The
reference notebooks' PF heuristic is a *separately built inference-mode artifact*:
`run_pf_lik_ensemble_scales` with 128 seeds × 500 particles × 4 scales. Our aw pipeline's
`pf_ancc` is generated once during feature building (low-quality single-seed feature extraction
run), NOT the reference's high-quality inference estimator. These are architecturally different.

**Finding — Part 3: Submission LB = 10.208 — regression of 0.244 vs BASELINE_V1 (9.964)**

Submission used α=0.95 (nearly pure Ridge) + polynomial projection postprocess applied to test.
The projection was applied to only 3 test wells (anomaly — see open_questions.md for root cause
hypothesis). Whether the projection helped, hurt, or was mostly inactive is unclear; the
regression is primarily attributable to the Ridge stacker.

**Evidence**: EXP-007 ablation output (exp007_ridge_stack__GPU.ipynb lines 779–878);
Submission LB score 10.208.

**Implication**:
1. Do NOT resubmit Ridge + pf_ancc blend in any form. The heuristic branch must first be built
   properly (EXP-014: 128-seed inference-mode PF ensemble).
2. Run EXP-013 IMMEDIATELY: apply projection postprocess to BASELINE_V1 predictions only (no
   stacking). This is the cheapest way to recover to/past 9.964 with a documented postprocess gain.
3. Diagnose the "3 test wells projected" anomaly (open_questions.md) before any projection-
   enabled submission.

**Action taken**: Updated `research/current_plan.md`, `research/experiment_queue.md`,
`research/open_questions.md`, `memory/failed_attempts.md` (FAIL-003),
`memory/leaderboard_progress.md` (EXP-007 row).

---

## 2026-06-10 (Part 2) -- Architecture corrected from notebook source; ravaghi artifacts discovered

See research/findings.md full entry. Key facts:
- sub_2 (70%) = run_pf_lik_ensemble_scales(128 seeds, 4 scales) per test well, NOT pf_ancc
- Correct Ridge: alpha=1.66, positive=True
- ravaghi dataset = pre-built train.csv + 5 koolbox Trainer pickles (LGB x3 + CB x2)
- EXP-007b = correct rebuild using ravaghi artifacts
