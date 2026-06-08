# Experiment Queue

## Template

Each experiment must fill ALL fields before running.

```
## EXP-[ID]: [Short Name]

- Objective: [what you want to test]
- Hypothesis: [why you expect improvement]
- Expected gain: [e.g., +0.002 AUC]
- Risk: Low / Medium / High — [reason]
- Compute estimate: [e.g., 20 min on T4 x2]
- Status: Queued / Running / Done / Cancelled
- Result: [fill after run]
- Notes: [fill after run]
```

---

## EXP-001: Baseline + CV harness reproduction

- Objective: Stand up StratifiedGroupKFold CV (signed azimuth / median-TVT / spatial-bin strata, per `rogii_cv.py`) and record dumb-baseline + default-LightGBM RMSE on it
- Hypothesis: H-002
- Expected gain: N/A (baseline) — but produces the CV harness every later experiment depends on
- Risk: Low — mechanical reproduction of a documented, working scheme
- Compute estimate: <5 min CPU on tiny sample; <15 min Kaggle T4 on full data
- Status: Queued
- Result: [FILL]
- Notes: [FILL]

---

## EXP-002: Domain feature group ablation

- Objective: Reproduce the toolkit's cumulative ablation, group by group: (a) multi-scale NCC + self-correlation, (b) Q-3D tortuosity, (c) trajectory/lateral-only structural baseline, (d) well-level time-series summaries (expected to *hurt* — see H-007)
- Hypothesis: H-001, H-004, H-007
- Expected gain: −0.1 to −0.5 RMSE cumulative (toolkit reports −0.107 from tortuosity alone); H-007 group expected to be net-negative under group CV
- Risk: Medium — typewell↔lateral joins are an easy place to leak eval-zone information; verify each group against the CV harness from EXP-001 before trusting any gain
- Compute estimate: ~15-30 min CPU per feature group on a tiny well subset
- Status: Queued
- Result: [FILL]
- Notes: [FILL]

---

## EXP-003: Lateral-only structural baseline feature

- Objective: Implement and test `tvt_linear_pred_delta`-style structural prior, fit ONLY on lateral-section known points (excluding build), and compare against the naive full-section fit the toolkit flagged as broken
- Hypothesis: H-003
- Expected gain: Naive full-section version should show large negative impact (hundreds-of-ft bias on extrapolation); corrected lateral-only version should give a small, stable CV gain
- Risk: Medium — easy to accidentally include build-section rows in the "lateral-only" mask; add an explicit unit check (slope should be near −0.14 population-average, not near −1)
- Compute estimate: ~10 min CPU
- Status: Queued
- Result: [FILL]
- Notes: [FILL]

---

## EXP-004: ~~Ridge baseline reproduction~~ — SUPERSEDED by EXP-007

- Objective (original): Reproduce a simple Ridge-regression pipeline in the 7.7-7.9 RMSE range
- Status: **Cancelled — premise was wrong.** Reading the actual top-score notebook source
  (2026-06-08) showed "Ridge" is not a standalone competitor model at all — it's a stacking
  meta-learner over 6 base-model OOF predictions, blended 0.30/0.70 with an independent PF/beam
  heuristic. There is no simple-Ridge-pipeline to reproduce; the real architecture is EXP-007.
- Notes: H-006 revised accordingly (see `research/hypothesis.md` Confirmed table). Moved to
  Cancelled/Deprioritized table below.

---

## EXP-007: Ridge meta-learner stack + 0.30/0.70 blend with PF/beam heuristic

- Objective: Replace our current fixed-weight blend of 3 selected base models
  (`{catboost-2: 0.272, catboost-3: 0.376, lightgbm-1: 0.352}`, OOF 10.4521 / LB 9.964) with
  the top-score architecture: (a) train a `Ridge` meta-learner on OOF predictions of ALL 6 base
  models (LGB×3 + CB×3) via `koolbox.Trainer`-style CV wrapper, producing `ridge_oof_preds`;
  (b) compute an independent model-free heuristic (`run_pf_lik_ensemble_scales` — 128-seed
  likelihood-weighted PF ensemble at `pf_scale_8`, or `tvt_from_contacts` for same-well-visible
  test wells); (c) final = `0.30 * ridge_stack_pred + 0.70 * heuristic`
- Hypothesis: H-006 (corrected/confirmed), H-008
- Expected gain: This is the architecture that scores 7.748-7.910 — likely the single largest
  lever toward closing the ~2.1 RMSE gap from our 9.964
- Risk: Medium — `koolbox` is a custom/private library (`ModuleNotFoundError` guarded in their
  code with a fallback `Trainer` stub); we'll need to either obtain it or write an equivalent
  sklearn-CV-wrapper. The PF-likelihood-ensemble heuristic (128 seeds × 500 particles) is
  expensive — budget compute carefully (their own code notes "DTW/PF feature building is
  RAM-bound; serial builds are safer")
- Compute estimate: ~30-60 min CPU for the stacking + heuristic build on a subset; full run
  likely needs Kaggle (their reference uses GPU LGB/CB + 128-seed PF ensembles)
- Status: Queued
- Result: [FILL]
- Notes: [FILL] — ablate (b) alone (heuristic-only) vs. (a) alone (stack-only) vs. the 0.3/0.7
  blend to attribute gain correctly per H-008; do not assume the blend ratio 0.30/0.70 is
  optimal for OUR base models — it was tuned for theirs

---

## EXP-008: Per-well adaptive selector mechanism

- Objective: Reproduce `selector_well_code` — classify each well by `n_eval` row count
  (threshold `4840.0`) and `Z`-span (`(136.73, 185.51)`) into 6 bins, each mapped to a distinct
  PF-scale/beam-hold variant string (`SELECTOR_BIN_VARIANTS`), with a fallback
  `SELECTOR_GLOBAL_VARIANT='pf_scale_8_hold_0.2'` and `SELECTOR_SCALES=(3.0, 5.0, 8.0, 12.0)`
- Hypothesis: H-009
- Expected gain: Unclear magnitude in isolation — depends on how much well-geometry variance
  exists in OUR train/test split (their thresholds were tuned on THEIR data/CV; may need
  re-deriving from scratch rather than copied verbatim)
- Risk: Medium — copying their literal threshold values (4840.0, 136.73, 185.51) without
  re-deriving them against our own well population risks cargo-culting numbers that don't
  transfer; treat as a *mechanism* to reproduce, not literal constants to copy
- Compute estimate: ~20 min CPU to compute `n_eval`/`Z`-span distributions and re-derive
  thresholds; PF-variant runs are the expensive part (shared cost with EXP-007's heuristic)
- Status: Queued
- Result: [FILL]
- Notes: [FILL]

---

## EXP-009: Robust per-well polynomial projection postprocess (U = TVT+Z space)

- Objective: Implement the final-step postprocess that overwrites the blended submission:
  per-well robust degree-5 polynomial fit (`_robfit`, 4 iterative-reweighting passes, Tukey-
  style scale `1.4826 * MAD`) of `U = TVT + Z - anchor` vs. normalized along-lateral position
  `s = (MD - MD_last) / (MD_end - MD_last)`, where `anchor = last_known_TVT + last_known_Z`.
  Compare against (and potentially layer on top of) our existing `sg_smooth` (Savitzky-Golay
  directly on TVT)
- Hypothesis: H-010
- Expected gain: Their own inline comment claims "CV-validated: raw PF -0.54, deployed
  components -0.33" RMSE — i.e. potentially the SECOND largest lever after EXP-007's stacking
- Risk: Low — purely a postprocess step, easy to A/B against current `sg_smooth` on the same
  OOF predictions without retraining anything
- Compute estimate: ~15 min CPU (operates on existing OOF/test predictions, no model training)
- Status: Queued
- Result: [FILL]
- Notes: [FILL] — this is the CHEAPEST experiment to run (no retraining) and has a documented
  claimed gain; consider running it BEFORE EXP-007 as a quick win on our existing blend

---

## EXP-010: Retune LightGBM toward Optuna heavy-regularization regime

- Objective: Add/replace one of our 3 LGB candidates with the Optuna-tuned config found in
  `rogii-sel15-forced-selector.ipynb`: `learning_rate≈0.00935, num_leaves=64, reg_alpha≈10.79,
  reg_lambda≈95.75, colsample_bytree≈0.393, min_child_samples=40, subsample≈0.474,
  n_estimators=10000`. Compare its OOF score AND its contribution as a stack input (per EXP-007)
  against our current hand-set `lgb_params_base` (`num_leaves=255, reg_lambda=3.0, lr=0.02-0.03`)
- Hypothesis: H-011
- Expected gain: Small standalone OOF delta likely, but may matter more as a LOWER-VARIANCE
  stack input (tighter Ridge meta-learner) than as a standalone scorer
- Risk: Low — single config swap, cheap to A/B
- Compute estimate: ~Same as one current LGB candidate run (already budgeted in our pipeline)
- Status: Queued
- Result: [FILL]
- Notes: [FILL] — run this AFTER EXP-007 stack is wired up so its effect can be measured both
  standalone and as a stack-input change

---

## EXP-011: Same-well `tvt_from_contacts` physical shortcut

- Objective: For test wells whose ID also appears in train (`PF_SELECTOR_USE_SAME_WELL_PHYSICAL`
  shortcut), compute a physical TVT baseline directly from formation-contact offsets
  (`tvt_from_contacts(hw_tr, tw_tr, ref_col='EGFDU')`: `offset = mean(TVT - (ref_tvt - (Z -
  ref_col_depth)))`, then `pred = ref_tvt - (Z - ref_col_depth) + offset`) and use it
  preferentially over the learned/heuristic blend for those wells
- Hypothesis: (none yet — needs a new H- entry once we know the overlap count)
- Expected gain: Entirely gated on **how many test wells share an ID with a train well** — if
  near-zero, this is a wasted reproduction effort; if substantial, could be a meaningful
  zero-cost win (perfect physical reconstruction beats any learned model on visible wells)
- Risk: Low — but HIGH information-value-of-checking-first; do the overlap count BEFORE
  investing in implementation
- Compute estimate: <5 min CPU for the overlap check; ~10 min to implement `tvt_from_contacts`
  if overlap is non-trivial
- Status: Queued — **blocked on overlap check** (see `research/open_questions.md`)
- Result: [FILL]
- Notes: [FILL]

---

## EXP-005: CV scheme comparison against first LB submission

- Objective: After the first LB submission from EXP-001, compare CV RMSE under StratifiedGroupKFold vs. plain GroupKFold vs. pure spatial blocking — measure which scheme's CV–LB gap is smallest
- Hypothesis: H-002
- Expected gain: A calibrated, trustworthy local CV signal — prerequisite for trusting any later "CV improved" claim without burning LB submissions
- Risk: Low
- Compute estimate: ~20 min CPU (re-running folds under 3 schemes)
- Status: Queued
- Result: [FILL]
- Notes: [FILL]

---

## EXP-006: AEON MiniROCKET / Catch22 features (Phase 2 stretch)

- Objective: Test whether MiniROCKET (convolutional, local-shape) and/or Catch22 (canonical TS features) features add signal that NCC/self-correlation/tortuosity do not already capture
- Hypothesis: H-005
- Expected gain: Small additive RMSE gain *if* genuinely orthogonal to NCC; otherwise treat as a documented dead end (like MassSNN)
- Risk: Medium — thousands of ROCKET features against ~773 wells risks overfitting; budget timeouts needed (toolkit's `rogii_features.py` already gates AEON/dcor features behind a budget)
- Compute estimate: ~30-60 min CPU for kernel generation on a subset; full run likely needs Kaggle T4
- Status: Queued (deprioritized to Phase 2 — only run after EXP-002 ablation establishes the NCC/tortuosity baseline to test "non-redundancy" against)
- Result: [FILL]
- Notes: [FILL]

---

## Cancelled / Deprioritized

| ID | Name | Reason |
|----|------|--------|
| — | MassSNN-based similarity features | Mathematically equivalent to scipy NCC (d² = 2n(1−ρ), see `AEON_evaluation.md`) — would add code complexity with zero new signal. Use scipy NCC; spend the AEON budget on MiniROCKET (EXP-006) instead |
| — | Naive per-well linear TVT~Z fit on full known section | Confirmed-broken-by-reasoning in toolkit methodology note before we ever ran it — recovers the global slope ≈ −1 and biases toe extrapolation by hundreds of ft. Superseded by EXP-003's lateral-only version |
| EXP-004 | "Ridge baseline reproduction" | Premise was wrong — there is no standalone simple-Ridge pipeline to reproduce. Reading the actual top-score notebook source (2026-06-08) revealed "Ridge" is a stacking meta-learner over 6 base-model OOF preds, blended 0.30/0.70 with an independent PF/beam heuristic. Superseded by EXP-007 (the real architecture) |
| EXP-001/002/003/005 | From-scratch toolkit reproduction (CV harness, NCC/tortuosity ablation, lateral-only structural baseline, CV-LB calibration) | Discovered (2026-06-08) we already have a scored, working `aw` pipeline (LB 9.964) with ~150 features/well including multi-scale NCC, DTW, PF, beam search, and formation-contact features — re-deriving these from the external mycarta toolkit would be redundant. Audit our existing feature set against these instead (folded into EXP-007 prep), don't re-run as fresh experiments |
