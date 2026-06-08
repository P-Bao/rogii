# Current Research Plan

Last updated: 2026-06-08
Status: Active

## Reality Check (supersedes the original "start from mycarta toolkit" framing)

We do **not** need to bootstrap from the external mycarta toolkit — we already have a working,
scored pipeline uploaded to `notebooks/reference/mine/` (the "Another-Work" / `aw` pipeline):
feature build → LightGBM×3 + CatBoost×3 → weighted blend → particle-filter-residual postprocess
→ Savitzky-Golay smoothing. **Real measured baseline: OOF RMSE 10.4521227, Public LB 9.964**
(see `memory/leaderboard_progress.md`, `memory/previous_runs.md` for full config).

Top public notebooks in `notebooks/reference/top_score/` score **7.748–7.910** — a **~2.0–2.2
RMSE gap**. Reading them in full shows the gap is *not* feature poverty (their feature core —
`BEAMS`, `_pf_ancc`, `_beam_jit`, multi-scale NCC/DTW, `FORMATIONS` — is near-identical to ours,
strongly suggesting shared lineage/reference). The gap is **architectural**: how predictions
are combined and post-processed. See `research/findings.md` 2026-06-08 entry for the full
side-by-side comparison.

## Active Sprint

**Goal**: Close the ~2.1 RMSE gap by reproducing the four specific architectural components the
top-score notebooks have and we don't (see Priority Queue). This is now an *ablation-toward-a-
known-target* sprint, not a from-scratch baseline build.
**Deadline**: Competition runs 2026-05-05 → 2026-08-05. Phase 1 target: complete by ~2026-06-22.

## Phase 1 — Close the Architecture Gap

1. **Stacking via meta-learner** (EXP-007): replace our fixed-weight blend
   (`{catboost-2: 0.27, catboost-3: 0.38, lightgbm-1: 0.35}`) with a Ridge meta-learner trained
   on OOF predictions of all 6 base models (3 LGB + 3 CB), per top-score "# 4. Ensembling with
   Ridge" section. Ridge is a STACKER here, not a standalone model — H-006 corrected.
2. **Blend learned stack with an independent physical/PF heuristic** (EXP-007 cont'd): top
   notebooks compute final = `0.30 * ridge_stack_pred + 0.70 * pf_beam_selector_heuristic`.
   This is a genuinely different ensembling philosophy — diversify across inductive bias
   (learned vs. model-free physics), not just across model families. New: H-008.
3. **Per-well adaptive "selector"** (EXP-008): reproduce `selector_well_code` — classifies each
   well by `n_eval` count (threshold 4840) and `Z`-span (thresholds 136.73 / 185.51) into 6 bins,
   each mapped to a different PF-scale / beam-hold parameter variant
   (`SELECTOR_BIN_VARIANTS`, `SELECTOR_GLOBAL_VARIANT='pf_scale_8_hold_0.2'`,
   `SELECTOR_SCALES=(3,5,8,12)`). New: H-009.
4. **Geometry-aware projection postprocess** (EXP-009): replace/augment our `sg_smooth`
   (Savitzky-Golay directly on TVT) with their robust per-well degree-5 polynomial fit in
   `U = TVT + Z - anchor` space, parameterized by normalized MD `s = (MD - MD_last)/(MD_end -
   MD_last)`, with iterative reweighting (Tukey-style, 4 iters, `c=2.0`). Their own comment
   claims this is **CV-validated worth ~0.3–0.5 RMSE** ("raw PF -0.54, deployed components
   -0.33"). New: H-010.
5. **Retune LightGBM toward heavier regularization** (EXP-010): their Optuna-tuned LGB variant
   (`lr≈0.0093, num_leaves=64, reg_alpha≈10.8, reg_lambda≈95.8, colsample_bytree≈0.39,
   min_child_samples=40`) is a very different regime from our hand-set params (`num_leaves=255,
   reg_lambda=3.0, lr=0.02-0.03`). New: H-011.
6. **Wire `tvt_from_contacts` as a same-well physical shortcut** (EXP-011): when a test well ID
   also appears in train, top notebooks compute a physical TVT baseline directly from formation-
   contact offsets (`tvt_from_contacts`, ref formation `EGFDU`) and use it preferentially
   (`PF_SELECTOR_USE_SAME_WELL_PHYSICAL=True`). Confirm whether/how many test wells overlap train
   IDs — if few, this is a low-leverage item; if many, it could be a meaningful chunk of the gap.

**Exit criteria**: Local OOF RMSE materially below 10.45 (toward the 7.7–7.9 reference range),
and each of the 5 components above individually ablated and logged in `research/findings.md` so
we know which one(s) actually drive the gain (do NOT assume all 5 contribute equally).

## Priority Queue

1. [ ] EXP-007: Ridge meta-learner stack + 0.3/0.7 blend with PF/beam heuristic — Expected gain: largest single lever; likely the bulk of the ~2.1 RMSE gap (H-006 corrected, H-008)
2. [ ] EXP-009: Robust per-well polynomial projection postprocess in U=TVT+Z space — Expected gain: ~0.3-0.5 RMSE per their own CV note (H-010)
3. [ ] EXP-008: Per-well selector mechanism (n_eval / Z-span thresholds → PF variant lookup) — Expected gain: unclear magnitude; adaptive > one-size-fits-all in principle (H-009)
4. [ ] EXP-010: Retune LightGBM toward Optuna-found heavy-regularization regime — Expected gain: small but compounds inside the stack (H-011)
5. [ ] EXP-011: Same-well `tvt_from_contacts` physical shortcut for overlapping test/train well IDs — Expected gain: depends entirely on train/test well-ID overlap count (open question)

### Deprioritized (from original from-scratch plan — now redundant)

- EXP-001/002/003/005 (CV harness, NCC/tortuosity ablation, lateral-only structural baseline,
  CV-vs-LB calibration): our `aw` pipeline already HAS multi-scale NCC, DTW, PF, beam search,
  and `tvt_from_contacts`-equivalent formation features at ~150 features/well. Re-deriving them
  from the external mycarta toolkit would be redundant churn. Keep the hypotheses (H-001/H-004)
  as background reading but do not re-run as fresh experiments — instead audit whether our
  existing ~150-feature set already covers them (fold into EXP-007 prep work).

## In Progress

- (none — sprint not started; EXP-007 is next up)

## Blocked

- Train/test well-ID overlap count (needed for EXP-011 prioritization) — requires a quick
  `set(train_wells) & set(test_wells)` check against the actual data, not yet run locally.
- `dataset_info.md` row counts / missingness still TBD — lower priority now that we have a
  working pipeline; back-fill opportunistically from the `aw` `prepare_meta.json` artifact.

## Recently Completed

| Date | Task | Result | Next Action |
|------|------|--------|-------------|
| 2026-06-08 | Repo research pass: read all `references/`, fetched mycarta toolkit repo + methodology notes | Found external reference solution w/ documented ablations | Superseded — see "Reality Check" above |
| 2026-06-08 | Read all 5 of our own uploaded notebooks (`eda-features`, `train-lightgbm`, `train-catboost`, `postprocess-research`, `submission`) end to end | Extracted real baseline numbers (OOF 10.4521, LB 9.964), winning blend weights, postprocess params, full feature-pipeline architecture | Recorded in `memory/leaderboard_progress.md`, `memory/previous_runs.md`, `research/findings.md` |
| 2026-06-08 | Read all 3 top-score reference notebooks (`ridge-sp45-proj` 7.748, `ridge-artifact-parameter-experiments` 7.881, `sel15-forced-selector` 7.807/7.839 ref) | Identified the 5 specific architectural deltas vs. our pipeline (stacking, blend-with-heuristic, selector, projection postprocess, hyperparameter regime) | Drove this rewrite of Phase 1 — EXP-007..011 |

## Strategic Direction

Stop treating this as a feature-engineering problem — our feature core is already comparable to
(and in some respects richer than) the top-score notebooks'. The ~2.1 RMSE gap lives in
**ensembling philosophy and postprocessing geometry**: (a) a learned Ridge-stack blended with an
independent model-free PF/beam heuristic at a fixed ratio, (b) per-well adaptive parameter
selection instead of global constants, and (c) a geometry-aware (U=TVT+Z space) projection
postprocess instead of naive TVT-space smoothing. Reproduce these in priority order (EXP-007 →
EXP-009 → EXP-008 → EXP-010 → EXP-011), ablating each individually so we can attribute the gain.

## Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-06-08 | Adopt StratifiedGroupKFold (signed azimuth, median TVT, spatial bins) as the default CV scheme instead of plain KFold or pure spatial blocking | Toolkit found train/test wells are spatially *interpolated*, not extrapolated — pure spatial blocking is overly pessimistic vs. the real test condition; well-level grouping prevents leakage. NOTE: our `aw` pipeline currently uses plain `GroupKFold(5)` — revisit whether switching helps once EXP-007..011 land |
| 2026-06-08 | Reject "fit per-well linear TVT = a·Z + b on the full known section" as a baseline feature | Methodology note `within_well_tvt_z_decoupling.md`: full-section fit is dominated by the build section (slope ≈ −1); within a single lateral, slope ≈ −0.14 |
| 2026-06-08 | Treat MassSNN (AEON) and scipy NCC as redundant; do not implement both | `AEON_evaluation.md` proves `d² = 2n(1−ρ)` — z-normalized Euclidean distance and Pearson correlation rank identically |
| 2026-06-08 | SUPERSEDED original "build from mycarta toolkit" plan — pivot to "close the gap to our own uploaded reference notebooks" | Discovered we already have a scored, working pipeline (LB 9.964) far more advanced than a from-scratch toolkit reproduction would produce in the same timeframe; the toolkit is now a secondary cross-reference, not the primary roadmap |
