# Current Research Plan

Last updated: 2026-06-10
Status: Active — Architecture-Corrected Rebuild Sprint

---

## Score Board

| Run | OOF RMSE | Public LB | Status |
|-----|----------|-----------|--------|
| BASELINE_V1 (aw pipeline) | 10.4521 | **9.964** | Best LB to date |
| EXP-007 (ridge stack α=0.95 + proj postprocess) | 10.550 | **10.208** | Submitted — **REGRESSED −0.244 LB** |

---

## Architecture Correction (2026-06-10 — from reading source of rogii-ridge-sp45-proj.ipynb)

Reading the actual source of `rogii-ridge-sp45-proj.ipynb` (Cell 21, 27, 30) revealed that our
EXP-007 implementation was **architecturally wrong in two ways**:

### Correct reference architecture

```
sub_1 = last_known_tvt + apply_pp(train_df,
            ridge_oof_preds,          # model residual
            pf_oof,                   # pf_ancc residual (soft constraint, w_pf=0.09 only)
            alpha=1.0, tau=85, w_pf=0.09)

sub_2 = per-well SELECTOR:
          code, variant = selector_well_code(hw_te)   # n_eval / Z-span → bin → variant name
          pf_by_scale = run_pf_lik_ensemble_scales(hw_te, tw_ref,
                            n_particles=500, n_seeds=128)  # 128-seed × 4-scale PF
          tvt_beam    = run_beam_ensemble(hw_te, tw_ref)   # 14-config beam search ensemble
          sub_2 = apply_selector_variant(variant, pf_by_scale, tvt_beam, last_known_tvt)

final = 0.30 * sub_1 + 0.70 * sub_2
        → then EXP-009 robust polynomial projection OVERWRITES this
```

### What EXP-007 got wrong

| Architecture item | Reference | Our EXP-007 |
|-------------------|-----------|-------------|
| `sub_1` postprocess | `apply_pp(ridge, pf_ancc, w_pf=0.09, tau=85)` — pf_ancc as 9% soft weight | None — pure ridge residual |
| `sub_2` heuristic | 128-seed PF ensemble at **selector-chosen scale** + beam blend | `0.5*pf_ancc + 0.5*pf_z` from train_df column (same weak feature!) |
| Ridge params | `alpha=1.66, tol=5e-4, positive=True` | `alpha=1.0, no positivity constraint` |
| Model set | 5 models: LGB×3 (1 default + 2 Optuna) + CB×2 | 6 models: LGB×3 + CB×3 (our own pipeline) |
| `koolbox.Trainer` CV wrapper | Groups-aware CV, stores OOF preds correctly | Custom `RidgeStacker` with KFold (no group awareness) |

The `pf_ancc` column only enters `sub_1` as a 9% soft weight via `apply_pp` — it was never
intended to be "the heuristic branch." The heuristic (sub_2 = 70% of the final) is the
128-seed PF ensemble computed **fresh at inference time per test well**.

### Why this matters

The 0.30/0.70 blend **only makes sense** when sub_2 (the 70% branch) is a quality PF ensemble.
In our EXP-007, sub_2 was the weak pf_ancc column — hence the 0.30/0.70 ratio gave 12.14 RMSE
(catastrophic). Using α=0.95 (near-pure ridge) was the correct fallback given what we had, but
the Ridge itself has its own issues (wrong params, wrong CV, not group-aware).

---

## Key External Resource: `ravaghi/wellbore-geology-prediction-artifacts`

Dataset: https://www.kaggle.com/datasets/ravaghi/wellbore-geology-prediction-artifacts
Size: 2.36 GB. Version 6 (updated 2026-06-04). Apache 2.0.

**Contents:**
```
artifacts_path/
├── data/
│   └── train.csv          ← Pre-built feature dataframe (all train wells, ~150+ features)
└── models/
    ├── lightgbm-1/        ← koolbox.Trainer pickle — default LGB (lr=0.03, n_leaves=255)
    ├── lightgbm-2/        ← koolbox.Trainer pickle — Optuna LGB (lr=0.0093, n_leaves=64)
    ├── lightgbm-3/        ← koolbox.Trainer pickle — Optuna LGB (seed=29)
    ├── catboost-1/        ← koolbox.Trainer pickle — CB (lr=0.02, seed=7)
    └── catboost-2/        ← koolbox.Trainer pickle — CB (lr=0.03, seed=123)
```

**How the reference notebook uses it** (Cell 8/13/15):
```python
# Feature load (avoids ~5-min feature build)
if (CFG.artifacts_path / "data" / "train.csv").exists():
    train_df = pd.read_csv(CFG.artifacts_path / "data" / "train.csv", low_memory=False)

# Model load (avoids ~2-hr GPU training)
trainer = joblib.load(list((CFG.artifacts_path / save_path).glob('*.pkl'))[0])
oof_preds[name] = trainer.oof_preds   # already has OOF from CV run
test_preds[name] = trainer.predict(X_test)
```

**Dependency**: requires `koolbox` Python package (private library). The reference notebook
installs it from `/kaggle/input/koolbox-offline` (a separate Kaggle dataset with the `.whl`).

**Impact on our plan**: Using the `ravaghi` artifacts + `koolbox-offline` means:
- No GPU run needed for EXP-007b (load pre-trained models directly)
- Same feature matrix as reference (no feature-engineering divergence)
- Correct OOF predictions for Ridge stacking
- Allows EXP-007b to run on CPU-only in ~30 min

---

## Phase 2 — Architecture-Corrected Rebuild

**Goal**: Correctly implement the reference architecture using `ravaghi` artifacts + `koolbox`.
**Deadline**: Competition ends 2026-08-05. Phase 2 target: working correct impl by ~2026-06-22.

### P2-A: Cheap quick wins (PRIORITY 1 — no GPU, no koolbox needed)

1. **EXP-013: BASELINE_V1 + projection postprocess** (~15 min CPU)
   Apply EXP-009 projection postprocess to the KNOWN-GOOD aw blend OOF predictions.
   Tests projection value independently of stacking. Reference claims ~0.3–0.5 RMSE gain.
   **Run this first** — can submit immediately if CV improves.

### P2-B: Correct architecture rebuild (PRIORITY 2 — requires koolbox + ravaghi)

2. **EXP-007b: Correct re-implementation** (~30 min CPU after artifact load)
   - Load `train.csv` from `ravaghi` (skip feature build)
   - Load 5 Trainer pickles (skip GPU training)
   - Ridge with correct params: `alpha=1.66, tol=5e-4, positive=True`
   - `sub_1 = apply_pp(ridge_preds, pf_ancc_residual, alpha=1.0, tau=85, w_pf=0.09)`
   - `sub_2 = selector → run_pf_lik_ensemble_scales(128 seeds × 4 scales) + beam ensemble`
   - `final = 0.30 * sub_1 + 0.70 * sub_2`
   - Then apply EXP-009 projection as final step
   - **This is the correct reproduction of the 7.748-score architecture.**

3. **EXP-008b: Selector mechanism** (bundled into EXP-007b — it's in Cell 27 of the reference)
   The selector is computed inline during the sub_2 heuristic build — not a separate experiment.

### P2-C: Understand koolbox dependency (PREREQUISITE for EXP-007b)

4. **koolbox investigation**: Determine whether we can:
   - (a) Use the `koolbox-offline` Kaggle dataset as-is
   - (b) Implement a minimal `koolbox.Trainer` equivalent (the notebook only uses:
     `trainer.oof_preds`, `trainer.overall_score`, `trainer.predict(X_test)`)
   - (c) Whether our own aw pipeline's `GroupKFold(5)` OOF preds are compatible enough to
     substitute (same fold structure needed for Ridge to generalize)

### P2-D: Ablation clarity (PRIORITY 3)

5. **EXP-014: 128-seed PF ensemble on OOF wells** — validate sub_2 standalone RMSE.
   Only run if EXP-007b shows sub_2 (selector PF) is still weak — unlikely given reference
   scores, but needed to attribute gain correctly.

6. **EXP-010: Optuna LGB** — already in `ravaghi` as lightgbm-2/3. Covered by EXP-007b.

7. **EXP-011: Same-well tvt_from_contacts** — Cell 27 handles this automatically
   (`if wid in train_wells: tvt_phys = tvt_from_contacts(...)`). Covered by EXP-007b.

---

## Priority Queue (Updated 2026-06-15)

1. [x] ~~EXP-007~~ — DONE, REGRESSED. Architecture was wrong.
2. [x] ~~koolbox check~~ — DONE. Bypassed by writing a local `koolbox.Trainer` stub in `koolbox/` to support offline deserialization.
3. [ ] **EXP-013**: BASELINE_V1 + projection postprocess — ~15 min CPU, submit if CV↑. Cauchy weights projection code implemented in `src/postprocessing/projection.py` and validated; run script ready at `scratch/run_projection_postprocess.py`.
4. [ ] **EXP-007b**: Correct architecture rebuild using `ravaghi` artifacts + correct Ridge params. Execution script ready at `scratch/run_exp007b.py` and currently validating.
5. [ ] EXP-014: 128-seed PF OOF validation (deferred — covered by EXP-007b)
6. [ ] EXP-005: CV scheme calibration (deferred until architecture stabilizes)
7. [ ] EXP-006: MiniROCKET features (Phase 3 stretch)

---

## Strategic Direction (Updated 2026-06-15)

We now have a clear, concrete path:

**Step 1 (today)**: Run EXP-013 (projection on BASELINE_V1) on Kaggle using `scratch/run_projection_postprocess.py`. If OOF improves → submit → new best LB.

**Step 2 (today/this week)**: Run EXP-007b on Kaggle using `scratch/run_exp007b.py` with `ravaghi` model artifacts. The script uses the correct Ridge stacking (positive=True, α=1.66), sub_1 postprocess (apply_pp at w_pf=0.09), sub_2 (128-seed PF ensemble + beam blend selector), and Cauchy projection postprocess.

**Step 3 (after EXP-007b)**: If EXP-007b scores ~7.7–7.9, ablate what actually helps vs our baseline. If it still regresses, investigate which sub-component fails.

---

## Blocked

- *None* — All blockers are resolved.
  - `koolbox-offline` blocker bypassed by local stub package.
  - train.csv column presence verified from reference source.
  - 3-well projection behavior verified as correct.

---

## Recently Completed

| Date | Task | Result | Next Action |
|------|------|--------|-------------|
| 2026-06-08 | Repo research pass, read all notebooks | Identified 5 architectural deltas | Drove Phase 1 plan |
| 2026-06-10 | EXP-007: Ridge stack + projection submission | **REGRESSED: LB 10.208 vs 9.964** | Architecture re-analysis |
| 2026-06-10 | Read full source of rogii-ridge-sp45-proj.ipynb | **Architecture corrected** — `sub_2` is 128-seed PF/beam selector, NOT pf_ancc column; `sub_1` is ridge + apply_pp at w_pf=0.09; Ridge params = alpha=1.66, positive=True | EXP-007b |
| 2026-06-10 | Discovered `ravaghi/wellbore-geology-prediction-artifacts` dataset | Pre-built `train.csv` + 5 koolbox Trainer pickles — eliminates GPU training for EXP-007b | Add to Kaggle notebook as data source |

---

## Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-06-08 | Adopt StratifiedGroupKFold as default CV | Train/test wells spatially interpolated, not extrapolated |
| 2026-06-08 | Reject full-section TVT~Z linear fit | Build-section dominated; within-lateral slope ≈ −0.14, not −1 |
| 2026-06-08 | Treat MassSNN and scipy NCC as redundant | d² = 2n(1−ρ), mathematically identical |
| 2026-06-08 | Pivot from mycarta toolkit rebuild to closing architecture gap | Already have LB 9.964 baseline |
| 2026-06-10 | Do NOT use pf_ancc column as the 70% heuristic branch | pf_ancc is only a 9% soft weight in apply_pp (sub_1); the 70% branch is the 128-seed PF selector ensemble (sub_2) |
| 2026-06-10 | Use `ravaghi` artifacts for EXP-007b instead of retraining | Eliminates 2+ hrs GPU, ensures identical OOF preds to reference, correct koolbox Trainer CV wrapper |
| 2026-06-10 | EXP-013 before EXP-007b | Projection on BASELINE_V1 is zero-risk, 15-min, submittable if OOF↑; EXP-007b needs koolbox setup which takes more time |
