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

## EXP-013: BASELINE_V1 + Projection Postprocess Only ← NEXT UP (CPU, ~15 min)

- Objective: Apply robust per-well degree-5 polynomial projection postprocess (EXP-009 logic,
  `_robfit` in U = TVT+Z space) to the KNOWN-GOOD aw pipeline OOF/test predictions directly —
  no stacking, no Ridge, just the existing aw blend → projection.
  Tests projection value independently of the EXP-007 stacking failure.
- Hypothesis: H-010 — projection in U=TVT+Z space removes jitter that sg_smooth cannot.
  Reference notebook inline comment: "CV-validated: raw PF -0.54, deployed components -0.33 RMSE."
- Expected gain: ~0.1–0.5 RMSE reduction on LB. Even 0.1 puts us at 9.86 (new best).
- Risk: **Low** — pure postprocess on existing predictions. If OOF worsens → don't submit.
- Compute estimate: **< 15 min CPU**. No model training. Operates on saved aw-blend OOF arrays.
- Status: **Queued — Script Ready** (validated locally via `scratch/validate_projection.py` which successfully reduced RMSE from 8.71 to 5.90 on synthetic well data).
- Result: [Pending Kaggle Run]
- Notes: Run script `scratch/run_projection_postprocess.py` on Kaggle. It will evaluate:
  (1) sg_smooth only
  (2) projection only
  (3) projection → sg_smooth (matching reference order)
  And automatically output OOF and test `.npy` predictions for the best performing variant.

---

## EXP-007b: Correct Architecture Rebuild via ravaghi Artifacts ← PRIORITY 2

- Objective: Correct re-implementation of the reference 7.748-score architecture using the
  `ravaghi/wellbore-geology-prediction-artifacts` Kaggle dataset. Specifically:
  (a) Load `data/train.csv` (pre-built feature dataframe — skips 5-min feature build)
  (b) Load 5 trainer pickles from `models/` — skips 2+ hrs GPU training
  (c) Ridge meta-learner with CORRECT params: `alpha=1.66, tol=5e-4, positive=True`
  (d) `sub_1 = apply_pp(train_df, ridge_oof_preds, pf_ancc_resid, alpha=1.0, tau=85, w_pf=0.09)`
  (e) `sub_2 = selector → run_pf_lik_ensemble_scales(128 seeds × 500 particles) + beam ensemble`
  (f) `final = 0.30 * sub_1 + 0.70 * sub_2`
  (g) Apply robust Cauchy projection postprocess as the final overwrite step
- Hypothesis: H-008 (revised) + H-009 + H-010
- Expected gain: Matches the 7.748-score architecture. Expect OOF RMSE < 10.0 and LB ~7.7–7.9.
- Risk: **Low** — bypassed `koolbox` package dependency by writing a local `koolbox.Trainer` stub.
- Compute estimate: **~30–60 min CPU** on Kaggle (due to 128 seeds PF run). No GPU needed.
- Status: **Queued — Script Ready** (validated locally via `scratch/run_exp007b.py` which ran successfully end-to-end on a dummy CPU fallback dataset in 16.2 seconds).
- Result: [Pending Kaggle Run]
- Notes: Run script `scratch/run_exp007b.py` on Kaggle. It will load the pickles using our local unpickling stub, fit the positive-constrained Ridge meta-learner, run the full 128-seed PF ensemble, blend, and apply robust projection.

---

## EXP-007: Ridge meta-learner stack + 0.30/0.70 blend with PF/beam heuristic

- Status: **DONE — 2026-06-10 — FAILED**
- Result:
  ```
  A_heuristic_only (pf_ancc+pf_z):  13.744  ← weak feature column, not inference-mode PF
  B_ridge_stack_only:                10.562  ← regressed vs baseline 10.452
  C_blend_030_070:                   12.143  ← catastrophic (heuristic branch was wrong)
  D_tuned_blend (α=0.95):            10.550  ← still regressed
  Baseline (aw):                     10.452  LB 9.964
  Submission LB:                     10.208  ← REGRESSED −0.244
  ```
- Notes: **Architecture was wrong in two ways:**
  1. sub_2 (70% branch) should be 128-seed PF ensemble per test well — NOT the pf_ancc feature column
  2. Ridge params wrong: used alpha=1.0, unconstrained; should be alpha=1.66, positive=True
  3. apply_pp missing from sub_1 (ridge residual was not post-processed before blend)
  4. CV not group-aware (used KFold, not GroupKFold — may leak between wells)
  See FAIL-003 and FAIL-004. Superseded by EXP-007b.

---

## EXP-009: Robust per-well polynomial projection postprocess (U = TVT+Z space)

- Objective: Implement the final-step postprocess: per-well robust degree-5 polynomial fit
  (`_robfit`, 4 iterative-reweighting passes, Tukey-style with `1/(1+(r/2*MAD)²)` weights)
  of `U = TVT_pred + Z - anchor` vs. normalized along-lateral position `s`.
- Hypothesis: H-010
- Expected gain: ~0.3–0.5 RMSE per reference inline comment
- Risk: **Low** — pure postprocess
- Compute estimate: ~15 min CPU
- Status: **Notebook template DONE** (`exp009_projection_postprocess__CPU.ipynb`).
  Full test via **EXP-013** (on BASELINE_V1) and as step (g) in **EXP-007b**.
- Result: [see EXP-013 and EXP-007b]
- Notes: Reference Cell 31 weight function is `1/(1+(r/(2*MAD))²)` (Cauchy/Lorentzian),
  not the binary Tukey hard-threshold we implemented. Verify implementation matches exactly.
  Reference anchor: `anchor = last_known_TVT_input + last_known_Z` (from raw test CSV, not train_df).

---

## EXP-012: PF heuristic quality audit — PARTIALLY ANSWERED

- Status: **Partially answered by architecture analysis (2026-06-10)**
- What we now know:
  - `train_df['pf_ancc']` IS a single-seed feature-extraction PF column (confirmed: produced
    by `run_pf_ancc()` in `build_well()` — one ANCC-based PF run, Numba JIT, not likelihood-weighted)
  - The reference architecture uses `pf_ancc` ONLY at `w_pf=0.09` in `apply_pp` (sub_1 branch,
    9% soft weight with exponential ramp `1-exp(-md_since/tau)`)
  - The real 70% heuristic (sub_2) is computed fresh at submission time via
    `run_pf_lik_ensemble_scales(n_particles=500, n_seeds=128, scales=(3,5,8,12))`
  - This is a different function: 128 seeds × 4 scales, likelihood-weighted ensemble
- Remaining questions: See open_questions.md — is `ravaghi/data/train.csv` identical in schema
  to our aw train_df? Specifically: are `pf_z`, `dz`, `md_since`, `well` columns present?
- Result: **Architecture question answered. Code quality audit no longer needed as blocker.**
- Notes: Do NOT run the remaining Ridge α sensitivity test until EXP-007b is attempted.

---

## EXP-014: 128-seed inference-mode PF ensemble

- Objective: Validate that `run_pf_lik_ensemble_scales(n_particles=500, n_seeds=128)` produces
  a materially better predictor than the pf_ancc feature column (standalone RMSE ≪ 13.74)
- Hypothesis: The 128-seed PF ensemble (sub_2) is the dominant contributor to the 7.748 score
- Expected gain: Diagnostic — if sub_2 standalone RMSE < 10.0, the 0.30/0.70 blend makes sense
- Risk: **High compute** — 128 seeds × 500 particles per well × ~100 test wells ≈ expensive
- Compute estimate: 60–120 min CPU (RAM-bound, serial per well recommended)
- Status: **Queued — DEFERRED** (EXP-007b runs the full sub_2 build; validate there first)
- Result: [FILL]
- Notes: [FILL] — if EXP-007b shows sub_2 is still weak (>12 RMSE standalone), then EXP-014
  must investigate whether our `run_pf_lik_ensemble_scales` implementation matches reference.

---

## EXP-008: Per-well adaptive selector mechanism

- Status: **BUNDLED INTO EXP-007b** — the selector is computed inline in Cell 27 of the
  reference notebook; it is not a separable experiment. EXP-007b implements it directly.
- Notes: `selector_well_code(hw_te)` → (code, variant) → `apply_selector_variant(variant,
  pf_by_scale, tvt_beam, last_known_tvt)`. Thresholds: n_eval=4840, Z-span=(136.73, 185.51).

---

## EXP-010: Retune LightGBM toward Optuna heavy-regularization regime

- Status: **COVERED BY ravaghi ARTIFACTS** — `lightgbm-2` and `lightgbm-3` Trainer pickles
  already contain the Optuna-tuned models (`lr=0.0093, num_leaves=64, reg_alpha=10.79,
  reg_lambda=95.75, colsample_bytree=0.39, min_child_samples=40`). Loading them in EXP-007b
  implicitly tests this — no separate experiment needed.

---

## EXP-011: Same-well `tvt_from_contacts` physical shortcut

- Status: **BUNDLED INTO EXP-007b** — Cell 27 of reference handles this:
  `if wid in train_wells: tvt_phys = tvt_from_contacts(hw_tr, tw_tr)`
  The check `set(train_wells) ∩ set(test_wells)` is implicit in the loop.
  EXP-007b will reveal how many test wells get the physical shortcut.

---

## EXP-005: CV scheme comparison against first LB submission

- Status: **Queued — DEFERRED** (after architecture stabilizes)

---

## EXP-006: AEON MiniROCKET / Catch22 features

- Status: **Queued — Phase 3 only** (after EXP-007b is resolved)

---

## Cancelled / Deprioritized

| ID | Name | Reason |
|----|------|--------|
| — | MassSNN-based similarity features | Mathematically equivalent to scipy NCC — zero new signal |
| — | Naive per-well linear TVT~Z fit on full known section | Confirmed broken before running |
| EXP-004 | "Ridge baseline reproduction" | Premise wrong — Ridge is a stacking meta-learner |
| EXP-001/002/003 | From-scratch toolkit reproduction | aw pipeline already has equivalent features |
| EXP-007 (as implemented) | Ridge stack with pf_ancc as 70% heuristic | Architecture wrong: pf_ancc is 9% soft weight in apply_pp, not the heuristic branch. Superseded by EXP-007b. |
