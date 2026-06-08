# Open Questions

Questions that need answers before research can proceed confidently.

---

## Critical (Blocking)

- [ ] **Q**: How many test wells share an ID with a train well (overlap count for the
  `tvt_from_contacts` "same-well physical shortcut", `PF_SELECTOR_USE_SAME_WELL_PHYSICAL`)?
  - Why blocking: EXP-011 is entirely gated on this number — near-zero overlap makes it a wasted
    reproduction effort, substantial overlap makes it a near-zero-cost win (perfect physical
    reconstruction beats any learned model on a visible well).
  - How to answer: `set(train_wells) & set(test_wells)` — a 5-minute check against the actual
    `train/`/`test/` directory listings. Do this FIRST, before investing in EXP-011.

- [ ] **Q**: What are the actual dataset sizes — row counts per well, missing-value %, target (TVT) distribution? `dataset_info.md` is still all `[FILL]`/`TBD`.
  - Why blocking: Lower priority now that we have a working pipeline (LB 9.964) — but still needed to back-fill `dataset_info.md` and to re-derive selector thresholds (EXP-008) against OUR well population rather than copying theirs verbatim.
  - How to answer: Our own `aw` pipeline already produces `prepare_meta.json` / `train_df.joblib` artifacts on Kaggle — pull those stats back into the repo rather than re-deriving from scratch.

- [ ] **Q**: Is the masked "evaluation zone" always the toe-end of the lateral, and is its extent (ΔMD / row count) consistent across wells?
  - Why blocking: Needed to re-derive `SELECTOR_N_EVAL_THRESHOLD`/`SELECTOR_Z_SPAN_THRESHOLDS`-equivalent values against our own data (EXP-008) rather than cargo-culting the top-score notebooks' literal constants (4840.0, 136.73, 185.51 — tuned on THEIR split).
  - How to answer: Inspect `TVT_input` (NaN in evaluation zone) directly per well; confirm the NaN block is always at the high-MD (toe) end and measure `n_eval` / `Z`-span distributions across all wells.

---

## Important (Non-blocking)

- [x] **Q**: Why do public Ridge-based notebooks (7.748, 7.807, 7.881 RMSE) reportedly do well? — **ANSWERED 2026-06-08**, see Answered table below.

- [ ] **Q**: Of the 5 architectural deltas identified (Ridge-stack, blend-with-heuristic, selector, U-space projection postprocess, Optuna-tuned LGB regime), which ones actually drive the ~2.1 RMSE gain, and which are marginal/noise?
  - Context: The top-score notebooks bundle all 5 together; their own internal note attributes ~0.3-0.5 RMSE to the projection postprocess alone (H-010), but the rest is unattributed. Cargo-culting the whole bundle risks copying dead weight (e.g., literal selector threshold constants tuned on a different CV split).
  - Possible answers: Run EXP-007/008/009/010/011 with INDIVIDUAL ablation logging (each on/off against the same base) rather than bundling — `research/findings.md` should end up with 5 separate before/after CV deltas.

- [ ] **Q**: Is freely-available external geological data (e.g., public formation-tops databases for the basin this dataset is drawn from) usable, and would it materially help beyond the provided typewells?
  - Context: Competition rules say "Freely & publicly available external data is allowed," and the toolkit's bimodal NW-SE azimuth rose suggests a specific field/basin with a knowable regional dip trend.
  - Possible answers: If the basin can be identified from formation names (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA — these look like named formation-top codes), a regional structure map could give a much stronger structural prior than per-well linear fits. Worth a focused search once formation names are confirmed against public basin databases.

- [ ] **Q**: How should we treat the `ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA` "predicted depth of various geological formations" columns — are they training-only inputs that establish a per-well formation-top structure usable as a feature, or potential leakage vectors?
  - Context: `dataset_info.md` flags them as "Training only," same caveat as TVT itself.
  - Possible answers: If they are organizer-provided structural interpretations available at training time only, they may be usable as auxiliary targets / multi-task signal but must be excluded from any feature pipeline that runs on test wells (where they won't exist) — confirm they are absent from `test/` files before building anything on them.

---

## Curiosity (Low priority)

- [ ] **Q**: Would a physics-based sequential stratigraphic-inversion approach (Tikhonov-regularized sequential curve fitting, as in [geosteering-no/inversion_school_geosteering](https://github.com/geosteering-no/inversion_school_geosteering)) beat or usefully complement an ML model, given how small the within-lateral ΔTVT range is (5-13 ft)?
- [ ] **Q**: Does AEON's MiniROCKET generalize with only ~773 wells, or does its thousands-of-features output overfit a dataset this size (small-N, high-dimensional feature risk — see EXP-006)?
- [ ] **Q**: Does the bimodal NW-SE drilling-azimuth pattern the toolkit found (fig2_azimuth_rose) hold in our copy of the data too, and does it correlate with the per-well TVT-Z slope the way the toolkit's `fig3c` suggests?

---

## Answered Questions

| Q | Answer | Source | Date |
|---|--------|--------|------|
| What is the evaluation metric? | RMSE (lower is better), on the masked-toe TVT predictions | `competition/evaluation.md` | 2026-06-08 |
| Is external data allowed? | Yes — "Freely & publicly available external data is allowed" | `competition/competition_info.md` | 2026-06-08 |
| Does the global TVT–Z correlation (r ≈ −0.96) imply a strong within-well linear relationship usable as a feature? | No — within a single lateral, the relationship is essentially decoupled (slope ≈ +0.057 to −0.14 population-mean, not −1). The global correlation is a between-well structural-elevation signal dominated by the build section; within-lateral TVT change is driven by formation-thickness variation independent of the well's vertical drift | `mycarta/rogii-geosteering-toolkit`, `methodology/within_well_tvt_z_decoupling.md` | 2026-06-08 |
| Is there a validated reference approach we can adapt rather than build from scratch? | Yes — `mycarta/rogii-geosteering-toolkit`: single LightGBM + StratifiedGroupKFold, feature groups = multi-scale NCC, self-correlation, Q-3D tortuosity, trajectory baseline, offset-well prior, landing-zone state, with a documented per-group ablation (tortuosity = −0.107 RMSE, the largest single gain) | `github.com/mycarta/rogii-geosteering-toolkit` README + `kaggle/rogii_features.py`, `kaggle/rogii_cv.py` | 2026-06-08 |
| Is MassSNN (AEON) worth implementing alongside scipy NCC? | No — proven mathematically equivalent up to a monotonic transform (d² = 2n(1−ρ)); adds code complexity with zero new signal. MiniROCKET/Catch22 are the genuinely-different AEON candidates | `methodology/AEON_evaluation.md` | 2026-06-08 |
| Why do public Ridge-based notebooks (7.748-7.910 RMSE) score well — strong model, postprocessing trick, or LB-probing? | Neither, exactly: "Ridge" is a STACKING META-LEARNER trained on OOF predictions of 6 base models (LGB×3 + CB×3) — not a standalone model. The actual score driver is the FULL architecture: `0.30*ridge_stack + 0.70*model-free PF/beam heuristic` blend, PLUS a per-well adaptive selector, PLUS a final geometry-aware (U=TVT+Z space) robust polynomial projection postprocess that their own inline comment claims is worth ~0.3-0.5 RMSE alone. H-006 corrected; H-008/009/010/011 added | Direct read of `rogii-ridge-sp45-proj.ipynb`, `rogii-sel15-forced-selector.ipynb`, `rogii-ridge-artifact-parameter-experiments.ipynb` source (2026-06-08) | 2026-06-08 |
| Do we actually have a working scored pipeline already, or do we need to build from scratch? | We have a working, scored `aw` ("Another-Work") pipeline: ~150 features/well (multi-scale NCC, DTW, particle filters, beam search, formation-contact features), LightGBM×3 + CatBoost×3, weighted blend, PF-residual postprocess + Savitzky-Golay smoothing. **Real measured baseline: OOF RMSE 10.4521227, Public LB 9.964.** This supersedes the original "build from mycarta toolkit" plan | `notebooks/reference/mine/*.ipynb` (read in full 2026-06-08); recorded in `memory/leaderboard_progress.md`, `memory/previous_runs.md` | 2026-06-08 |
