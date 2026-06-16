# Open Questions

Questions that need answers before research can proceed confidently.

---

## Critical (Blocking)

- *None* — All blocking questions have been answered and resolved.

---

## Important (Non-blocking)

- [ ] **Q**: Is the `ravaghi` dataset `train.csv` schema identical to our aw pipeline's
  `train_df` (from `rogii-eda-features.ipynb`)? Are there feature columns we have that they
  don't, or vice versa?
  - Context: If the schemas differ, we can't directly substitute `ravaghi/train.csv` into
    our EXP-013 pipeline. More importantly for EXP-007b, we need the Ridge stacker features
    (the 5 base model OOF predictions) to be trained on the SAME feature set as the pickles.
  - Possible answers: The `build_well()` function in `rogii-ridge-sp45-proj.ipynb` (Cell 7)
    is very similar to our aw `build_well` but has additional features (sc_ens ensemble,
    new `bw_early_*`, `bw_mid_*` formation segment features, `gr_env`, `gr_nrg`, etc.).
    The `ravaghi` pickle models were trained on THIS feature set — so we MUST use
    `ravaghi/train.csv` as input to ridge stacking, not our own aw train_df.

- [ ] **Q**: Of the 5 architectural components (Ridge+apply_pp, sub_2 selector+PF, projection,
  Optuna LGB, selector thresholds), which actually drives the gap from 9.964 to 7.748?
  - Context: EXP-007b implements all 5 together. After it runs, ablate each.
    Reference comment suggests projection alone = −0.33 RMSE. Sub_2 PF selector likely
    contributes most of the remaining 2.0 RMSE gap (it is 70% of the final prediction).
  - How to answer: After EXP-007b, run:
    (1) sub_1 only (ridge+apply_pp, no blend)
    (2) sub_2 only (PF selector, no ridge)
    (3) 0.30/0.70 blend without projection
    (4) 0.30/0.70 blend with projection (= full EXP-007b)

- [ ] **Q**: How many test wells share an ID with a train well?
  - Context: EXP-007b Cell 27 handles this via `if wid in train_wells: tvt_phys = tvt_from_contacts(...)`.
    Running EXP-007b will answer this implicitly (count how many wells get the physical shortcut).
  - How to answer: `set(train_wells) & set(test_wells)` — 5-min check.

- [ ] **Q**: Is the projection `_robfit` weight function Tukey hard-threshold or Cauchy soft-threshold?
  - Context: Our `exp009` implementation uses binary Tukey weights `w = (|r|/scale <= c)`.
    Reference Cell 31 uses `w = 1 / (1 + (r / (2 * MAD))²)` — Cauchy/Lorentzian soft weights.
    These give different results on wells with a few large outliers.
  - How to answer: Read Cell 31 of `rogii-ridge-sp45-proj.ipynb` closely. Update
    `exp009_projection_postprocess__CPU.ipynb` weight function if they differ.

- [ ] **Q**: Is freely-available external geological data (public formation-top databases) usable?
  - Context: Competition allows it. Formation names (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA)
    may correspond to a specific basin. Worth a focused search once core architecture is working.

---

## Curiosity (Low priority)

- [ ] **Q**: Would a physics-based stratigraphic-inversion approach complement the ML model?
- [ ] **Q**: Does AEON MiniROCKET generalize with ~773 wells (small-N, high-dimensional)?
- [ ] **Q**: Does the bimodal NW-SE drilling-azimuth pattern hold in our data copy?

---

## Answered Questions

| Q | Answer | Source | Date |
|---|--------|--------|------|
| What is the evaluation metric? | RMSE (lower is better), on masked-toe TVT predictions | `competition/evaluation.md` | 2026-06-08 |
| Is external data allowed? | Yes — "Freely & publicly available external data is allowed" | `competition/competition_info.md` | 2026-06-08 |
| Does global TVT–Z correlation imply useful within-well feature? | No — within-lateral slope ≈ −0.14, not −1. Build-section dominated. | `methodology/within_well_tvt_z_decoupling.md` | 2026-06-08 |
| Is there a validated reference approach? | Yes — mycarta toolkit: LGB + StratifiedGroupKFold, documented ablation | `github.com/mycarta/rogii-geosteering-toolkit` | 2026-06-08 |
| Is MassSNN worth implementing? | No — d² = 2n(1−ρ), mathematically identical to scipy NCC | `methodology/AEON_evaluation.md` | 2026-06-08 |
| Why do Ridge-based notebooks score well? | "Ridge" = stacking meta-learner; win comes from correct 0.30/0.70 architecture | Read of top-score notebooks | 2026-06-08 |
| Do we have a working scored pipeline? | Yes — aw pipeline OOF 10.452, LB 9.964 | `notebooks/reference/mine/*.ipynb` | 2026-06-08 |
| Did EXP-007 improve over baseline? | **NO — LB 10.208 vs 9.964. Architecture was wrong in 2 ways (see below).** | EXP-007 ablation + submission score | 2026-06-10 |
| Is `pf_ancc` the 70% heuristic branch? | **NO — pf_ancc is a 9% soft weight in apply_pp (sub_1 branch). The 70% branch (sub_2) is the 128-seed PF ensemble computed fresh per test well via `run_pf_lik_ensemble_scales`.** | Direct read of `rogii-ridge-sp45-proj.ipynb` Cells 20–21, 27, 30 | 2026-06-10 |
| What Ridge params does the reference use? | **`alpha=1.66, tol=5e-4, positive=True`** (not alpha=1.0, no positivity constraint as in EXP-007) | `rogii-ridge-sp45-proj.ipynb` Cell 10 `ridge_params` dict | 2026-06-10 |
| What is the correct sub_1 formula? | `apply_pp(train_df, ridge_oof_preds, pf_ancc_resid, alpha=1.0, tau=85, w_pf=0.09)` — ridge residual blended with 9% pf_ancc via exponential ramp `1-exp(-md_since/tau)` | `rogii-ridge-sp45-proj.ipynb` Cells 20–21 | 2026-06-10 |
| What is the correct sub_2 formula? | `selector → run_pf_lik_ensemble_scales(hw_te, n_particles=500, n_seeds=128, scales=(3,5,8,12))` → `apply_selector_variant(variant, pf_by_scale, tvt_beam, last_known_tvt)` | `rogii-ridge-sp45-proj.ipynb` Cells 27–28 | 2026-06-10 |
| Does `ravaghi/wellbore-geology-prediction-artifacts` exist and is it useful? | **Yes** — 2.36 GB dataset with `data/train.csv` (pre-built features) + 5 koolbox Trainer pickles (LGB×3 + CB×2). Eliminates GPU training for EXP-007b. Requires `koolbox-offline` to load pickles. | https://www.kaggle.com/datasets/ravaghi/wellbore-geology-prediction-artifacts + notebook source | 2026-06-10 |
| How many base models in the reference vs our pipeline? | Reference: 5 models (LGB×3 + CB×2, per ravaghi pickles). Our aw: 6 models (LGB×3 + CB×3). Ridge stacking set differs slightly. | `ravaghi` artifacts structure + EXP-007 config | 2026-06-10 |
| Can we load `koolbox.Trainer` pickles from `ravaghi` artifacts without installation? | **Yes** — We implemented a local `koolbox.Trainer` stub in `koolbox/__init__.py` that successfully unpickles pre-trained model pickles and delegates `.predict()` to the base estimators. | Workspace stub implementation | 2026-06-15 |
| Does `ravaghi/data/train.csv` contain the necessary columns? | **Yes** — The reference uses `train_df['md_since']` and `train_df['pf_ancc']` directly in stacking/PP, indicating they are present. Test-time projection builds metadata fresh from raw CSVs, bypassing test_df dependencies. | Direct read of `rogii-ridge-sp45-proj.ipynb` Cell 21 and 31 | 2026-06-15 |
| Why did the submission apply projection to only 3 test wells? | **No bug** — The local test folder only contains 3 wells for quick validation (so local runs only show 3 projected wells). In the Kaggle environment, it naturally glob-detects and processes all 100+ wells. | `test_hw_paths` search check in `rogii-submission-ridge-stack-proj.ipynb` | 2026-06-15 |
