# Previous Runs

Full history of all training runs. Append only. Never delete rows.

---

## Run Log

| Run ID | Date | Model | Features | CV Score | Public LB | Private LB | Config | Notes |
|--------|------|-------|----------|----------|-----------|------------|--------|-------|
| AW-LGB | [pre-existing] | LightGBM ×3 (seeds 42/7/123, lr 0.025/0.020/0.030, n_est=8000) | `aw` ~150-feature/well core (multi-scale NCC, DTW multiscale+stochastic, particle filters `_pf_ancc`/`_pf_z`, beam search ×7 configs, formation-contact/spatial-KNN imputation) on residual target `TVT - last_known_tvt` | per-candidate OOF (see `lightgbm_scores.csv`) | — | — | `notebooks/reference/mine/rogii-train-lightgbm.ipynb`: `lgb_params_base = dict(boosting_type="gbdt", num_leaves=255, min_child_samples=15, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05, max_bin=255)`, `GroupKFold(5)`, early-stop 250 rounds | Saved as `lightgbm-{1,2,3}` artifacts; `lightgbm-1` (seed 42, lr 0.025) is one of the 3 models in the winning blend |
| AW-CB | [pre-existing] | CatBoost ×3 (seeds 42/7/123, lr 0.025/0.020/0.030, iterations=8000) | same `aw` feature core as AW-LGB | per-candidate OOF (see `catboost_scores.csv`) | — | — | `notebooks/reference/mine/rogii-train-catboost.ipynb`: `cb_params_base = dict(iterations=8000, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254, loss_function="RMSE", od_type="Iter", od_wait=300)`, `GroupKFold(5)` | Saved as `catboost-{1,2,3}` artifacts; `catboost-2` (seed 7) and `catboost-3` (seed 123) are 2 of the 3 models in the winning blend |
| **BASELINE_V1** | [pre-existing] | Weighted blend of `lightgbm-1` + `catboost-2` + `catboost-3` + PF-residual postprocess + Savitzky-Golay smoothing | `aw` core + `pf_ancc` (particle-filter TVT track) as postprocess input | **OOF RMSE = 10.4521227** | **9.964** | — | blend weights `{catboost-2: 0.2718483, catboost-3: 0.3763824, lightgbm-1: 0.3517693}`; `apply_pp(model_resid, alpha=1.04, tau=65, w_pf=0.07)`; `sg_smooth(sg_w=27, sg_p=3)` | **THIS IS OUR REAL CURRENT BASELINE.** Source: `notebooks/reference/mine/rogii-postprocess-research.ipynb` `BASELINE_V1_OOF_RMSE`/`BASELINE_V1_LB`/`WINNING_WEIGHTS`/`WINNING_PP_PARAMS` constants (read+recorded 2026-06-08). Gap to top-score reference notebooks (7.748-7.910): ~2.0-2.2 RMSE — see `research/findings.md` for architecture-delta analysis |

---

## Best Configurations

### Best CV
- Run ID: BASELINE_V1
- Score: OOF RMSE 10.4521227
- Config: `notebooks/reference/mine/rogii-postprocess-research.ipynb` — see Run Log row above for full weights/params

### Best Public LB
- Run ID: BASELINE_V1
- Score: 9.964
- Config: same as above — `submission.csv` produced by `notebooks/reference/mine/rogii-submission.ipynb` (loads `selection_postprocess_research_v2.json`, requires `use_candidate=True` guardrail before writing)

### Selected for Final Submission
- Run ID: BASELINE_V1 (current default until EXP-007/008/009/010/011 produce a beating candidate)
- Reason: Only scored, validated submission path we have right now. The `submission.csv` writer
  itself enforces this via a guardrail (`use_candidate` flag in `selection_postprocess_research_v2.json`
  — raises if the candidate hasn't beaten `BASELINE_V1_OOF_RMSE`), so any future replacement must
  prove a CV win before it can even be submitted through the existing pipeline.

---

## Config Snapshots

### R-001 Config
```python
# Paste key config here after each run
params = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 1000,
}
```

---

## Ensemble Log

| Ensemble ID | Components | Weights | CV | LB | Notes |
|-------------|-----------|---------|----|----|-------|
| [FILL] | [Run IDs] | [FILL] | [FILL] | [FILL] | [FILL] |
