# Useful Links

## Competition

| Description | URL |
|-------------|-----|
| Competition page | [FILL] |
| Data download | [FILL] |
| Submission page | [FILL] |

## Our Notebooks

| Description | URL |
|-------------|-----|
| Submit | https://www.kaggle.com/code/baopv05/rogii-submission?scriptVersionId=323128087 |
| EDA | https://www.kaggle.com/code/baopv05/rogii-eda-features |
| Train model (CatBoost) | https://www.kaggle.com/code/baopv051/rogii-train-catboost |
| Train model (LightGBM) | https://www.kaggle.com/code/baopv051/rogii-train-lightgbm |
| Select (Postprocess research) | https://www.kaggle.com/code/baopv051/rogii-postprocess-research |

## Top Notebooks (Public)

| Score | Author | Description | URL |
|-------|--------|-------------|-----|
| 7.748 | sarpal465 | rogii-ridge-sp45-proj | https://www.kaggle.com/code/sarpal465/rogii-ridge-sp45-proj |
| 7.881 | pilkwang | rogii-ridge-artifact-parameter-experiments | https://www.kaggle.com/code/pilkwang/rogii-ridge-artifact-parameter-experiments |
| 7.807 | yaroslavkholmirzayev | rogii-sel15-forced-selector | https://www.kaggle.com/code/yaroslavkholmirzayev/rogii-sel15-forced-selector |

## Secondary Public Notebooks (cited BY the top-score notebooks themselves)

Found inside `rogii-sel15-forced-selector.ipynb`'s "References" markdown cell (read 2026-06-08)
— these are the lineage chain the 7.7-7.9 notebooks drew their PF/selector/Ridge-stack ideas
from. Worth reading if reproducing EXP-007/008 hits a wall:

| Description | URL |
|-------------|-----|
| PF selector / physical model (likely shared lineage source — `BEAMS`/`_pf_ancc` near-identical to ours) | https://www.kaggle.com/code/aiwody/physical-model-less-overfitting-noise |
| PF selector rerun | https://www.kaggle.com/code/aidensong123/rogii-sel15-rerun |
| Ridge artifact reference | https://www.kaggle.com/code/overvalueawareness/wellbore-geology-prediction-ridge/notebook |
| Ridge artifact reference | https://www.kaggle.com/code/ravaghi/wellbore-geology-prediction-ridge |
| Ridge SP45 variant | https://www.kaggle.com/code/needless090/rogii-ridge-sp45 |
| Better solution LB 9.956 | https://www.kaggle.com/code/romantamrazov/rogii-better-solution-lb-9-956 |
| Super solution LB top 3 | https://www.kaggle.com/code/romantamrazov/rogii-super-solution-lb-top-3 |
| Physics-informed baseline | https://www.kaggle.com/code/karnakbaevarthur/physics-informed-baseline?scriptVersionId=317950936 |
| Triple-signal beam search / dual PF / LightGBM | https://www.kaggle.com/code/shinyanagai123/triple-signal-beam-search-dual-pf-lightgbm |
| Plane-fit formation-top KNN | https://www.kaggle.com/code/konbu17/rogii-plane-fit-formation-top-knn |
| Wellbore geology prediction baseline | https://www.kaggle.com/code/vishwasmishra1234/rogii-wellbore-geology-prediction |
| XGB starter CV 15 | https://www.kaggle.com/code/cdeotte/xgb-starter-cv-15 |

## Discussion Threads (Key Insights)

| Topic | URL | Key takeaway |
|-------|-----|-------------|
| [FILL] | [FILL] | [FILL] |

## Tools & Libraries

| Tool | Purpose | URL |
|------|---------|-----|
| LightGBM | Gradient boosting | https://lightgbm.readthedocs.io |
| XGBoost | Gradient boosting | https://xgboost.readthedocs.io |
| HuggingFace Accelerate | Multi-GPU training | https://huggingface.co/docs/accelerate |
| Optuna | Hyperparameter tuning | https://optuna.readthedocs.io |
| Polars | Fast DataFrames | https://docs.pola.rs |
| timm | PyTorch image models | https://timm.fast.ai |

## Internal References

| File | Description |
|------|-------------|
| [competition/competition_info.md](../competition/competition_info.md) | Competition overview |
| [research/findings.md](../research/findings.md) | All findings |
| [memory/failed_attempts.md](../memory/failed_attempts.md) | What NOT to repeat |
