# Reference Repositories

Public repos with useful code to adapt (not copy verbatim — understand then reimplement).

---

## Template

```
## [Repo Name]

- URL: [FILL]
- Author: [FILL]
- Stars: [FILL]
- What it has: [FILL]
- Useful files: [FILL paths]
- Adapted for: [FILL experiment or feature]
```

---

## General Purpose

### rapids-ai/cuml
- URL: https://github.com/rapidsai/cuml
- What it has: GPU-accelerated sklearn-compatible ML
- Useful for: Dropping in as sklearn replacement on Kaggle GPU envs

### microsoft/LightGBM
- URL: https://github.com/microsoft/LightGBM
- Useful files: `examples/` — complete training scripts
- Adapted for: Baseline training loop

### huggingface/accelerate
- URL: https://github.com/huggingface/accelerate
- Useful files: `examples/complete_nlp_example.py`
- Adapted for: Multi-GPU training template

---

## Competition-Specific

### mycarta/rogii-geosteering-toolkit
- URL: https://github.com/mycarta/rogii-geosteering-toolkit
- What it has: Full reference solution — single LightGBM + StratifiedGroupKFold (strata: signed azimuth, median TVT, spatial XY bins), feature groups = multi-scale NCC vs typewell, self-correlation, Q-3D wellbore tortuosity (`wellbore_tortuosity.py`), trajectory baseline, offset-well prior, landing-zone state. Documented per-group ablation table — tortuosity alone gave the largest single gain (−0.107 RMSE).
- Useful files: `kaggle/rogii_features.py` (feature extraction), `kaggle/rogii_cv.py` (CV scheme), `toolkit/wellbore_tortuosity.py`, `toolkit/despike.py`, `toolkit/zone_monitor.py`, `methodology/within_well_tvt_z_decoupling.md` (critical: explains why naive linear TVT~Z structural baselines are biased), `methodology/AEON_evaluation.md` (proves MassSNN ≈ scipy NCC mathematically; flags MiniROCKET/Catch22 as the genuinely-novel AEON candidates)
- Adapted for: EXP-001 through EXP-003, EXP-006 (see [research/experiment_queue.md](../research/experiment_queue.md))

### geosteering-no/inversion_school_geosteering
- URL: https://github.com/geosteering-no/inversion_school_geosteering
- What it has: Lecture notes + exercises on real-time geological inversion — sequential curve fitting (`curvefit`) and Tikhonov-regularized inversion (`minimize`) that map expected stratigraphy to offset-log data. A physics-based alternative/complement to ML regression.
- Useful files: `playground_curve_fit.py`, `playground_curve_fit_with_minimize.py`, `competition_plotter.py`
- Adapted for: Curiosity item in [research/open_questions.md](../research/open_questions.md) — not yet planned as an experiment
