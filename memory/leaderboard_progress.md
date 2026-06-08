# Leaderboard Progress

Track position and score over time.

---

## Score History

| Date | Public LB Score | Rank | Total Teams | Percentile | Notes |
|------|----------------|------|-------------|-----------|-------|
| [pre-existing, exact date unknown] | **9.964** | [FILL] | [FILL] | [FILL]% | `BASELINE_V1` — our `aw` ("Another-Work") pipeline: LightGBM×3 + CatBoost×3 weighted blend (`{catboost-2: 0.2718483, catboost-3: 0.3763824, lightgbm-1: 0.3517693}`) + PF-residual postprocess (`apply_pp`: `alpha=1.04, tau=65, w_pf=0.07`) + Savitzky-Golay smoothing (`sg_w=27, sg_p=3`). OOF RMSE = 10.4521227. Source: `notebooks/reference/mine/rogii-postprocess-research.ipynb` `WINNING_WEIGHTS`/`WINNING_PP_PARAMS`/`BASELINE_V1_*` constants (read 2026-06-08) |
| [FILL] | [FILL] | [FILL] | [FILL] | [FILL]% | (next submission — pending EXP-007/009 reproduction of top-score architecture) |

## Reference Point — Top Public Notebooks (NOT our scores; for gap-tracking)

Read in full from `notebooks/reference/top_score/` on 2026-06-08:

| Notebook | Score | Architecture summary |
|---|---|---|
| `rogii-ridge-sp45-proj.ipynb` | **7.748** | `0.30*ridge_meta_stack + 0.70*PF/beam_heuristic`, then robust deg-5 U=TVT+Z projection postprocess |
| `rogii-sel15-forced-selector.ipynb` | 7.807 (champion before this run was 7.910; reference target 7.839) | Same architecture family, "reference-exact" reproduction attempt with `koolbox`/`Trainer` artifact-first loading |
| `rogii-ridge-artifact-parameter-experiments.ipynb` | 7.881 | Parameter sweep around the ridge-artifact blend (`w_r`, PF particle/seed counts, init spread, optional projection degree) |

**Gap: our 9.964 vs. their 7.748-7.910 ⇒ ~2.0-2.2 RMSE.** See `research/findings.md`
2026-06-08 entry and `research/current_plan.md` "Reality Check" for the full architectural
delta analysis (5 components: Ridge-stack, blend-with-heuristic, per-well selector, U-space
projection postprocess, Optuna-tuned LGB hyperparameters).

---

## Final Submissions

Select 2 submissions for final evaluation. Record both.

| Submission | CV | Public LB | Strategy |
|-----------|----|-----------|----|
| Sub A | [FILL] | [FILL] | Best Public LB |
| Sub B | [FILL] | [FILL] | Best CV / Safer |

## Private LB Result

| Submission | Public LB | Private LB | Shake-up |
|-----------|-----------|------------|---------|
| Sub A | [FILL] | [FILL] | +/- [FILL] places |
| Sub B | [FILL] | [FILL] | [FILL] |

---

## Shake-up Analysis

- Expected shake-up: Low / Medium / High
- Actual shake-up: [FILL places]
- Cause: [FILL — e.g., overfit to public split, distribution shift]
- Lesson: [FILL]

---

## Competition Milestones

| Date | Event | Score | Notes |
|------|-------|-------|-------|
| [FILL] | First submission | [FILL] | |
| [FILL] | Beat dumb baseline | [FILL] | |
| [FILL] | Top 50% | [FILL] | |
| [FILL] | Top 25% | [FILL] | |
| [FILL] | Medal zone | [FILL] | |
