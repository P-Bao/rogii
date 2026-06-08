# Papers

Relevant papers with key takeaways. No need to read full paper — just extract the useful part.

---

## Template

```
## [Paper Title]

- Authors: [FILL]
- Year: [FILL]
- Venue: NeurIPS / ICML / arXiv / [FILL]
- Link: [FILL]
- Key method: [1-2 sentences]
- Relevant to: [what aspect of current competition]
- Tried: Yes / No / Planned (EXP-[ID])
- Result if tried: [FILL]
```

---

## Foundational

### LightGBM: A Highly Efficient Gradient Boosting Decision Tree

- Authors: Ke et al.
- Year: 2017
- Venue: NeurIPS
- Link: https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- Key method: Gradient-based one-side sampling + exclusive feature bundling for fast tree boosting
- Relevant to: All tabular tasks
- Tried: Yes — baseline

### Attention Is All You Need

- Authors: Vaswani et al.
- Year: 2017
- Venue: NeurIPS
- Link: https://arxiv.org/abs/1706.03762
- Key method: Transformer self-attention — eliminates recurrence for sequence modeling
- Relevant to: NLP / sequence tasks

---

## Competition-Specific

| Paper | Method | Relevance | Planned Experiment |
|-------|--------|-----------|-------------------|
| Jing, Ye, Cao, Ran (2022) — "Actual wellbore tortuosity evaluation using a new quasi-three-dimensional approach", *Petroleum* 8:118-127 | Q-3D tortuosity index from survey/trajectory data (T_incline, Gamma_incline, TQG_Q3D, etc.) | High — mycarta toolkit's single best feature group (−0.107 RMSE in their ablation); high-tortuosity sections mark active steering correlated with formation deviation | EXP-002 (H-001) |
| Zeng, Bhaidasna, Zou (2026) — "Integrated Automation: Combining Automated Geosteering and Well-Placement", IADC/SPE-230729-MS | Automated geosteering / well-placement integration | Background context on how TVT prediction feeds real drilling decisions; cited by mycarta toolkit | Not planned — background reading only |
