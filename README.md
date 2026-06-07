# Research Repository Template

Reusable template for Kaggle competitions, ML research, and LLM training projects.

Designed to work with Claude Code and OpenRouter-based LLM agents.

---

## Repository Structure

```
.
├── competition/          # Competition metadata
│   ├── competition_info.md
│   ├── dataset_info.md
│   └── evaluation.md
├── data/
│   ├── raw/              # Original competition files (DO NOT MODIFY)
│   ├── external/         # External datasets
│   └── processed/        # Feature-engineered outputs
├── notebooks/
│   ├── 01_eda.ipynb      # Exploratory data analysis (CPU, < 5 min)
│   ├── 02_training.ipynb # Training with 5-fold CV
│   ├── 03_inference.ipynb# Blend predictions
│   └── 04_submission.ipynb # Generate submission.csv
├── src/
│   ├── features/         # Feature engineering functions
│   ├── models/           # Model wrappers (LGB, Accelerate)
│   └── utils/            # Metrics, reproducibility
├── research/
│   ├── current_plan.md   # Active sprint plan
│   ├── experiment_queue.md
│   ├── findings.md       # All discoveries (append only)
│   ├── hypothesis.md     # All hypotheses
│   └── open_questions.md
├── memory/               # Persistent knowledge base
│   ├── previous_runs.md  # All run results
│   ├── failed_attempts.md# What NOT to repeat
│   └── leaderboard_progress.md
├── references/           # External resources
│   ├── useful_links.md
│   ├── kaggle_discussions.md
│   ├── papers.md
│   └── repos.md
├── prompts/              # LLM agent prompts
│   ├── planner_prompt.md
│   ├── executor_prompt.md
│   └── reviewer_prompt.md
├── results/
│   ├── experiments/      # OOF/test .npy files, configs
│   ├── submissions/      # submission.csv files
│   └── plots/            # EDA and analysis plots
├── skills/               # Claude Code custom skills
├── mcp/                  # MCP server configs
├── docs/                 # Documentation and reports
├── CLAUDE.md             # Instructions for Claude Code
└── PLAN.md               # Repository creation plan
```

---

## Quickstart

### 1. Fill competition metadata

Edit these first:
- `competition/competition_info.md`
- `competition/dataset_info.md`
- `competition/evaluation.md`

### 2. Place data

```bash
cp /path/to/train.csv data/raw/
cp /path/to/test.csv  data/raw/
cp /path/to/sample_submission.csv data/raw/
```

### 3. Run EDA

```bash
cd notebooks
jupyter nbconvert --to notebook --execute 01_eda.ipynb --output 01_eda_out.ipynb
```

Local CPU: completes in < 5 min with SAMPLE_SIZE = 1000.

### 4. Run baseline training

```bash
jupyter nbconvert --to notebook --execute 02_training.ipynb --output 02_training_out.ipynb
```

### 5. Blend and submit

```bash
jupyter nbconvert --to notebook --execute 03_inference.ipynb
jupyter nbconvert --to notebook --execute 04_submission.ipynb
```

---

## Research Workflow

```
session start
    └─► read: memory/failed_attempts.md (avoid repeating mistakes)
    └─► read: research/current_plan.md  (pick up where you left off)
    └─► pick next EXP from research/experiment_queue.md

run experiment (notebooks/02_training.ipynb)
    └─► record result in research/experiment_queue.md
    └─► append finding to research/findings.md
    └─► update memory/previous_runs.md

if public LB submission:
    └─► update memory/leaderboard_progress.md

session end
    └─► update research/current_plan.md
    └─► update research/open_questions.md if new questions arose
```

---

## Planner / Executor Workflow

Use these prompts with Claude Code or any LLM:

1. **Plan**: paste `prompts/planner_prompt.md` + current context files
2. **Execute**: paste `prompts/executor_prompt.md` + experiment spec
3. **Review**: paste `prompts/reviewer_prompt.md` + experiment results

Always paste file contents directly — do not rely on the model's memory across sessions.

---

## Kaggle Workflow

### Local (CPU) validation

- `SAMPLE_SIZE = 500–1000`
- Target: < 5 min
- Notebooks: `01_eda.ipynb`, `02_training.ipynb`

### Kaggle GPU training (2x T4)

- `SAMPLE_SIZE = -1`
- Use `accelerate launch` for neural networks
- Use LightGBM/XGBoost for tabular (no Accelerate needed)
- Template: `src/models/accelerate_trainer.py`

---

## Compute Rules

| Environment | Hardware | SAMPLE_SIZE | Max Runtime |
|-------------|----------|-------------|-------------|
| Local dev   | CPU only | 500–1000    | 30 min      |
| Kaggle      | 2x T4    | full (-1)   | 9 hours     |

**Never assume GPU locally.**

---

## LLM Training Workflow

1. Prepare data in `data/processed/`
2. Configure `src/models/accelerate_trainer.py`
3. Run locally with `SAMPLE_SIZE = 500` to verify code
4. Push to Kaggle notebook for full training
5. Record results in `memory/previous_runs.md`

---

## Key Principles

1. **Reproducibility first** — set seeds, save configs, version outputs
2. **Memory over chat** — write everything to files; never rely on chat history
3. **Fail fast locally** — validate on tiny sample before Kaggle run
4. **Record failures** — `memory/failed_attempts.md` prevents repeated mistakes
5. **CV-LB correlation** — track gap; if gap > threshold, investigate leakage
