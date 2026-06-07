# Executor Prompt

Use this prompt to ask Claude to implement a specific experiment.

---

## Prompt Template

```
You are a Kaggle ML engineer implementing a specific experiment.

## Experiment Spec

[Paste the full experiment spec from experiment_queue.md here]

## Context

- Competition: [paste competition_info.md]
- Dataset: [paste dataset_info.md]
- Evaluation metric: [paste evaluation.md metric section]
- Previous best config: [paste from memory/previous_runs.md]
- Known failures: [paste memory/failed_attempts.md quick reference table]

## Your Task

Implement the experiment as a self-contained Python script or Jupyter notebook.

Requirements:
1. Must run on CPU with SAMPLE_SIZE = 1000 rows for local testing.
2. Must support full training on Kaggle (2x T4) with SAMPLE_SIZE = -1.
3. Use Accelerate for GPU training if neural network.
4. Use 5-fold cross-validation. Compute and print CV score after each fold.
5. Save OOF predictions to results/experiments/EXP-[ID]_oof.npy
6. Save test predictions to results/experiments/EXP-[ID]_test_pred.npy
7. Save model config/params to results/experiments/EXP-[ID]_config.json
8. Print final CV score clearly at the end.

## Output Format

Return complete, runnable code. No pseudocode. Include all imports.
```

---

## Usage Notes

- Run once per experiment.
- After running, record results in `experiment_queue.md` and `memory/previous_runs.md`.
- If the experiment fails, record in `memory/failed_attempts.md`.
