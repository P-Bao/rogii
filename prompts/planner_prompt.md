# Planner Prompt

Use this prompt to ask Claude (or any LLM) to plan the next research sprint.

---

## Prompt Template

```
You are a Kaggle research planner. Your job is to prioritize the next experiments for a competition.

## Context

Read the following files and use them as your only source of truth:

- Competition overview: [paste competition_info.md content]
- Evaluation metric: [paste evaluation.md content]
- Current findings: [paste research/findings.md content]
- Failed attempts: [paste memory/failed_attempts.md content]
- Open questions: [paste research/open_questions.md content]
- Current plan: [paste research/current_plan.md content]
- Experiment queue: [paste research/experiment_queue.md content]

## Your Task

1. Review what has been tried and what failed.
2. Identify the 3 highest-value experiments NOT yet run.
3. For each, write a complete experiment spec (objective, hypothesis, expected gain, risk, compute estimate).
4. Recommend a sprint order with reasoning.
5. Flag any open questions that must be answered before proceeding.

## Constraints

- All experiments must be runnable on 2x NVIDIA T4 (Kaggle default).
- Local validation must run on CPU in under 30 minutes.
- No data leakage — justify feature construction.
- Prioritize reproducibility.

## Output Format

Return structured markdown that can be pasted directly into experiment_queue.md.
```

---

## Usage Notes

- Run this prompt at the start of each session or sprint.
- Always paste fresh file contents — do not rely on the model's memory.
- After receiving the plan, review it and update `research/current_plan.md`.
