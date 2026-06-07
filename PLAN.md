# Repository Creation Plan

## Goal

Create a reusable repository template for:

- Kaggle competitions
- Notebook optimization
- LLM training research
- Long-term experimentation

---

# Phase 1

Create directory structure.

Required:

competition/
data/
notebooks/
src/
research/
memory/
references/
prompts/
results/
skills/
mcp/
docs/

---

# Phase 2

Create documentation templates.

Required:

competition_info.md
dataset_info.md
evaluation.md

previous_runs.md
failed_attempts.md

useful_links.md
papers.md
repos.md

current_plan.md
experiment_queue.md
open_questions.md

---

# Phase 3

Create notebook templates.

Required:

EDA notebook

training notebook

inference notebook

submission notebook

Requirements:

- CPU runnable
- tiny sample support
- modular design

---

# Phase 4

Create prompt templates.

planner_prompt.md

executor_prompt.md

reviewer_prompt.md

These prompts must support:

- research planning
- experiment execution
- result review

---

# Phase 5

Create experiment tracking system.

Include:

experiment history

leaderboard history

failed experiments

research findings

open questions

---

# Phase 6

Create starter README.

README must explain:

repository structure

research workflow

planner/executor workflow

Kaggle workflow

LLM workflow

---

# Phase 7

Review

Verify:

- all folders exist
- all files exist
- templates contain examples
- notebooks run on CPU
- no file is empty

Generate a final report summarizing the repository structure.

---

# Phase 8

Integrate skills from `.claude/skills/ai-research-skills/`.

## 8a — Skill Inventory

Document all 54 available skills in `docs/skills_index.md`.

For each skill record:

- skill name
- invoke command
- use case
- dependencies
- Kaggle compatibility (T4 yes/no)

## 8b — Skill-to-Workflow Mapping

Map skills to research workflow stages:

| Stage | Recommended Skills |
|-------|-------------------|
| Research planning | `autoresearch`, `brainstorming-research-ideas`, `creative-thinking-for-research` |
| Data preparation | `nemo-curator`, `ray-data`, `huggingface-tokenizers`, `sentencepiece` |
| Fine-tuning (Kaggle T4) | `trl-fine-tuning`, `axolotl`, `peft`, `unsloth`, `llama-factory` |
| RL training | `grpo-rl-training`, `openrlhf`, `verl`, `simpo` |
| Distributed training | `torchtitan`, `slime`, `miles`, `torchforge` |
| Quantization | `awq`, `gptq`, `bitsandbytes`, `gguf`, `hqq`, `flash-attention` |
| Evaluation | `lm-evaluation-harness`, `nemo-evaluator`, `bigcode-evaluation-harness` |
| LLM apps / RAG | `langchain`, `llamaindex`, `dspy`, `instructor`, `outlines` |
| Agentic research | `autoresearch`, `crewai`, `autogpt`, `a-evolve` |
| Safety | `llamaguard`, `prompt-guard`, `nemo-guardrails`, `constitutional-ai` |
| Paper writing | `ml-paper-writing`, `academic-plotting`, `presenting-conference-talks` |
| Session epilogue | `research-manager`, `rigor-reviewer` |

## 8c — Kaggle T4 Compatibility Matrix

Identify which skills run on Kaggle 2×T4 without modification:

Compatible (tested patterns):
- `trl-fine-tuning` with LoRA/QLoRA
- `peft` (LoRA fine-tuning)
- `unsloth` (memory-efficient)
- `axolotl` (YAML config, T4 supported)
- `lm-evaluation-harness` (inference only)
- `bitsandbytes` (4-bit inference)
- `flash-attention` (T4 supported)

Requires adaptation (large memory):
- `torchtitan` (prefer A100/H100)
- `openrlhf` (requires Ray cluster)
- `verl` (scales to large clusters)
- `slime` / `miles` (Megatron-based)

CPU-only compatible:
- `gguf` (llama.cpp)
- `lm-evaluation-harness` (small models)
- `academic-plotting`
- `ml-paper-writing`
- `research-manager`

## 8d — Skill Prompt Integration

Update `prompts/planner_prompt.md` to reference skill routing.

Update `prompts/executor_prompt.md` to specify which skill to invoke per task type.

Update `prompts/reviewer_prompt.md` to invoke `rigor-reviewer` for epistemic audit.

## 8e — Session Start Protocol

Add to `research/current_plan.md` a standard session start checklist:

1. Read `memory/failed_attempts.md`
2. Read `research/current_plan.md`
3. Invoke `/autoresearch` for new project or `/research-manager` to resume
4. Select EXP from `research/experiment_queue.md`
5. Choose skill from CLAUDE.md skill table matching the EXP type

## 8f — Verify

- All 54 skills listed in `docs/skills_index.md`
- Skill routing table in CLAUDE.md is accurate
- Prompts reference skill invocations
- No duplicate skill mappings