# Repository Generator

You are creating a reusable GitHub template repository for:

- Kaggle competitions
- Machine learning research
- LLM training projects
- Notebook improvement workflows

The repository will later be used by Claude Code and OpenRouter-based models.

---

# Primary Goal

Generate a complete research repository template.

The repository must support:

1. Competition research
2. Experiment tracking
3. Notebook iteration
4. Kaggle workflows
5. LLM training workflows
6. Long-term project memory

---

# Required Directory Structure

Create and maintain:

.
├── competition/
├── data/
│   ├── raw/
│   ├── external/
│   └── processed/
├── notebooks/
├── src/
├── research/
├── memory/
├── references/
├── prompts/
├── results/
├── skills/
├── mcp/
└── docs/

---

# Required Files

competition/

- competition_info.md
- dataset_info.md
- evaluation.md

research/

- current_plan.md
- experiment_queue.md
- findings.md
- hypothesis.md
- open_questions.md

memory/

- previous_runs.md
- failed_attempts.md
- leaderboard_progress.md

references/

- useful_links.md
- kaggle_discussions.md
- papers.md
- repos.md

prompts/

- planner_prompt.md
- executor_prompt.md
- reviewer_prompt.md

root/

- README.md
- PLAN.md
- CLAUDE.md

---

# Compute Rules

Assume local development machine:

- CPU only
- limited resources

All generated notebooks must:

- run on CPU
- use a tiny subset
- complete quickly

Maximum local validation target:

30 minutes CPU

Preferred target:

5 minutes CPU

Never assume access to powerful local GPUs.

---

# Kaggle Rules

Default Kaggle environment:

2 × NVIDIA T4

When generating training code:

- support multi-GPU execution
- prefer Accelerate
- prefer DDP
- avoid single-GPU assumptions

Generate code that is ready for Kaggle execution but do not execute full training.

Only create:

- notebook templates
- training templates
- validation templates

---

# Experiment Rules

Every experiment must contain:

- objective
- hypothesis
- expected gain
- risk
- compute estimate
- status

---

# Memory Rules

The repository must support continuing research after weeks or months.

Every finding must be stored.

Never rely on chat history.

All important information must be written into files.

---

# Research Rules

Always prioritize:

1. Reproducibility
2. Experiment tracking
3. Knowledge accumulation
4. Compute efficiency

---

# Deliverable

Create a complete reusable repository template.

Generate all required files with meaningful starter content.

Do not leave placeholder empty files.

Every file should contain instructions and examples.

---

# Context Isolation

This repository is self-contained.

Always rely only on:

- files in this repository
- explicitly provided URLs

Do not assume knowledge from previous projects,
competitions, repositories, or sessions.

If required information is missing,
request it or record it in repository files.

---

# Available Skills

Skills are in `.claude/skills/ai-research-skills/.claude/skills/`.
Invoke via `/skill-name` or reference by task below.

## Research Orchestration

| Skill | Invoke | Use When |
|-------|--------|----------|
| `0-autoresearch-skill` | `/autoresearch` | Starting autonomous research project; running inner/outer loop experiment cycles; managing multi-hypothesis effort |
| `research-manager` | `/research-manager` | End of session — extract decisions, experiments, pivots from conversation into `ara/` provenance files |
| `brainstorming-research-ideas` | `/brainstorming-research-ideas` | Exploring new directions; pivoting between projects; seeking novel angles |
| `creative-thinking-for-research` | `/creative-thinking-for-research` | Structured ideation with cognitive science frameworks when stuck on a problem |
| `rigor-reviewer` | `/rigor-reviewer` | Epistemic review of research artifacts before publishing or sharing |
| `compiler` | `/compiler` | Compiling papers, GitHub repos, experiment logs, or code directories into a structured research summary |

## LLM Fine-tuning

| Skill | Invoke | Use When |
|-------|--------|----------|
| `trl-fine-tuning` | `/fine-tuning-with-trl` | SFT instruction tuning, DPO preference alignment, PPO/GRPO reward optimization with HuggingFace TRL |
| `axolotl` | `/axolotl` | Fine-tuning with YAML configs; LoRA/QLoRA/DPO/KTO/ORPO/GRPO; 100+ model support |
| `peft` | `/peft-fine-tuning` | LoRA/QLoRA on 7B–70B models with limited GPU memory; multi-adapter serving |
| `grpo-rl-training` | `/grpo-rl-training` | GRPO reward-based fine-tuning; enforcing output formats; math/code verifiable tasks |
| `unsloth` | `/unsloth` | Fast LoRA/QLoRA training (2–5× faster, 50–80% less memory) on Llama/Mistral/Gemma/Qwen |
| `llama-factory` | `/llama-factory` | WebUI no-code fine-tuning; 100+ models; 2/3/4/5/6/8-bit QLoRA |
| `litgpt` | `/litgpt` | Training with Lightning AI LitGPT; 20+ pretrained architectures |
| `simpo` | `/simpo-training` | Reference-free preference alignment; faster/simpler than DPO with +6.4 pts AlpacaEval |
| `ml-training-recipes` | `/ml-training-recipes` | PyTorch training loops, optimizer selection, LR scheduling, debugging loss spikes/OOM |

## Distributed / Large-scale Training

| Skill | Invoke | Use When |
|-------|--------|----------|
| `openrlhf` | `/openrlhf` | High-performance RLHF (PPO, GRPO, RLOO, DPO) with Ray+vLLM at scale |
| `verl` | `/verl-rl-training` | Volcano Engine RL for flexible RLHF/GRPO/PPO infrastructure backends |
| `torchtitan` | `/torchtitan` | PyTorch-native distributed LLM pretraining with 4D parallelism (FSDP2, TP, PP, CP) |
| `torchforge` | `/torchforge` | Meta's PyTorch-native agentic RL; separates infra from algorithm |
| `slime` | `/slime` | LLM post-training RL using Megatron+SGLang framework |
| `miles` | `/miles` | Enterprise-grade RL training; production fork of slime |

## Quantization & Memory Efficiency

| Skill | Invoke | Use When |
|-------|--------|----------|
| `awq` | `/awq` | 4-bit activation-aware quantization; 3× speedup, minimal accuracy loss |
| `bitsandbytes` | `/bitsandbytes` | 8-bit or 4-bit quantization; 50–75% memory reduction; drop-in for transformers |
| `flash-attention` | `/flash-attention` | 2–4× attention speedup, 10–20× memory reduction; standard for long sequences |
| `gguf` | `/gguf` | GGUF format and llama.cpp quantization for CPU/consumer GPU inference |
| `gptq` | `/gptq` | Post-training 4-bit quantization for 70B+ deployment |
| `hqq` | `/hqq` | Half-Quadratic Quantization at 4/3/2-bit without calibration data |

## Evaluation & Benchmarking

| Skill | Invoke | Use When |
|-------|--------|----------|
| `lm-evaluation-harness` | `/evaluating-llms-harness` | 60+ academic benchmarks (MMLU, HumanEval, GSM8K, TruthfulQA); comparing models |
| `nemo-evaluator` | `/nemo-evaluator` | 100+ benchmarks from 18+ harnesses; safety and VLM evaluation |
| `bigcode-evaluation-harness` | `/bigcode-evaluation-harness` | Code model benchmarks (HumanEval, MBPP, DS-1000) |

## Data Processing

| Skill | Invoke | Use When |
|-------|--------|----------|
| `nemo-curator` | `/nemo-curator` | GPU-accelerated data curation; deduplication; text/image/video/audio pipelines |
| `ray-data` | `/ray-data` | Scalable CPU/GPU data processing; batch inference; distributed ETL |
| `huggingface-tokenizers` | `/huggingface-tokenizers` | Fast Rust-based tokenizers; 1GB/s tokenization; BPE/WordPiece/Unigram |
| `sentencepiece` | `/sentencepiece` | Language-independent BPE/Unigram tokenization; multilingual models |

## Architecture & Models

| Skill | Invoke | Use When |
|-------|--------|----------|
| `mamba` | `/mamba` | State-space model O(n) vs Transformer O(n²); 5× faster inference; million-token sequences |
| `rwkv` | `/rwkv` | RWKV architecture; linear attention alternative |
| `nanogpt` | `/nanogpt` | Minimal GPT (~300 lines); reproduce GPT-2; clean hackable baseline |

## LLM Applications & Agents

| Skill | Invoke | Use When |
|-------|--------|----------|
| `langchain` | `/langchain` | Building LLM apps with agents, chains, RAG; multi-provider support |
| `llamaindex` | `/llamaindex` | RAG with 300+ data connectors; document Q&A; knowledge graphs |
| `dspy` | `/dspy` | Declarative AI programming; automatic prompt optimization; modular RAG |
| `instructor` | `/instructor` | Extracting structured data from LLM responses with Pydantic validation |
| `outlines` | `/outlines` | Guaranteed JSON/XML/code structure during generation |
| `autogpt` | `/autogpt` | Autonomous AI agent platform; continuous agent workflows |
| `crewai` | `/crewai` | Multi-agent teams; specialized role delegation; collaborative tasks |
| `a-evolve` | `/a-evolve` | LLM-driven evolution/optimization of AI agents across any domain |

## Safety & Alignment

| Skill | Invoke | Use When |
|-------|--------|----------|
| `constitutional-ai` | `/constitutional-ai` | Anthropic's two-phase self-improvement for training harmless AI |
| `llamaguard` | `/llamaguard` | Meta's 7–8B content moderation for LLM input/output safety |
| `prompt-guard` | `/prompt-guard` | Meta's 86M prompt injection and jailbreak detector |
| `nemo-guardrails` | `/nemo-guardrails` | NVIDIA's runtime safety: jailbreak detection, input/output validation |

## Academic Output

| Skill | Invoke | Use When |
|-------|--------|----------|
| `academic-plotting` | `/academic-plotting` | Publication-quality figures for ML papers; architecture diagrams; data plots |
| `ml-paper-writing` | `/ml-paper-writing` | Writing ML/AI papers for NeurIPS, ICML, ICLR, ACL, AAAI, COLM |
| `systems-paper-writing` | `/systems-paper-writing` | Systems papers for OSDI, SOSP, ASPLOS, NSDI, EuroSys |
| `presenting-conference-talks` | `/presenting-conference-talks` | Conference slides from a paper (Beamer LaTeX PDF + PPTX) |

---

# Skill Usage Rules

1. **Invoke skills by task, not by name.** Match the task to the table above.
2. **For LLM training on Kaggle 2×T4**: prefer `trl-fine-tuning` or `axolotl` with `peft` (LoRA). Add `unsloth` for speed.
3. **For evaluation after training**: use `lm-evaluation-harness` for academic benchmarks; `nemo-evaluator` for safety + VLM.
4. **For autonomous research sessions**: invoke `autoresearch` first; use `research-manager` at session end.
5. **For paper writing**: run `academic-plotting` before `ml-paper-writing`.
6. **Record skill invocations** in `research/experiment_queue.md` and findings in `research/findings.md`.