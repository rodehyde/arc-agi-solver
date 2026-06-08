---
name: nanoGPT fine-tuning — ABANDONED
description: GPT-2 fine-tune approach tried and abandoned; model too small to learn ARC rules
type: project
originSessionId: e487d252-10cb-4121-a4a5-aef8e3f8d060
---
## Status: ABANDONED (2026-05-12)

GPT-2 small (124M params) cannot learn ARC rule induction from 880 examples.
Best run (500 steps, LARC+Claude descriptions): model output garbled text on 4/5 eval tasks,
produced training data format instead of rules, gave vague/wrong descriptions.
Root cause: fundamental capacity issue, not data or training length.

## What was built (kept in repo for reference)
- `notebooks/finetune_colab.ipynb` — nanoGPT training + Cell 8 evaluation
- `notebooks/generate_descriptions_colab.ipynb` — calls Claude API for descriptions
- `scripts/prepare_arc_finetune.py` — builds LARC+Claude dataset (880 examples)
- `scripts/generate_claude_descriptions.py` — Claude API descriptor (400 tasks done)
- `data/claude_descriptions.json` — 400 Claude-generated TYPE/RULE/STEPS/RELATIONSHIP descriptions
- `nanogpt/config/finetune_arc.py` — training config

## Decision
Pivoted back to custom transformer (in-context learning). The Claude descriptions in
data/claude_descriptions.json may be useful later as additional context for a Stage 2
grid generator, but Stage 1 LLM approach is shelved indefinitely.
