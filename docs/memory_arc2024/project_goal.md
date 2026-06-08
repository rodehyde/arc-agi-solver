---
name: ARC-AGI project goal and roadmap
description: The overarching goal, three-stage roadmap, and current stage of the arc-agi-solver project
type: project
originSessionId: 7ef6f1f1-c886-4850-94e8-9dac9e4763f1
---
The project investigates whether a computer can mimic how humans solve ARC puzzles — by building up a bank of pattern experiences and combining them for novel tasks.

**Three-stage roadmap:**
1. **Categorise** (current stage) — Python algorithms build a growing taxonomy of task categories from features detectable in training pairs. A task can belong to multiple categories. The richer the taxonomy, the better the routing.
2. **Solve per category** — For each category, develop an ML model that predicts the correct output. Target: models that together cover ~50–60% of all tasks.
3. **Merge models** — Combine models for tasks spanning multiple categories or that resist categorisation, mimicking how humans combine prior experiences for novel puzzles.

**Solvers will be machine-learned** — no hand-coded rules. The category taxonomy is used to route tasks to specialised ML models, each trained on RE-ARC data for that category.

**Next immediate step:** Build the first end-to-end ML solver for the `FIXED_OUTPUT + SINGLE_COLOUR_OUTPUT` intersection (67 tasks). The output is a single colour value (0–9) — a 10-class classification problem. Use RE-ARC data for training. This proves the full pipeline: data → model → prediction → evaluation.

**Why:** ARC puzzles are trivial for humans (who improve by recognising and combining patterns) but very hard for computers. The project tests if a structured, experience-accumulating approach can close that gap.

**How to apply:** Keep suggestions focused on the current stage. Don't jump to Stage 2/3 work until Stage 1 coverage is satisfactory. Always define categories on training pairs only — never the test pair. Solvers are ML-based, not rule-based.

**Research discipline (decided 2026-05-22):**
- Training set (400 tasks) — train and iterate freely
- Evaluation set (400 tasks) — permanently held out as a benchmark; never train on it; run against it periodically to measure genuine progress
- ARC Prize 2025 — the target competition; submit when results are strong enough
- Rationale: keeping evaluation set clean gives honest, comparable measurements of real improvement over time
