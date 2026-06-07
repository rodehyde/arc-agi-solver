# ARC-AGI Solver

An attempt to build a rule-based solver for the [ARC-AGI benchmark](https://arcprize.org/) — the Abstraction and Reasoning Corpus for Artificial General Intelligence.

---

## What is ARC-AGI?

ARC puzzles present a small set of input/output grid pairs and ask you to infer the rule that connects them, then apply that rule to a new input. The grids use up to 10 colours on a black background, and the rules involve spatial, geometric, and logical reasoning.

The puzzles are trivially easy for humans and extremely hard for computers. A score of ~85% is considered state-of-the-art. Most ML approaches score well below that without task-specific fine-tuning. The benchmark is specifically designed to resist pattern-matching without understanding: every task requires deriving a new rule from a handful of examples.

---

## Our approach

The central question: **can a computer mimic the human process of recognising patterns and deriving rules?**

We are not training a model end-to-end. Instead, we are building a growing library of task-level rule-based solvers — one per task family — guided by a structured analytical process applied task by task.

### The analytical process

For each new task, we apply a fixed protocol before writing any code:

**Pre-step — scan known patterns.** A growing list of named structural families (separator grids, two-halves-with-separator, template-and-stamp, etc.) is checked first. If the task matches a known family, go straight to implementation.

**7 decomposition lenses.** If no pattern match:
1. Which cells are fixed vs. changing between input and output?
2. How many distinct connected components are there?
3. What is the *nature* of the change — translation, rotation, recolouring, deletion, completion, scaling?
4. Is the change self-determined by the shape, or driven by an indicator (a marker cell or colour)?
5. Where do the parts end up spatially — touching, aligned, symmetric?
6. Is the output *more* structured than the input? (suggests repair/complete)
7. Is the output *smaller* than the input? (suggests extract/filter)

**4-step protocol.** Applied in order:
1. What is this input *almost*? (near-regular, near-symmetric, near-tiled)
2. What doesn't belong? (the anomaly identifies *where* the rule acts)
3. Hold input and output together — what rule connects them?
4. What is the shortest *complete* rule that fits all training pairs?

**Mandatory prediction test.** Before writing code, the rule derived from pair 0 must correctly predict pair 2's output. A rule that can't predict unseen pairs is an observation, not a rule.

**Code verification.** A rule described in words is a hypothesis. It only becomes verified when a Python implementation produces zero mismatches across all training pairs.

### What we learned about ARC tasks

Tasks do not cluster neatly into large families. Most solvers end up covering one task or a very small family. The real leverage is not finding group-level shortcuts but developing the analytical process so that any new task can be solved quickly from first principles.

The primary output is therefore not a classification system but a growing *library of solved examples* — each one exercising the process and adding to a bank of recognised patterns.

---

## Results

### Training set (400 tasks)

Approximately **141 tasks** solved by verified rule-based solvers as of June 2025. Each solver was verified against all training pairs before registration.

Sample of solver families found:
- Geometric transforms (rotation, reflection, scaling): ~15 tasks
- Logical operations (AND/OR/XOR of two halves): ~10 tasks
- Colour remapping and substitution: ~10 tasks
- Separator-grid operations (stamp, fill, connect): ~8 tasks
- Gravity, flood-fill, frame operations: many 1–3 task solvers
- Dozens of highly specific single-task rules

### Evaluation set (400 tasks, held out)

We inspected 10 randomly selected evaluation tasks and attempted to solve each from scratch, with no reference to the training set. Results:

| Task | Solver | Result |
|------|--------|--------|
| `516b51b7` | CONCENTRIC_RINGS | ✓ |
| `2a5f8217` | MATCH_RECOLOR_ONES | ✓ |
| `f0afb749` | DOUBLE_DIAGONAL | ✓ |
| `1e97544e` | STAIRCASE_DIAGONAL | ✓ |
| `5af49b42` | LEGEND_ALIGN | ✓ |
| `f21745ec` | FRAME_STAMP | ✓ |
| `4c177718` | T-arrow direction → partner placement | ✓ |
| `d56f2372` | EXTRACT_LR_SYMMETRIC | ✓ |
| `0a2355a6` | RECOLOUR_BY_HOLE_COUNT | ✓ |
| `212895b5` | 8-block with diagonal rays + staircase arms | ✗ |

**9/10 solved** on a blind sample of evaluation tasks.

---

## Process insights

Several things we learned about *how* to apply the process well:

**Exhaust hypothesis variants before discarding.** If "symmetry" is a candidate but one variant fails, test all sub-variants (LR, UD, 180°, 90°, diagonal) before concluding that symmetry is not the rule. One failed variant is evidence against that variant, not against the whole family. This lesson came from `d56f2372`: an earlier attempt tried "symmetry" and got a false negative; a later attempt tested LR-symmetry specifically and solved it immediately.

**Role-based rules generalise better than literal-value rules.** A rule that says "the anchor block" generalises to unseen tasks. A rule that says "colour 3 in the bottom-right corner" does not. Always ask: does this need to name a specific value, or can it refer to a role?

**Predict before coding.** Attempting to predict pair 2 from a rule derived from pair 0 catches ~half of wrong hypotheses before any implementation effort is spent.

**When stuck, stop.** The time limit is approximately one minute per task. Past that, a human pattern-recogniser outperforms continued solo analysis. The right call is to surface the difficulty rather than continue grinding.

---

## Repository structure

```
data/
  training/          400 ARC training tasks (JSON)
  evaluation/        400 ARC evaluation tasks (JSON, held out)
scripts/
  solvers.py         All registered solvers + find_solver() dispatcher
src/
  categories/        Individual solver modules
  loader.py          Grid utilities
  visualise.py       Plot tasks and predictions
results/
  solver_backlog.md  Tasks whose rules are partially understood
notebooks/
  exploration.ipynb  Interactive visual exploration
CLAUDE.md            Full analytical protocol and conventions
```

---

## Running

```bash
conda activate arc-agi

# Run all tests
pytest

# Check how many training tasks are solved
python src/explore.py

# Try the solver on evaluation tasks
python src/explore.py --split evaluation
```

---

## Status

Active development. The analytical process is the core asset — each new task exercises it and refines the library of recognised patterns. The goal is to reach a solve rate on the evaluation set that is competitive with the ARC Prize 2025 leaderboard.
