# CLAUDE.md
*Last updated: 2026-06-06*

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

ARC puzzles are easy for humans but very hard for computers. Humans improve by recognising previously seen patterns and building up a bank of pattern experiences. When a new task doesn't match anything seen before, they combine existing experiences to form a new one. This project asks: **can a computer mimic that process?**

### The real target: the evaluation set

The 400 training tasks are **not** a completion checklist. They are 400 opportunities to discover, implement, and test generalizable solving methods. The true measure of success is coverage on the **400 evaluation tasks** — tasks we have never seen and will never inspect individually during development.

When we work through a training task, the goal is not to tick it off. The goal is to extract a pattern that is likely to recur in the evaluation set and encode it as a robust, generalizable solver. A training task that yields no reusable insight is worth less than one that uncovers a broad family of rules, even if both get "solved."

**The 400 training tasks run out eventually — when they do, the work stops. Every solver we build should be earning its place by being likely to fire on unseen tasks, not just the one that motivated it.**

### What we have learned about the approach

The original plan assumed that tasks would cluster into groups sharing a common rule, and that a solver per category would scale. In practice this hasn't held: most solvers end up specific to one task or a very small family. **Each ARC task tends to need its own solver.**

This is not a failure — it reflects the nature of ARC. The tasks are deliberately diverse. The real leverage is not in finding group-level patterns but in developing good thinking tools: a process that lets us look at any new task and derive its rule quickly and reliably. The growing library of solvers is evidence that the process is working, not a route to a group-level shortcut.

**The primary goal is therefore to equip Claude with the best possible analytical process** — the recurring structural patterns, the decomposition lenses, the 4-step protocol — so that each new task can be solved stand-alone, without needing a category to belong to.

### Current state

Platform is set up: VS Code + Claude Code on macOS, Python environment, GitHub repo, ARC training/evaluation data loaded. 400 training tasks; ~100 currently solved by rule-based solvers (as of 2026-06-06).

**Active approach:** Work through unsolved training tasks one at a time using the triage process. For each task, the primary question is not "can I make this pass?" but "what is the rule, and can I derive it from first principles using the process?" If stuck after one revision attempt, stop and ask the user immediately — do not stall alone.

The primary output is a growing library of task-level solvers registered in `scripts/solvers.py`. Where sub-operations recur across solvers, extract them as shared primitives. But do not force generalisation — a solver that correctly handles one task is better than an abstraction that handles none well.

### Principles
- Categories are defined on training pairs only — never the test pair.
- A task can belong to multiple categories; categories are not mutually exclusive.
- Start simple; grow complexity only when simpler approaches fail.
- **Write a solver for every task whose rule can be verified** — but ask first whether the rule is general enough to justify a solver. Coverage of 1 training task is sufficient *if the rule is role-based and likely to recur*. A highly specific rule (e.g. one that only works for one exact grid size and colour) should be noted in the backlog rather than registered, unless it's a clear instance of a broader family.
- **Build composable primitives.** When implementing a solver, check whether the sub-operations (bounding box, flood fill, line extension, hollow interior, etc.) already exist in `src/loader.py` or a utility module. If a new primitive is needed and likely to recur, extract it to the shared library rather than duplicating code. This keeps new solvers short.
- **Generalise before you specify.** Before committing to a rule that names a specific colour, count, or position, ask: *does this need to be specific?* Prefer a rule that refers to a role (e.g. "the anchor block", "the unique colour", "the largest region") over one that names a literal value (e.g. "colour 1", "3 blocks", "bottom row"). A role-based rule generalises better to the evaluation set.
- **Ask "will this fire on an unseen task?"** before registering any solver. If the answer is "only if the task is identical to the training one", the solver is too narrow. Widen the detection condition or move it to the backlog.

### Task triage cycle

For each task:

1. **Pattern match** — Scan the recurring structural patterns list first. If the task matches a named family, skip straight to implementation.
2. **Decompose** — If no pattern match, apply the 7 decomposition lenses (see below).
3. **Verbal 4-step** — Apply the protocol in text. If no hypothesis emerges, stop and ask the user immediately.
4. **Solver + verification** — Write `solve(inp)`, run against all training pairs.
5. **Decision:**
   - All pairs match → write module, register in `ALL_PRIMITIVES`, move on.
   - Some pairs fail → make one revision attempt. If still failing, stop and ask the user immediately.
   - No hypothesis after pattern match + decomposition + 4-step → stop and ask the user immediately.

**No conclusion without process.** Before stating a rule or pattern, identify which step of the triage cycle produced it. If you can't, you've skipped the process — go back and run it. A confident-sounding answer reached by implicit pattern matching is not a conclusion; it's a hypothesis that hasn't been examined.

**Time limit: ~1 minute per task.** If the rule isn't clear within that window, bring the user in rather than stalling. The user is a faster pattern-recogniser for novel tasks and should not be kept waiting.

**Do not reach for a mathematical toolkit before the data has suggested it.** When a task has structure (symmetry, periodicity, a group action), that structure will be *visible in the pairs* — you will see it before you name it. The right order is: (1) read the pairs, (2) ask "what is this grid almost?", (3) notice the structure concretely (e.g. "row r and column r seem to mirror each other"), (4) only then identify the mathematical abstraction that names what you observed (e.g. "that is transposition symmetry"). Running this in reverse — starting from a pre-formed toolkit of transforms (D4 group, symmetry families, rotation matrices) and testing each one — is a search strategy that can take minutes and may still miss the answer. It also bypasses the user, who can often see the structure in seconds. If you find yourself iterating through abstract cases without a concrete observation from the data anchoring you, stop immediately and ask the user.

### Autonomous operation

Proceed without asking for confirmation on all local work:
- Running analysis code, reading task data, printing grids
- Writing or editing solver modules, tests, and utility files
- Updating `results/solver_backlog.md`, `CLAUDE.md`, or memory files
- Any local file read, write, or edit

**Always ask before:**
- `git push` (to any remote branch)
- Deleting files or branches
- Any operation affecting shared or remote state

Do not run long autonomous batches that leave the user uninformed for more than ~1 minute. Prefer short task-by-task cycles with the user present.

## Recurring structural patterns

These patterns appear across dozens of tasks. Recognising the structure immediately suggests the transformation — scan this list **before** applying the 7 lenses or the 4-step protocol. If the task matches a named family, skip straight to implementation.

**Regular grid of separator lines.** If a distinct colour forms continuous horizontal and vertical lines dividing the grid into a regular array of same-size cells, the rule operates per-cell. Common variants:
- One cell is the "master" (densest or most complex); the rule stamps a recoloured copy of the master into every other cell.
- Cells are selected, filtered, or ranked by a property (count, colour, uniformity, size).
- Related: if one cell is already fully coloured and others have single indicator cells, the indicator colour maps to a position within the master template (STAMP_MASTER pattern).

**Two equal halves with separator.** If the input is split into two equal halves by a uniform single-colour separator line (row or column), the output is almost certainly a logical combination of the two halves: AND (output is non-zero where *both* halves are non-zero), OR (where *either* is non-zero), or XOR (where *exactly one* is non-zero). Check AND first — it is most common.

**Template + indicator map (MOVE_TO_STATIC family).** If the input contains a small isolated template shape *and* a sparse map of coloured indicator cells at specific positions on a background grid, the output places a copy of the template at every grid position marked by an indicator cell (often recoloured to match the indicator colour). This "stamp-where-indicated" pattern covers the majority of MOVE_TO_STATIC tasks (~76 tasks).

**Extension by period.** If the output is longer than the input in one dimension and begins with the same content, find the repeating unit in the input sequence and continue it. The period is usually short (2–4 cells). Check rows, columns, and diagonals independently. Task 017c7c7b exemplifies this along a diagonal.

**Extract the unique object.** When multiple objects/regions exist in the input and one is distinguished by a property not shared by any other (unique colour, unique size, unique shape, unique hole, only one touching the border), the output is that unique object, typically cropped to its bounding box or centred in the output. Apply step 2 ("What doesn't belong?") to find it — but note that "unique" here means structurally unique, not just visually prominent.

**Template + isolated markers encoding orientation (STAMP_ROTATED).** If the input contains one or more connected multi-colour template shapes AND isolated single cells whose colours match the template's special (non-skeleton) colours, the output places each template in the D4 rotation/reflection that aligns its special cells with the corresponding marker positions. Everything else is cleared. Works even when there are two templates and two marker sets — match each template to the marker set whose colours it can align to under some D4 transform. Task 0e206a2e exemplifies this.

*Implementation note — try all 8 D4 transforms.* When implementing any orientation-encoding task, do not try to derive which D4 transform applies analytically. Instead, define the 8 transforms as a list of lambdas and loop over all of them, checking which one satisfies the constraints (skeleton match AND special-cell offset match). This is 8 iterations of a lambda — not complex — and it works uniformly for all shape types without any shape-type detection. DECORATE_SHAPE_BY_2_ORIENT (36d67576) exemplifies this.

*Process note — compute key quantities numerically before estimating complexity.* For any D4/orientation task, before reasoning about what the implementation requires, extract the actual centroid-normalised offsets for the skeleton cells and the special cell from pair 0 and print them. The structure visible in the numbers (e.g. whether offsets are integer or half-integer, how many 4-cells there are) often reveals a simpler path than abstract reasoning suggests. Estimating complexity before seeing the numbers risks dramatically overestimating it.

**Near-rotationally-symmetric shape (ROTATION_COMPLETE).** If a single-colour shape is almost 180° rotationally symmetric but has extra cells on one side, the output adds colour 2 at the 180°-rotated positions of those extra cells. Find the rotation centre algebraically (midpoint of matched cell pairs). Task 1b60fb0c exemplifies this.

**Template + directional marker arms (STAMP_WITH_ARMS).** If the input contains a template shape and small clusters of other colours positioned adjacent to it (one cluster per direction), the output stamps copies of the template in each marker's colour, extending in the direction the marker indicates, repeating with gap=1 until the grid edge. Each marker cluster is a partial copy of the template placed at the first copy's position — this is how the direction and colour are encoded. Task 045e512c exemplifies this.

**Input encodes its own output size via a border legend.** If the input contains a marginal annotation — an L-shaped border, a full edge row/column, or a corner region — whose colours or segment counts vary across training pairs while the core content stays the same, that annotation is a scale key. Count the distinct colours (or segments) in the annotation to derive the expansion factor; the first two training pairs together establish the mapping. Task 469497ad exemplifies this: the right column and bottom row form an L-border whose distinct-colour count equals (scale − 1), and the output is the full input scaled by that factor with diagonal marker rays added.

## Decomposition pre-step

Before running the 4-step protocol, apply these 7 lenses to decompose the input. They answer "what kind of problem is this?" and dramatically narrow the hypothesis space.

**"Background" is a conclusion, not an observation.** Do not label any colour as background until the output confirms it plays no active role. The most common colour is often background — but it may instead be a wall, a constraint, or a medium that the transformation acts through. Ask: *what does the path avoid? what stops the change?* A colour that appears unchanged in every output cell is not necessarily inert — it may be structurally load-bearing. Only call something background when you have verified it has no role.

1. **Fixed vs. moving** — Which cells are identical between input and output? The remainder is the change. (If one object stays and another shifts, the rule is probably a translation or alignment.)
2. **How many parts?** — Can non-zero cells be split into 2+ distinct connected components? If yes, do they behave differently?
3. **Nature of the change** — For the part(s) that change, is the change: translation / rotation (90°, 180°, 270°) / reflection / recolouring / deletion / completion / scaling? Check rotation explicitly — shapes that look like reflections are often 180° rotations, and near-symmetric shapes may have a rotational gap being filled.
4. **What drives the change?** — Is it self-determined (the shape implies the transformation) or indicator-driven (a marker cell or colour specifies what to do)? Specifically: are there **isolated single-cell markers** whose colours match special cells in a nearby template? If yes, the markers are encoding position/direction/orientation for where the template should be placed or extended.
5. **Spatial relationship after** — Do parts end up touching, adjacent, aligned to a grid, or symmetric?
6. **More structure than input?** — If the output is more symmetric, complete, or regular, the rule is likely *repair* or *complete*. If the output contains additional copies of a shape already present in the input, the rule is likely *stamp* or *extend*. Check this before analysing the markers — once you see "copies added", the markers only need to answer direction and colour.
7. **Less content than input?** — If the output has fewer cells, colours, or a smaller grid, the rule is likely *extract* or *filter*.

Only after these 7 lenses: run the 4-step protocol below.

## ARC task analysis protocol

When analysing an unknown task or bucket of tasks, run these steps **in order** before enumerating pixel-level features. The ordering is the point — it prevents defaulting to bottom-up feature cataloguing before higher-level patterns have been checked.

**Before step 1: read the training pairs in order.** If pair 0 looks simpler or more structured than the rest, treat it as a legend — its output defines the tile types, stamps, or tools available for all later pairs. The later pairs are then instances of the same rule applied to different inputs.

1. **What is this input ALMOST?**
   Check for near-regularity: almost uniform (a few contaminating cells), almost symmetric (one region breaks it), almost tiled (one tile is wrong or missing), almost identical to another region. If found, the transformation is likely *repair, extract, or complete* the near-regular structure.

2. **What doesn't belong?**
   Look for anomalous cells, colours out of place, broken regularity, or a single shape that violates an otherwise consistent pattern. The anomaly tells you *where* the rule acts — but not *what* the rule is. Its role (remove, repair, extend, or structural pivot) only becomes clear in step 3. Never act on a step-2 anomaly without completing step 3.

3. **Hold input and output together — what rule connects them?**
   The input and output together are the demonstration; the answer you are seeking is how to get from one to the other. The older framing ("what question is the input posing?") is a useful special case, but the general form is: treat each training pair as a worked example and ask what rule makes this output the inevitable consequence of this input. This matters because the role of individual cells often only becomes clear when you see both sides — the input alone does not tell you what a marker cell means.

4. **What is the shortest *complete* rule that fits all training pairs?**
   If the rule requires more than one sentence, the abstraction is probably wrong. Prefer rules with zero special cases over rules with one, and rules with one over rules with two. Critically: any observation from step 3 that distinguishes this rule from a plausible alternative must survive the compression. Brevity cannot come at the cost of correctness — "shortest" means no unnecessary words, not missing clauses.

If steps 1–4 fail to yield a hypothesis, stop and ask the user immediately.

**Before coding, predict pair 2 from the rule derived from pair 0.** State what pair 2's output should look like under the hypothesis, then check it against the actual output. If it doesn't match exactly — including the turn point, endpoint, colour, and extent — the rule is incomplete. Do not proceed to code until the prediction matches. A rule that fits pair 0 but cannot predict pair 2 is an observation, not a rule.

**Verification is mandatory before claiming HIGH confidence.** A rule described in words is a hypothesis. It only becomes HIGH confidence when a Python implementation produces zero mismatches across all training pairs. Write the solver inline, run it, and report the per-pair match results. If any pair fails, revise the rule — do not report HIGH confidence on a partial match. MEDIUM confidence means the rule has not been code-verified or has known gaps.

**Batch analysis with a subagent.** When delegating a batch of tasks to an Explore subagent, the prompt must require code verification — not just verbal description. The subagent has Bash access and can run Python. A well-formed subagent prompt should ask it to: (1) read the raw grid numbers for each task, (2) apply the 4-step protocol, (3) write a candidate `solve(inp)` function, (4) run it against all training pairs, and (5) report the per-pair match results. Any task that doesn't produce `True` for every pair must be marked MEDIUM or lower. Tasks returned as HIGH confidence without code verification should be treated as MEDIUM until verified.

## Environment setup

```bash
# Create the environment (first time)
conda env create -f environment.yml

# Activate before working
conda activate arc-agi
```

In VS Code: `Cmd+Shift+P` → `Python: Select Interpreter` → choose `arc-agi`.

The codebase uses `str | Path` and other Python 3.10+ syntax — always use the `arc-agi` environment, not the system Python.

## Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_loader.py

# Run a single test by name
pytest tests/test_loader.py::test_grid_dims_square

# Categorise all training tasks and save results to results/
python src/explore.py

# Categorise evaluation tasks
python src/explore.py --split evaluation

# Jupyter notebook (for visual exploration)
jupyter notebook notebooks/exploration.ipynb

# Download RE-ARC dataset (400 tasks × 1,000 synthetic examples each → data/re_arc/)
python scripts/download_re_arc.py

# Categorise RE-ARC tasks
python src/explore.py --split re_arc
```

## Architecture

The project is structured as a pipeline: **load → categorise → (solve) → visualise**.

### Data
Raw ARC tasks live in `data/training/` and `data/evaluation/` as JSON files named by task ID (e.g. `007bbfb7.json`). Each file has `train` (list of `{input, output}` pairs) and `test` (list of `{input}` dicts). `src/loader.py` attaches `task_id` from the filename when loading.

### Categorisation (`src/categories/`)
`size_features.py` contains all category logic. `categorise_task(task)` returns a list of matching category labels for a task — a task can belong to multiple categories. Categories are defined by inspecting training pairs only (never the test pair). Current categories are size/area-based; the intent is to grow this into a richer taxonomy to route tasks to specialised solvers.

### Exploration
`src/explore.py` is a CLI script that loads all tasks, categorises them, prints a summary table, and saves `{category: [task_ids]}` to `results/categories_{split}.json`. The notebook (`notebooks/exploration.ipynb`) does the same interactively with visual grid output via `src/visualise.py`.

### Visualisation (`src/visualise.py`)
`plot_task(task)` shows all train pairs + test input for a single task. `plot_category_sample(tasks, category, n)` shows the first train pair from `n` tasks in a category. Uses the standard ARC 10-colour palette (indices 0–9 map to black, blue, red, green, yellow, grey, magenta, orange, azure, maroon).

### Conventions
- All grid utility functions (`grid_dims`, `count_nonzero`, `grid_area`) live in `src/loader.py`.
- New categories go in `src/categories/size_features.py` (or a new module imported via `src/categories/__init__.py`).
- The `results/` directory is gitignored and generated by running `explore.py`; `results/solver_backlog.md` is the exception and is tracked.
- **Commit when significant work is done.** This includes: any new category module, any update to `results/solver_backlog.md`, any change to CLAUDE.md, or after completing a batch of 4-step analyses. Prompt the user to commit if they haven't after a meaningful session.
