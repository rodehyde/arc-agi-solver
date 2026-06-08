---
name: Strategy re-think signal (2026-06-04)
description: User flagged that the current approach needs rethinking after ERASE group yielded 0 new solvers
type: project
originSessionId: 4bd354e2-8375-4d79-869d-9a1220edc506
---
After completing full analysis of the 17-task ERASE group (all tasks where AND(input,output)=output), every single task had coverage=1 — no new solver families, only backlog entries. This pattern of diminishing returns on family search suggests the long-tail unsolved tasks are genuinely diverse and may not cluster into ≥2-task rule families the way the earlier batches did.

**Why:** The user ended the session with "we're going to have to have a re-think."

**How to apply:** At the start of the next session, discuss the strategic direction before diving into more triage. Options include:
1. Pivot to neural/TTT approaches for the long-tail instead of continuing family search
2. Focus on the larger identified groups (SEPARATOR_GRID 31 tasks, SINGLE_COLOUR_OUTPUT 80 tasks) rather than the ERASE/AND-analysis residual
3. Accept that rule-based coverage may plateau around 68–80 tasks and invest more in ML coverage

The backlog now has 30+ entries — plenty of material. The question is whether to continue expanding it or shift effort toward implementing the TTT/neural pipeline for the remaining ~330 unsolved tasks.
