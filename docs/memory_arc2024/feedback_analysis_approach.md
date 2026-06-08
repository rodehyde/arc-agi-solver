---
name: ARC analysis approach — cognitive heuristics
description: User identified three levers for improving analysis: context, tools, and cognitive heuristics (search-order changes). Protocol now in CLAUDE.md.
type: feedback
originSessionId: 7ef6f1f1-c886-4850-94e8-9dac9e4763f1
---
Use the 4-step ARC analysis protocol (now in CLAUDE.md) before any feature enumeration.

**Why:** Bottom-up feature cataloguing leads to getting stuck on tasks where the rule is obvious to humans (e.g. 5bd6f4ac — grey cells as anomalies in uniform blocks). The protocol restructures search order to catch high-level patterns first.

**Three levers for improvement (user's framing):**
1. Context — gives priors (human-accumulated knowledge)
2. Tools — pre-built detectors
3. Cognitive heuristics — compressed human intuitions about HOW to look, not WHAT to look for

**How to apply:** Always run the 4 steps in order before pixel-level analysis. When a task is solved by step 1 or 2, note it as a worked example. When the protocol fails, ask the user "how did you see that?" — the answer is likely a new heuristic to add.

**Known heuristics to potentially formalise later:**
- "The output is the answer to a question the input is posing"
- "Elegant puzzles have short rules" (one sentence = good signal)
- "What's invariant across pairs is as informative as what varies"
