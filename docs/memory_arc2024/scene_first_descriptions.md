---
name: Process-based description approach (current)
description: LLM prompting strategy for ARC descriptions — read/operation/write structure, saves to descriptions_process.json
type: project
originSessionId: e487d252-10cb-4121-a4a5-aef8e3f8d060
---
The original description approach (5-field controlled vocabulary: TASK TYPE, SIZE CHANGE, STEPS, LOCATING NEEDED, OPERATION) was bottom-up and output-focused. It forced tasks into 4 predefined types and had no room for novel mechanisms like guided navigation or compartment fill.

## New approach: scene-first prompting

Key insight from manually describing tasks 2dd70a9a (guided navigation) and 272f95fa (compartment fill) with the user:
- Reading the INPUT as a scene first (before touching the output) surfaces functional roles of all elements
- Grey cells in 2dd70a9a looked like random noise but were actually steering obstacles for a path
- Questioning apparent randomness is the critical step: "Look at what seems random and ask whether it has meaning"

## New prompt structure (5 fields)

**SCENE**: Describe each input as a scene, naming the role of each colour. Then look at whatever seems random and ask whether it might be functional rather than decorative.

**INVARIANTS**: What is the same across ALL input examples?

**VARIABLES**: What differs between examples?

**MECHANISM**: What rule maps input to output? Describe functionally (why cells were added, not what shape they make).

**TYPE**: 1–4 words naming the kind of task, invented freely (e.g. "guided navigation", "compartment fill", "conditional pattern tiling"). Not from a controlled vocabulary.

## Key improvements over old approach

- No controlled vocabulary — TYPE is invented per task
- Explicitly questions apparent noise/randomness
- Focuses on mechanism (why) not geometry (what)
- max_tokens=600 (was 150 — descriptions were being cut off)
- Output file: `data/descriptions_scene.json` (old: `data/descriptions_training.json`)

## Validation results (Sonnet 4.6 on 5 tasks)

- 272f95fa: correctly identified central room + fixed colour assignments per position
- 00d62c1b: "Interior enclosure marking" / flood fill from edge — more precise than EXTEND_RECT
- 29c11459: "Symmetric inward fill" — complete accurate mechanism
- 007bbfb7: "Conditional pattern tiling" — precise and correct
- 2dd70a9a: richer than before but still doesn't fully capture grey-cell steering mechanism

## Script

`scripts/generate_descriptions.py` — updated with new prompt. Run with Haiku for all 400 tasks.
`scripts/test_scene_descriptions.py` — test harness comparing old vs new on 5 benchmark tasks.

## Further evolution: process-first prompting (current)

Scene-first was good but still produced appearance-based descriptions in some cases. The prompt was tightened further:
- MECHANISM explicitly requires (a) what the rule READS, (b) OPERATION name (e.g. flood-fill, majority-vote), (c) what it WRITES
- Added critical warning: "do NOT describe what the output looks like — describe the rule's operations"
- TYPE must be verb-first: e.g. "boundary flood fill", not "compartment fill"
- Output file: `data/descriptions_process.json` (Sonnet 4.6, all 400 tasks complete)
- Embeddings cached to `data/embeddings_process.npz`
- Produced 30 clusters (vs 26 old); 335/400 clustered; 65 noise
