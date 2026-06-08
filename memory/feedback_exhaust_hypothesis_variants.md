---
name: Exhaust hypothesis variants before discarding
description: When a broad hypothesis (symmetry, size, etc.) partially suggests itself but one variant fails, test all sub-variants before moving on
type: feedback
originSessionId: 4bd354e2-8375-4d79-869d-9a1220edc506
---
When a broad hypothesis category (e.g. "symmetry") is a candidate but doesn't immediately confirm, enumerate and test ALL variants within that category before discarding it. One failed variant is evidence against that variant, not against the whole family.

**Why:** On d56f2372, "tried symmetry — didn't fit" led to marking the task as unsolved. The rule was LR-symmetry specifically. Testing a different symmetry variant first (possibly rotational or LR+UD combined) produced a false negative and the whole hypothesis was dropped.

**How to apply:** When symmetry is a candidate: test LR, UD, 180° rotation, 90° rotation, diagonal — each independently. Same principle for any broad hypothesis: size (cell count? bbox area? bbox dimensions?), uniqueness (unique colour? unique shape? unique hole?), etc. Only discard the category when all specific sub-variants have been tested and failed.
