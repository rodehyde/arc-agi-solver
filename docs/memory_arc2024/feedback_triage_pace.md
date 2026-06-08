---
name: Triage pace and user involvement
description: Keep task analysis under ~1 minute; stop and ask user immediately when stuck; no long autonomous batches
type: feedback
originSessionId: 4bd354e2-8375-4d79-869d-9a1220edc506
---
Keep each task analysis to ~1 minute. If more than a minute passes without a solution, stop and describe what was found so far — do not continue autonomously.

**Why:** User was interrupted multiple times when analysis exceeded 1–2 minutes. The right model is: present the 4-step analysis, form a hypothesis, attempt verification, and if stuck after one revision attempt, surface it immediately rather than spinning.

**How to apply:** After the 7-lens decomposition + 4-step protocol, if no clear rule emerges within the first attempt, state what was found and ask the user rather than trying more approaches. For coding/verification, if the first implementation fails and a quick fix isn't obvious, flag it immediately.
