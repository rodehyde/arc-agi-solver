---
name: The silent code trap
description: After data loads, always write the verbal analysis as text before calling any tool — never call a tool silently first
type: feedback
originSessionId: 4bd354e2-8375-4d79-869d-9a1220edc506
---
After a data-loading tool call returns, the next action must always be to write visible text (the 7-item mandatory structure), never to call another tool.

**Why:** Repeatedly caught reaching for Bash to "quickly check" hypotheses before completing the verbal steps. This produces no visible output, consumes time invisibly, bypasses the user, and leads to failed code iterations. Task a096bf4d: loaded data, called no tools silently, was interrupted with nothing to show after 8 minutes. Same pattern on 692cd3b6 (25 minutes of silent iteration). When the verbal steps ARE done first, correct code arrives on the first attempt (a096bf4d solved 3/3 first try when process was followed).

**How to apply:** The moment data prints, write the 7-item mandatory structure (output dimensions, all 7 lenses, Steps 1–4 + held-out prediction) as text in the response. No tool call until all seven items are visibly written above it. The check "have I written these seven items?" must happen before every non-data-loading tool call.
