---
name: Don't ask permission for local work
description: Never ask "shall I proceed?" or similar for local file/code operations — just do it
type: feedback
originSessionId: 4bd354e2-8375-4d79-869d-9a1220edc506
---
Do not end a message with "shall I proceed?", "ready to continue — want me to go on?", or any equivalent. For all local operations (reading tasks, writing solvers, editing files, moving to the next task in the triage queue), proceed without confirmation.

**Why:** We explicitly agreed that autonomous local work doesn't need confirmation. Asking creates unnecessary back-and-forth and slows the triage pace. The user's instruction in CLAUDE.md covers this.

**How to apply:** After finishing one task, immediately move to the next. Only stop and ask when genuinely stuck (no hypothesis after running the full protocol) or when approaching a non-local operation (git push, delete, remote state).
