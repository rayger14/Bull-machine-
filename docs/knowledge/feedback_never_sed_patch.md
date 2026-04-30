---
name: never_sed_patch_server
description: Never patch the server directly via sed — always commit to git first then deploy. Sed patches get overwritten on next deploy.
type: feedback
---

Never apply fixes to the server directly via sed or vim. Always commit the fix to git and deploy through deploy.sh.

**Why:** On Apr 12 the dd_score NameError was patched on the server via sed. On Apr 20, deploy.sh overwrote the patched file with branch code that never had the fix committed. The engine ran broken for 2+ days, crashing on every signal entry attempt. Missed a validated $2-3K trade (confluence_breakout at $74,848 during Wyckoff bearish 0.928).

**How to apply:** If something is broken live, the workflow is: fix locally → commit → push → deploy. Even if urgent, the commit takes 30 seconds. The deploy takes 2 minutes. Total: under 3 minutes. A sed patch saves 2 minutes but creates a time bomb that explodes on the next deploy.
