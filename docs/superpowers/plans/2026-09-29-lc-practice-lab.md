# LC practice lab execution ledger

User explicitly authorized direct implementation on September29 and waived
remaining spec/plan approval pauses. Native implementation, one final independent
software review; paid market-role launch still requires separate authorization.

Spec: `docs/superpowers/specs/2026-09-29-lc-practice-lab-design.md`.
Base: `85923a4`; existing branch `quant/archetype-evidence-audit`.

## Tasks

- [x] 1. Source preparation: archive alignment, fixed policy, mechanical control,
  structure-only requests and immutable manifest. `lc_practice_sources.py`.
- [x] 2. Capture: single persistent owner, authorization/reservation, exact bytes,
  measured timing, restart/null handling and all-case lock. `lc_practice_runtime.py`.
- [x] 3. Replay: first eligible entry, structural exits, legacy reference,
  independent cases and subtype/matched accounting. `lc_practice_replay.py`.
- [x] 4. Report and CLI: self-contained charts/HTML, JSON/Markdown, owner bridge,
  integration tests and source-only real preparation. `lc_practice_report.py`
  and `lc_practice.py`.
- [x] 5. One independent review, regression verification and launch handoff.

Each component receives public synthetic tests before implementation. Interfaces:
preparation produces manifest/case input/request files; runtime produces immutable
terminals and terminal lock; replay consumes only a verified terminal lock and
archive; report consumes the resulting case records. Neither preparation nor
capture can read future prices for scoring. All model dispatch is host-controlled.

## Rulings and evidence

- User waiver: implement from the existing written spec without another plan
  review. This file records execution, not another approval request.
- Workspace: retain user's explicitly selected existing research branch.
- Preserve existing source/pre-entry/outcome modules and all frozen experiments.
- No automatic commit/push/PR; deliver working-tree changes for the requested
  implementation. Source preflight/handoff changes predate this step and remain.
- Review focus: partial writes/restarts; wrong/missing role identity and payload
  bytes; gaps before first fill; unsupported post-entry invalidation; HTML injection
  and unknown values incorrectly counted as avoided losses.

Progress and actual verification are appended as work completes.

Initial RED→GREEN: source tests9 passed; runtime+source22 passed; replay15 passed
after correcting a test fixture whose two supposedly different lows both equaled99.
Combined components37 passed6.32s; integration3 passed14.04s (14 existing
matplotlib/pyparsing warnings). Baseline structure regressions159 passed7.77s.
One reviewer `/root/lc_practice_review` dispatched read-only, Astra/high, no market
assessments. Broader regression/repo collection checks still pending.
Source-only real control check:10 proposals,2 no-setup;12 complete request payloads
total1,860,603bytes (151,836–156,227 each). No future outcomes or market calls.

Final checkpoint: one independent review completed, no Critical findings and
two Important findings. Root reproduced/fixed nullable future-cell crashes and
missing matched coverage/attribution by subtype, then verified both regressions.
A report string continuation and Markdown table spacing were also corrected;
final end-to-end tests exercise the scorecards and contiguous Markdown table.
Final fresh focused check:368 passed78.71s (43 new practice tests),14 installed
matplotlib/pyparsing warnings. Whole-repo collection remains blocked by the
unchanged missing baseline config. Broader research run was interrupted after
665 passes/478.44s, not a full-suite success. No processes remain running.

Real source-only preparation passed all12 cases/1,320 candle comparisons.
Manifest1d51c49b68732f821160a2fd22a39bf9f2559826f47d62e5210fb5e75e53a86c,
`results/lc_practice_2026_09_29/run_v1`. Status12 pending, zero attempts,
authorization false, no terminal lock or outcomes. Actual dispatch and economic
report are the next separately authorized execution step, not an unfinished
implementation task. No paid market-role call or live change occurred.

Review exclusions: no profitability/semantic thesis, provider billing/attestation,
host-isolation or funded-execution certification. Public test success is not an
edge claim. See [implementation checkpoint](../../knowledge/lc_practice_checkpoint_2026_09_29.md)
for evidence, exact launch scope, local dependencies and continuation instructions.
