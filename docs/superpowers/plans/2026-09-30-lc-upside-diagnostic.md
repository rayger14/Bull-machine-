# Upside LC diagnostic implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. This implements
> the already approved next mechanical research stage, without a new approval loop.

**Goal:** Finish the source-backed winner/loser and matched-breakout comparison.

**Architecture:** Add a pure diagnostic helper and a two-stage offline runner.
Reuse frozen context and replay primitives. Prepare writes source-only selections;
score requires that exact immutable preflight. No native engine modifications.

**Tech Stack:** Existing Python, pandas, NumPy, TA-Lib, pytest and JSON hash locks.

**Spec:** `docs/knowledge/lc_upside_diagnostic_protocol_2026_09_30.md`.

## Global constraints

Remain on quant/archetype-evidence-audit per user preference. Preserve all dirty
work and frozen files. No new paid market roles, live/config/fusion changes,
dependencies, commits, pushes or PRs. Existing full-suite collection failures are
reported rather than silently counted as passes. No fresh-holdout claims.

## Review focus

- Future candle changes cannot alter earlier matching features or selected controls.
- Unknown structure or outcomes must not become favorable evidence or zero losses.
- Duplicate or future controls, reuse and missing matches must be explicit.
- R denominators include costs, and nonentries differ from missing outcomes.
- Modified source/code bindings must prevent outcome scoring or publication.

## Task 1 Diagnostic helpers and two stage runner

Files: create `scripts/research/lc_upside_diagnostic.py`,
`scripts/research/run_lc_upside_diagnostic.py` and
`tests/research/test_lc_upside_diagnostic.py`.

Interfaces: `hourly_features(hourly, start, end)` returns a decision-indexed
DataFrame; `match_controls(cases, features, native_decisions)` returns one record
per LC case. `context_labels(facts, close, stop)` returns the six fixed categories.
`event_result(plan, bars)` returns a resolved or null economic record.
`summarize_events(rows)` and `paired_summary(pairs, events, first_month, last_month)`
retain missingness. Runner `prepare` freezes inputs; `score` verifies them and
uses the existing conditional scorer.

- [x] Write synthetic tests for pre-setup features, future mutation, matching
  exclusions/ties/reuse/unmatched cases, missing structure, boundaries, cost R,
  null accounting and source mutation. Expected examples: a $200 net gain on
  $90 stop risk plus $10 costs is +2R; unknown is not zero; an upward broken
  parent is labeled broken_up, not automatically rejected.
- [x] Run `python3 -m pytest -q -o addopts='' tests/research/test_lc_upside_diagnostic.py`.
  Expect failure because the new API is absent; implement only the specified API.
- [x] Repeat the focused command until all new behavior passes. Then run relevant
  context, scorecard and execution regression tests and attempt bare repo pytest.
- [x] Run `python3 -m scripts.research.run_lc_upside_diagnostic prepare`, inspect
  the source-only manifest, then run the same module with `score`.
- [x] Independently check arithmetic, input integrity and source cutoff invariance;
  use one narrow software/accounting review, not a market assessment campaign.
- [x] Save results report and update PROJECT.md plus MEMORY.md with finished
  work, actual verification, next decision and local dependencies.

## Execution ledger

Start: branch and HEAD verified, 85923a4. Existing changes preserved.
Ruling: retain the user's existing research checkout and file-backed ledger;
do not auto-create a worktree or commit, because the user chose this branch and
the project withholds commit/push authority. Cost: new work remains local until
explicitly checkpointed to Git. The existing baseline protocol supplies the
approved direction; this document records its bounded execution details.

Pre-flight: features and matching feed the immutable preflight; scoring consumes
that preflight without recomputing selection from outcomes. No interface conflict.

Development checkpoint: 22 initial tests passed after observed RED runs; focused
regression 258 passed. Source-only run_v1 reconstructed 142 cases, 68 upside
contexts, 65 controls and 3 unmatched cases. Independent exhaustive selection
audit agrees with all 68 match decisions. No control outcomes scored.

Final review found one Important issue: copying a run could verify the original
absolute artifact paths but consume different local artifacts. Two reproducing
tests reached the forbidden outcome-read boundary (RED). Fix requires exact
consumed paths, including prices/baseline/code, in the preflight bindings.
Run_v1 is retained as superseded source-only evidence; corrected run_v2 will be
prepared before outcomes. No market selection or economic rule changed.

Ruling: the reviewer's exclusions of prior campaigns, unrelated archetypes and
live parity remain out of this bounded scope. Main controller verifies actual
historical arithmetic separately; no software review certifies an economic edge.
Cost if wrong: an offline pattern could fail to transfer to the native live engine;
no live promotion is permitted here.

Task 1 complete: 24 new tests after observed RED→GREEN, focused regression260
passed13.75s. Bare repo test collection still aborts at the pre-existing missing
integration config; no full-suite pass. Corrected run_v2 completed133 event
replays, repeated with exact result equality; all brackets independently checked
against raw minutes. All six partitions and paired bootstrap reconcile. Review
fix verified locally; no second review or market-role calls. Results and current
continuity updated, no process running. Nothing committed or pushed.
