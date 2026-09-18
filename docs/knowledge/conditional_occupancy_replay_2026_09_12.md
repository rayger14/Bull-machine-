# Multi-candidate conditional replay — September 12, 2026

## Accomplished

The research replay now handles competing candidates, not just independent
trade outcomes. A delayed entry can be overtaken by a later signal; the position
that actually enters occupies the slot until its causally observed exit.
This is an execution/accounting component, not evidence of a profitable agent.

[`conditional_occupancy.py`](../../scripts/research/conditional_occupancy.py)
adds `replay_sleeve` and `replay_reviewed_sleeve`. The latter uses the real
compiler/review gate before passing any released research plan to the book.
Hourly and minute tracks run separately with equal standing: one fixed-notional
long position per track, shared cost assumptions within each track.

- Pending intentions reserve no capacity. Entries sort by actual readiness,
  original decision time, then candidate ID. Busy entries are skipped permanently.
- Opening-gap and original-deadline exits release before another same-open
  entry. Intrabar exits release only when that minute has completed. Both-touch
  bars use stop-first; stops gap at the open and favorable target gaps fill at
  the target. Stops and actual-entry-derived 2R targets are fixed.
- The causal position scanner stops consuming prices at a known exit. Missing
  later diagnostic data cannot erase a previously observed closed trade.
- Missing pending-intent or active-position observations stop the trustworthy
  admission prefix. Subsequent entries are indeterminate, not presumed tradable.
- Every supplied candidate remains in the ledger. An unavailable assessment is
  a fail-closed nonorder, not a successful rejection. Known closed-trade PnL is
  a subtotal; complete-policy PnL stays null when records are unresolved.

The caller must still establish source population completeness, prior locks and
actual independent assessor/reviewer receipts. Hand-authored reviews in unit
tests do not demonstrate model behavior. Starting equity is unspecified; this
is unfunded research, without funding, impact, margin or inference costs and
without certified live fills. No claim is made about a combined two-track book.

## Verification and review

The delegated quant reviewer approved the prospective
[contract](../superpowers/specs/2026-09-12-conditional-occupancy-replay.md)
and independently reran all **25 focused tests**, finding no correctness blocker.
Tests were introduced before the module and initially failed because it was
absent. They cover event ordering, ties, permanent skips, exit phases, missing
observations, prefix invariance, malformed manifests, four single-position
economic parity cases and the actual reviewed-plan adapter.

Final root verification: **660 research tests passed**, one existing
urllib3/LibreSSL warning, in 12.35 seconds:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research
```

Spec commit `f8116f9`; implementation commit `26cef32`.

| Artifact | SHA256 |
|---|---|
| Frozen contract | `42abc98a1768b2d96415edd0531aa2407b5298b701c9d28d57d99872b3c8b569` |
| Replay module | `a9cb64ebed5c92570e582c3b519f039daa9d8619bc2441fb21c255634024a711` |
| Tests | `bd173176cdb5fbd50b511b22ec5f443648c53f905a5d19f244e8a9e07353280e` |

No new market outcomes, model market-assessment calls, optimization, dependency
installation, production/live/all-seventeen-archetype/fusion changes, push or PR.
Full walk-forward and CPCV evaluation remain **unrun**.

## Next concrete deliverable

Read-only inventory of the existing local April pilot found 245 candidate IDs
and classifications (3 hourly, 242 minute), but only three selected complete
candidate records in `results/evidence_id_pilot_2026_09_12/prepared_sources.json`.
Its `prepare_sources.py` constructs the complete candidate objects, then saves
only selected full objects plus the classification ledger; the persisted
`observations` list is empty. This is an artifact-retention limitation, not a
claim that the underlying historical archive is absent.

Build a new source-only derivative that persists every candidate's causal
evidence and provenance; preserve the frozen pilot and its hashes. Reconcile
the derivative against all 245 existing IDs/classifications before assessing
any outcomes. Do not run the old preparer in place and overwrite its artifacts.

Then prospectively freeze matched code-immediate, code-confirmation and agent
arms, costs/latency, chronological folds and complete-population accounting.
Lock fresh isolated assessments and reviews before revealing their outcomes.
April and previously revealed cases are development evidence, not a new
untouched holdout. The comparison must establish incremental agent value on
both timeframe tracks before any broader validation or deployment claim.
