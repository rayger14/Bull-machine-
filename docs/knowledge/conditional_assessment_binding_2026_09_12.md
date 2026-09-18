# Agent-choice to replay binding — September 12, 2026

## Completed

The research adapter now connects a structured specialist choice to the
conditional-entry replay through an exactly bound review. This closes the code
interface between the previous evidence-ID assessment and conditional resolver.
It does **not** mean fresh agents have run through the new contract, a persistent
master is operating, or full trading validation has completed.

[`conditional_assessment.py`](../../scripts/research/conditional_assessment.py)
implements these interfaces:

- `compile_menu(packet, settings)`: enter, reject, and waits above the latest
  completed 1m/5m/15m high. The compiler owns prices, stop, expiry, delays,
  original deadline, notional and costs. Source timeframe identifies the anchor;
  confirmation is always a subsequent eligible completed 1m close above it.
- `build_assessor_request`: supplies the source packet as evidence, sealed menu
  and explicit response schema. The old immediate-entry plan is only a reference
  comparator/transport field, not the chosen conditional execution instruction.
- `grade_choice`: rejects altered menu/settings, unsupported IDs, wrong facts
  or evidence IDs and malformed claims. Null means insufficient evidence, not
  rejection. No profit probability is allowed for null/reject choices.
- `build_review_request` / `grade_review`: bind review to the canonical contents
  of the evidence, settings, menu, exact response object, resolved claims,
  selected plan and reviewer rules. A same-case review cannot authorize a
  changed choice, claim or source. These are canonical-content hashes, not a
  proof of raw message provenance or independent model execution.
- `gate_choice` / `replay_choice`: independently regenerate the chosen policy
  and release it only after valid contracts and a passing bound review. Other
  cases retain null plan/PnL. All outputs remain execution-unauthorized.

The compatibility projection into the old fact/claim validator stays internal.
It cannot become the selected immediate-entry plan. Frozen old modules, answers,
grades and outcomes are preserved.

## Reviewer rules clarified

The critic has a review-only instruction. It must identify an actual claim and
evidence when alleging factual, citation or status-polarity defects. Explicit
response-contract violations are distinguished from ambiguity and discretionary
judgment. Material findings fail; incomplete reviews are not assessable;
ambiguity/opinion alone cannot manufacture a failed factual review.

Status now explicitly evaluates the **whole proposition**, not whether a
condition mentioned inside it passes. The requested probability is explicitly
uncalibrated, for chosen-policy net PnL>0 including no-fill=0, with no invented
probability citation slot. This menu has no extra H3 gate; a future arm requiring
one needs an explicit versioned contract. Required teaching citations still
matter. These changes address the previously observed review ambiguities; their
effect on actual model behavior has not been measured yet.

## Verification

The quant reviewer approved the
[prospective contract](../superpowers/specs/2026-09-12-conditional-assessment-binding.md)
and final code after three findings were addressed: unrepresentable delays,
assessor language accidentally included in reviewer instructions, and factual
findings without a claim index. Delay/index counterexamples failed before the
fix and passed afterward. Prompt wording was inspected, not behaviorally proven.

**39 focused tests pass**, independently rerun by the reviewer. **635 full
research tests pass**, one existing LibreSSL warning, root run 12.49 seconds.
The count is the prior 596 plus 39; it excludes old private pilot tests.
Synthetic integration uses the real transport-envelope builder, review adapter,
conditional resolver and future scorer. Its hand-authored choices/reviews are
test fixtures, not evidence of model skill or reviewer truth.

The final assessor-request builder was also exercised read-only on the existing
E01/E02/E03 source packets. Each produced five menu IDs, no omitted options and
ten transport chunks: 38,791 / 38,972 / 39,165 bytes respectively. These chunks
were constructed, **not delivered to new model roles**. E01's three timeframe
highs are the same price; three labels do not represent three distinct economic
policies or independent trials. No new outcomes were scored in this unit.

| Artifact | SHA256 |
|---|---|
| Frozen spec | `f723675787d5d655674e5c40d385d83c70498e5d6bb2f6a47da762e158562045` |
| Adapter | `4e27c5f8b237a5dbe062e7f0a47214ffc5952844a300421985f07c925d500096` |
| Tests | `f9d1fb21243de8ef5af6c3877fe72a427e5e9807fcda2aff528ab8771475c308` |

No production/live/fusion/archetype changes, new paid market-assessment calls,
new dependency, push or PR. All seventeen archetypes remain untouched.

## Next concrete deliverable

Build the complete candidate ledger and replay entries in actual readiness
order, so one delayed trade can correctly block or yield to another. Freeze
same-time ties, whether pending intentions reserve capacity, and release rules
before economic comparison. The current helper scores independent cases; it is
not yet that occupancy-aware replay.

Then freeze the matched code-immediate, code-confirmation and agent arms plus
calendar folds, lock actual isolated role decisions/reviews before outcomes,
and evaluate hourly and minute tracks separately with equal priority. Include
all invalid/missing cases, latency and cost stress. The full walk-forward and
CPCV evaluations remain unrun; this unit makes no profitability claim.
