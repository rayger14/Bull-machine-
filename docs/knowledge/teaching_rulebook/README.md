# Trader teaching rulebook

This is the durable, developer-facing record of the teachings Bull Machine is
trying to implement. It connects original sources to explicit rules, current
code gaps and small repair chunks. A trader's claim, our interpretation, a
working implementation and a profitable strategy are four different things.

Updated October 5, 2026. The files live outside the ignored `reports/` directory.
They are eligible for Git tracking, but are not committed or on GitHub merely
because they exist here. Historical source ledgers and experiment artifacts
remain unchanged. This is a curated index, not a complete trader archive.

## Start here

1. Read [sources](sources.md) for what was actually inspected and what it supports.
2. Read [rules](rules.md) for the meaning we must preserve and prohibited shortcuts.
3. Read [chunks](chunks.md) for all 17 archetypes and the current repair queue.
4. Review the [first chunk design](../../superpowers/specs/2026-10-05-developing-pullback-episode-design.md).
   It specifies a developing post-breakout pullback, not the complete trading strategy.

The [independent review receipt](review_2026_10_05.md) records the first design's
missing-data correction and readiness for user review, not implementation approval.

The existing [September source ledger](../trader_primary_source_ledger_2026_09_12.md)
and [Moneytaur access record](../moneytaur_primary_sources_2026_09_12.md) remain
authoritative about their original inspection scope. Newer explicit corrections
in this rulebook take precedence over stale identity or coverage statements.

## Evidence and progress are separate

Every rule identifies its basis:

- **Teaching:** a paraphrase supported by an inspected original, with reading limits.
- **Interpretation:** our proposed relationship or data representation, not the author's algorithm.
- **Convention:** an explicit mechanical choice needed to run code.
- **Hypothesis:** a proposed advantage that still needs economic testing.
- **Unresolved:** information we do not have; it cannot silently become a passing gate.

Track source coverage, contract review, implementation, recognition and economic
validation separately. A passing software test must not upgrade an unverified
trader attribution. A correct label must not upgrade a strategy to profitable.

## Completion standard for each chunk

- Original URLs and inspected scope are preserved locally in paraphrase.
- Rules identify the object, location, direction, sequence, availability,
  confirmation, expiry and invalidation, where applicable.
- Every numerical choice has units and is labeled teaching or project convention.
- Genuine, misleading and ambiguous recognition cases have expected decisions
  recorded before running the implementation. Existing exposed cases stay regression.
- Actual selected-path code and downstream consumers are checked, not only a
  helper function or an inactive module with an attractive name.
- Evidence of implementation and verification is recorded against that revision.
- Entry, sizing, management and economic acceptance remain separate when outside scope.

## How to add or correct knowledge

Use stable `SRC-*` and `RULE-*` identifiers. Add a new source record rather than
silently changing the meaning of an old receipt. Record corrections and link
the superseded claim. Preserve disagreements: two setup families may need
different retest rules. Never infer numerical rules from a handle, cropped chart
or unverified local summary.

For another CLI: read `AGENTS.md`, `PROJECT.md`, this index, the current chunk and
its review status before changing code. Do not load project handoffs, known-outcome
research or this developer progress record into outcome-hidden market assessors;
their allowed teaching packets must be separately prepared and reviewed.
