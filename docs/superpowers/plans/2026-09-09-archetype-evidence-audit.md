# Archetype Evidence Audit Implementation Plan

> **For agentic workers:** Use inline execution for this research task. Preserve the running paper book.

**Goal:** Reconcile all 17 archetypes with their trader origins, executable rules, and current live evidence; audit the minute candidate on the same standard.

**Architecture:** A read-only scorecard consumes captured dashboard JSON and the actual configured roster. It separates exit records, identified completed positions, partial open positions, and unidentified legacy records. A human-reviewed report joins these measurements to rule provenance. No trading rules change in this stage.

**Tech Stack:** Python standard library for accounting; PyYAML for roster loading; unittest; Markdown.

**Spec:** User instructions in this conversation: audit all 17, consider hourly and minute BTC equally, target at least one dependable strategy before synthesizing an All-Seeing Eye.

## Global constraints

- All 17 remain in scope, including zero-trade and ungraduated archetypes.
- Preserve production configs, bypass mode, and enabled archetypes.
- Current research base is the original local checkout at d2fe814; this working checkout is 9a9d12e. Record that difference.
- Use March 9 entry cutoff only as a separately labeled view; preserve the entire ledger.
- Do not equate scale-outs with independent observations, or open partial exits with completed trades.
- Legacy missing IDs remain unidentified; never silently merge them or present them as known positions.
- September data inspected in this audit is research data now; it is not an untouched holdout for a subsequently designed rule.
- Old feature-store results and newer parity-store results are not interchangeable.
- No new fusion-quality filter, parameter optimization, or production deployment.

## Task 1 — Reproducible live scorecard

Status: complete. Nine accounting tests passed; captured API data reconciled,
including two-cent rounding difference and the open partial position. Input and
config hashes are in the generated JSON. This does not validate a strategy.

Files: `scripts/research/archetype_evidence_audit.py`, `tests/research/test_archetype_evidence_audit.py`.

Interface: `audit(trades: list[dict], status: dict, roster: list[str], cutoff: str) -> dict`.

- [ ] Test a scaled winner (+40, +60) against one loser (-50): two completed positions, 50% position win rate, PF 2.0, realized +50.
- [ ] Test an open position with a partial exit: retain its realized P&L but exclude it from completed-position statistics.
- [ ] Test two missing IDs: count two unidentified exit rows, zero identified completed positions.
- [ ] Test a pre-cutoff entry exiting after cutoff: exclude it from the entry-cutoff view.
- [ ] Test a zero-trade archetype remains in the roster and a conflicting ID fails loudly.
- [ ] Run these tests before implementation, implement accounting, and rerun.
- [ ] Load all 17 names from the configured directory, skipping example files.
- [ ] Emit JSON + CSV; include input SHA-256, source paths, timestamp coverage, and explicit limitations.
- [ ] Compute position-clustered weekly bootstrap intervals for mean realized P&L as descriptive uncertainty, not a strategy acceptance test.
- [ ] Reconcile all-history and March-9 views with the captured API values.

## Task 2 — Complete translation audit

Status: first-pass report complete in the file below. All 17 identities and gate
modes are covered. New blocking findings: minute selector fails prefix invariance;
V23 extended funding_Z is constant zero, and effort_result_ratio disappears in
2025–2026. No entry/exit tuning or causal-selector repair performed.

File: `docs/knowledge/archetype_translation_audit_2026_09_09.md`.

- [ ] Read all 17 operative YAMLs and associated structural methods.
- [ ] For each: distinguish sourced trader concept, inferred association, current rule, execution caveat, and evidence status.
- [ ] Explain selection interactions and risk behavior in collection mode.
- [ ] Audit the minute study's executable assumptions against its declared final specification; identify reproduction prerequisites.
- [ ] Include all 17 scorecard rows, date windows, ungraduated candidates, and zero-trade archetypes.
- [ ] Choose next experiments by evidence quality; no deployment recommendation without independent validation.

## Task 3 — Verification and handoff

Status: accounting tests and original minute cache reproduction pass. Minute
causality checks FAIL as documented; production files remain untouched. Full
runtime feature-parity/golden-master work and independent strategy validation
remain outstanding. Checklist items below are the original plan, not claims that
all subsequent research has been completed.

- [ ] Run the new accounting tests and inspect generated artifacts.
- [ ] Run deterministic structural probes where a translation discrepancy can be demonstrated without market-fitting.
- [ ] Confirm the trading-engine and production-config diff is empty.
- [ ] Record completed work, unresolved evidence, and the next executable study.

This plan does not claim to have produced a consistently profitable strategy. That requires a candidate passing independent, causal, cost-aware validation and forward execution checks.
