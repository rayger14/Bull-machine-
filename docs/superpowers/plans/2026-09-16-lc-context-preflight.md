# LC Context and Chronological Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. Execute inline for the user's cost preference; retain the explicitly requested existing research branch.

**Goal:** Deliver the outcome-free classification and coverage check needed before the approved one-assessor comparison, without changing native eligibility or live orders.

**Architecture:** Add a versioned research adapter around the existing sealed-packet context extractor; do not modify modules pinned by previous runs. It describes subtype and returns all candidates with readiness/exposure status. Original packets and their hashes remain the evidence source.

**Tech Stack:** Python, existing pandas/context helpers, pytest. No dependency installation.

**Spec:** `docs/knowledge/lc_september_distinguishing_factors_2026_09_16.md`, proposed engine application; user's subsequent approval of all four requirements and instruction to proceed.

## Global Constraints

- No production/config/order changes, no remote writes, no automatic model calls or new credit authorization.
- Preserve dirty worktree and existing evidence/response/runtime locks.
- Low-RSI, high-RSI or a parent label alone cannot become an entry veto.
- Candle geometry labels a candidate thesis, not proven exhaustion or continuation.
- Parent absence, directional breaks and untrusted data remain distinct.
- Future confirmation cannot be read as entry-time evidence.
- Keep Jan19/20/25 outcome exposure recorded. Do not claim the remaining roster is untouched history.
- This milestone is not the completed backtest or live shadow integration.

## Task1: Descriptive subtype adapter

**Files:** create `scripts/research/lc_setup_preflight.py`; create `tests/research/test_lc_setup_preflight.py`.

**Interface:** `annotate_lc_setup(packet: dict) -> dict`; consumes an existing sealed LC packet and returns its bound context plus a subtype and explicit `execution_authorized=False`.

- [ ] Write failing tests using `tests.research.test_lc_context_facts.price_case` and `reseal`. Hand-derived expectations:

```python
assert annotate(price_case(95., 112., 98.))['subtype'] == 'downside_rebound_candidate'
assert annotate(price_case(101., 115., 114.))['subtype'] == 'upside_expansion_candidate'
assert annotate(price_case(101., 112., 108.))['subtype'] == 'unresolved'
```

- [ ] Test that changing RSI does not relabel identical price geometry; unknown hourly evidence stays unresolved; a low sweep/reclaim is labeled downside-rebound candidate; input remains byte-equivalent and native eligibility unchanged.
- [ ] Run `python3 -m pytest -q tests/research/test_lc_setup_preflight.py` and observe missing-implementation failure.
- [ ] Implement classification using the existing trusted context only:

```python
context = describe_lc_context(packet)
hourly = context['hourly']
subtype = 'unresolved'
if hourly['status'] == 'known':
    if hourly['close_relation'] == 'above_prior_high':
        subtype = 'upside_expansion_candidate'
    elif hourly['close_relation'] == 'below_prior_low' or hourly['reclaimed_prior_low']:
        subtype = 'downside_rebound_candidate'
```

- [ ] Run focused tests and existing context tests. Keep outcome values and new entry permission out of this adapter.

## Task2: Chronological readiness/exposure report

**Files:** extend the same adapter/tests.

**Interface:** `build_preflight(packets: list[dict], *, outcome_exposed_ids: set[str]) -> dict`. Returns chronological records, counts, evidence readiness and explicit unknown reasons. No file IO/model bridge inside this function.

- [ ] Test unsorted inputs sort by decision time/case ID, duplicate IDs fail, exposure IDs outside supplied roster fail, known absent parents do not exclude candidates, unknown minute evidence marks not ready without dropping the candidate.
- [ ] Implement readiness from known hourly, known1m/5m context, known4H/daily parent evidence, and native_long=True. No parent state or subtype permits/rejects a trade. Keep every input row in output.
- [ ] Store exposure as `outcome_exposed` versus `no_new_reveal_recorded`, never `untouched`. Emit aggregate subtype/readiness counts and `execution_authorized=False`.
- [ ] Run new/context tests. No fixed score threshold or post-outcome field may enter readiness.

## Task3: Real-data preflight and handoff

**Files:** create local ignored `results/lc_context_preflight_2026_09_16/report.json`; create `docs/knowledge/lc_context_preflight_2026_09_16.md`; update `PROJECT.md` and `docs/knowledge/MEMORY.md`.

- [ ] Read the20 frozen original packets in `results/lc_consolidated_2026_09_15/judgment_v1/evidence/`; identify Jan20/25 by exact IDs as exposed; verify returned source hashes without changing packets.
- [ ] Generate readiness report with all20 retained; separate assessed/exposed cases from the18 still awaiting decisions. Do not invoke a model or reveal further outcomes.
- [ ] Document September live-data gap separately: current Coinbase adapter label does not identify historical perp-versus-spot instrument; Binance minute archive endsAug31. Native logged HTF scores are not a replacement for verified parent structure or missing1m observations.
- [ ] Run regression tests; report exactly which of the four milestones is implemented. No commit/push unless separately requested.

## Downstream experiment contract

**Execution checkpoint:** Tasks1–3 implemented inline and verified.12 new tests RED before code,77 combined regression tests GREEN.20 real packet hashes matched;20 evidence-ready,13 downside/7 upside; two outcome-exposed and18 awaiting decisions. Report: `docs/knowledge/lc_context_preflight_2026_09_16.md`. No model calls, live changes, commits or pushes. The downstream work below remains pending, not silently included in this completion claim.

Historical comparison will use separate immediate, mechanical-wait, one-assessor and reject-all books. Freeze single-role validation, sampled-critic selection, legal menu, outcome exposure and spend cap before dispatch. Keep original stops/24h deadline/$50k notional and12/24bps,90/300s plus measured-delay sensitivity for comparability; live-management equivalence is not claimed. Use prior data for development, later chronological blocks for evaluation with overlapping outcome windows removed at boundaries. Report missed winners, avoided losses, net returns, drawdown, coverage, latency and missing results. CPCV is not a substitute for enough independent episodes.

Single-assessor execution needs a new versioned contract: the old controller requires a critic. Do not fake a critic response or mark an unreviewed answer independently reviewed. Live shadow remains a separate integration after historical feasibility; no live deployment is authorized by this plan.
