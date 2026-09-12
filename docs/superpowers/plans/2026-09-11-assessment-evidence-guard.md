# Assessment Evidence Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small deterministic boundary for future assessment evidence delivery and arithmetic.

**Architecture:** A standalone research module builds canonical chunk envelopes,
validates caller-supplied transport records against the original packet, computes
indicative economics and resolves typed evidence paths. It never calls models or
touches live trading. The existing frozen study remains unchanged.

**Tech Stack:** Python3.9 standard library; existing unittest/pytest test environment.

**Spec:** `docs/superpowers/specs/2026-09-11-assessment-evidence-guard.md`

## Global Constraints

- Standard-library-only pure functions; no model calls, file/network I/O or trading integration.
- Preserve all frozen study artifacts and the current eight responses; no retries.
- Transport checks are caller-attested only, not proof of rendering or model comprehension.
- Citation path existence is not claim entailment or teacher authenticity.
- Current research branch only; no push, PR, merge, deployment or dependency installation.

### Task 1: Deterministic packet/receipt/economics boundary

**Files:**
- Create `scripts/research/assessment_evidence_guard.py`.
- Create `tests/research/test_assessment_evidence_guard.py`.
- Do not modify older assessment/replay code or private frozen files.

**Interfaces:**
- `build_envelope(packet, max_chunk_bytes=4096) -> dict`
- `validate_delivery(packet, envelope, records, *, case_id) -> dict`
- `resolve_evidence(packet, path) -> JSON value`
- Envelope includes `case_id`, `packet_sha256`, `packet_bytes`, `sections`,
  `max_chunk_bytes`, `chunks`, `indicative_economics`, `execution_authorized=False`.
- Each chunk has `index`, `text`, `bytes`, `sha256`.
- Economics keys: `basis`, `quantity`, `initial_risk`, `target`, `cost_R`,
  `breakeven`, `net_stop`, `net_target`, `actual_fill_known=False`.

- [x] **Step 1: Write failing tests.** Use a literal fixture:

```python
packet = {'case_id':'C1', 'plan':{'indicative_close':100., 'stop':98.,
          'notional':1000., 'roundtrip_cost':1.2},
          'evidence':{'parent':{'low':90.}, 'bars':[[100.,101.]]}}
# Expected economics, derived independently: qty10, risk20, target104,
# cost_R0.06, breakeven100.12, net_stop-21.2, net_target38.8.
# build with max_chunk_bytes=32; every chunk must fit and rejoin to
# json.dumps(packet, sort_keys=True, separators=(',', ':'), ensure_ascii=True).
# Exact record = {'index':i,'text':chunk['text'],'truncated':False}.
# Missing/extra/swapped/duplicate/modified records and true/null truncation fail.
# Changed envelope/hash/sections/economics or case ID fail.
# resolve_evidence(packet,['evidence','bars',0,1]) ==101.; returned nested
# values can be changed without mutating packet. Missing/wrong-type/bool/
# negative index paths fail. Nonfinite/bool/invalid stops/overflow fail.
```

- [x] **Step 2: Confirm red.** Import the module within tests after explicitly
  asserting its path exists, so absence is an expected failing assertion, not
  a collection/import error. Run `python3 -m pytest -q tests/research/test_assessment_evidence_guard.py`.
- [x] **Step 3: Implement minimal functions.** Validate JSON/numeric types first.
  Serialize canonical ASCII text, compute SHA256, slice bounded chunks, derive
  economics with explicit finite checks. Receipt validation rebuilds the expected
  envelope from the trusted packet and bound, compares it exactly, then requires
  a complete ordered record list and exact text/index/boolean matches. Resolve
  paths by explicit dictionary/list type dispatch and return deepcopy(value).

```python
quantity = notional / indicative_close
risk = (indicative_close - stop) * quantity
breakeven = indicative_close * (1 + roundtrip_cost / notional)
# Use no floating-point rounding until presentation; validate derived finiteness.
# Validation failures return valid=False/errors; invalid builders/locators raise ValueError.
```

- [x] **Step 4: Verify.** Run focused tests, then
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -q tests/research`.
  No mocks or new dependencies. Report exact commands, red/green evidence and
  remaining scope limits in the task report. Review spec and code quality.
- [x] **Step 5: Scoped local commit.** Commit only the new module/tests with
  message `research: add deterministic assessment evidence guard`.
  Root separately commits results/docs and performs final scoped review.

Completed implementation: `deb675a`; review correction: `b18120b` rejects lossy
integer conversion and independently tests packet/chunk hashes.33 focused tests;
task spec/quality review approved after one fix round. Actual tool delivery and
reviewer routing remain explicitly outside this helper's integration scope.
