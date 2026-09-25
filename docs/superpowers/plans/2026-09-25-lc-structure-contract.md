# LC Structure Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testable offline contract for source-backed LC proposals without model calls, outcome scoring or order placement.

**Architecture:** A separate projection converts a validated legacy source request into a causal level catalog without its fixed trade menu. A proposal validator binds one answer to that catalog and an external policy. A pure pre-entry checker exercises hypothetical timing and risk constraints; it does not replay exits or report PnL.

**Tech Stack:** Existing Python, pandas, pytest and standard-library JSON/hash utilities; no new dependencies.

**Spec:** [2026-09-24-lc-structure-first-design.md](../specs/2026-09-24-lc-structure-first-design.md)

**Status:** September 25: design approved by the user's “proceed”; this implementation plan awaits review. No task below has been implemented. Recommend native execution with one independent software review at the end to limit repeated context costs.

## Global Constraints

- “All 17 native archetypes, live orders, fusion configuration and frozen research remain unchanged.”
- “The immediate implementation candidate is an **offline proposal contract and deterministic validator**, not an order service.”
- “`execution_authorized: false` in every record in this offline slice.”
- “V1 proposals select exact catalog levels; tick rounding is code-owned, and discretionary buffers are excluded.”
- “Synthetic fixtures are not economic policy approval.”
- “All 142 previously scored cases are development data now.”
- Keep `quant/archetype-evidence-audit`, as explicitly requested; preserve unrelated untracked files. Do not push, launch market agents, add outcome lessons to curriculum, or edit existing research modules.
- Reading a source seal verifies integrity, not independent authentication of its archive or live receipt times. Preserve that limitation in every projected packet.

## Review Focus

1. Resealed but source-inconsistent catalog or altered policy must fail exact reconstruction, not merely hash-format checks (Tasks 1–2).
2. NaN, infinity, booleans as prices, duplicate JSON keys, and naive timestamps must fail explicitly (Tasks 1–3).
3. Equal-price levels with distinct provenance must retain their IDs; obstacles cannot disappear through deduplication or selective reporting (Tasks 1–2).
4. Known absent/broken parents must not be conflated with missing data or a universal trade veto (Tasks 1–2).
5. A trigger at expiry, a partially completed minute, a gap in pre-entry history or a stop breach during response time must never become an eligible fill (Task 3).

---

## File map and existing sources

Create only:

- `scripts/research/lc_structure_packet.py`: source projection, level catalog, packet reconstruction.
- `scripts/research/lc_structure_proposal.py`: strict answer parsing, policy binding and structural proposal validation.
- `scripts/research/lc_structure_preentry.py`: pure hypothetical pre-entry eligibility and geometry checks.
- `tests/research/lc_structure_fixtures.py`: public synthetic helpers, no private artifact dependency.
- `tests/research/test_lc_structure_packet.py`, `test_lc_structure_proposal.py`, `test_lc_structure_preentry.py`: tests for those respective modules.
- `docs/knowledge/lc_structure_contract_checkpoint_2026_09_25.md`: actual results, limitations and next action after execution.

Update `PROJECT.md` and `docs/knowledge/MEMORY.md` at handoff. Do not add a runner, database, CLI, model transport, trading strategy or new graph build.

Read before implementation: `lc_context_assessment.py` (`validate_context_request`), `lc_context_facts.py`, `lc_published_assessment.py`, `conditional_assessment.py` (`digest`), `conditional_entry.py`, and `tests/research/test_lc_context_assessment.py` (`context_request`). These are references/read-only dependencies. Existing publication helpers enforce the old contract: do not call them to publish the new answer schema.

## Shared contract details

Use plain JSON-compatible dictionaries and deterministic ordering. Reject unknown fields at the response, plan and policy boundaries. All prices must be finite positive numbers, not booleans. All timestamps must carry a timezone and normalize to UTC. Version strings: `lc_structure_packet_v1`, `lc_structure_proposal_v1`, `lc_structure_policy_v1`, `lc_structure_preentry_v1`.

Packet fields: `version`, `contract_sha256`, `case_id`, `setup_open`, `decision_time`, `source_binding`, `curriculum_sha256`, `instrument`, `data_stream_id`, `context`, `candles`, `levels`, `citation_catalog`, `curriculum`, `limitations`, `execution_authorized`, `seal`. Hash all fields except `seal` with existing `digest`. `source_binding` contains source request version and digest. `curriculum_sha256` hashes the exact projected curriculum plus master brief, not the unrelated project memory. `contract_sha256` hashes a literal JSON contract descriptor containing the four version strings, field sets, trigger kinds and decision enums specified here; define that descriptor in the packet module so the proposal module can import it without a circular dependency.

The projection's `context` retains hourly facts and daily/4H lifecycle, but removes old-stop-derived `ceiling_distance_r` and `distance_basis`. Keep `distance_is_unobstructed_room: false`. Exclude old `plan`, `plan_menu`, `indicative_economics`, scores, outcomes, source prompts and response schemas. Do not copy arbitrary top-level source fields.

Each level has `id`, `price`, `kind`, `timeframe`, `instrument`, `data_stream_id`, `source_locator`, `observation_start`, `observation_end`, `available_at`, `lifecycle`, and `relation_to_setup`. Use raw bar high/low IDs `bar:<tf>:<row-index>:<high|low>` for every supplied eligible candle in `1d,4h,1h,15m,5m,1m`; do not silently truncate. Parent boundary IDs are `parent:<1d|4h>:<bound-id>:<low|high>`. Raw levels are `candle_extreme`, not pivots. A parent is `parent_boundary` only when existing context says its provenance is known and its bound predates setup. Preserve equal-price entries with different sources. Standalone pivot/Fibonacci extraction is unsupported in v1; known parent pivot lineage is preserved, not fabricated.

The citation catalog maps IDs to exact paths within the new packet: one ID per candle, level, context group and curriculum item. It must resolve without a second catalog namespace. Raw bars use derived close time as reconstructed availability; parent levels use their recorded availability. Record missing intervals, missing timeframes and context uncertainties rather than filling prices. For this first projection, optional raw fusion/Fibonacci/derivative features are deliberately omitted: no unverified default becomes a fact. That is reduced coverage, not support for such signals.

Proposal fields: `version`, `contract_sha256`, `case_id`, `packet_sha256`, `curriculum_sha256`, `policy_sha256`, `decision`, `thesis`, `parent_child`, `sequence`, `supporting`, `opposing`, `competing_explanation`, `unknowns`, `plan`, `execution_authorized`. Claims contain nonempty `text` and unique `evidence_ids`; lists contain claims. `sequence` also records observation-end timestamps that code checks against the cited candle/level. Mandatory parent/child and competing-explanation claims, and at least one opposing claim for enter/wait. Unknowns may be empty. A citation cannot certify the truth of prose. Require the exact packet contract hash, not merely a recognizable version label.

For enter/wait, `plan` contains `trigger`, `confirmation_evidence_ids`, `invalidation_level_id`, `invalidation_operator`, `stop_level_id`, `destination_level_id`, `obstacle_level_ids`, `horizon_minutes`, `expiry_minutes`. `invalidation_operator` is `touch_or_below`; an invalidation and protective stop may differ, but both must be below indicative entry. Allowed triggers: `{"kind":"immediate"}` and `{"kind":"close_above","level_id":ID}`. No free numeric prices, buffers, retest programs or alternate management. Immediate requires completed predecision confirmation citations. Wait requires its future post-arm condition; it cannot claim this has already occurred.

For reject/insufficient-evidence, `plan` is null and its explanation belongs in opposing/unknown claims respectively. Unknown thesis is valid only for non-entry decisions. Invalid output is a separate status, never converted to reject. Every answer is non-authorizing.

Policy is external and fully explicit: `version`, `instrument`, `data_stream_id`, `max_entry_price`, `entry_expiry_minutes`, `horizon_minutes`, `minimum_net_rr`, `risk_budget_usd`, `max_notional_usd`, `equity_usd`, `max_leverage`, `roundtrip_cost_bps`, `processing_seconds`, `routing_seconds`, `tick_size`. Numeric zero is permitted only for costs, minimum RR and delays; durations are positive integers. Missing policy leaves a diagnostic proposal non-executable; never supply an economic default. Policy values here are fixtures, not strategy recommendations.

### Task 1: Causal source projection and public fixtures

**Files:** create packet module, fixture helper and packet tests listed above.

**Interfaces:**

```python
def build_structure_packet(source_request: dict) -> dict: ...
def validate_structure_packet(source_request: dict, packet: dict) -> None: ...
# Test helper: returns a fresh legacy synthetic source request.
def structure_source() -> dict: ...
```

- [ ] **Write failing tests** using the existing public fixture, not private price data:

```python
from copy import deepcopy
from scripts.research.conditional_assessment import digest
from tests.research.test_lc_context_assessment import context_request
from scripts.research.lc_structure_packet import build_structure_packet, validate_structure_packet

def test_projection_is_source_bound_and_does_not_mutate():
    source = context_request()
    before = deepcopy(source)
    packet = build_structure_packet(source)
    assert source == before
    assert packet == build_structure_packet(source)
    assert packet['levels']['bar:5m:11:high']['price'] == 110.0
    assert packet['levels']['bar:5m:11:low']['price'] == 99.0
    assert packet['execution_authorized'] is False
    assert not {'plan', 'plan_menu', 'indicative_economics'} & packet.keys()
    validate_structure_packet(source, packet)

def test_resealed_catalog_tampering_fails():
    import pytest
    source = context_request()
    packet = build_structure_packet(source)
    packet['levels']['bar:5m:11:high']['price'] = 111.0
    packet['seal'] = digest({k:v for k,v in packet.items() if k != 'seal'})
    with pytest.raises(ValueError):
        validate_structure_packet(source, packet)
```

- [ ] Run `env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research/test_lc_structure_packet.py`; first failure must be the missing new module/API, not a broken old fixture.
- [ ] **Implement projection**: validate the legacy request first; build an allowlisted detached payload; iterate all supplied rows by declared column names; validate OHLC/volume, unique increasing UTC opens and completed intervals; derive raw-level availability at bar close. Unknown parent context stays unknown and has no executable parent levels. Strip old economics, rebuild local citations and hash last. Validation recomputes the packet from the original request and requires exact equality. Copy, never mutate input.

```python
# Core source binding and reconstruction pattern for the new module:
from scripts.research.conditional_assessment import digest
from scripts.research.lc_context_assessment import validate_context_request

def validate_structure_packet(source_request: dict, packet: dict) -> None:
    expected = build_structure_packet(source_request)
    if packet != expected:
        raise ValueError('structure packet differs from validated source')
```

- [ ] Add parameterized mutations with explicit expected rejection: future bar close; naive time; duplicated/reversed open; high below close; NaN/boolean price; wrong source seal; unknown timeframe; unknown candle column. For gaps, assert a limitation and no fabricated rows, not automatic whole-case rejection. Assert absent parent differs from missing/unknown; post-setup parent is never pre-existing; raw highs never become confirmed pivots; equal prices retain separate IDs; every emitted citation resolves. Use existing fixture builders to make internally consistent alternative parent states; do not reseal one source field and assume deeper legacy validation will accept it.
- [ ] Run the new packet tests plus `tests/research/test_lc_context_assessment.py` and `tests/research/test_lc_published_assessment.py`; record exact results. Commit only Task 1 files with `feat: add causal LC structure packet`.

### Task 2: Strict proposal and external-policy validation

**Files:** create proposal module/tests; extend shared synthetic fixtures.

**Interfaces:** consumes Task 1's original source and projected packet.

```python
def validate_structure_proposal(source_request: dict, packet: dict,
                                raw_response: str, policy: dict | None) -> dict: ...
# Test helpers:
def structure_policy() -> dict: ...
def structure_answer(packet: dict, policy: dict | None, decision: str = 'enter_proposal') -> dict: ...
```

Return exactly `status`, `errors`, `proposal`, `policy_sha256`, `execution_authorized`.
Statuses: `valid_proposal`, `valid_reject`, `insufficient_evidence`, `invalid`.
Errors are stable strings; invalid has `proposal: null`. Missing policy may
preserve a well-formed proposal with `policy_sha256: null`, but Task 3 cannot
mark it eligible. Invalid source reconstruction raises `ValueError` rather than
being blamed on the assessor. Malformed agent JSON returns `invalid`.

- [ ] **Write failing tests** and helper fixtures. The source fixture is the existing synthetic `context_request()`. Its close is 105; use `bar:5m:11:low` at 99 as stop/invalidation and `bar:5m:11:high` at 110 as destination. Include every catalog ID strictly between entry and destination in `obstacle_level_ids` (not only unique prices). Claims explicitly say “synthetic schema fixture,” not a real setup assessment. Use policy values: BTC-USD/same-stream; max entry 106; expiry 15min; horizon 60min; minimum net RR 0.5; risk budget $100; max notional $2,000; equity $1,000; leverage 2; costs 12bps; processing 90s; routing 0; tick 0.1. No fixture value becomes a production default.

```python
import json
from scripts.research.lc_structure_packet import build_structure_packet
from scripts.research.lc_structure_proposal import validate_structure_proposal
from tests.research.lc_structure_fixtures import structure_source, structure_policy, structure_answer

def test_literal_proposal_valid_but_never_authorizes_order():
    source = structure_source(); packet = build_structure_packet(source)
    policy = structure_policy(); answer = structure_answer(packet, policy)
    result = validate_structure_proposal(source, packet, json.dumps(answer), policy)
    assert result['status'] == 'valid_proposal'
    assert result['errors'] == []
    assert result['execution_authorized'] is False

def test_wrong_bound_policy_is_not_a_rejection():
    source = structure_source(); packet = build_structure_packet(source)
    policy = structure_policy(); answer = structure_answer(packet, policy)
    policy['max_entry_price'] = 107
    result = validate_structure_proposal(source, packet, json.dumps(answer), policy)
    assert result['status'] == 'invalid'
    assert 'policy_binding' in result['errors']
```

- [ ] Run the new proposal test file and observe the missing API failure.
- [ ] **Implement** strict JSON parsing (reject duplicate keys and non-finite constants), exact field sets, bindings, decision/plan consistency, claim citation membership, chronological sequence checks, source-backed level roles and indicative geometry. Entry proposals require known validated current/hourly facts, a known native long and trustworthy cited price operands; unavailable mandatory facts permit only insufficient-evidence output, not an executable proposal. Missing optional parent structure is not missing mandatory price data. Validate policy type/units and stream identity. Compute obstacle membership from the whole catalog; no model-supplied subset may omit an intervening level. Do not enforce a bullish HTF gate or numerical fusion threshold. Do not parse prose as proof of semantic truth.

```python
def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result

def reject_constant(value):
    raise ValueError('non-finite JSON constant: ' + value)

# json.loads(raw_response, object_pairs_hook=unique_object,
#            parse_constant=reject_constant)
```

- [ ] Parameterize answer mutations: wrong case/packet/curriculum/policy hash; nonexistent citation; empty opposing/competing claim; invented numeric price; unknown management field; retest trigger; future confirmation; out-of-order sequence; stop at/above entry; target at/below entry; omitted obstacle; unresolved entry thesis; reject with a plan; insufficient evidence without unknowns; execution_authorized true. Assert `invalid`, never silent repair. Check policy null separately from invalid policy; a missing policy cannot prevent a valid rejection explanation. Valid broken-parent and absent-parent packets do not fail solely for those states.
- [ ] Run packet/proposal and old context/published tests. Commit Task 2 files with `feat: validate bounded LC structure proposals`.

### Task 3: Pure pre-entry checks and acceptance checkpoint

**Files:** create preentry module/tests; extend fixtures; create checkpoint report and update project handoff.

**Interfaces:** consumes original source, packet, raw response and external policy; revalidates them, never trusts caller-supplied `valid: true`.

```python
def check_structure_preentry(source_request: dict, packet: dict,
                            raw_response: str, policy: dict | None,
                            execution: dict) -> dict: ...
```

`execution` has `response_available_at`, `proposed_fill_at`, `fill_open`, `completed_minutes`. Minutes contain UTC `open_time`, OHLC and volume, from decision through the minute before proposed fill. No high/low/close of the fill minute is accepted. Every supplied bar must be completed by fill and form uninterrupted 1m coverage; this is a hypothetical fixture interface, not a claim of authenticated fills. Result fields: `status`, `reason`, `geometry`, `execution_authorized`; statuses `eligible_hypothetical`, `cancelled`, `not_ready`, `invalid`. All are non-authorizing and contain no PnL.

- [ ] **Write failing tests** using the same 105/99/110 fixture:

```python
import json
from scripts.research.lc_structure_packet import build_structure_packet
from scripts.research.lc_structure_preentry import check_structure_preentry
from tests.research.lc_structure_fixtures import structure_source, structure_policy, structure_answer

def test_hypothetical_fill_recomputes_room_without_authorizing():
    source = structure_source(); packet = build_structure_packet(source)
    policy = structure_policy(); raw = json.dumps(structure_answer(packet, policy))
    execution = {'response_available_at':'2026-01-01T04:01:30Z',
                 'proposed_fill_at':'2026-01-01T04:02:00Z', 'fill_open':105.0,
                 'completed_minutes':[
                     {'open_time':f'2026-01-01T04:0{i}:00Z', 'open':105.0,
                      'high':105.5, 'low':104.5, 'close':105.0, 'volume':1.0}
                     for i in range(2)]}
    result = check_structure_preentry(source, packet, raw, policy, execution)
    assert result['status'] == 'eligible_hypothetical'
    assert result['geometry']['risk_per_unit'] == 6.0
    assert result['geometry']['reward_per_unit'] == 5.0
    assert result['execution_authorized'] is False
    execution['fill_open'] = 107.0
    assert check_structure_preentry(source, packet, raw, policy, execution)['reason'] == 'entry_cap'
```

- [ ] Run the new preentry test file and observe missing API failure.
- [ ] **Implement**: reject malformed inputs; validate answer/source/policy; no policy returns `not_ready/missing_policy`; non-entry decisions return `not_ready/no_entry_proposal`. Arm is minute-ceiling of the later of actual response availability and decision plus policy processing, plus routing. Expiry is decision plus policy expiry; eligible fill must be strictly before expiry. Immediate permits only the first arm open. Wait requires a completed 1m close strictly above the frozen trigger, with bar open at/after arm, and fill at the next open. A trigger before arm, equality at trigger, or skipped eligible fill does not qualify. Cancel on any stop or structural-invalidation touch before entry, including during response; check fill open similarly. A gap in the evidence is `not_ready/coverage_gap`, not permission to assume no breach.
- [ ] Round long protective stop and destination down to tick using `Decimal(str(price))`, never move anchors in the source catalog. Round entry cap down. This models a farther stop and no inflated target reward; it does not establish fill quality. Recompute at `fill_open`; cancel at/above destination, above cap, or below required net RR. Conservative round-trip cost per unit is `fill_open * roundtrip_cost_bps / 10000`. Use:

```python
risk_per_unit = fill_open - rounded_stop
reward_per_unit = rounded_destination - fill_open
net_rr = (reward_per_unit - cost_per_unit) / (risk_per_unit + cost_per_unit)
quantity_limit = min(policy['risk_budget_usd'] / (risk_per_unit + cost_per_unit),
                     policy['max_notional_usd'] / fill_open,
                     policy['equity_usd'] * policy['max_leverage'] / fill_open)
```

Report that quantity only as `quantity_upper_bound`, not an executable size: lot-size rounding, stop gaps and live slippage are outside this checker. The budget is modeled risk, not a guaranteed loss cap. If structural invalidation is above the protective stop, cancellation uses that nearer level before entry; post-entry management is not implemented. Recompute/report intervening levels at actual fill; do not assume a crossed level has become support.

- [ ] Test immediate/wait positive controls, both theses, exact trigger equality, pre-arm close, partial bar, skipped first fill, actual response later than assumed delay, expiry boundary, future response time, missing/duplicate/reversed minute, stop touch during processing, invalidation touch without stop touch, gap below stop, gap beyond cap/target, tick rounding, fees erasing room, NaN/boolean prices and policy absence. Assert no output has `execution_authorized: true`, order fields, exit outcomes or PnL. An `enter_proposal` positive test checks contract validity, not truth of its synthetic rationale.
- [ ] Run all three new test files and focused frozen regressions:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research/test_lc_structure_packet.py tests/research/test_lc_structure_proposal.py tests/research/test_lc_structure_preentry.py tests/research/test_lc_context_assessment.py tests/research/test_lc_published_assessment.py tests/research/test_lc_context_facts.py tests/research/test_conditional_entry.py tests/research/test_lc_single_assessment.py
git diff --check
```

- [ ] Obtain one independent software review after implementation (if native execution is chosen); address factual defects and rerun affected tests. Reviewer sees source/tests, not a new market case to assess. Do not count that review as model-trading evidence.
- [ ] Write actual commands, counts, failures and review outcome in the checkpoint. State: contract/pre-entry fixture readiness only; structural-target exit replay, capital accounting, paid agent evaluation and live service remain unimplemented. Update PROJECT/MEMORY with first unfinished action, private-data boundary and local commit/push status. Commit Task 3 files with `test: verify LC proposal pre-entry guardrails`.

## Completion boundary

This plan is complete when valid public synthetic proposals pass, adversarial cases fail with stable reasons, old focused tests remain passing and review findings are closed. It makes no profitability claim. No private data or network access is required for its tests. The next separate work item is structural-target outcome replay and its numerical study protocol—not another source census, trader-teaching audit or paid campaign launched automatically.
