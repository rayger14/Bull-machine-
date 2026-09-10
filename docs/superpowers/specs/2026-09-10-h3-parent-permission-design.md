# H3 fixed-event parent permission

## Scope and approval

Research-only architectural addition after the causal parent ledger. The user delegated routine research approval to quant agents; `/root/parent_ledger_task_review` approved the pure sidecar approach on 2026-09-10. This implements the H3 contract in `docs/knowledge/candidate_rule_contracts_2026_09_10.md`, not a trading strategy or a fusion replacement.

No production `engine/`, `bin/`, or `configs/` changes, live changes, optimizer, public push, entry/exit simulation, or performance claim. Preserve all 17 native archetypes. Use the current research branch in place, preserving unrelated graph outputs. Local research commits are allowed.

## Alternatives and choice

1. **Chosen:** pure annotation of fixed events. This answers whether parent permission was present without changing event selection.
2. Inject permission into native selection now. Deferred because it changes cooldown/dedup populations and answers a different question.
3. Filter completed trades. Rejected because omitted opportunities and path-dependent selection make this an invalid counterfactual.

## Interfaces

Create `scripts/research/parent_context_policy.py`:

```python
evaluate_h3_permission(ledger, *, policy_id, child_event) -> dict
annotate_h3_events(ledgers, *, policy_id, child_events) -> dict
```

`ledgers` is a list of exactly four ledgers with unique configurations `(4H,3)`, `(4H,5)`, `(1D,3)`, `(1D,5)`. Batch input with an incomplete/duplicate configuration set raises `ValueError`; no winning configuration is selected. Each event yields four records. The batch returns `rows`, counts by configuration and status, `certified=False`, and limitations. No PnL is accepted or calculated.

Each child event contains nonempty strings `id`, `contract_id`, `kind`, `instrument`, `data_stream_id`, `evidence_id`; aware timestamps `first_sweep_open`, `decision_time`, `available_at`; and `values`.

Supported contracts:

- `lc_fixed_parent_reclaim_v1`: kind `hourly_lc`, values `low`, `close`, decision exactly first-sweep open plus one hour, sweep open on an hour boundary.
- `minute_child_sweep_parent_location_v1`: kind `minute_equal_low_sweep`, values `child_level`, `sweep_low`, `reclaim_close`; sweep and decision on minute boundaries, decision strictly after sweep. The selector's sweep-depth/touch/spacing rules are not reimplemented.

All numeric values must be finite positive numbers, not booleans or numeric strings. LC `low <= close`; minute `sweep_low < child_level` and `sweep_low <= reclaim_close`. Evidence IDs attest frozen-event provenance only; the evaluator does not authenticate raw candles or reproduce native detection. `available_at == decision_time` is mandatory. Unknown policy IDs raise `ValueError`; invalid/missing event, ledger or temporal evidence yields `unknown`, never permission.

Return fields: `id`, `policy_id`, `status` (`pass`, `reject`, `unknown`), `would_allow` (true only for pass), `reasons`, `child_event_id`, `child_event_contract_id`, `decision_time`, `parent_config`, `parent_contract_id`, `parent_data_stream_id`, `binding`, `interval_transition_ids`, `bound_lineage_broken`, `bound_version_superseded`, `evaluated_values`, `comparison_contract`, `certified=False`, `limitations`.

The deterministic record ID binds policy semantics, event identity/content, parent contract, frozen binding, and consumed causal transitions. It must not hash future ledger contents or an entire later-extended input manifest. Inputs and returned nested state must not alias. Invalid unsupported value objects may raise `ValueError` rather than be serialized implicitly.

## Causal parent semantics

- Verify exact instrument and data-stream match with manifest. No symbol-only or cross-venue equivalence.
- Use existing `bind_parent` with availability strictly before the first sweep open. A parent formed at equal time cannot authorize the event.
- Validate coverage through decision: the ledger must include the latest completed hour at decision, with ordered contiguous hourly transitions across its declared processed interval up to decision. Do not carry past a missing completed hour. Decision may occur between completed hours, with no incomplete-hour evidence consumed. Input cap remains 2,048 hours.
- Use only transitions with availability `<= decision_time`; interval diagnostics cover `[first_sweep_open, decision_time]`. Equal-time transitions at both boundaries count.
- Any source break of the bound lineage in the interval rejects, including a break after floor tightening. This is **lineage-lifecycle permission**, not a counterfactual test of whether a close broke the old frozen floor. Explicitly report that distinction in limitations/comparison contract.
- Tightening alone does not reject or rewrite geometry. Set `bound_version_superseded=True` when the same lineage changes versions during the interval.
- Break followed by reformation cannot rescue the event. Reformation strictly before first sweep can be bound normally.
- A lineage disappearance/change without its matching break transition is incomplete/malformed evidence: return `unknown`.
- Missing parent with otherwise valid coverage is `reject`; insufficient or malformed coverage is `unknown`.
- Parent bounds must be finite positive with low < high. Bound anchor/version/lineage identities must be present. Validate references used by the causal prefix; do not accept dangling or temporally inconsistent parent versions.

## Frozen geometry predicates

LC: `low < parent_low` AND `parent_low < close < parent_high`.

Minute: `parent_low <= child_level <= (parent_low + parent_high)/2` AND `parent_low < reclaim_close < parent_high`.

Equality at LC breach or reclaim bounds rejects. Minute lower-half level boundaries are inclusive; reclaim boundaries strict. No new RSI, trend veto, threshold, stop, trailing, or sizing rule.

## Fixed populations and later evidence

Hourly adapters must eventually freeze earliest observed native LC structural passes before cooldown/gates/fusion/dedup, not final selected trades. Minute adapters freeze existing `detect_events` output, already post-60-minute selector spacing. This implementation has no native adapter and does not claim these event populations have been recovered.

First local acceptance uses synthetic hand-derived temporal cases and real recovered-source ledger integration. Historical batch annotation is a separate follow-up after same-stream event witnesses are assembled. V23 hourly parents must never silently parent minute-venue children. Minute parents require 60-complete-minute aggregation from that stream and explicit ATR lineage. Do not manufacture arbitrary multi-year chunks: the parent constructor only supports bounded contiguous histories and full-history restart.

## Tests and acceptance

Test formation before/equal/after sweep; break at sweep/between/decision/after decision; tightening at those boundaries; break/reform with reused anchors; eligible reformation before sweep; strict LC and inclusive minute level boundaries; missing/NaN/infinity/boolean inputs; mismatched instrument/stream; wrong clocks; missing completed hour; malformed lineage/version history; input/output immutability; fresh rebuild/prefix/future-append identity; four records per event without winner selection. Include a real recovered-source ledger case, skipped only if the explicitly hashed sibling source is unavailable.

No historical return or new safe/profitable archetype conclusion follows from acceptance.
