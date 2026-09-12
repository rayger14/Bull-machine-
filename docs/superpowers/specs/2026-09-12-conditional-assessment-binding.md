# Conditional assessment binding — prospective research contract

User approved the next integration step, with routine research decisions
delegated to quant review. The quant reviewer approved this bounded extension.
Keep the old evidence-ID pilot, original graders and all production untouched.

## Compiler-owned choices

Given an original outcome-hidden packet and separately frozen caller settings
(`entry_expiry_minutes`, `processing_seconds`, `routing_seconds`), compile a
sealed menu: enter, reject, and wait above the latest completed 1m/5m/15m high.
Read highs using named candle columns; record source candle evidence ID and
availability, distinct from the always-1m confirmation timeframe. Omit missing
or stale anchor options explicitly. Malformed/future supplied candles or a
tampered catalog invalidate compilation; never send future candles to the agent.

The source packet's fixed plan supplies stop, notional, costs, original absolute
exit deadline and 2R target convention. It is a reference comparator, NOT the
agent's selected execution instruction. The new menu overrides its immediate
entry convention. No free-form model prices, stops, targets, expiry or delay.
This menu imposes no extra H3 permission; a future arm explicitly requiring H3
needs its own versioned compiler contract, not reviewer discretion.

Hash the source packet, settings and generated menu. Agent response has exactly
case_id, plan_id, probability_net_positive, facts and claims. Null plan_id means
insufficient evidence, not reject. The probability is an uncalibrated estimate
of chosen-policy net PnL>0, counting unfilled policy outcomes as zero; require
null for reject/null choices. Reuse the old fact/claim checks by an internal
compatibility projection only; do not expose that projection as the chosen plan.

## Reviewer correction and exact binding

Reviewer sees the original packet, sealed menu, exact raw choice, resolved
claims and chosen plan, plus versioned critic rules. Bind its response to the
canonical digest of this entire review request. A stale pass on the same case
must fail after any choice, claim, settings, source or rule change.

Closed rule IDs distinguish material factual, citation, explicit-contract and
claim-status-polarity errors from nonblocking ambiguity and subjective judgment.
Claim status describes evidentiary support for the ENTIRE proposition; whether
a named condition passes is separate. An uncalibrated estimate is permitted,
not proof of calibration. Do not invent a probability citation slot, universal
H3/collector gate, or new trigger. Missing required teaching citations remain
real support defects. Ambiguity is not contradiction unless reasonable readings
are contradicted; unsupported interpretation is not automatically invented fact.

Review includes case_id, reviewed_sha256, assessment_complete, findings, verdict.
Each finding names rule_id, nullable claim_index, evidence_ids and explanation.
Claim-level findings must point to an existing claim and valid evidence IDs.
Factual/citation/polarity findings require evidence; response-level explicit
contract findings may have null claim_index. Material rule findings => fail;
incomplete without material findings => not_assessable; otherwise pass. This is
verdict consistency, NOT proof of semantic truth or reviewer independence.

## Handoff boundary

The final adapter independently recompiles the menu, validates the raw choice,
rebuilds the exact review request and checks its bound verdict. Only a valid,
review-passed nonnull selection produces a research replay plan. Null choices,
malformed responses, stale reviews and review failures retain null plan/PnL,
never silently becoming reject or avoided-loss credit. Always report
execution_authorized=false; transport capture and isolated role provenance
remain requirements of the later orchestration harness, not facts proved here.

Tests must exercise menu/settings tampering, stale review, source/column/clock
validation, unsupported/null choices, probability semantics, materiality/verdict
consistency, reviewer envelope construction, and real adapter-to-resolver
execution on synthetic bars. This unit makes no paid model calls, repairs no
old answers, and claims no completed walk-forward/CPCV evaluation.

Next after this unit: complete actual-readiness-ordered candidate occupancy
replay and freeze matched chronological evaluation arms. Old case diagnostics
are not substitutes for that experiment.
