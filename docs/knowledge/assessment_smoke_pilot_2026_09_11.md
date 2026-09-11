# Tiny agent-assessment handshake pilot

## Result and scope

Completed a three-case **synthetic compliance smoke test**, using one context-free assessor invocation and no critic, optimizer, paid API integration or live runner. It is not an evaluation of market interpretation, raw-data provenance, trade selection, management quality or profit. No archetype behavior was changed.

| Synthetic case | Expected | Agent response | Validator |
|---|---|---|---|
| Hourly LC, five supplied passing witnesses | supported | supported | valid |
| Minute sweep/reclaim, supplied parent rejection | contradicted | contradicted | valid |
| Hourly LC, missing input witness | unresolved | unresolved | valid |

All 15 stage findings had valid references and matched supplied statuses. The agent paraphrased already-evaluated witnesses; ordinary code can produce the same conclusions. This establishes a working file-based handshake, **not incremental agent value**. Do not call it three successful trades or a 100% trading accuracy result.

## What was implemented

`scripts/research/agent_assessment.py` provides:

- `build_packet(case_id, archetype, decision_time, witnesses)`: copies allowlisted caller-attested witnesses; validates timezone-aware availability; omits future witnesses; hashes packet content.
- `validate_assessment(packet, response)`: requires all five stages exactly once, correct packet identity, exact stage evidence references and status-consistent conclusions. Rejects malformed responses and always returns `execution_authorized=False`.
- `PROMPT`: the bounded status/citation assessment instruction.

Witness schema: `id`, `stage`, `status`, `available_at`, `source_ref`, `summary`. Stages: inputs, parent, sequence, trade_plan, management. Statuses: pass/reject/unknown. Missing stage evidence means unresolved. Contradiction dominates unresolved overall. No TTL, source authentication, independent event sequencing, native policy adapter, numerical trade-plan evaluator or prose truth checker has been implemented. This deliberately narrow contract must not be used as an admission gate. Future-exclusion counts are diagnostic inventory, not independently verified as-of observations.

Hashing establishes content identity, not historical truth. Validation must receive the original trusted packet, not one returned by a model. Natural-language summaries remain caller-supplied; free-text assertions can be misleading even when the response validates. Prompt-injection robustness, repeatability and latency were not measured.

The real assessor was launched with no conversation history and read only the saved input file. Its final JSON was preserved verbatim in content (formatting is not an API wire capture) and checked with the validator. This is manual orchestration through the session's agent tool, not an autonomous callable-model service. The tool does not expose a pinned model snapshot or exact token accounting; no such reproducibility or usage claim is made. One assessor invocation may involve multiple internal model steps.

## Tests and reproduction

23 new focused tests were observed failing before implementation because the adapter was absent; then all passed. Post-change full research suite: **382 passed in 12.89s**, one existing urllib3/LibreSSL warning. Full legacy repository suite was not run because it can invoke runner scripts and move private result files.

Run `PYTHONPATH=. python3 -m pytest tests/research/test_agent_assessment.py -o addopts='' -q`.

The three inputs can be regenerated without local market data using `runpy.run_path('tests/research/test_agent_assessment.py')['pilot_packets']()`. Pass them with `PROMPT` to an assessor, then validate each returned item against its original packet. This fixture helper is test-only.

Private artifacts: `results/assessment_smoke_2026_09_11/{input,response,validation}.json`.

- Input SHA256: `f118ea57115d2ab464e1a48e4b1fd263f60c1e977d9e72ba5871353d19e0a889`.
- Saved response SHA256: `1dd875e6a1fc10de2ff0eca4456515046d14e07f852734919b2223077bb2cc02`.

## Next smallest step

One real, outcome-hidden evidence packet with actual numeric values and a separately frozen rulecard; independently compute reference predicates without exposing their answers to the assessor. Preserve unknown provenance rather than manufacturing a complete five-stage case. Hourly and minute need separate semantics. This would start testing reasoning rather than status paraphrasing, still without trades. Do not scale to all 17, add a critic or run economic experiments from this smoke result alone.

Implementation remains local pending scoped code review; no automatic push or PR update in this pilot.
