# Assessment Smoke Pilot Implementation Plan

> **For agentic workers:** Use the existing approved agentification advisory and test-driven implementation. Execute inline to respect the user's small-token request; one context-free assessor invocation, no critic or additional model review in this pilot.

**Goal:** Test a file-based, annotation-only agent handshake on three synthetic cases, not market interpretation or profitability.

**Architecture:** A pure research adapter packages caller-supplied stage witnesses with cutoff checks and a content hash. An agent summarizes each stage with evidence references; a deterministic validator checks coverage, references and agreement with the supplied witness statuses. This is deliberately a compliance baseline, not an independent semantic judge.

**Tech Stack:** Python standard library, pytest, one hosted session subagent call; no new dependency, API key, external service integration or live runner.

**Spec:** `docs/knowledge/agentification_design_2026_09_11.md`, narrowed by the user's explicit request to start small and test assessment.

## Constraints

- Preserve current feature branch and unrelated files; no production or live changes.
- Five stages: inputs, parent, sequence, trade_plan, management. Pass/reject/unknown witnesses map to supported/contradicted/unresolved. Rejection dominates unknown overall.
- Only witnesses available by the cutoff are usable. Missing/future witnesses become unknown; invalid timestamps and malformed records reject packet construction.
- Hash binds packet content. Responses must cover each stage once and cite exactly its usable witnesses. Validation does not certify witness truth or prose semantics.
- Fixtures explicitly synthetic. No raw data, profit labels, order authority, selection changes, semantic edge or live fidelity claim.

## One implementation unit

- [x] Write `tests/research/test_agent_assessment.py`: literal expected conclusions for supported hourly, contradictory minute and missing-input hourly cases; future and exact-cutoff boundaries; tampering; unknown/wrong-stage citations; omitted stages; wrong overall conclusion; nonmutation; malformed input/response.
- [x] Run `PYTHONPATH=. python3 -m pytest tests/research/test_agent_assessment.py -q`; establish failure from missing adapter (23 expected failures).
- [x] Implement `scripts/research/agent_assessment.py`: `build_packet(case_id, archetype, decision_time, witnesses)` and `validate_assessment(packet, response)`. SHA256 canonical JSON; timezone-aware timestamps; exact schemas; no I/O or tools.
- [x] Pass focused tests and run the complete research suite (23 focused; 382 full research tests).
- [x] Generate three synthetic packets using the test fixture helper; save input/prompt before one blind assessor invocation. Preserve response and deterministic validation in ignored results. No separate expected conclusions supplied; witness statuses are visible by design. All three responses validate; exact internal model-call/token counts unavailable.
- [x] Document actual results, limitations and the next step; no additional assessor calls or automatic PR update.
