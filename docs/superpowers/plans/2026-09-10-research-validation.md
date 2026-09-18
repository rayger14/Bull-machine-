# Research validation implementation plan

> **For agentic workers:** Execute inline using executing-plans and test-driven-development. Use requesting-code-review for an independent read-only review. Preserve production and original data.

**Goal:** Implement the research-only validation build approved by the user on September 10.

**Architecture:** Three separable diagnostics: causal minute event/position replay, gate-input observability using the actual production evaluator, and behavioral comparison of the unmodified live/backtest threshold loops. Results are local JSON and a human-readable report. These tools establish measurement contracts; they do not optimize or deploy a strategy.

**Tech Stack:** Existing Python, pandas, numpy, PyYAML, unittest; no new dependencies.

**Spec:** `docs/knowledge/archetype_translation_audit_2026_09_09.md`, section 4 and user's explicit approval of research-only validation build.

## Global constraints

- Keep all 17 archetypes, live bypass, production engine/configs, and input data unchanged.
- No network calls from research utilities; no public push of the local-data report.
- Preserve old minute evidence separately. No parameter search; historical results are diagnostic, not untouched OOS.
- Fixed minute setup: 15 bars either side of pivot, prior touch >=30 minutes old within trailing 24h and 0.1%, sweep below 0.02%, first reclaim within 30 minutes. Use confirmed pivots only.
- Candidate ordering: reclaim index, then pivot index. Apply the original 60-minute sweep-spacing condition only when a candidate becomes observable. An expired/slow/unconfirmed event must not reserve future cooldown.
- Minute baseline: stop 0.15% below sweep low, fixed $50k notional, 4h lockout from entry even after early stop, 12bps round-trip costs. Default next-open entry includes the entry minute's stop check and adverse gap fills. Close-fill is an explicitly labeled diagnostic. Do not force-close unresolved trades at end of data. Record open trades separately.
- Dollar drawdown uses minute-close mark-to-market including transaction costs; this is not tick-level maximum drawdown or a funded-margin account simulation. Report starting equity as unspecified, not as trade notional.
- Gate coverage is across input bars, not just entered trades. Distinguish raw/derived input absence, runtime defaults, actual skips, failures, and errors.
- Threshold loop extraction executes source AST unchanged with controlled boundary inputs. Report this limited stage explicitly, never as full pipeline parity.

## Task 1 — Causal minute replay

Files: `scripts/research/minute_sweep_validation.py`, `tests/research/test_minute_sweep_validation.py`.

Interfaces: `detect_events(bars) -> list[dict]` with pivot_idx/sweep_idx/reclaim_idx/level/sweep_low/touches; `simulate_events(bars, events, entry_mode='next_open', ...) -> dict`.

- [x] Write tests first: synthetic two-touch level produces a literal expected reclaim; unconfirmed pivots cannot fire; changing future cannot change prior events; missing/duplicate/nonfinite/malformed bars fail; next-open fill checks same-minute stop; stop gaps fill adversely; incomplete horizon stays open; fixed lockout prevents reentry after an early stop.
- [x] Run unittest; verify missing implementation fails before building.
- [x] Enumerate independently observable candidate events, sort by reclaim time, then apply sweep-spacing. Validate a unique complete UTC minute index and finite valid OHLC. Enforce sorted unique candidates in simulation.
- [x] Implement fixed-notional replay with immutable initial risk, explicit entry/exit fees, gap handling, and separate open positions. Emit parameter and source hashes through CLI `--bars --out`.
- [x] Run synthetic tests, then the recovered January/March counterexamples and monthly prefix checks on actual bars. Run frozen full-period diagnostic and report yearly counts, dollar P&L, risk and drawdown.
- [x] Commit this independently testable deliverable.

## Task 2 — Gate observability

Files: `scripts/research/gate_observability.py`, `tests/research/test_gate_observability.py`.

Interfaces: `gate_observation(gate, features) -> dict`; `audit_frame(frame, configs, source) -> list[dict]` grouped by archetype/gate/year.

- [x] Tests first: missing raw input with skip is skipped; constant zero funding fails its numeric gate rather than skipping; missing RSI converted to zero by a derived gate is flagged despite pass; unknown derived input/unsupported operator is visible; empty source still preserves roster.
- [x] Run failing tests, then use real `ArchetypeInstance._evaluate_gates` with single-gate configs. Record derived dependencies via literal dict.get keys in registered derived functions. Do not mutate their behavior.
- [x] Aggregate pass/fail/skip/error and missing/defaulted/constant input coverage. Honor main-config global gate-mode and gate-value overrides. Report disabled/zero-input status explicitly.
- [x] CLI resolves champion YAML directory from root config and profiles selected parquet plus JSONL live snapshots; output local JSON with hashes and date coverage. Avoid loading unnecessary full-store columns.
- [x] Verify tests and actual store/live observations, then commit.

## Task 3 — Live/backtest threshold boundary checks

Files: `scripts/research/decision_boundary_parity.py`, `tests/research/test_decision_boundary_parity.py`.

Interfaces: `run_boundary(source_path, kind, case) -> dict`, `compare_boundaries(root) -> dict`.

- [x] Tests first: below-threshold clean signal with global bypass is allowed live but rejected by current standalone backtester; without bypass both reject; above threshold both accept; per-archetype bypass distinguishes the backtester path; failed gates under live bypass block unless opted out.
- [x] Execute the actual AST `for s in signals` loop containing the adaptive-threshold comparison, rejecting ambiguous extraction. Only boundary dependencies are supplied; no runner constructor, network, wallet, or order-placement calls occur.
- [x] Generate the complete finite branch matrix and record mismatches, source locations/hashes and limited scope. A mismatch is a finding, not permission to repair production.
- [x] Run all research tests, independent read-only code review, CLI smoke tests and production-diff check. Commit source/tests and update report with what passed, failed, and remains unvalidated.

## Acceptance examples

```python
# Funding zero must not masquerade as missing:
assert gate_observation({'feature': 'funding_Z', 'op': 'max', 'value': -.5,
                         'nan_policy': 'skip'}, {'funding_Z': 0.0})['status'] == 'fail'
# No reinterpretation of the observed path:
assert run_boundary(live_path, 'live', clean_bypass_case)['accepted'] is True
assert run_boundary(backtest_path, 'backtest', clean_bypass_case)['accepted'] is False
```

Full historical backtesting/optimization and production repairs are deliberately excluded. Existing accounting regression tests must remain green throughout.

## Execution record — September 10

All scoped deliverables implemented and exercised; 35 research tests pass. Actual-data minute replay passed 68 monthly prefix comparisons and both recovered counterexample dates. The threshold boundary diagnostic correctly fails parity (48/192 cases), and minute historical profitability fails. These findings do not mean the strategy or full pipeline has been validated. Independent review corrections and exact results are recorded in `docs/knowledge/research_validation_results_2026_09_10.md`. Production/configs remain unchanged; branch and raw results stay local. No end-to-end golden-master certification, optimization, graduation or deployment performed.
