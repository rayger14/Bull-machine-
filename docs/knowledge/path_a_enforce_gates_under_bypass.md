# Path A — Enforce Hard Gates Under Bypass

**Date**: 2026-05-19
**Branch**: `feat/enforce-gates-under-bypass`
**Status**: Implemented + backtested. **Backtest result mixed; needs your call before deploy.**

## What it does

When a signal is about to skip the fusion-threshold check (either via the global `bypass_threshold: true` in live, or per-archetype `bypass_fusion_threshold: true` in YAML), Path A re-checks the archetype's configured `hard_gates`. If any hard_gate failed during signal generation, the signal is dropped.

Implementation: `engine/archetypes/archetype_instance.py` now stores `hard_gates_passed` and `hard_gates_failed_reason` in the signal's metadata. The bypass branches in `bin/live/v11_shadow_runner.py:1014` and `bin/backtest_v11_standalone.py:756` honor this flag.

New config knob: `adaptive_fusion.enforce_gates_under_bypass: true` (defaults to True).

## Why

Per the May 19 gate-enforcement audit: 22 of 67 live trades (33%) fired despite explicit hard_gate violations. `oi_divergence` and `long_squeeze` were 100% gate-immune in live. Violators net **−$5,543**; clean trades net **+$3,449**.

The root cause is Rule 10's "gate-immune architecture": `gate_mode: soft` + global `bypass_threshold: true` means hard_gate violations only penalize fusion, but fusion is then bypassed, so the gates never actually block anything.

Path A restores the YAML's intent without changing any archetype YAML or the bypass behavior itself.

## Backtest result (2020-2024, $100K start)

| Metric | Baseline | Path A | Δ |
|--------|---------:|-------:|--:|
| Trades | 3,384 | 2,909 | **−475 (−14%)** |
| PnL | $264,651.80 | $255,062.43 | **−$9,589 (−3.6%)** |
| **PF** | **1.42** | **1.47** | **+0.05 ✓** |
| **MaxDD** | **−17.53%** | **−13.90%** | **+3.63 pts ✓** |
| **Sharpe** | **1.47** | **1.51** | **+0.04 ✓** |

Risk-adjusted metrics (PF, Sharpe, MaxDD) all improved. PnL dropped 3.6%.

### Per-archetype impact — the story is entirely about CB

| Archetype | Baseline trades | Path A trades | Δ trades | Δ PnL |
|-----------|----------------:|--------------:|---------:|------:|
| confluence_breakout | 577 | **13** | **−564** | **−$22,455** |
| trap_within_trend | 511 | 531 | +20 | +$4,231 |
| exhaustion_reversal | 325 | 344 | +19 | +$3,971 |
| spring | 656 | 675 | +19 | +$3,014 |
| funding_divergence | 19 | 24 | +5 | +$2,495 |
| wick_trap | 103 | 106 | +3 | +$1,161 |
| liquidity_vacuum | 34 | 41 | +7 | +$600 |
| liquidity_sweep | 912 | 929 | +17 | −$130 |
| retest_cluster | 72 | 71 | −1 | −$2,475 |
| liquidity_compression / long_squeeze / oi_divergence | (no change) | | 0 | 0 |

**CB collapsed**: 577 → 13 trades, +$19,744 → −$2,711 PnL. The remaining 13 CB trades had WR 30.8% / PF 0.30.

### Why backtest only shows the CB regression

Only `confluence_breakout.yaml` has `bypass_fusion_threshold: true` set. In backtest, the per-archetype bypass affects only CB. In live, the **global** `bypass_threshold: true` affects ALL archetypes (the gate-immune state the audit documented). The backtest cannot replicate the live problem, so it only exposes the CB regression — not the oi_divergence/long_squeeze benefits Path A would deliver in production.

This means **the backtest is the wrong test for Path A's main benefit**. The live benefit (~$5,543 of violator losses prevented, per the May 19 audit) is not visible here.

## Interpretation

Path A is working exactly as designed. The CB collapse is a real finding: **most of CB's profitable trades historically VIOLATED its own configured hard_gates** (especially `volume_zscore min 0.5` which was a stale calibration). The compliant CB trades (gates pass) actually perform WORSE (PF 0.30, WR 30.8%).

This is the same regression that the May 17 CB hard-mode test caught and rejected (over-blocked 78%, lost $17K in backtest). Path A reproduces that result for CB because CB has `bypass_fusion_threshold: true` in YAML, which Path A now treats as needing the same hard_gate enforcement.

Reading this generously: CB's `gate_mode: soft + bypass_fusion_threshold: true` was deliberate compensation for Lesson #54 (CB's fusion is artificially low). Its hard_gates were never recalibrated to be production-strict. Path A naively assumes they are.

## Decision required

Three options:

**A. Ship Path A as-is, default ON**
- Live: prevents the 22 violator trades from the audit (~$5,543 net protection)
- Backtest: regresses CB by $22K, but improves PF/Sharpe/DD overall
- Net production risk: blocks CB's bypass route entirely, which is intentional bypass

**B. Ship Path A, exempt CB explicitly**
- Add `enforce_gates_under_bypass: false` to `confluence_breakout.yaml` (per-archetype override, requires modifying CB YAML)
- Live: oi_divergence, long_squeeze, etc. get gate enforcement. CB keeps current bypass behavior.
- Backtest: matches baseline (CB unchanged). Live: gets the audit benefit minus CB.
- Cleanest separation: CB's bypass was an explicit compensation for Lesson #54, treat it as a different case from the global-bypass mess.

**C. Don't ship Path A — fix CB's gates instead**
- Recalibrate CB's `volume_zscore min 0.5` based on what its actual winners use
- Re-enable hard_gate enforcement once gates pass on winners
- More work, but addresses the root cause rather than working around it.

**My recommendation: Option B.** The audit problem (oi_divergence/long_squeeze in live) is distinct from CB's intentional bypass. The fix should be surgical, not coupled. If we later re-tune CB's gates (Option C), we can remove the per-archetype exemption.

## How to enable / disable

```jsonc
// configs/bull_machine_isolated_v11_fixed.json
"adaptive_fusion": {
  "enforce_gates_under_bypass": true   // default. Set false to revert to old gate-immune behavior.
}
```

Per-archetype opt-out (Option B implementation — not yet applied):
```yaml
# configs/archetypes/confluence_breakout.yaml
enforce_gates_under_bypass: false   # CB exemption — its hard_gates need recalibration
```

## Files modified

- `engine/archetypes/archetype_instance.py` — attach `hard_gates_passed` + `hard_gates_failed_reason` to Signal metadata
- `bin/live/v11_shadow_runner.py` — global-bypass branch now consults the flag and drops violators
- `bin/backtest_v11_standalone.py` — per-archetype bypass branch consults the flag

## Files (artifacts)

- Baseline backtest: `/tmp/baseline_backtest.txt`
- Path A backtest: `/tmp/path_a_backtest.txt`
- Compare script: `/tmp/compare_backtest_runs.py`
- Audit source: `docs/knowledge/gate_enforcement_audit_2026_05_19.md`
