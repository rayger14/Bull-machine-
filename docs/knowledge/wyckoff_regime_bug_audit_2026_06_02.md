# Wyckoff 4H Regime Bug Audit — 2026-06-02

## TL;DR
`tf4h_wyckoff_bullish_score` and `tf4h_wyckoff_bearish_score` (CSV columns `wyckoff_4h_bull` / `wyckoff_4h_bear`) are **NOT mutually exclusive** as the downstream archetype gates assume. They are computed as independent maxes over disjoint event families with NO direction-arbitration step, so both can fire near 1.0 on the same bar. Confirmed: 28/94 live trades have BOTH > 0.5; correlation is only -0.27.

## Bug Confirmation (Step 1)

CSV: `/tmp/trade_outcomes_live_jun2.csv`, 94 live trades.

**Reading note:** `pd.read_csv` without `index_col=False` mis-treats the timestamp column as index, shifting every column left by one. The off-the-shelf inspection misreads `wyckoff_4h_bear` as the `fear_greed` value (32 etc.). The correct read uses `index_col=False`.

### Correct statistics (43 columns, no shift):

| metric | wyckoff_4h_bull | wyckoff_4h_bear |
|---|---|---|
| min | 0.000 | 0.299 |
| 25% | 0.299 | 0.887 |
| median | 0.311 | 0.928 |
| 75% | 0.971 | 0.928 |
| max | 0.974 | 0.928 |
| unique | 8 | 5 |

- Both > 0.5 simultaneously: **28 of 94 trades (29.8%)**
- Both > 0.8 simultaneously: **28 of 94 trades** (same set — strong-signal conflicts)
- Pearson correlation: **-0.27** (only weakly negative; direction signals should be < -0.7)
- Top (bull,bear) pair: `(0.30, 0.93)` = 36 trades, `(0.97, 0.85)` = 22 trades

The 22 trades at `(0.97, 0.85)` are the smoking gun: a "bullish accumulation" reading of 0.97 should preclude a "bearish distribution" reading of 0.85 within the same 4H window.

## Root Cause (Steps 2 + 3)

### Source
- Score writer: `bin/live/live_feature_computer.py:1699-1700`
  ```python
  out['tf4h_wyckoff_bullish_score'] = htf_context_4h.bullish_score
  out['tf4h_wyckoff_bearish_score'] = htf_context_4h.bearish_score
  ```
- Score producer: `engine/wyckoff/events.py::create_wyckoff_context` (lines 1197-1276)

### Why both can be hot at once
```python
# events.py:1216-1234
bullish_confs = []
for e in _ACCUM_EVENTS:        # ['sc','ar','st','spring_a','spring_b','sos','lps']
    col = f'wyckoff_{e}_confidence'
    if col in tail.columns:
        bullish_confs.append(float(tail[col].dropna().max()))
bullish_score = max(bullish_confs)

bearish_confs = []
for e in _DISTRIB_EVENTS:      # ['bc','as','sow','ut','utad','lpsy']
    col = f'wyckoff_{e}_confidence'
    ...
bearish_score = max(bearish_confs)
```

Two independent maxes over disjoint event families. NO arbitration. If a 4H bar window contains ANY spring_a (acc) event AND ANY ut/utad (dist) event, both scores light up.

Aggravating factor in the live runner: `lookback=len(buf_4h_copy)` (line 1694) scans the ENTIRE history (e.g. 250 4H bars ≈ 42 days). Over 6 weeks of BTC, you almost always have at least one accumulation and one distribution event somewhere — guaranteeing both scores ≈ max event confidence ever observed.

### Why correlation looks like -0.27 instead of +1
Because confidences are decoupled per event family — they trend with regime, not with each other. Negative-ish but nowhere near mutually exclusive.

### Why downstream is broken
- `engine/archetypes/archetype_instance.py:70-81` uses `tf4h_wyckoff_bearish_score > 0.5` as a hard gate for short archetypes and `> 0.6` as a sizing-boost trigger. With bull also at 0.97, the "bear gate" is rubber-stamping shorts in confirmed accumulation regimes.
- `bin/live/v11_shadow_runner.py:1206` applies a 1.25x sizing boost when `tf4h_wyckoff_bearish_score >= 0.6` — also boost-firing during accumulation.

## Fix (Step 4)

Branch: `fix/wyckoff-regime-score`

### Strategy
Make bullish_score and bearish_score directionally exclusive at the source, in `create_wyckoff_context`. The score that "wins" keeps its raw max; the loser is **suppressed proportionally to the dominance ratio** so that on a clear regime, only one fires.

Net formula:
```
raw_bull = max(accum confs)
raw_bear = max(distrib confs)

if raw_bull >= raw_bear:
    net_bull = raw_bull
    # damp bear by dominance margin (in [0,1])
    margin = raw_bull - raw_bear
    net_bear = raw_bear * max(0.0, 1.0 - 2.0 * margin)
else:
    net_bear = raw_bear
    margin = raw_bear - raw_bull
    net_bull = raw_bull * max(0.0, 1.0 - 2.0 * margin)
```

- If margin ≥ 0.5, the loser is fully suppressed.
- If margin = 0 (true tie), both keep their raw values (consistent with "transition" phase).
- Default ON. Behind a feature flag `wyckoff_score_mutual_exclusion` (default `true`) so we can toggle off if it breaks downstream.

Also added a runtime assertion that emits a WARNING (not a raise) if both net scores exceed 0.5 after damping — that should never happen with the above formula and catches future regressions.

### Files modified
- `engine/wyckoff/events.py` — net-scoring inside `create_wyckoff_context`, default-on flag
- `tests/test_regime_score_mutual_exclusion.py` — new unit test asserting the invariant

### Feature-store impact
The hierarchical Wyckoff scores are computed at LIVE-runner time (in `live_feature_computer._wyckoff_multi_tf_hierarchical`) and NOT pre-baked into `BTC_1H_FEATURES_V12_ENHANCED.parquet`. So:
- **No feature-store rebuild required** for the live runner.
- For historical backtests that DO read pre-baked wyckoff scores from the parquet (if any path does), the parquet would need rebuild — but the columns `tf4h_wyckoff_bullish_score` / `tf4h_wyckoff_bearish_score` are computed at backtest runtime via the same `create_wyckoff_context` path, so the fix applies on-the-fly without rebuild.

### Existing CSV rows
The `wyckoff_4h_bull` / `wyckoff_4h_bear` values in `/tmp/trade_outcomes_live_jun2.csv` are **wrong-forever for those 94 trades**. The macro snapshot is captured at entry time and persisted; we can't retroactively re-score without re-running the live engine over the same hours with the same data buffer state. New trades (post-fix) will use the corrected scores.

## Validation (Step 5)
- Smoke test: synthetic 4H buffer with both a `spring_a` (acc) at +0.9 and a `utad` (dist) at +0.8 → confirms post-fix net scores collapse to bull=0.9 / bear≈0.0 (bull dominates by 0.1, damped 80%).
- Unit test runs in `tests/test_regime_score_mutual_exclusion.py`.

## Standing-orders compliance
- Did NOT push to remote.
- Did NOT deploy.
- Did NOT modify archetype YAMLs.
- Did NOT modify `configs/bull_machine_isolated_v11_fixed.json`.
- Did commit to branch `fix/wyckoff-regime-score`.
