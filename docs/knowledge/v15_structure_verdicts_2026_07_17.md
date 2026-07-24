# V15 structure patch — first honest verdicts (2026-07-17)

## What V15 is
`data/features_mtf/BTC_1H_FEATURES_V15_STRUCTURE.parquet` = V14L with 33
structure columns recomputed + 16 added (repaired SMC BOS/CHoCH, BOMS,
graded Wyckoff scores) via `scripts/rebuild/patch_v15_structure.py`
(12-chunk replay of the live code over the store's own OHLCV, 73,829 bars).
Faithfulness checks: BOMS 0.68% vs 0.7% and tf1d_wyckoff 53.5% vs 53.8%
(unchanged-logic columns reproduce); BOS 1.8-1.9%/bar, CHoCH 1.65% (alive
for the first time).

## Second engine repair found during validation
**Structural checks used the GLOBAL bar index on a rolling lookback window**
(structural_check.py): `_check_B`/`_check_C` sliced
`df.iloc[global_index-10:global_index]` on a 501-row frame → empty → auto-
reject. order_block_retest and fvg_continuation were structurally dead in
backtest AND live even after the feature repairs (double-buried). Fixed by
normalizing to `len(lookback_df)-1`; wick_trap V15 holdout re-ran
bit-identical (n=81 PF 1.7103) proving the fix is surgical. Deployed live
2026-07-17. Regression tests: tests/test_structural_index_alignment.py.

## Verdicts (battery: configs/champion variant, thresholds enforced, V15)

### order_block_retest — VALIDATED (third edge)
| window | n | PF | PnL |
|---|---|---|---|
| wfo_train 2018-22 | 148 | 1.29 | +$9,717 |
| y2023 | 41 | 1.29 | +$2,139 |
| y2024 | 46 | 2.01 | +$5,772 |
| holdout 2025-26 | 31 | **2.08** | +$2,647 |
| **CPCV k=6** | 266 | **15/15 positive**, median 1.39, worst combo **+$2,939** | |
OOS ≥ train everywhere (anti-overfit shape). Never tuned — zero optimization
ever ran on it. Only red: 2022 −$2.5K (universal long bear-bleed). ~31
trades/yr. Caveat: holdout n=31 small; anchor expectations to CPCV median
1.39, not 2.08. Live: capable of firing since 2026-07-17 deploy; scorecard
expectation added to bin/live_evidence.py.

### fvg_continuation — REJECTED (honestly, this time)
Fires at volume post-repair (n=338 full) but holdout PF 0.71 (−$4,001),
y2024 0.68. Train 1.17 → OOS collapse = the classic overfit/no-edge shape.
The 2026-07-16 audit's "highest revival probability" call was wrong on the
edge (right that the old verdict measured a bug).

### liquidity_vacuum — REJECTED (first honest test)
Fired for the first time ever (V15 battery): full PF 0.40, −$9,065.
Crisis-capitulation thesis does not survive contact.

### Champions on V15 (re-baseline, different signal population — fusion now
sees real structure scores)
- wick_trap: holdout **1.71** (+$11.3K, n=81) vs 1.43 on V14 — improved OOS.
  Train 1.10 (down from 1.37): shape inverted vs overfit.
- liquidity_compression: holdout 1.23 vs 1.14 — improved; train 1.35.

## Book after V15
wick_trap (champion) + order_block_retest (validated third edge, structure
family ≠ liquidity family — breadth the industry study ranked #1) +
liquidity_compression (thin second). Next: watch OBR live via
live_evidence.py; correlation check wick_trap×OBR once live sample grows.
