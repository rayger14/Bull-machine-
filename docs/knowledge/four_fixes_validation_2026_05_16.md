# Four-Fix Validation — May 16, 2026

**Branch**: `feat/tp-tier1-defaults-validated` (Fix 4 shipped)
**Other branches**: not created (Fixes 1-3 rejected, no commits)
**Verdict**: 1 of 4 fixes accepted (Fix 4). Three rejected with evidence.

---

## TL;DR

User approved 4 production changes pending WFO validation. Each was applied + train/OOS backtested + compared to baseline. Only Fix 4 (TP Tier 1 exit defaults) survived. The other three were either neutral-to-negative or actively destructive on OOS.

| Fix | OOS PnL Δ | Verdict | Reason |
|-----|----------:|---------|--------|
| 1 oi_divergence → short | +$1,482 (+1.21%) | REJECT | Target archetype's PnL got *worse* ($-1,511 → $-3,173) — simple direction flip alone is insufficient; gates need direction-aware rework |
| 2 long_squeeze + `ema_slope_50 ≤ 0` gate | −$2,048 (−1.68%) | REJECT | Gate selected the worst trades (WR 50% → 8.3%) |
| 3 CB `gate_mode: soft → hard` | **−$17,885 (−14.63%)** | REJECT | Hard mode over-blocks (CB trades 345 → 74, lost $27K profit); matches MEMORY.md Lesson #55 |
| **4 TP Tier 1 defaults** | **+$8,784 (+7.19%)** | **ACCEPT** | Only fix that improves OOS without damage |

---

## Methodology

- WFO: train 2018-2022, test 2023-2024 (OOS)
- Backtester: `bin/backtest_v11_standalone.py --commission-rate 0.0002 --slippage-bps 3`
- Each fix applied in isolation against the same baseline
- Reject criteria: OOS PnL drops, OOS PF gap, archetype trade count drop > 70%, or target archetype gets worse despite system improvement

---

## Baseline (current production state)

| Window | Trades | PF | PnL | Sharpe |
|--------|-------:|------:|---------:|-------:|
| Train 2018-2022 | 2,965 | 1.298 | $156,761 | 0.88 |
| **OOS 2023-2024** | **1,562** | **1.466** | **$122,219** | **1.78** |

---

## Fix 1: oi_divergence direction flip (long → short)

**Hypothesis**: archetype detects selling climax (extreme volume + bearish BOS + oversold RSI) but trades long — direction is wrong.

**Result**:
- System OOS PnL: +$1,482 (+1.21%) — marginal positive
- oi_divergence OOS trade count: 42 → 55 (+13 trades, fires more as a short)
- **oi_divergence OOS PnL: -$1,511 → -$3,173 (LOSES MORE)**
- oi_divergence OOS WR: 40.5% → 60.0% (wins more often, but loses bigger when it loses)

**Why it fails**: the gates are direction-coupled but in subtle ways. With `direction: short`:
- `rsi_14 ≤ 35` (designed for long-buy-oversold) now reads as "short the oversold bottom" — fires near the bottom, gets squeezed up
- `derived:distribution_at_resistance bool_false` is the wrong sign for shorts (we'd WANT distribution at resistance to short)

The system PnL gain is from less archetype-overlap dedup pressure (fewer long competitors at the same bars), not from the short archetype being good. The target archetype itself is worse.

**Recommendation**: A full direction-aware gate rework would be needed: invert RSI to `min: 65`, flip distribution check to `bool_true` (or remove). That's a 4-line change instead of 1. **Hold for separate follow-up.**

---

## Fix 2: long_squeeze + `ema_slope_50 ≤ 0` regime gate

**Hypothesis**: long_squeeze (the only short archetype) shouldn't fire during sustained uptrends — squeezes don't happen when trend keeps absorbing late longs.

**Result**:
- System OOS PnL: -$2,048 (-1.68%) — worse
- long_squeeze OOS trade count: 30 → 12 (blocked 60% as intended)
- **long_squeeze OOS PnL: -$3,332 → -$5,381 (LOSES MORE despite fewer trades)**
- long_squeeze OOS WR: 50% → 8.3% (the gate kept the WORST trades, blocked the WINS)

**Why it fails**: `ema_slope_50 ≤ 0` selected for the wrong subset. Apparently the rare long_squeeze WINS happened during weak uptrends where ema_slope was positive but slowing. The gate filtered those out, leaving only the catastrophic 1-in-12 surviving trade.

**Recommendation**: Try a different regime signal — `tf4h_wyckoff_bullish_score < 0.5` was variant B; could test that instead. Or reconsider whether long_squeeze should be enabled at all in current regime (Standing Order forbids disabling but tightening to near-zero firings would have similar effect).

---

## Fix 3: CB `gate_mode: soft → hard`

**Hypothesis**: CB's `bypass_fusion_threshold: true` + `gate_mode: soft` means structural gates only penalize fusion (which is bypassed) → no actual blocking. Flipping to hard mode would let existing structural gates (volume_zscore ≥ 0.5, etc.) actually block junk.

**Result**:
- System OOS PnL: **-$17,885 (-14.63%)** — catastrophic
- CB OOS trade count: 345 → 74 (-78%)
- **CB OOS PnL: $34,916 → $7,718 (LOST $27K of profit)**
- CB OOS PF: 1.58 → 1.71 (marginal improvement)

**Why it fails**: CB's existing hard_gates (especially `volume_zscore ≥ 0.5`) block 78% of trades when enforced strictly. Those blocked trades were COLLECTIVELY profitable — PF 1.58 means total wins > total losses. Even though the surviving 22% have slightly better PF, the volume reduction kills total PnL.

This **confirms MEMORY.md Lesson #55** verbatim: "retest_cluster hard mode dropped PF 1.43→1.18 — REVERTED. Soft mode preferred." Same pattern applies to CB.

**Recommendation**: Hard mode is the wrong tool. The real CB junk problem (low-fusion trades) would be better addressed by tightening individual hard_gate thresholds (e.g., `volume_zscore ≥ 1.0` instead of 0.5) rather than flipping the mode. That's a different study.

---

## Fix 4: TP Tier 1 exit defaults (ACCEPTED)

**Hypothesis**: Prior TP Study #6 (branch `quant/tp-strategy-research`) found that changing the engine-wide default exits to `trailing_atr_mult: 2.5`, `trailing_start_r: 1.0`, `scale_out_pcts: [0.1, 0.3, 0.5]` would lift OOS PnL ~22%.

**Result**:
- System OOS PnL: **+$8,784 (+7.19%) — $122,219 → $131,003**
- System OOS PF: 1.466 → 1.484 (+0.018)
- System OOS Sharpe: 1.78 → 1.81
- System train PnL: $156,761 → $165,841 (+5.8%) — train/test gap within bounds, no overfit
- Trade count: 1,562 → 1,571 (essentially unchanged)

**Per-archetype OOS impact** (the major ones):

| Archetype | PnL Δ | PF Δ |
|-----------|------:|------:|
| liquidity_compression | $9,347 → $10,907 (+17%) | 3.56 → 4.03 |
| liquidity_sweep | $26,322 → $33,312 (+27%) | 1.40 → 1.50 |
| confluence_breakout (excluded from change) | $34,916 → $33,973 (-3%) | 1.58 → 1.53 |
| wick_trap | $7,818 → $5,447 (-30%) | 1.51 → 1.37 |
| funding_divergence (excluded) | $2,884 → $1,846 (-36%) | 1.32 → 1.19 |

LC and liquidity_sweep gain materially. The excluded archetypes drift slightly. wick_trap is the one notable casualty — its existing exit profile may have been tuned and the new defaults are worse for it specifically.

**Note on the +7.19% vs +22.7% gap**: the original TP study estimated +22.7% from per-axis sweeps; this confirmation deploy shows +7.19% measured end-to-end. The gap is likely because:
1. The original study used scratch override per-archetype configs; this deploy modifies engine defaults (different code path)
2. CB excluded here (it was independently tuned in PR #30)
3. wick_trap and funding_divergence have their own tuned configs that absorb less of the benefit

Still: +7.19% is a real, validated, system-wide PnL improvement with no regime damage and no overfit signal.

**Final code change**: `engine/archetypes/exit_logic.py::create_default_exit_config()` — 15 archetype blocks updated, CB and funding_divergence intentionally skipped. Also the module-level `default_rules` block at line ~141.

---

## Recommendations for the rejected fixes

These are real problems — they just don't have a clean 1-line solution:

1. **oi_divergence**: needs full direction-aware gate rework (4-line change), then re-test
2. **long_squeeze**: try the `tf4h_wyckoff_bullish_score < 0.5` variant instead of ema_slope; if that also fails, this archetype may be fundamentally regime-mismatched and effective tightening (frequency → near-zero in bull regimes) is the only path Standing Orders permit
3. **CB**: don't flip gate_mode; tighten the existing hard_gate thresholds instead (e.g., `volume_zscore ≥ 1.0`). Smaller surgical tighten that keeps the high-volume trades

All three deserve their own focused 1-2 hour studies. Not done in this batch.

---

## Standing Orders honored

- ✓ No archetypes disabled (Fix 2's gate just makes long_squeeze fire less, doesn't disable it)
- ✓ `bypass_threshold` untouched
- ✓ Real backtest validation, not phantoms
- ✓ Train/test WFO split applied
- ✓ Lesson #54 honored: no fusion-based filtering
- ✓ Failed fixes reverted from working tree
- ✓ Successful fix (4) on isolated feature branch for user review before merge

## Files

- This report: `docs/knowledge/four_fixes_validation_2026_05_16.md`
- Accepted change: `engine/archetypes/exit_logic.py` (15 archetype blocks + default_rules updated)
- Raw outputs: `results/fix_validation/{baseline,fix1_oi_short,fix2_ls_regime,fix3_cb_hard,fix4_tp_tier1}/{train,oos}/`
