# Bear Patterns Quick Reference

**Last Updated:** 2025-11-13

---

## Approved Patterns (Ready to Implement)

### S2: Failed Rally Rejection

**Signal:** RSI >70 + vol_z <0.5 + upper_wick >40%
**Win Rate:** 58.5% (2022 data)
**Estimated PF:** 1.4
**Frequency:** 2.3% of bars (205 instances in 2022)

**Config:**
```json
{
  "enable_S2": true,
  "thresholds": {
    "S2": {"rsi_min": 70.0, "vol_z_max": 0.5, "wick_ratio_min": 0.4, "fusion": 0.36}
  },
  "exits": {"S2": {"target_rr": 1.5, "trail_atr_mult": 1.0, "max_bars": 48}},
  "routing": {"risk_off": {"rejection": 2.0}, "crisis": {"rejection": 2.5}}
}
```

---

### S5: Long Squeeze Cascade (CORRECTED)

**Signal:** funding_Z >1.0 + rsi >65 (OI filter disabled)
**Estimated PF:** 1.3
**Frequency:** TBD (pattern was too strict with original thresholds)

**CRITICAL FIX:** User's original logic was BACKWARDS!
- Original claim: "funding >+0.08 = short squeeze UP"
- **Corrected:** Positive funding = longs pay shorts = **long squeeze DOWN**

**Config:**
```json
{
  "enable_S5": true,
  "thresholds": {
    "S5": {"funding_z_min": 1.0, "rsi_min": 65.0, "fusion": 0.38, "require_oi_filter": false}
  },
  "exits": {"S5": {"target_rr": 2.0, "trail_atr_mult": 0.8, "max_bars": 24}},
  "routing": {"risk_off": {"long_squeeze": 2.2}, "crisis": {"long_squeeze": 2.8}}
}
```

---

## Deferred Patterns (Blocked or Needs Work)

### S1: Liquidity Vacuum Cascade

**Status:** BLOCKED (liquidity_score column missing)
**Action:** Backfill liquidity_score using: `0.5*BOMS + 0.25*FVG + 0.25*displacement`

---

### S4: Distribution Climax

**Status:** PARTIAL APPROVAL (needs tightening)
**Issue:** Too frequent (11.3% of bars), edge too thin (PF 1.05)
**Fix:** Tighten to vol_z >1.8, liq <0.20, add RSI >60 + 4H trend = down

---

## Rejected Patterns

| Pattern | Reason | Data |
|---------|--------|------|
| S3: Wyckoff Upthrust | 70% overlap with S2 | Merge into S2 |
| S6: Alt Rotation Drain | Too rare (<1% of bars) | No clear edge |
| S7: Yield Curve Panic | Event too persistent (66% of 2022) | Use as regime filter, not trigger |
| S8: Exhaustion Fade | **Wrong bias** (positive returns) | 1h: +0.03%, 4h: +0.11% |

---

## Critical System Fixes

### 1. Regime Routing (Suppress Trap Within Trend in Bear Markets)

```json
{
  "routing": {
    "risk_off": {"trap_within_trend": 0.2, "rejection": 2.0, "long_squeeze": 2.2},
    "crisis": {"trap_within_trend": 0.1, "rejection": 2.5, "long_squeeze": 2.8}
  }
}
```

### 2. Macro Veto (Tighten Thresholds)

```json
{
  "macro_veto_threshold": 0.75,
  "dxy_extreme_threshold": 1.0,
  "crisis_fuse": {"dxy_z_trigger": 1.0, "yc_inverted": true, "vix_z_trigger": 1.0}
}
```

### 3. Feature Engineering

| Feature | Priority | Effort | Impact |
|---------|----------|--------|--------|
| Backfill `liquidity_score` | CRITICAL | 1 day | Unblocks S1, S4 |
| Fix `OI_CHANGE` pipeline | CRITICAL | 2-3 days | Validates S5 |
| Add `btc_spy_corr` | HIGH | 1 day | Enables Macro Risk-Off |

---

## 2022 Market Regime Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| DXY Z-Score | +1.18 (>0.8σ for 72.8% of year) | Strong dollar = crypto headwind |
| Yield Curve | -0.053 (inverted 66% of year) | Recession pricing |
| VIX Z-Score | +0.095 (elevated 34% of time) | Fear regime |
| Funding Rate | -0.014 (neutral) | No overcrowding |
| Volume Z-Score | +0.03 (below avg) | Low conviction |

**Key Insight:** 2022 was persistent macro risk-off, not opportunistic bear signals.

---

## Baseline Performance (2022)

| Metric | Value | vs 2024 |
|--------|-------|---------|
| Trades | 13 | 330 (2024) |
| Profit Factor | 0.11 | 1.16 (2024) |
| Win Rate | 23.1% | 48.8% (2024) |
| Avg R-Multiple | -0.99 | Positive (2024) |

**Root Cause:** Trap Within Trend archetype dominated 96.5% of triggers but only ~30% win rate.

---

## Implementation Files

**Analysis:**
- `docs/BEAR_MARKET_ANALYSIS_2022.md` (full analysis)
- `docs/BEAR_PATTERNS_FEATURE_MATRIX.md` (feature availability)
- `results/bear_patterns/BEAR_ARCHETYPE_VALIDATION_SUMMARY.md` (validation results)

**Code:**
- `bin/analyze_2022_bear_market.py` (analysis script)
- `engine/archetypes/bear_patterns_phase1.py` (S2 + S5 implementation)

---

## Next Steps

1. [ ] Review S5 funding logic fix (CRITICAL)
2. [ ] Integrate S2 + S5 into `logic_v2_adapter.py`
3. [ ] Add regime routing weights
4. [ ] Backtest on 2022-2024
5. [ ] Fix OI_CHANGE data pipeline
6. [ ] Backfill liquidity_score

---

**End of Quick Reference**
