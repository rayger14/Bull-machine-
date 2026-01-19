# S5 Long Squeeze Configuration Fix

**Date**: 2026-01-08
**Status**: READY TO IMPLEMENT
**Priority**: CRITICAL

---

## Problem Summary

S5 (long_squeeze) archetype losing money with 17% win rate:
- 6 trades, 5 stopped out (83% stop-out rate)
- Total PnL: -$314.95
- Profit factor: 0.31

See `S5_PERFORMANCE_DIAGNOSIS_REPORT.md` for full analysis.

---

## Configuration Changes

### BEFORE (Current - Broken)

```json
{
  "long_squeeze": {
    "fusion_threshold": 0.45,
    "funding_z_min": 1.5,
    "rsi_min": 70,
    "atr_stop_mult": 3.0,
    "cooldown_bars": 8
  },
  "exits": {
    "long_squeeze": {
      "trail_atr": 1.5,
      "time_limit_hours": 24
    }
  },
  "routing": {
    "risk_on": {
      "weights": {"long_squeeze": 0.0}  // DISABLED
    }
  }
}
```

### AFTER (Fixed - Proposed)

```json
{
  "long_squeeze": {
    "direction": "short",

    "_comment": "FIXED v2 - 4 critical changes",

    "fusion_threshold": 0.55,         // +0.10 (more selective)
    "final_fusion_gate": 0.55,

    "funding_z_min": 2.5,             // +1.0 (extreme funding only)
    "min_oi_change_24h": 5.0,         // NEW: Rising OI requirement
    "rsi_min": 75,                    // +5 (deeper overbought)
    "liquidity_max": 0.2,

    "adx_min": 25,                    // NEW: Trending market required
    "adx_max": 50,                    // NEW: Avoid parabolic moves
    "require_trend_weakening": true,  // NEW: Exhaustion filter

    "cooldown_bars": 12,              // +4 (more patience)
    "max_risk_pct": 0.012,            // -0.003 (lower risk per trade)

    "atr_stop_mult": 6.0,             // +3.0 (CRITICAL: wider stops)

    "_calibration_metadata": {
      "version": "v2_fixed_2026_01_08",
      "changes": [
        "FIX #1: Wider stops (3.0x → 6.0x ATR) - address 83% stop-out rate",
        "FIX #2: Trend exhaustion filter (ADX 25-50, weakening) - avoid mid-uptrend shorts",
        "FIX #3: Tighter thresholds (funding_Z 1.5→2.5, fusion 0.45→0.55) - better discrimination",
        "FIX #4: Enable in risk_on regime (with higher bar) - fix regime alignment"
      ],
      "target_metrics": {
        "win_rate": ">40%",
        "profit_factor": ">1.5",
        "trades_per_year": "4-8",
        "max_stop_out_rate": "<50%"
      }
    }
  },

  "exits": {
    "long_squeeze": {
      "trail_atr": 2.0,               // +0.5 (wider trailing too)
      "time_limit_hours": 240         // 10 days (winner held 236h)
    }
  },

  "routing": {
    "risk_on": {
      "weights": {"long_squeeze": 1.5},  // ENABLED (was 0.0)
      "final_gate_delta": 0.1            // Require extra conviction
    },
    "neutral": {
      "weights": {"long_squeeze": 1.0}
    },
    "risk_off": {
      "weights": {"long_squeeze": 2.2}
    },
    "crisis": {
      "weights": {"long_squeeze": 2.5}
    }
  }
}
```

---

## Code Changes Required

### 1. Update `engine/strategies/archetypes/bear/long_squeeze.py`

#### Add Trend Filter to Vetoes

```python
def _check_vetoes(self, row: pd.Series, regime_label: str) -> Optional[str]:
    """Check safety vetoes - ENHANCED with trend filters."""

    # VETO 1: Don't short strong uptrends (new)
    adx = row.get('adx_14', 0)
    trend_4h = row.get('tf4h_external_trend', 0)

    if trend_4h == 1 and adx > 50:
        # Parabolic uptrend - too risky to short
        return 'parabolic_uptrend_veto'

    if trend_4h == 1 and adx > 35:
        # Strong uptrend - only allow if EXTREME funding
        funding_z = row.get('funding_Z', 0)
        if funding_z < 3.0:
            return 'strong_uptrend_low_funding_veto'

    # VETO 2: Prefer downtrends or exhaustion (new)
    if self.config.get('thresholds', {}).get('require_trend_weakening', False):
        # Check if ADX is declining (trend losing steam)
        adx_prev = row.get('adx_14_prev', adx)  # Need lagged ADX
        if adx >= adx_prev and adx > 30:
            return 'trend_strengthening_veto'

    # VETO 3: Risk_on regime requires extra confirmation (new)
    if regime_label == 'risk_on':
        funding_z = row.get('funding_Z', 0)
        if funding_z < 2.5:
            return 'risk_on_regime_insufficient_funding'

    return None
```

#### Add OI Divergence Check

```python
def _compute_oi_score(self, row: pd.Series) -> float:
    """Compute OI divergence score (rising OI = more longs entering)."""
    score = 0.0

    # Check OI change with HIGHER threshold
    oi_change = row.get('oi_change_pct_24h', 0.0)
    min_oi_change = self.config.get('thresholds', {}).get('min_oi_change_24h', 5.0)

    if oi_change > min_oi_change:
        # Rising OI with positive funding = longs building
        # More extreme = higher score
        score = min(1.0, (oi_change - min_oi_change) / 10.0 + 0.5)

    return score
```

#### Update Constructor with New Thresholds

```python
def __init__(self, config: Optional[Dict[str, Any]] = None):
    """Initialize Long Squeeze archetype."""
    self.config = config or {}
    thresholds = self.config.get('thresholds', {})

    # Core thresholds - TIGHTENED
    self.min_funding_rate = thresholds.get('min_funding_rate', 0.0001)
    self.min_funding_z = thresholds.get('min_funding_z', 2.5)  # Was 2.0
    self.min_fusion_score = thresholds.get('min_fusion_score', 0.55)  # Was 0.40
    self.min_oi_change = thresholds.get('min_oi_change_24h', 5.0)  # NEW

    # Trend filters - NEW
    self.adx_min = thresholds.get('adx_min', 25)
    self.adx_max = thresholds.get('adx_max', 50)
    self.require_trend_weakening = thresholds.get('require_trend_weakening', True)

    # Domain weights - unchanged
    self.funding_weight = thresholds.get('funding_weight', 0.40)
    self.smc_weight = thresholds.get('smc_weight', 0.30)
    self.liquidity_weight = thresholds.get('liquidity_weight', 0.20)
    self.oi_weight = thresholds.get('oi_weight', 0.10)

    logger.info(f"[S5 Long Squeeze] Initialized FIXED v2")
    logger.info(f"  - Min funding_Z: {self.min_funding_z}")
    logger.info(f"  - Min OI change: {self.min_oi_change}%")
    logger.info(f"  - ADX range: {self.adx_min}-{self.adx_max}")
    logger.info(f"  - DIRECTION=SHORT (contrarian in bull exhaustion)")
```

### 2. Update Config File: `configs/variants/s5_full.json`

Replace entire `archetypes.thresholds.long_squeeze` section with AFTER config above.

### 3. Update Stop Loss Logic (if needed)

Check `engine/backtesting/engine.py` or signal generation to ensure `atr_stop_mult: 6.0` is used:

```python
# In signal generation or position creation:
atr = row.get('atr_14', entry_price * 0.025)  # Fallback to 2.5% of price
atr_mult = config.get('thresholds', {}).get('long_squeeze', {}).get('atr_stop_mult', 6.0)

if direction == 'short':
    stop_loss = entry_price + (atr * atr_mult)  # Price goes UP to hit stop
else:
    stop_loss = entry_price - (atr * atr_mult)  # Price goes DOWN to hit stop
```

---

## Implementation Steps

### Step 1: Update Configuration File

```bash
# Backup current config
cp configs/variants/s5_full.json configs/variants/s5_full_BACKUP_pre_fix.json

# Apply changes to s5_full.json
# (Manual edit or script to update JSON)
```

### Step 2: Update Python Code

```bash
# Edit long_squeeze.py with new veto logic and thresholds
vim engine/strategies/archetypes/bear/long_squeeze.py

# Changes:
# 1. Update __init__ with new threshold defaults
# 2. Add _check_vetoes() logic (trend filters)
# 3. Update _compute_oi_score() with min_oi_change threshold
```

### Step 3: Run Backtest with Fixes

```bash
# Run full backtest (2022-2024) with fixed config
python bin/backtest_full_engine_replay.py \
  --config configs/variants/s5_full.json \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --output results/s5_fix_backtest/

# Check results
cat results/s5_fix_backtest/final_report.json
```

### Step 4: Compare Before/After

```bash
# Compare performance metrics
python -c "
import pandas as pd

# Load before
before = pd.read_csv('results/full_engine_backtest/trades_full.csv')
s5_before = before[before['archetype'] == 'long_squeeze']

# Load after
after = pd.read_csv('results/s5_fix_backtest/trades_full.csv')
s5_after = after[after['archetype'] == 'long_squeeze']

print('BEFORE FIX:')
print(f'  Trades: {len(s5_before)}')
print(f'  Win rate: {(s5_before[\"pnl_net\"] > 0).sum() / len(s5_before):.2%}')
print(f'  PnL: ${s5_before[\"pnl_net\"].sum():.2f}')
print(f'  Stop-out rate: {(s5_before[\"exit_reason\"] == \"stop_loss\").sum() / len(s5_before):.2%}')

print('\nAFTER FIX:')
print(f'  Trades: {len(s5_after)}')
print(f'  Win rate: {(s5_after[\"pnl_net\"] > 0).sum() / len(s5_after):.2%}')
print(f'  PnL: ${s5_after[\"pnl_net\"].sum():.2f}')
print(f'  Stop-out rate: {(s5_after[\"exit_reason\"] == \"stop_loss\").sum() / len(s5_after):.2%}')
"
```

---

## Expected Results

### Before Fix (Current)
```
Trades: 6
Win Rate: 16.67%
PnL: -$314.95
Profit Factor: 0.31
Stop-out Rate: 83%
```

### After Fix (Target)
```
Trades: 4-8
Win Rate: >40%
PnL: >$0 (profitable)
Profit Factor: >1.5
Stop-out Rate: <50%
```

### Key Improvements

1. **Fewer but better trades** (6 → 4-8): Tighter filters = quality over quantity
2. **Higher win rate** (17% → 40%+): Better entry timing + trend filters
3. **Lower stop-out rate** (83% → <50%): Wider stops survive volatility
4. **Profitable** (-$315 → >$0): Fixes address root causes

---

## Rollback Plan

If fixes make performance WORSE:

```bash
# Restore backup config
cp configs/variants/s5_full_BACKUP_pre_fix.json configs/variants/s5_full.json

# Revert code changes
git checkout engine/strategies/archetypes/bear/long_squeeze.py

# Re-run backtest to confirm restoration
python bin/backtest_full_engine_replay.py --config configs/variants/s5_full.json
```

---

## Testing Checklist

- [ ] Config updated with new thresholds
- [ ] Python code updated with veto logic
- [ ] Backtest runs without errors
- [ ] S5 win rate >40%
- [ ] S5 profit factor >1.5
- [ ] Stop-out rate <50%
- [ ] 4-8 trades generated (not 0, not 20)
- [ ] SHORT direction maintained (not flipped to long)
- [ ] Regime routing working (not all "unknown")

---

## Risk Mitigation

### Wider Stops = Larger Max Loss

**Concern**: 6.0x ATR stops mean max loss per trade ~15% (vs 7.5% current)

**Mitigation**:
1. Lower position size: `max_risk_pct: 0.012` (was 0.015) = -20% risk per trade
2. Tighter entry filters: Fewer trades means less total risk exposure
3. Better win rate: Higher win rate offsets occasional larger losses

**Math**:
- Before: 6 trades × 0.015 risk × 83% stop-out = 7.5% portfolio risk
- After: 6 trades × 0.012 risk × 50% stop-out = 3.6% portfolio risk (50% reduction!)

### Fewer Trades = Opportunity Cost

**Concern**: Tighter filters may reduce trades to 2-3/year

**Mitigation**:
1. Target is 4-8 trades/year (realistic for SHORT archetype)
2. Quality > quantity: One good trade beats five mediocre ones
3. S5 is meant for SPECIFIC conditions (overleveraged longs), not always-on

---

## Next Steps After Implementation

1. **Monitor live paper trading** (1 month)
2. **Validate on 2025 data** (if available)
3. **Compare to other SHORT strategies** (if any)
4. **Consider ensemble approach**: S5 + another SHORT archetype for diversification

---

## Summary

**Problem**: S5 losing money due to tight stops + poor entry timing + weak filters

**Solution**: 4 critical fixes:
1. Widen stops (3.0x → 6.0x ATR)
2. Add trend exhaustion filter (ADX, weakening check)
3. Tighten feature thresholds (funding_Z, fusion score, OI)
4. Fix regime routing (enable risk_on with higher bar)

**Expected Outcome**: Transform S5 from money-losing (PF 0.31) to profitable (PF >1.5)

**Timeline**: 1-2 hours implementation + 10 min backtest + 30 min validation

---

**Created**: 2026-01-08
**Status**: Ready to implement
**Author**: Claude Code (Backend Architect)
