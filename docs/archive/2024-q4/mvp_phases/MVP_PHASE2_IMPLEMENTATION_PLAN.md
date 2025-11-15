# Phase 2: Complete Exit Stack - Implementation Plan

**Date**: 2025-10-20 04:00 AM  
**Status**: Ready to implement (no feature store rebuilds needed)  
**Complexity**: Policy-layer only - 2-3 hours

---

## TL;DR

Phase 2 adds 6 robust, trend-friendly exit mechanisms to the existing backtest engine. All are **policy-layer changes** requiring no feature store rebuilds. Expected to lift Profit Factor from 2.39 → 3.0+ without relying on noisy 1H structure flips.

---

## Phase 2 Exit Stack (6 Mechanisms)

### 1. Partial Profit Ladder ✅ (Partially Implemented)
**Status**: NEEDS ENHANCEMENT  
**File**: `bin/backtest_knowledge_v2.py:615-629`

**Current Implementation**:
```python
# Lines 621-629: Basic partial exit detection
if pnl_r >= 1.0 and not any(p['level'] == 'TP1' for p in trade.partial_exits):
    pass  # No actual sizing reduction!
```

**Required Changes**:
```python
# Phase 2.1: Enhanced Partial Profit Ladder
if pnl_r >= 1.0 and not hasattr(trade, 'tp1_taken'):
    # Take 1/3 off
    trade.position_size *= (2/3)
    trade.tp1_taken = True
    trade.partial_exits.append({'level': 'TP1', 'price': current_price, 'pnl': pnl_r})
    
    # Move stop to BE-ε (breakeven minus small epsilon)
    if self.params.breakeven_after_tp1:
        eps = trade.atr_at_entry * 0.1  # Small epsilon
        trade.initial_stop = trade.entry_price - eps * trade.direction
    
    logger.info(f"PARTIAL EXIT TP1 (+1R): 1/3 position closed, stop → BE-{eps:.2f}")

if pnl_r >= 2.0 and hasattr(trade, 'tp1_taken') and not hasattr(trade, 'tp2_taken'):
    # Take another 1/3 off (1/2 of remaining)
    trade.position_size *= 0.5
    trade.tp2_taken = True
    trade.partial_exits.append({'level': 'TP2', 'price': current_price, 'pnl': pnl_r})
    
    # Tighten trailing stop (ATR × k down to k - 0.5)
    self.params.trailing_atr_mult = max(1.5, self.params.trailing_atr_mult - 0.5)
    
    logger.info(f"PARTIAL EXIT TP2 (+2R): 1/3 position closed, trail tightened to {self.params.trailing_atr_mult}×ATR")
```

**Expected Impact**:
- Lock in profits early (reduce MAE)
- Let runners continue with tightened risk
- ~10% PNL improvement

---

### 2. Volatility-Aware Trailing Stop ✅ (Partially Implemented)
**Status**: NEEDS ENHANCEMENT  
**File**: `bin/backtest_knowledge_v2.py:636-647`

**Current Implementation**:
```python
# Lines 636-647: Static ATR trailing
trailing_stop = trade.entry_price + (trade.peak_profit - self.params.trailing_atr_mult * atr) * trade.direction
```

**Required Changes**:
```python
# Phase 2.2: Volatility-Aware Trailing with KAMA/ADX Bias
atr = row.get('atr_14', trade.atr_at_entry)
adx = row.get('adx_14', 20)
kama = row.get('kama_10', row['close'])
vix = row.get('vix', 20)  # From macro features

# Adaptive ATR multiple based on trend strength
if adx > 25 and kama > row['close'].shift(1):  # Strong uptrend
    atr_mult = self.params.trailing_atr_mult  # Allow looser (2.0)
elif adx < 20 or vix > 25:  # Weak trend or high VIX
    atr_mult = max(1.5, self.params.trailing_atr_mult - 0.5)  # Tighten
else:
    atr_mult = self.params.trailing_atr_mult  # Default

# Apply trailing with adaptive multiple
trail_distance = atr_mult * atr
trailing_stop = current_price - trail_distance * trade.direction

logger.debug(f"Trail: ADX={adx:.1f}, VIX={vix:.1f}, ATR mult={atr_mult:.1f}")
```

**Expected Impact**:
- Ride strong trends longer (ADX > 25)
- Exit choppy markets faster (ADX < 20)
- ~5% profit factor improvement

---

### 3. Adaptive Time Exit with M1/M2 Extension ✅ (Partially Implemented)
**Status**: NEEDS ENHANCEMENT  
**File**: `bin/backtest_knowledge_v2.py:662-673`

**Current Implementation**:
```python
# Lines 662-673: Basic max_hold or adaptive (not well-defined)
if bars_held >= max_hold_adjusted:
    return ("max_hold", current_price)
```

**Required Changes**:
```python
# Phase 2.3: Adaptive Time Exit with M1/M2 Extension Logic
bars_held = (row.name - trade.entry_time).total_seconds() / 3600  # Hours

# Base max_hold from asset type (BTC: 168h, ETH: 120h, SPY: 24h)
if 'BTC' in self.df.attrs.get('symbol', ''):
    base_max_hold = 168  # 7 days
elif 'ETH' in self.df.attrs.get('symbol', ''):
    base_max_hold = 120  # 5 days
elif 'SPY' in self.df.attrs.get('symbol', ''):
    base_max_hold = 24   # 1 day (adaptive 24-72h logic from v3.1)
else:
    base_max_hold = self.params.max_hold_bars

# Extend if M1/M2 strong and trade > +0.5R
m1_score = context.get('wyckoff_m1_score', 0.0)
m2_score = context.get('wyckoff_m2_score', 0.0)

if (m1_score > 0.5 or m2_score > 0.5) and pnl_r > 0.5:
    extension_factor = 1.5  # 50% extension
    max_hold_adjusted = base_max_hold * extension_factor
    logger.info(f"Max hold extended: {base_max_hold}h → {max_hold_adjusted}h (M1/M2 strong, PNL={pnl_r:.2f}R)")
else:
    max_hold_adjusted = base_max_hold

if bars_held >= max_hold_adjusted:
    return ("max_hold", current_price)
```

**Expected Impact**:
- Ride strong Wyckoff setups longer
- Exit weak setups faster
- ~8% win rate improvement

---

### 4. Signal Neutralization & Regime Flip ✅ (Partially Implemented)
**Status**: NEEDS ENHANCEMENT  
**File**: `bin/backtest_knowledge_v2.py:649-660`

**Current Implementation**:
```python
# Lines 649-652: Basic fusion drop check
if fusion_score < self.params.tier3_threshold:
    return ("signal_neutralized", current_price)

# Lines 658-660: Macro crisis only
if context.get('macro_regime') == 'crisis':
    return ("macro_crisis", current_price)
```

**Required Changes**:
```python
# Phase 2.4: Enhanced Signal Neutralization with Regime Flip
fusion_score, context = self.compute_advanced_fusion_score(row)

# Exit if fusion drops below re-entry threshold - δ
fusion_exit_delta = 0.05
fusion_exit_threshold = self.params.tier3_threshold - fusion_exit_delta

# AND check macro alignment flip
entry_macro_regime = trade.macro_regime  # From Trade.macro_regime
current_macro_regime = context.get('macro_regime')

if fusion_score < fusion_exit_threshold and current_macro_regime != entry_macro_regime:
    logger.info(f"Signal neutralized: fusion={fusion_score:.3f} (< {fusion_exit_threshold:.3f}), "
                f"macro flip {entry_macro_regime} → {current_macro_regime}")
    return ("signal_neutralized", current_price)
```

**Expected Impact**:
- Exit when thesis invalidates (fusion + regime)
- Avoid holding through regime changes
- ~5% drawdown reduction

---

### 5. Drawdown Guard (60% Retrace from Max Runup) ⚠️ NOT IMPLEMENTED
**Status**: MISSING  
**File**: `bin/backtest_knowledge_v2.py` - needs new method

**Implementation**:
```python
# Phase 2.5: Drawdown Guard (60% Retrace from Max Runup)
# Add BEFORE structure invalidation check (priority 2b)

def _check_drawdown_guard(self, row: pd.Series, trade: Trade) -> bool:
    """
    Exit if open PNL retraces > 60% from max runup after +1R.
    
    Protects locked-in profits from melting away.
    """
    current_price = row['close']
    pnl_pct = (current_price - trade.entry_price) / trade.entry_price * trade.direction
    pnl_r = pnl_pct / (self.params.atr_stop_mult * trade.atr_at_entry / trade.entry_price)
    
    max_runup = trade.peak_profit
    
    # Only apply after +1R achieved
    if max_runup < 1.0:
        return False
    
    # Check if current PNL has retraced > 60% from peak
    retrace_pct = (max_runup - pnl_r) / max_runup if max_runup > 0 else 0
    
    if retrace_pct > 0.6:
        logger.info(f"Drawdown guard: PNL retraced {retrace_pct*100:.1f}% from peak {max_runup:.2f}R → {pnl_r:.2f}R")
        return True
    
    return False
```

**In check_exit_conditions():**
```python
# 2b. Drawdown Guard (NEW - Phase 2.5)
if self._check_drawdown_guard(row, trade):
    return ("drawdown_guard", current_price)
```

**Expected Impact**:
- Protect winners from melting away
- Reduce max drawdown by ~10%
- ~3% profit factor improvement

---

### 6. Pattern Exits (2-Leg Pullback, Inside-Bar Expansion) ⚠️ NOT IMPLEMENTED
**Status**: MISSING  
**File**: `bin/backtest_knowledge_v2.py` - needs new method

**Implementation**:
```python
# Phase 2.6: Pattern Exits (Simple 2-Leg Pullback Break, Inside-Bar Expansion)
# Add AFTER drawdown guard (priority 2c)

def _check_pattern_exit(self, row: pd.Series, prev_row: pd.Series, trade: Trade) -> Tuple[bool, Optional[str]]:
    """
    Detect 2-leg pullback break or inside-bar expansion against position.
    
    Returns:
        (should_exit, pattern_name) or (False, None)
    """
    if prev_row is None:
        return False, None
    
    # 2-leg pullback break (failure of higher low / lower high)
    if trade.direction == 1:  # Long
        # Check for lower low after pullback
        if row['low'] < prev_row['low'] and prev_row['low'] < prev_row.shift(1).get('low', float('inf')):
            logger.info(f"Pattern exit: 2-leg pullback break (lower low)")
            return True, 'pattern_2leg_break'
    else:  # Short
        # Check for higher high after pullback
        if row['high'] > prev_row['high'] and prev_row['high'] > prev_row.shift(1).get('high', 0):
            logger.info(f"Pattern exit: 2-leg pullback break (higher high)")
            return True, 'pattern_2leg_break'
    
    # Inside-bar then expansion against position (fakeout end)
    atr = row.get('atr_14', prev_row.get('atr_14', 0))
    prev_range = abs(prev_row['high'] - prev_row['low'])
    
    # Inside bar: range < 50% ATR
    if prev_range < atr * 0.5:
        if trade.direction == 1 and row['close'] < prev_row['low']:
            logger.info(f"Pattern exit: inside-bar expansion (bearish)")
            return True, 'pattern_inside_expansion'
        elif trade.direction == -1 and row['close'] > prev_row['high']:
            logger.info(f"Pattern exit: inside-bar expansion (bullish)")
            return True, 'pattern_inside_expansion'
    
    return False, None
```

**In check_exit_conditions():**
```python
# 2c. Pattern Exits (NEW - Phase 2.6)
prev_row = self.df.iloc[bar_idx - 1] if bar_idx > 0 else None
is_pattern_exit, pattern_name = self._check_pattern_exit(row, prev_row, trade)
if is_pattern_exit:
    return (pattern_name, current_price)
```

**Expected Impact**:
- Exit on technical pattern failures
- Reduce false breakout losses
- ~2% win rate improvement

---

## Implementation Checklist

### Phase 2.1: Partial Profit Ladder
- [ ] Update partial exit logic to actually reduce position_size
- [ ] Add BE-ε stop adjustment after TP1
- [ ] Tighten trailing_atr_mult after TP2
- [ ] Add tp1_taken / tp2_taken flags to Trade

### Phase 2.2: Volatility-Aware Trailing
- [ ] Add ADX/KAMA/VIX checks in trailing stop logic
- [ ] Implement adaptive ATR multiple (1.5-2.0)
- [ ] Add debug logging for trail adjustments

### Phase 2.3: Adaptive Time Exit
- [ ] Implement asset-specific base_max_hold (BTC: 168h, ETH: 120h, SPY: 24h)
- [ ] Add M1/M2 extension logic (1.5x if strong + PNL > 0.5R)
- [ ] Log extension events

### Phase 2.4: Signal Neutralization Enhancement
- [ ] Add fusion_exit_delta (0.05)
- [ ] Check macro regime flip (entry vs current)
- [ ] Log neutralization with both conditions

### Phase 2.5: Drawdown Guard
- [ ] Create `_check_drawdown_guard()` method
- [ ] Insert before structure invalidation (priority 2b)
- [ ] Log retrace percentage and max runup

### Phase 2.6: Pattern Exits
- [ ] Create `_check_pattern_exit()` method
- [ ] Implement 2-leg pullback break detection
- [ ] Implement inside-bar expansion detection
- [ ] Insert after drawdown guard (priority 2c)

### Testing & Validation
- [ ] Run BTC 2024 backtest with Phase 2 complete stack
- [ ] Compare to baseline (31 trades, $5,715 PNL, 2.39 PF)
- [ ] Validate exit reason distribution (target 15-20% each for stop_loss, max_hold, partials)
- [ ] Check that structure exits remain < 20%

---

## Expected Results (Phase 2 Complete)

| Metric            | Baseline | Phase 2 Target | Improvement |
|-------------------|----------|----------------|-------------|
| Total PNL         | $5,715   | $6,500-7,000   | +15-20%     |
| Profit Factor     | 2.39     | 3.0-3.5        | +30-45%     |
| Win Rate          | 54.8%    | 60-65%         | +10-15%     |
| Max Drawdown      | ~0%      | -2% to -3%     | Controlled  |
| Trade Count       | 31       | 28-35          | ±10%        |

**Exit Reason Distribution (Target)**:
- stop_loss: 20-25%
- max_hold: 20-25%
- partial_tp1: 15-20%
- partial_tp2: 10-15%
- signal_neutralized: 10-15%
- drawdown_guard: 5-10%
- pattern_exit: 5-10%
- structure_invalidated: < 10%

---

## Acceptance Gates

1. ✅ PNL improvement ≥ +10% vs baseline
2. ✅ Profit factor ≥ 2.8
3. ✅ Win rate ≥ 58%
4. ✅ Max drawdown ≤ 5%
5. ✅ Exit diversity: No single reason > 30%
6. ✅ Structure exits < 20% of total

---

## Timeline

- **Phase 2.1-2.4** (enhancements): 1 hour
- **Phase 2.5-2.6** (new methods): 1 hour
- **Testing & validation**: 30 min
- **Total**: 2.5-3 hours

---

## Notes

- All changes are **policy-layer only** - no feature store rebuilds needed
- Existing SMC columns from Phase 1 remain available for future use
- Phase 1 structure exits can be disabled via config flag if Phase 2 proves superior
- After Phase 2 validation, can tune both Phase 1 + Phase 2 together for optimal mix

---

**Status**: Implementation plan ready ✅  
**Next Step**: Implement Phase 2.1-2.6 in sequence, test after each
**ETA**: 2.5-3 hours
