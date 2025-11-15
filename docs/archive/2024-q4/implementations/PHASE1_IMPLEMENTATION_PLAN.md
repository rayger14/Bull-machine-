# Phase 1: Structure Invalidation - Implementation Plan

Status: Ready to implement
Target: 5-10% DD reduction, maintain win rate

---

## Summary

Implementing structure invalidation exit checks to cut losers earlier when critical support/resistance breaks (OB/BB/FVG). This fills the Zeroika/Moneytaur gap of "exit on structure melt" before price runs further against us.

---

## Step 1: Verify Existing Structure Detection

**Files to Check:**
- `engine/smc/smc_engine.py` - SMC signal generation (OB, FVG detection)
- `bin/build_mtf_feature_store.py` - Feature extraction from SMC

**What to Look For:**
- OrderBlock detection (already exists based on grep results)
- FVG (Fair Value Gap) detection
- BOS (Break of Structure) flags
- What structure levels are already being computed

**Action:** Read these files to understand what's already available vs. what needs to be added.

---

## Step 2: Add Structure Level Features to Feature Store

**File:** `bin/build_mtf_feature_store.py`

**New Columns to Add (for 1H timeframe):**

```python
# In the section that processes SMC signals:

# Order Block levels (from SMC engine)
df['tf1h_ob_low'] = ...  # Nearest OB support level
df['tf1h_ob_high'] = ...  # Nearest OB resistance level

# Breaker Block levels (if SMC engine supports, else use OB)
df['tf1h_bb_low'] = ...  # Nearest BB support
df['tf1h_bb_high'] = ...  # Nearest BB resistance

# Fair Value Gap boundaries
df['tf1h_fvg_low'] = ...  # FVG support boundary
df['tf1h_fvg_high'] = ...  # FVG resistance boundary
df['tf1h_fvg_present'] = ...  # Boolean: Is there an active FVG?

# Break of Structure flags
df['tf1h_bos_bearish'] = ...  # BOS to downside occurred
df['tf1h_bos_bullish'] = ...  # BOS to upside occurred
```

**Implementation Notes:**
- These should be forward-looking (the CURRENT nearest levels, not historical)
- OB/BB levels should be the most recent refined levels (not all historical)
- FVG present = there's a gap within last N bars that hasn't been filled
- BOS flags = structure break occurred in last 1-3 bars

**Test:** Rebuild BTC feature store for a small window (e.g., 2024-09-01 to 2024-09-07) and verify new columns exist with reasonable values.

---

## Step 3: Implement Structure Invalidation Check

**File:** `bin/backtest_knowledge_v2.py`

**New Method to Add (after line ~557, before `run()` method):**

```python
def _check_structure_invalidation(self, row: pd.Series, trade: Trade) -> bool:
    """
    Exit if critical support/resistance structures break.

    Checks (in order):
    1. Order Block (OB) invalidation: BOS close below refined OB low (longs)
    2. Breaker Block (BB) penetration: Full body through BB without recovery
    3. Fair Value Gap (FVG) fill: Price melts through FVG with momentum

    Returns:
        True if any structure invalidated (exit immediately)
    """
    import logging
    logger = logging.getLogger(__name__)

    # Get structure levels from feature store
    ob_low = row.get('tf1h_ob_low', None)
    bb_low = row.get('tf1h_bb_low', None)
    fvg_low = row.get('tf1h_fvg_low', None)

    current_close = row['close']

    # Long trade checks
    if trade.direction == 1:
        # OB invalidation: Close below OB with BOS confirmation
        if ob_low is not None and not pd.isna(ob_low) and current_close < ob_low:
            bos_confirmed = row.get('tf1h_bos_bearish', False)
            if bos_confirmed:
                logger.info(f"Structure invalidation: OB broken at {current_close:.2f} < {ob_low:.2f}")
                return True

        # BB penetration: Full body below BB
        if bb_low is not None and not pd.isna(bb_low) and current_close < bb_low:
            body_penetration = (row['open'] + row['close']) / 2 < bb_low
            if body_penetration:
                logger.info(f"Structure invalidation: BB penetrated at {current_close:.2f}")
                return True

        # FVG melt: Price through FVG with momentum (RSI < 40)
        if fvg_low is not None and not pd.isna(fvg_low) and current_close < fvg_low:
            rsi = row.get('rsi_14', 50)
            if rsi < 40:  # Momentum confirmation
                logger.info(f"Structure invalidation: FVG melted with momentum (RSI={rsi:.1f})")
                return True

    # Short trade checks (inverse logic)
    else:
        ob_high = row.get('tf1h_ob_high', None)
        bb_high = row.get('tf1h_bb_high', None)
        fvg_high = row.get('tf1h_fvg_high', None)

        if ob_high is not None and not pd.isna(ob_high) and current_close > ob_high:
            bos_confirmed = row.get('tf1h_bos_bullish', False)
            if bos_confirmed:
                logger.info(f"Structure invalidation: OB broken at {current_close:.2f} > {ob_high:.2f}")
                return True

        if bb_high is not None and not pd.isna(bb_high) and current_close > bb_high:
            body_penetration = (row['open'] + row['close']) / 2 > bb_high
            if body_penetration:
                logger.info(f"Structure invalidation: BB penetrated at {current_close:.2f}")
                return True

        if fvg_high is not None and not pd.isna(fvg_high) and current_close > fvg_high:
            rsi = row.get('rsi_14', 50)
            if rsi > 60:
                logger.info(f"Structure invalidation: FVG melted with momentum (RSI={rsi:.1f})")
                return True

    return False
```

---

## Step 4: Insert Check in Exit Hierarchy

**File:** `bin/backtest_knowledge_v2.py`
**Method:** `check_exit_conditions()` (around line 482-557)

**Modification:** Insert structure check AFTER stop loss but BEFORE trailing stop (priority 2a):

```python
def check_exit_conditions(self, row: pd.Series, trade: Trade) -> Optional[Tuple[str, float]]:
    """Enhanced with structure invalidation check."""

    # 1. Stop loss hit (UNCHANGED)
    if trade.direction == 1:
        if current_price <= trade.initial_stop:
            return ("stop_loss", trade.initial_stop)
    else:
        if current_price >= trade.initial_stop:
            return ("stop_loss", trade.initial_stop)

    # 2. Partial exits (UNCHANGED)
    if self.params.use_smart_exits:
        # ... existing partial exit logic ...
        pass

    # 🆕 2a. Structure Invalidation Exit (NEW - Phase 1)
    if self._check_structure_invalidation(row, trade):
        return ("structure_invalidated", row['close'])

    # 3. Trailing stop (UNCHANGED)
    if pnl_r > 1.0 and self.params.use_smart_exits:
        # ... existing trailing logic ...
        pass

    # ... rest of existing checks unchanged ...
```

---

## Step 5: Backtest Phase 1 on BTC 2024

**Command:**
```bash
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import pandas as pd
import json
from bin.backtest_knowledge_v2 import KnowledgeParams, KnowledgeAwareBacktest

# Load BTC feature store (with new structure columns)
df = pd.read_parquet('data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet')

# Load BTC best config
with open('configs/v3_replay_2024/BTC_2024_best.json', 'r') as f:
    config = json.load(f)

# Convert to params
params = KnowledgeParams(
    wyckoff_weight=config['wyckoff_weight'],
    liquidity_weight=config['liquidity_weight'],
    momentum_weight=config['momentum_weight'],
    macro_weight=config['macro_weight'],
    pti_weight=config['pti_weight'],
    tier1_threshold=config['tier1_threshold'],
    tier2_threshold=config['tier2_threshold'],
    tier3_threshold=config['tier3_threshold'],
    require_m1m2_confirmation=config['require_m1m2_confirmation'],
    require_macro_alignment=config['require_macro_alignment'],
    atr_stop_mult=config['atr_stop_mult'],
    trailing_atr_mult=config['trailing_atr_mult'],
    max_hold_bars=config['max_hold_bars'],
    max_risk_pct=config['max_risk_pct'],
    volatility_scaling=config['volatility_scaling']
)

# Run backtest with Phase 1 enhancements
backtest = KnowledgeAwareBacktest(df, params, starting_capital=10000.0)
results = backtest.run()

# Compare to baseline
print('PHASE 1 RESULTS vs. BASELINE')
print('='*60)
print(f'Baseline (v2.0):')
print(f'  Total Trades: 31')
print(f'  Total PNL: \$5,715.29')
print(f'  Win Rate: 54.8%')
print(f'  Max DD: ~0%')
print()
print(f'Phase 1 (with structure invalidation):')
print(f'  Total Trades: {results[\"total_trades\"]}')
print(f'  Total PNL: \${results[\"total_pnl\"]:,.2f}')
print(f'  Win Rate: {results[\"win_rate\"]*100:.1f}%')
print(f'  Max DD: {results[\"max_drawdown\"]*100:.2f}%')
print()

# Count structure invalidation exits
structure_exits = [t for t in results['trades'] if t.exit_reason == 'structure_invalidated']
print(f'Structure Invalidation Exits: {len(structure_exits)} ({len(structure_exits)/results[\"total_trades\"]*100:.1f}%)')

# Expected improvement: DD reduction 5-10%, maintain or improve PNL
dd_reduction = (0.0 - results['max_drawdown']) / max(0.01, 0.0) * 100
pnl_change = (results['total_pnl'] - 5715.29) / 5715.29 * 100

print()
print('TARGET METRICS:')
print(f'  DD Reduction: {dd_reduction:.1f}% (target: 5-10%)')
print(f'  PNL Change: {pnl_change:+.1f}% (target: maintain or improve)')
print()

if dd_reduction >= 5 and pnl_change >= -5:
    print('✅ PHASE 1 ACCEPTANCE GATES PASSED')
else:
    print('❌ PHASE 1 NEEDS TUNING')
"
```

---

## Step 6: Validation & Analysis

**Check:**
1. Structure invalidation exits are firing (should see new exit reason in logs)
2. DD reduced by 5-10% compared to baseline
3. PNL maintained or improved (±5% acceptable)
4. Win rate maintained or improved
5. Trade count may decrease slightly (exiting earlier = fewer trades, expected)

**If Acceptance Gates Pass:**
- Commit changes with message: `feat(exits): Phase 1 structure invalidation exit`
- Update ENHANCED_EXIT_STRATEGIES_DESIGN.md with actual results
- Mark Phase 1 complete in todo list
- Proceed to Phase 2 (Pattern-Triggered Exits)

**If Tuning Needed:**
- Adjust thresholds (e.g., RSI momentum confirmation from 40 to 35)
- Try requiring 2/3 structure breaks instead of any 1
- Add minimum hold time (e.g., don't exit via structure in first 3 bars)

---

## Files to Modify

1. `bin/build_mtf_feature_store.py` - Add structure level features
2. `bin/backtest_knowledge_v2.py` - Add `_check_structure_invalidation()` method
3. `bin/backtest_knowledge_v2.py` - Insert check in `check_exit_conditions()`
4. `data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet` - Rebuild with new features

---

## Estimated Time

- Step 1 (Verify): 15 minutes
- Step 2 (Feature store): 1-2 hours (depending on SMC engine complexity)
- Step 3 (Implement method): 30 minutes
- Step 4 (Insert check): 15 minutes
- Step 5 (Backtest): 5 minutes
- Step 6 (Analysis): 30 minutes

**Total:** 3-4 hours

---

## Next Session Checklist

- [ ] Read `engine/smc/smc_engine.py` to understand OB/FVG detection
- [ ] Read `bin/build_mtf_feature_store.py` to see current SMC integration
- [ ] Add structure level columns to feature builder
- [ ] Rebuild BTC 2024 feature store
- [ ] Implement `_check_structure_invalidation()` method
- [ ] Insert check in exit hierarchy
- [ ] Run backtest and compare to baseline
- [ ] Analyze results and validate acceptance gates
- [ ] Commit if passed, tune if needed

---

**Status:** Design approved ✅ | Ready for implementation
