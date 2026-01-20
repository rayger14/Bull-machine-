# QUICK FIX GUIDE - Critical Blockers

**Purpose**: Step-by-step instructions to fix the 5 critical blockers identified in validation.

---

## FIX #1: Regime Data Bug (30 minutes)

### Problem
All trades showing `regime="unknown"` in CSV. Regime detection working but not passed to trade layer.

### Root Cause
`regime` field not in `PendingOrder` dataclass and not being passed from `RegimeService`.

### Fix Steps

1. **Edit backtest_full_engine_replay.py (line 52)**:
```python
@dataclass
class PendingOrder:
    """Order scheduled for execution on next bar."""
    archetype_id: str
    direction: str
    entry_bar_index: int
    signal_bar_index: int
    signal_time: pd.Timestamp
    confidence: float
    regime: str = "unknown"  # ADD THIS LINE
    entry_price: Optional[float] = None
    ...
```

2. **Edit backtest_full_engine_replay.py (line 69)**:
```python
@dataclass
class Position:
    """Active position being tracked."""
    symbol: str
    archetype_id: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    entry_bar_index: int
    size: float
    stop_loss: float
    take_profit: Optional[float]
    confidence: float
    regime: str = "unknown"  # ADD THIS LINE
    metadata: Dict = field(default_factory=dict)
```

3. **Edit backtest_full_engine_replay.py (around line 450 - when creating PendingOrder)**:
```python
# Find where PendingOrder is created (search for "PendingOrder(")
# Add regime parameter:
pending_order = PendingOrder(
    archetype_id=archetype_id,
    direction=direction,
    entry_bar_index=bar_index + 1,
    signal_bar_index=bar_index,
    signal_time=row['timestamp'],
    confidence=confidence,
    regime=self.regime_service.get_current_regime() if self.regime_service else "unknown",  # ADD THIS
    ...
)
```

4. **Edit backtest_full_engine_replay.py (around line 550 - when creating Position)**:
```python
# Find where Position is created from PendingOrder
position = Position(
    symbol=self.symbol,
    archetype_id=order.archetype_id,
    direction=order.direction,
    entry_time=row['timestamp'],
    entry_price=order.entry_price,
    entry_bar_index=bar_index,
    size=order.size,
    stop_loss=order.stop_loss,
    take_profit=order.take_profit,
    confidence=order.confidence,
    regime=order.regime,  # ADD THIS
    metadata=order.metadata
)
```

5. **Test**:
```bash
python3 bin/backtest_full_engine_replay.py
cat results/full_engine_backtest/trades_full.csv | grep -v "unknown"
# Should see actual regime values (risk_on, crisis, etc.)
```

---

## FIX #2: Reduce Regime Penalties (1 hour)

### Problem
Archetypes generating signals but not executing. Regime penalties likely too harsh.

### Current Penalties
- Wrong regime: -50% to -75% confidence penalty
- Transition period: -25% penalty
- Result: Confidence drops below execution threshold

### Fix Steps

1. **Find regime penalty config**:
```bash
grep -r "regime_penalty" engine/archetypes/
# or
grep -r "penalty.*0.50\|penalty.*0.75" engine/
```

2. **Reduce penalties** (in archetype logic or config):
```python
# OLD:
REGIME_PENALTIES = {
    'wrong_regime': 0.50,  # -50%
    'mismatch': 0.75,      # -75%
    'transition': 0.25     # -25%
}

# NEW (gentler):
REGIME_PENALTIES = {
    'wrong_regime': 0.20,  # -20%
    'mismatch': 0.40,      # -40%
    'transition': 0.10     # -10%
}
```

3. **Add debug logging** (in apply_regime_penalties function):
```python
logger.info(f"[Regime Penalty] {archetype}: {confidence:.3f} → {new_confidence:.3f} (penalty={penalty*100}%)")
```

4. **Rerun backtest**:
```bash
python3 bin/backtest_full_engine_replay.py 2>&1 | tee backtest_reduced_penalties.log
```

5. **Compare results**:
```bash
# Check if more archetypes firing
grep "ENTRY:" backtest_reduced_penalties.log | wc -l
# Should see 20-30+ trades instead of 9
```

---

## FIX #3: Disable S5 (30 minutes)

### Problem
S5 (long_squeeze) has 17% win rate, losing -$314.95. Dragging down performance.

### Fix Steps

1. **Edit backtest_full_engine_replay.py (line 154)**:
```python
self.archetype_logic = ArchetypeLogic(config={
    'use_archetypes': True,
    'enable_A': True,
    'enable_B': True,
    'enable_C': True,
    'enable_G': True,
    'enable_K': True,
    'enable_S1': True,
    'enable_S4': True,
    'enable_S5': False  # DISABLE S5
})
```

2. **Or disable in archetype_registry.yaml**:
```yaml
S5_long_squeeze:
  enabled: false  # Set to false
  direction: short
  ...
```

3. **Rerun backtest**:
```bash
python3 bin/backtest_full_engine_replay.py
```

4. **Check results**:
```bash
# Should see no long_squeeze trades
grep "long_squeeze" results/full_engine_backtest/trades_full.csv
# Empty result = success
```

5. **Analyze performance impact**:
```python
# Before: 9 trades, +$75 PnL, 33% win rate, 0.102 Sharpe
# After (expected): 3 trades, +$390 PnL, 67% win rate, higher Sharpe
```

---

## FIX #4: Add Debug Logging (2 hours)

### Problem
Need to identify which filter is blocking archetype trades (regime penalties? circuit breaker? confidence threshold?).

### Fix Steps

1. **Add logging to regime penalty application**:
```python
# In engine/archetypes/logic_v2_adapter.py or wherever regime penalties applied
def apply_regime_penalties(archetype, confidence, regime):
    original = confidence
    # ... apply penalty logic ...
    logger.info(f"[Regime Filter] {archetype}: {original:.3f} → {confidence:.3f} (regime={regime}, penalty={penalty*100:.1f}%)")
    return confidence
```

2. **Add logging to circuit breaker decisions**:
```python
# In engine/risk/circuit_breaker.py
def check_trade_allowed(self, signal):
    allowed = # ... decision logic ...
    if not allowed:
        logger.warning(f"[Circuit Breaker VETO] {signal.archetype}: Blocked (reason={reason})")
    else:
        logger.info(f"[Circuit Breaker PASS] {signal.archetype}: Approved")
    return allowed
```

3. **Add logging to confidence threshold check**:
```python
# In backtest_full_engine_replay.py (around signal generation)
if confidence < EXECUTION_THRESHOLD:
    logger.info(f"[Confidence Filter] {archetype}: {confidence:.3f} < {EXECUTION_THRESHOLD} - REJECTED")
else:
    logger.info(f"[Confidence Filter] {archetype}: {confidence:.3f} >= {EXECUTION_THRESHOLD} - APPROVED")
```

4. **Add logging to direction balance filter**:
```python
# In engine/risk/direction_balance.py
def scale_position_size(self, direction, size):
    original_size = size
    # ... scaling logic ...
    if scale_factor < 1.0:
        logger.warning(f"[Direction Balance] {direction}: size {original_size:.2f} → {size:.2f} (scaled by {scale_factor:.2f})")
    return size
```

5. **Rerun with debug logging**:
```bash
python3 bin/backtest_full_engine_replay.py 2>&1 | tee backtest_debug.log
```

6. **Analyze logs**:
```bash
# Check regime filter rejections
grep "\[Regime Filter\]" backtest_debug.log | grep "→ 0\." | head -20

# Check circuit breaker vetoes
grep "\[Circuit Breaker VETO\]" backtest_debug.log

# Check confidence rejections
grep "\[Confidence Filter\].*REJECTED" backtest_debug.log | wc -l

# Check direction balance scaling
grep "\[Direction Balance\]" backtest_debug.log | grep "scaled"
```

7. **Identify bottleneck**:
```bash
# Count rejections by filter
echo "Regime Filter:"
grep "\[Regime Filter\]" backtest_debug.log | grep "→ 0\." | wc -l

echo "Circuit Breaker:"
grep "\[Circuit Breaker VETO\]" backtest_debug.log | wc -l

echo "Confidence Threshold:"
grep "\[Confidence Filter\].*REJECTED" backtest_debug.log | wc -l

# Whichever has highest count is the bottleneck
```

---

## FIX #5: Investigate Archetype A & K (1 hour)

### Problem
Archetypes A and K supposed to be optimized but only K is firing. Check if production configs loaded.

### Fix Steps

1. **Check if production configs exist**:
```bash
ls -la configs/*production*.json
# Should see:
# - configs/system_a_production.json (or similar)
# - configs/system_k_production.json
```

2. **Check if backtest loading production configs**:
```bash
grep "production" bin/backtest_full_engine_replay.py
# Should see config loading logic
```

3. **If configs not loaded, edit backtest script**:
```python
# In backtest_full_engine_replay.py
self.archetype_logic = ArchetypeLogic(config={
    'use_archetypes': True,
    'enable_A': True,
    'enable_B': True,
    'enable_C': True,
    'enable_G': True,
    'enable_K': True,
    'enable_S1': True,
    'enable_S4': True,
    'enable_S5': False,
    # ADD PRODUCTION CONFIG LOADING:
    'A_config': load_json('configs/system_a_production.json'),
    'K_config': load_json('configs/system_k_production.json')
})
```

4. **Check Archetype A signal generation**:
```bash
# Add temporary debug logging
# In archetype A logic (engine/archetypes/a_spring.py or similar):
logger.info(f"[Archetype A Debug] Checking signal at {timestamp}: score={score}, threshold={threshold}")
```

5. **Rerun and check A signals**:
```bash
python3 bin/backtest_full_engine_replay.py 2>&1 | grep "Archetype A Debug" | head -20
# Should see A checking signals
```

6. **If A generating signals but not executing**:
```bash
# Go back to Fix #4 (debug logging) to identify bottleneck
```

---

## VALIDATION AFTER FIXES

### Run Complete Backtest
```bash
python3 bin/backtest_full_engine_replay.py 2>&1 | tee backtest_after_fixes.log
```

### Check Target Metrics
```bash
# 1. Total trades (target: 50-100)
grep "ENTRY:" backtest_after_fixes.log | wc -l

# 2. Archetypes active (target: 8+)
grep "ENTRY:" backtest_after_fixes.log | awk '{print $4}' | sort -u | wc -l

# 3. Short percentage (target: 30-40%)
echo "Short %:"
grep "ENTRY:.*short" backtest_after_fixes.log | wc -l
grep "ENTRY:" backtest_after_fixes.log | wc -l
# Calculate: shorts / total * 100

# 4. Win rate (target: 50%+)
grep "EXIT:" backtest_after_fixes.log | grep "PnL=\$[^-]" | wc -l  # Wins
grep "EXIT:" backtest_after_fixes.log | wc -l  # Total

# 5. Sharpe ratio (target: 0.6+)
tail -50 backtest_after_fixes.log | grep "sharpe_ratio"

# 6. Regime data populated?
tail -5 results/full_engine_backtest/trades_full.csv | cut -d',' -f13
# Should show regime names (not "unknown")
```

### Success Criteria
- [ ] 50+ trades (vs 9 before)
- [ ] 8+ archetypes active (vs 2 before)
- [ ] 30-40% shorts (vs 67% before)
- [ ] 50%+ win rate (vs 33% before)
- [ ] 0.6+ Sharpe (vs 0.102 before)
- [ ] Regime data populated (vs "unknown")

### If Still Not Meeting Targets
1. Review debug logs (Fix #4)
2. Further reduce regime penalties (try -10%/-20%)
3. Loosen circuit breaker thresholds
4. Lower confidence threshold
5. Disable other underperforming archetypes

---

## TROUBLESHOOTING

### Issue: Still only 2 archetypes firing
**Check**:
```bash
grep "\[Dedup\]" backtest_after_fixes.log | head -20
# Are other archetypes generating signals?
```
- If YES: Signals generated but blocked → Check debug logs (Fix #4)
- If NO: Archetypes not generating signals → Check archetype logic/configs

### Issue: Regime still showing "unknown"
**Check**:
```bash
grep "regime_service" bin/backtest_full_engine_replay.py
# Is regime_service being initialized?
```
- Verify `self.regime_service` not None
- Verify `get_current_regime()` method exists
- Add print statement: `print(f"Current regime: {self.regime_service.get_current_regime()}")`

### Issue: Sharpe still low
**Check**:
```bash
# Low Sharpe = low return relative to volatility
# Possible causes:
# 1. Still too few trades → Fix archetypes
# 2. Win rate still low → Check signal quality
# 3. S5 still active and losing → Verify disabled
# 4. Large losses → Tighten stop losses
```

### Issue: Too many shorts
**Check**:
```bash
grep "ENTRY:.*short" backtest_after_fixes.log | awk '{print $4}' | sort | uniq -c
# Which archetypes generating shorts?
```
- If S5 still active → Verify disabled (Fix #3)
- If other bear archetypes → Disable temporarily
- Focus on bull archetypes first (A, B, C, G, K)

---

## NEXT STEPS AFTER FIXES

1. **Validate Improvements** (1 hour)
   - Run analysis script
   - Compare before/after metrics
   - Document improvements

2. **Walk-Forward Validation** (1 week)
   - Test on 2022 crisis period
   - Test on 2023 Q1 bull recovery
   - Test on 2023 H2 mixed conditions
   - Verify parameter stability

3. **Paper Trading** (2-4 weeks)
   - Deploy with $1K capital
   - Monitor daily performance
   - Compare paper vs backtest

4. **Production Launch** (1 week)
   - Start with $1K real capital
   - Scale gradually
   - Monitor closely

---

**Document Created**: 2026-01-08
**Purpose**: Quick reference for fixing critical validation issues
**Estimated Time**: 4-6 hours total for all fixes
