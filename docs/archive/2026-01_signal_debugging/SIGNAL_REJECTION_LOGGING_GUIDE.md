# Signal Rejection Logging Implementation Guide

## Problem
97.5% of signals are being rejected (357 signals → 9 trades). We need comprehensive logging to understand why.

## Solution: Multi-Layered Logging Approach

### Already Implemented ✅
The backtest file (`bin/backtest_full_engine_replay.py`) already has:
1. **Rejection stats tracking** (lines 154-169):
   - `self.rejection_stats` dictionary
   - `self.signal_generation_count`
   - `self.signal_acceptance_count`

2. **Signal generation logging** (line 447):
   - Logs when signals are generated with confidence, direction, regime

3. **Cooldown rejection logging** (lines 407-411):
   - Logs when signals rejected due to cooldown

4. **Regime mismatch logging** (lines 417-421):
   - Logs hard regime rejections

5. **Already in position logging** (lines 426-430):
   - Logs when archetype already has active position

6. **Feature data missing logging** (line 461-462):
   - Logs archetype evaluation failures

### Needs to be Added 🔧

#### 1. Waterfall Logging in `_process_signal` (line ~570)

Add after `direction = signal['direction']`:

```python
regime = signal['regime']
base_confidence = signal['confidence']

# LOG CONFIDENCE WATERFALL
logger.info(f"[Confidence Waterfall] {archetype_id} {direction}:")
logger.info(f"  Base confidence: {base_confidence:.3f}")
```

#### 2. Regime Confidence Rejection Tracking (line ~580)

Replace:
```python
if regime_confidence < 0.50:
    confidence_scale = 0.65
    logger.warning(
        f"[Regime Confidence] LOW {regime_confidence:.2f} → "
        f"scaling {archetype_id} confidence to {confidence_scale*100}%"
    )
```

With:
```python
if regime_confidence < 0.50:
    confidence_scale = 0.65
    confidence *= confidence_scale
    logger.warning(
        f"[REJECT: Regime Confidence Low] {archetype_id}: "
        f"regime_conf={regime_confidence:.2f} → scale to {confidence_scale:.0%}"
    )
    logger.info(f"  After regime confidence scaling: {confidence:.3f}")
    logger.info(f"  Result: REJECT (low regime confidence)")
    self.rejection_stats['regime_confidence_low'] += 1
    return  # Early exit
```

#### 3. Regime Transition Rejection Tracking (line ~600)

Replace:
```python
if regime_transition:
    confidence *= 0.75
    logger.warning(
        f"[Regime Transition] Detected - further reducing {archetype_id} "
        f"confidence by 25%"
    )
    signal['metadata']['transition_penalty'] = 0.75
```

With:
```python
if regime_transition:
    transition_scale = 0.75
    confidence *= transition_scale
    logger.warning(f"[REJECT: Regime Transition] {archetype_id}: transition detected")
    logger.info(f"  After transition penalty ({transition_scale:.0%}): {confidence:.3f}")
    logger.info(f"  Result: REJECT (regime transition)")
    self.rejection_stats['regime_transition'] += 1
    return  # Early exit
```

#### 4. Regime Penalty Logging Enhancement (line ~615)

Replace:
```python
if 'all' not in allowed_regimes and regime not in allowed_regimes:
    confidence *= 0.5
    logger.debug(f"Regime penalty: {archetype_id} in {regime} (expected {allowed_regimes})")
```

With:
```python
if 'all' not in allowed_regimes and regime not in allowed_regimes:
    regime_penalty = 0.50
    confidence *= (1 - regime_penalty)
    logger.warning(
        f"[Regime Penalty] {archetype_id}: regime={regime}, "
        f"allowed={allowed_regimes}, penalty={regime_penalty:.0%}"
    )
    logger.info(f"  After regime penalty (-{regime_penalty:.0%}): {confidence:.3f}")
    self.rejection_stats['regime_penalty'] += 1
    # Continue through pipeline (don't return)
```

#### 5. Circuit Breaker Rejection (line ~624)

After `logger.warning(f"Circuit breaker HALT - rejecting {archetype_id}")`:
```python
logger.info(f"  Result: REJECT (circuit breaker)")
self.rejection_stats['circuit_breaker'] += 1
```

#### 6. Low Confidence Rejection (line ~632)

Replace:
```python
if confidence < 0.3:
    logger.debug(
        f"[Pipeline Reject] {archetype_id} confidence too low after pipeline: {confidence:.2f}"
    )
    return
```

With:
```python
min_confidence = 0.3
logger.info(f"  Final confidence: {confidence:.3f}")
logger.info(f"  Minimum required: {min_confidence:.3f}")

if confidence < min_confidence:
    logger.warning(
        f"[REJECT: Low Confidence] {archetype_id}: {confidence:.3f} < {min_confidence:.3f}"
    )
    logger.info(f"  Result: REJECT (too low after pipeline)")
    self.rejection_stats['low_confidence'] += 1
    return
```

#### 7. Position Limit Rejection (line ~639)

Replace:
```python
if len(self.positions) >= self.max_positions:
    logger.debug(f"Max positions reached ({self.max_positions}) - rejecting {archetype_id}")
    return
```

With:
```python
if len(self.positions) >= self.max_positions:
    logger.warning(
        f"[REJECT: Position Limit] {archetype_id}: "
        f"{len(self.positions)}/{self.max_positions} positions"
    )
    logger.info(f"  Result: REJECT (max positions)")
    self.rejection_stats['position_limit'] += 1
    return
```

#### 8. Signal Acceptance Logging (line ~643)

Before `order = PendingOrder(...)`:
```python
# SUCCESS - Signal accepted
logger.info(f"  Result: ACCEPT - scheduling for execution")
self.signal_acceptance_count += 1

logger.info(
    f"[ACCEPT] {archetype_id} {direction}: final_conf={confidence:.3f}, "
    f"scheduled for bar {bar_index+1}"
)
```

#### 9. Print Rejection Statistics at End (after line 369)

Add method to class:
```python
def print_rejection_analysis(self):
    """Print comprehensive rejection analysis."""
    logger.info("\n" + "="*80)
    logger.info("SIGNAL REJECTION ANALYSIS")
    logger.info("="*80)

    total_signals = self.signal_generation_count
    total_accepted = self.signal_acceptance_count
    total_rejected = total_signals - total_accepted

    logger.info(f"\n## Summary")
    logger.info(f"- Total signals generated: {total_signals}")
    logger.info(f"- Signals accepted: {total_accepted} ({total_accepted/total_signals*100:.1f}%)")
    logger.info(f"- Signals rejected: {total_rejected} ({total_rejected/total_signals*100:.1f}%)")

    logger.info(f"\n## Rejection Breakdown")
    rejection_total = sum(self.rejection_stats.values())

    sorted_rejections = sorted(
        self.rejection_stats.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for i, (reason, count) in enumerate(sorted_rejections, 1):
        if count == 0:
            continue
        pct = count / rejection_total * 100 if rejection_total > 0 else 0
        marker = " ← TOP CAUSE" if i == 1 else ""
        logger.info(f"{i}. {reason.replace('_', ' ').title()}: {count} ({pct:.1f}%){marker}")

    logger.info("="*80)
```

Then call it at line 369 (after `results = self._compute_results(test_data)`):
```python
# Print rejection analysis
self.print_rejection_analysis()
```

## Usage

### Option 1: Use Enhanced Debug Wrapper (Recommended)
```bash
# Sanity test with debug logging
python3 bin/backtest_debug_enhanced.py --debug --period sanity

# Full analysis
python3 bin/backtest_debug_enhanced.py --debug --period q1_2023
```

This wrapper:
- ✅ Already implemented and ready to use
- ✅ Captures all logging to `backtest_debug.log`
- ✅ Prints rejection analysis automatically
- ✅ Exports rejection stats to JSON
- ✅ No need to modify core backtest file

### Option 2: Manual Logging Modifications
Apply the 9 changes listed above to `bin/backtest_full_engine_replay.py`.

Then run:
```bash
python3 bin/backtest_full_engine_replay.py 2>&1 | tee backtest_debug.log
```

## Expected Output

```
[Confidence Waterfall] Archetype_B long:
  Base confidence: 0.850
  After regime confidence (HIGH - no penalty): 0.850
  After regime penalty (-50%): 0.425
  Final confidence: 0.425
  Minimum required: 0.300
  Result: ACCEPT - scheduling for execution

[ACCEPT] Archetype_B long: final_conf=0.425, scheduled for bar 1234
```

Or for rejections:
```
[Confidence Waterfall] Archetype_H long:
  Base confidence: 0.650
  After regime confidence scaling (80%): 0.520
  After regime penalty (-50%): 0.260
  Final confidence: 0.260
  Minimum required: 0.300
  Result: REJECT (too low after pipeline)

[REJECT: Low Confidence] Archetype_H: 0.260 < 0.300
```

## Analysis Report Template

After running, create `SIGNAL_REJECTION_ANALYSIS_REPORT.md` with:

```markdown
# Signal Rejection Analysis

## Summary
- Total signals: 357
- Accepted: 9 (2.5%)
- Rejected: 348 (97.5%)

## Rejection Breakdown
1. Regime Penalty: XXX (XX.X%) ← TOP CAUSE
2. Low Confidence: XXX (XX.X%)
3. Regime Confidence Low: XXX (XX.X%)
4. Circuit Breaker: XXX (XX.X%)
5. Position Limit: XXX (XX.X%)
6. Cooldown: XXX (XX.X%)
7. Already in Position: XXX (XX.X%)

## Root Cause Analysis
[Based on the breakdown, identify the top 2-3 causes]

## Recommendations
1. [Action item based on top cause]
2. [Action item based on 2nd cause]
3. [Expected improvement]
```

## Success Criteria
- ✅ Can identify top 3 rejection causes with percentages
- ✅ Waterfall logging shows exact confidence degradation path
- ✅ Debug log saved for analysis
- ✅ Diagnostic report created
- ✅ Ready to validate fixes from parallel agent
