# Domain Engine Debug Investigation Results

**Date:** 2025-12-11
**Mission:** Debug why domain engines aren't affecting S1 behavior despite complete implementation

## INVESTIGATION SUMMARY

### Root Cause: YES - IDENTIFIED

**Problem:** Feature flags ARE being read from config, but domain boost multipliers modify SCORES, not GATES.

### Evidence

1. **Feature Flags Are Working:**
   - Added logging at line 1756 in `logic_v2_adapter.py`
   - Config loading verified: `s1_core.json` has `enable_wyckoff=true`, all others false
   - Config loading verified: `s1_full.json` has all 6 engines enabled

2. **Domain Boost IS Calculated:**
   - Code at lines 1755-1953 executes when `use_wyckoff=true`
   - Wyckoff events (spring, SC, ST, LPS, etc.) trigger boost multipliers (1.3x - 2.5x)
   - SMC, temporal, HOB, macro engines similarly apply boosts when enabled

3. **Score Modification Confirmed:**
   - Line 1956: `score = score * domain_boost`
   - This modifies the score AFTER all gates have already passed
   - Score is used for ranking/sizing, NOT for pattern detection gates

### Architecture Issue

**Current Flow:**
```
1. Basic gates check (capitulation_depth, crisis, volume, wick, etc.) → True/False
2. IF gates pass → calculate base score
3. IF gates pass → apply domain_boost to score
4. Return (True, boosted_score, telemetry)
```

**Problem:**
- Domain engines ONLY apply multipliers to an already-passing pattern
- They NEVER veto patterns (except Wyckoff distribution hard veto at line 1769)
- Core (Wyckoff only) and Full (all 6 engines) produce SAME TRADES because:
  - Same basic gates → Same True/False decisions
  - Domain boost affects score ranking, not pattern matching

**Why Core = Full:**
- Both configs use identical S1 threshold parameters
- Both pass same `capitulation_depth_max`, `crisis_composite_min`, etc.
- Domain boosts  don't create NEW trades, just re-rank existing ones
- Result: 110 trades @ PF 0.32 (identical)

## FIX REQUIRED

**Option 1: Make Domain Engines Affect Gates (Recommended)**

Add domain-aware gate adjustments BEFORE score calculation:

```python
# BEFORE calculating score, apply domain-informed gate relaxation
if use_wyckoff and (wyckoff_spring_a or wyckoff_spring_b):
    # Spring events = relax capitulation gate by 20%
    cap_depth_threshold *= 0.80
    crisis_min_threshold *= 0.80

if use_smc and tf4h_bos_bullish:
    # 4H bullish structure = relax gates
    volume_z_min_threshold *= 0.85
```

This makes domain engines affect WHICH patterns fire, not just their scores.

**Option 2: Add Domain-Specific Entry Paths (Alternative)**

Create separate detection paths for high-conviction domain signals:

```python
# Add BEFORE main S1 logic
if use_wyckoff and wyckoff_spring_a:
    # Spring A = independent entry (looser gates)
    if capitulation_depth < -0.15:  # vs -0.20 normally
        return True, score * 2.5, {...}
```

**Option 3: Keep Current Architecture, Accept Limitation (Not Recommended)**

- Accept that domain engines only affect score/sizing
- Document that Core vs Full will have similar trade counts
- Rely on domain boost for position sizing differentiation only

## NEXT STEPS

1. **Decide on fix approach** (Option 1 recommended)
2. **Implement domain-aware gate adjustment** in `logic_v2_adapter.py`
3. **Re-run comparison:** Core should have ~110 trades, Full should have ~140 trades (estimate)
4. **Validate that Full catches more patterns** due to relaxed gates under domain confluence

## TEST ARTIFACTS

- **Debug logging added:** `engine/archetypes/logic_v2_adapter.py` lines 1756, 1958-1962
- **Test script created:** `bin/test_domain_engine_debug.py`
- **Config comparison:** `configs/variants/s1_core.json` vs `s1_full.json`

## CODE CHANGES

### 1. Added debug logging to logic_v2_adapter.py

**Line 1756:**
```python
logger.info(f"[DOMAIN_DEBUG] S1 Feature Flags: wyckoff={use_wyckoff}, smc={use_smc}, ...")
```

**Lines 1958-1962:**
```python
if domain_boost != 1.0 or len(domain_signals) > 0:
    logger.info(f"[DOMAIN_DEBUG] S1 Domain Boost Applied: {domain_boost:.2f}x | ...")
else:
    logger.info(f"[DOMAIN_DEBUG] S1 No Domain Boost: ...")
```

### 2. Created test_domain_engine_debug.py

- Loads s1_core.json and s1_full.json
- Scans first 200 bars of 2022 data
- Compares detection counts
- Purpose: Prove feature_flags are read but don't affect gates

## CONCLUSION

**Root cause identified:** Domain engines modify scores (multiplicative boosts) but NOT gates (True/False decisions). Feature flags work correctly, but architecture limits their impact to post-detection score adjustment.

**Recommended fix:** Implement Option 1 (domain-aware gate relaxation) to make engines affect WHICH patterns fire, creating meaningful difference between Core and Full variants.

**Impact:** This explains why variants show minimal performance differences in backtests - they detect the same patterns with different scores, rather than detecting different pattern sets.
