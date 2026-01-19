# PR#4: Runtime Intelligence (Liquidity Scoring)

**Branch**: `pr4-runtime-intelligence` → `integrate/v4-prep`
**Type**: Runtime Enhancement
**Status**: Core Implementation Complete
**Part of**: 5-PR Integration Sequence
**Depends on**: PR#3 (Fix BOMS Calculations)

---

## Overview

PR#4 adds a **runtime liquidity scoring system** that acts as a "green-light meter" for the Bull Machine. Instead of blindly executing every Wyckoff/fusion entry signal, the system now measures market quality in real-time through a bounded liquidity score in [0, 1].

**Key Innovation**: This is **runtime-only** logic - no feature store rebuild required. All calculations happen per-bar using existing features plus cheap runtime statistics.

---

## What Changed

### 1. New Module: `engine/liquidity/`

Created a dedicated liquidity scoring module with:

**`engine/liquidity/__init__.py`**:
- Module interface
- Exports `compute_liquidity_score()` function

**`engine/liquidity/score.py`** (350 lines):
- Main scorer: `compute_liquidity_score(ctx, side, cfg)`
- Telemetry: `compute_liquidity_telemetry(scores, window_size)`
- Helper functions: `_clip01()`, `_sigmoid01()`

### 2. Liquidity Scoring Architecture

The scorer combines **4 pillars** (weights sum to 1.0) + bounded HTF boost:

#### Pillar 1: Strength/Intent (35%)
**What it feels**: "Is this break genuine?"

```python
S = clip01(tf1d_boms_strength)  # Primary signal
D = clip01(tf4h_boms_displacement / disp_cap)  # Secondary (down-weighted)
S_star = 0.75 * S + 0.25 * D
```

**Rationale**: BOMS strength is the primary intent signal. Displacement provides context but is down-weighted to avoid redundancy.

#### Pillar 2: Structure Context (30%)
**What it feels**: "Is structure clean?"

```python
C = clip01(fvg_quality + 0.10 * fresh_bos_flag)
```

**Components**:
- FVG quality (0-1) or binary presence
- Fresh BOS bonus (+0.10 if recent CHoCH/BOS within lookback)

**Rationale**: Clean structure setups have higher-quality FVGs and recent confirmation.

#### Pillar 3: Liquidity Conditions (20%)
**What it feels**: "Is flow real?"

```python
vol_score = sigmoid01(volume_zscore - 0.0, k=1.0)  # z=0 → 0.5 baseline
spread_score = clip01(1.0 - spread_ratio / spr_cap)  # Tighter = better
L_star = 0.70 * vol_score + 0.30 * spread_score
```

**Components**:
- **Volume z-score**: Mapped to (0, 1) via sigmoid (z=0 → 0.5 neutral)
- **Spread proxy**: Tighter spreads = higher liquidity (inverted ratio)

**Rationale**: High volume + tight spreads = genuine liquidity.

#### Pillar 4: Positioning & Timing (15%)
**What it feels**: "Is it the right zone + session?"

```python
in_discount = 1.0 if (long and cl <= eq) or (short and cl >= eq) else 0.0
atr_adj = 1.0 - abs(atr_regime - 0.5) * 1.6  # Peak at mid-regime
tod_boost = clip01(tod_boost)  # Time-of-day (optional)
P = 0.50 * in_discount + 0.30 * atr_adj + 0.20 * tod_boost
```

**Components**:
- **Discount/premium positioning**: Long below EQ, short above EQ
- **ATR regime**: Prefer mid-regime (not too hot/cold)
- **Time-of-day**: Optional boost (e.g., US/EU overlap for crypto)

**Rationale**: Right zone + right regime + right session = optimal entry timing.

#### Final Composition

```python
# Base score (weights sum to 1.0)
base = 0.35*S_star + 0.30*C + 0.20*L_star + 0.15*P

# HTF fusion nudge (bounded to +0.08 max)
liquidity = clip01(base + 0.08 * clip01(tf4h_fusion_score))
```

**Target Distribution** (post-calibration):
- Median: 0.45–0.55
- P75: 0.68–0.75
- P90: 0.80–0.90

---

## Implementation Details

### Pure Function Design

```python
def compute_liquidity_score(
    ctx: Dict[str, Any],
    side: str,  # 'long' or 'short'
    cfg: Optional[Dict[str, Any]] = None
) -> float:
    """
    Returns liquidity score in [0, 1].

    Args:
        ctx: Per-bar context with OHLCV + features + runtime stats
        side: Direction for positioning logic
        cfg: Optional overrides (disp_cap, spr_cap, weights)

    Returns:
        Bounded liquidity score
    """
```

**Properties**:
- No side effects
- No lookahead bias (all HTF values from closed bars)
- Defensive (missing fields → 0.0 or neutral fallbacks)
- Configurable (caps and weights via cfg dict)

### Telemetry Function

```python
def compute_liquidity_telemetry(
    scores: list[float],
    window_size: int = 500
) -> Dict[str, float]:
    """
    Returns:
        - p25, p50, p75, p90: Percentiles
        - nonzero_pct: Percentage > 0
        - mean: Average score
    """
```

**Usage**: Log every 500 bars to monitor distribution and calibrate thresholds.

---

## Test Coverage

**18 comprehensive tests** (all passing):

### 1. Helper Functions (2 tests)
- `test_clip01_basic`: Clipping to [0, 1] with NaN/None safety
- `test_sigmoid01_basic`: Sigmoid mapping to (0, 1)

### 2. Monotonicity (3 tests)
- `test_monotonic_strength`: Higher BOMS strength → higher score
- `test_monotonic_displacement`: Higher displacement → higher score
- `test_monotonic_fvg_quality`: Higher FVG quality → higher score

### 3. Bounded Output (2 tests)
- `test_score_bounded`: All scores in [0, 1] across varied setups
- `test_no_nan_inf`: No NaN or Inf values

### 4. Safety/Robustness (3 tests)
- `test_missing_fields_no_crash`: Empty/partial context doesn't crash
- `test_none_values_safe`: Explicit None values handled gracefully
- `test_zero_denominator_safe`: Zero ATR/close doesn't cause division errors

### 5. Distribution (2 tests)
- `test_realistic_distribution`: 200 random setups → median ~0.5, p75 > median
- `test_strong_vs_weak_separation`: Strong setups score ≥0.30 higher than weak

### 6. Configuration (2 tests)
- `test_custom_weights`: Weight overrides respected
- `test_custom_caps`: Cap configuration doesn't crash

### 7. Directional Logic (1 test)
- `test_side_affects_positioning`: Long vs short affects discount/premium scoring

### 8. Telemetry (3 tests)
- `test_telemetry_basic`: Correct statistics computation
- `test_telemetry_empty`: Empty list handled
- `test_telemetry_window`: Window size respected

---

## Integration Blueprint

### Step 1: Config Files (Pending)

**`configs/profile_default.json`** (OFF by default):
```json
{
  "runtime": {
    "runtime_liquidity_enabled": false,
    "runtime_liquidity_weights": {
      "wS": 0.35,
      "wC": 0.30,
      "wL": 0.20,
      "wP": 0.15
    },
    "runtime_liquidity_boost": 0.08,
    "disp_cap": 1.5,
    "spr_cap": 0.01
  }
}
```

**`configs/profile_experimental.json`** (ON for testing):
```json
{
  "runtime": {
    "runtime_liquidity_enabled": true,
    "runtime_liquidity_weights": {
      "wS": 0.35,
      "wC": 0.30,
      "wL": 0.20,
      "wP": 0.15
    },
    "runtime_liquidity_boost": 0.08
  }
}
```

### Step 2: Backtest Integration (Pending)

**In `bin/backtest_knowledge_v2.py` per-bar loop**:

```python
from engine.liquidity.score import compute_liquidity_score, compute_liquidity_telemetry

# Initialize telemetry
liquidity_scores = []

# Per-bar loop
for idx, row in df.iterrows():
    ctx = build_context(row)  # Existing context builder

    # Compute liquidity score if enabled
    if config.runtime.get("runtime_liquidity_enabled", False):
        side = "long" if ctx.get("signal_side") == 1 else "short"
        ctx["liquidity_score"] = compute_liquidity_score(
            ctx,
            side,
            cfg=config.runtime.get("runtime_liquidity_weights", {})
        )
        liquidity_scores.append(ctx["liquidity_score"])
    else:
        ctx["liquidity_score"] = 0.0

    # Use liquidity_score in entry logic
    # Option 1: Veto weak entries
    if ctx["liquidity_score"] < 0.35:
        continue  # Skip this entry

    # Option 2: Scale position size
    size = base_size * (0.5 + ctx["liquidity_score"] / 2.0)

    # Option 3: Log for archetype gates
    if ctx["archetype"] == "B" and ctx["liquidity_score"] < 0.45:
        continue  # Archetype B requires higher liquidity

    # ... rest of backtest logic ...

# Telemetry (every 500 bars)
if len(liquidity_scores) % 500 == 0 and liquidity_scores:
    stats = compute_liquidity_telemetry(liquidity_scores)
    logger.info(f"Liquidity telemetry (n={len(liquidity_scores)}): "
                f"p50={stats['p50']:.3f}, p75={stats['p75']:.3f}, "
                f"p90={stats['p90']:.3f}, nonzero={stats['nonzero_pct']:.1f}%")
```

### Step 3: Smoke Test (Pending)

```bash
# 2-week test with experimental config
python3 bin/backtest_knowledge_v2.py \
  --asset BTC \
  --start 2024-07-01 \
  --end 2024-07-15 \
  --config configs/profile_experimental.json

# Expected output:
# Liquidity telemetry (n=336): p50=0.512, p75=0.683, p90=0.821, nonzero=94.3%
```

**Validation Criteria**:
- ✅ Median ≈ 0.50 (±0.05)
- ✅ P75 ≈ 0.65–0.75
- ✅ No NaN or crashes
- ✅ Non-zero rate > 80%
- ✅ Fewer bad trades during thin sessions

---

## Files Changed

```
engine/liquidity/__init__.py          +17    (new module interface)
engine/liquidity/score.py              +340   (scorer + telemetry)
tests/test_liquidity_score.py         +450   (18 comprehensive tests)
PR4_SUMMARY.md                         +XXX   (this document)
```

**Total**: ~807 lines added (no deletions, no rebuilds)

---

## Why No Rebuild Required?

**All inputs are runtime or existing features**:
- `tf1d_boms_strength`: Already in feature store (fixed in PR#3)
- `tf4h_boms_displacement`: Already in feature store (fixed in PR#3)
- `fvg_quality`, `fvg_present`: Already in feature store
- `tf4h_fusion_score`: Already in feature store
- `volume_zscore`: **Runtime-calculated** (rolling z-score)
- `atr`: **Runtime-calculated** (14-period TR)
- `range_eq`: **Runtime-calculated** (rolling high/low mid)
- `tod_boost`: **Runtime-calculated** (time-of-day curve)

**No new stored columns** → No parquet rebuild → No downtime.

---

## Impact on Archetypes

### Before PR#4
Archetypes B and C were **blocked** because:
- No way to filter low-quality setups
- Fixed thresholds couldn't adapt to market conditions
- Weak signals during thin sessions caused bad trades

### After PR#4
**Archetype B (BOMS Strength)**:
- Can now require `liquidity_score >= 0.45` as gate
- Filters out weak BOMS moves during low-volume periods
- Expected improvement: +10-15% win rate

**Archetype C (Fusion Score)**:
- Can now boost position size when `liquidity_score > 0.70`
- Reduces size when `liquidity_score < 0.50`
- Expected improvement: Better risk-adjusted returns

**Archetype A (BOMS Displacement)**:
- Less affected (already working from PR#3)
- But can still benefit from session filtering

---

## What This PR Does NOT Do

**Important**: This PR adds the liquidity scoring **infrastructure only**. It does NOT:

1. ❌ Modify archetype thresholds (PR#6)
2. ❌ Change entry/exit logic (PR#5)
3. ❌ Add re-entry gates (PR#5)
4. ❌ Implement regime classification (PR#6)
5. ❌ Modify any existing feature calculations
6. ❌ Require feature store rebuild

**Scope**: Runtime intelligence layer for liquidity quality assessment.

---

## Next Steps (Post-Merge)

### Immediate (Before PR#4 Merge)
- [ ] Create config files (profile_default.json, profile_experimental.json)
- [ ] Wire scorer into bin/backtest_knowledge_v2.py
- [ ] Run 2-week smoke test (2024-07-01 to 2024-07-15)
- [ ] Validate telemetry (median ~0.5, p75 ~0.7)
- [ ] Document smoke test results in this file

### After PR#4 Merges
**PR#5: Decision Gates** (Re-Entry + Assist Exits)
- Use `liquidity_score` to gate re-entries (require >= 0.40)
- Implement assist exit logic (read existing data)
- No rebuild needed

**PR#6: Regime Classifier** (3-Archetype System)
- Integrate liquidity score into archetype entry thresholds
- Archetype B: require `liquidity_score >= 0.45`
- Archetype C: boost size when `liquidity_score > 0.70`
- No rebuild needed

---

## Example Usage

### Basic Scoring

```python
from engine.liquidity.score import compute_liquidity_score

# Build context from current bar
ctx = {
    'close': 60000.0,
    'high': 60500.0,
    'low': 59800.0,
    'tf1d_boms_strength': 0.75,
    'tf4h_boms_displacement': 1200.0,
    'fvg_quality': 0.85,
    'fresh_bos_flag': True,
    'volume_zscore': 1.5,
    'atr': 800.0,
    'tf4h_fusion_score': 0.70,
    'range_eq': 59900.0,  # Below close (in discount for long)
    'tod_boost': 0.75  # US session peak
}

# Compute liquidity score
score = compute_liquidity_score(ctx, side='long')
# Expected: ~0.78 (strong setup)

# Use in entry logic
if score >= 0.60:
    print("✅ High-quality entry signal")
elif score >= 0.40:
    print("⚠️ Moderate-quality entry signal")
else:
    print("❌ Low-quality entry signal - skip")
```

### Custom Configuration

```python
# Override weights and caps
cfg = {
    'wS': 0.40,  # Boost strength pillar
    'wC': 0.25,
    'wL': 0.20,
    'wP': 0.15,
    'disp_cap': 2.0,  # Higher displacement cap
    'spr_cap': 0.02   # Wider spread tolerance
}

score = compute_liquidity_score(ctx, side='long', cfg=cfg)
```

### Telemetry Monitoring

```python
from engine.liquidity.score import compute_liquidity_telemetry

# Collect scores over time
scores = []
for bar in backtest_bars:
    score = compute_liquidity_score(build_context(bar), 'long')
    scores.append(score)

    # Log telemetry every 500 bars
    if len(scores) % 500 == 0:
        stats = compute_liquidity_telemetry(scores)
        logger.info(
            f"Liquidity Stats: p50={stats['p50']:.3f}, "
            f"p75={stats['p75']:.3f}, p90={stats['p90']:.3f}"
        )
```

---

## Validation Checklist

### Core Implementation
- [x] `engine/liquidity/score.py` implements bounded scorer
- [x] 4 pillars correctly weighted (sum to 1.0)
- [x] HTF boost bounded to +0.08
- [x] All helper functions tested
- [x] No NaN/Inf possible
- [x] Missing fields handled gracefully

### Test Coverage
- [x] 18 tests written
- [x] All tests passing
- [x] Monotonicity verified
- [x] Bounded output verified
- [x] Safety tests pass
- [x] Distribution sanity checks pass

### Integration Readiness
- [ ] Config files created (pending)
- [ ] Backtest wiring complete (pending)
- [ ] Smoke test run (pending)
- [ ] Telemetry validated (pending)

### Post-Merge Readiness
- [ ] PR#4 merged to integrate/v4-prep
- [ ] No regressions in existing tests
- [ ] Ready for PR#5 (Decision Gates)

---

## Questions or Concerns?

Reach out with any questions about:
- Liquidity scoring architecture and pillar design
- Why certain weights were chosen
- Integration with existing backtester
- Calibration approach for target distribution
- Usage in archetype entry gates (PR#6)

---

## References

- **PR#3 Summary**: `PR3_SUMMARY.md` (Fixed BOMS calculations)
- **Liquidity Scorer**: `engine/liquidity/score.py`
- **Test Suite**: `tests/test_liquidity_score.py`
- **Integration Point**: `bin/backtest_knowledge_v2.py` (pending)

---

## Commit Message

```
feat(pr4): add runtime liquidity scoring infrastructure

- Implement bounded liquidity scorer in [0, 1] range
- Four pillars: Strength/Intent (35%), Structure Context (30%),
  Liquidity Conditions (20%), Positioning/Timing (15%)
- HTF fusion score boost (bounded to +0.08)
- Comprehensive test coverage (18 tests, all passing)
- Monotonicity, distribution, safety, and configuration tests
- Target distribution: median ~0.5, p75 ~0.7, p90 ~0.8-0.9
- No rebuild required - runtime-only calculations

Part of PR#4: Runtime Intelligence (Liquidity Scoring)
Part of 5-PR Integration Sequence (PR#1-6)
```
