# Temporal Fusion Layer Specification

## Executive Summary

**Purpose**: Apply small, multiplicative adjustments (±5-15%) to fusion scores based on temporal context (Fibonacci time clusters, Wyckoff phase, session timing). NOT a hard veto—soft nudges only.

**Status**: GHOST (Referenced in Phase 2 plan, not yet implemented)

**Implementation Location**: `engine/fusion/temporal.py` (NEW)

**Integration Point**: AFTER base fusion (K2), BEFORE final archetype threshold comparison

**Philosophy**: "Time adds weight, not walls"—temporal context enhances existing signals but never kills them outright.

---

## Design Principles

### 1. Soft Adjustments Only (No Hard Vetoes)
**Problem**: Hard vetoes (e.g., "no trades outside NYC session") create blind spots and miss edge cases.

**Solution**: Multiplicative penalties/boosts (0.85x - 1.15x range) that allow marginal signals to compete.

**Example**:
```python
# BAD (hard veto):
if hour >= 22 or hour < 8:
    return None  # Kills signal completely

# GOOD (soft penalty):
if hour >= 22 or hour < 8:
    fusion_score *= 0.90  # 10% penalty, signal still viable
```

### 2. Chain Adjustments Multiplicatively (Not Additively)
**Reasoning**: Prevents adjustment accumulation from dominating base signal.

```python
# BAD (additive):
fusion_score += 0.10  # Fib cluster boost
fusion_score += 0.05  # Wyckoff phase boost
# Result: +0.15 total (too much!)

# GOOD (multiplicative):
fusion_score *= 1.10  # Fib cluster boost
fusion_score *= 1.05  # Wyckoff phase boost
# Result: 1.10 * 1.05 = 1.155 (15.5% total, compounding is natural)
```

### 3. Bounded Adjustments (Prevent Runaway)
**Min Multiplier**: 0.85 (max 15% penalty)
**Max Multiplier**: 1.15 (max 15% boost)

**Enforcement**:
```python
# After all adjustments
fusion_score = max(0.0, min(1.0, fusion_score))  # Clip to [0, 1]
```

---

## Integration Architecture

### Execution Flow
```
[Base Fusion Score K2]
       ↓
[Global Soft Filters]  ← Liquidity, regime, session penalties (in archetype logic)
       ↓
[Temporal Fusion Layer] ← THIS MODULE (Fib time, Wyckoff phase, PTI)
       ↓
[Archetype Threshold Check]
       ↓
[Trade Execution]
```

**Key Point**: Temporal fusion runs AFTER global soft filters but BEFORE archetype-specific scoring.

### Code Location
**File**: `engine/archetypes/logic_v2_adapter.py`, in `detect()` method

```python
# CURRENT (without temporal fusion)
fusion_score = self._fusion(context.row)
fusion_score, wyckoff_meta = self._apply_wyckoff_event_boosts(context.row, fusion_score)

# Apply global soft filters (liquidity, regime, session)
if use_soft_liquidity and liquidity_score < self.min_liquidity:
    fusion_score *= 0.7  # 30% penalty

# PROPOSED (with temporal fusion)
from engine.fusion.temporal import apply_temporal_adjustments

fusion_score = self._fusion(context.row)
fusion_score, wyckoff_meta = self._apply_wyckoff_event_boosts(context.row, fusion_score)

# Apply global soft filters (liquidity, regime, session)
if use_soft_liquidity and liquidity_score < self.min_liquidity:
    fusion_score *= 0.7  # 30% penalty

# NEW: Apply temporal fusion adjustments
fusion_score, temporal_meta = apply_temporal_adjustments(
    fusion_score,
    context.row,
    self.config.get('temporal_fusion', {})
)

# Continue with archetype dispatch...
```

---

## Adjustment Rules (v1 Baseline)

### Rule 1: Fibonacci Time Cluster Boost
**Condition**: High Fib cluster score + favorable Wyckoff phase

```json
{
  "rule": "fib_cluster_wyckoff_boost",
  "condition": "fib_time_cluster_score > 0.7 AND wyckoff_phase_abc IN ['C', 'D']",
  "adjustment": 1.10,
  "description": "10% boost near Fib time cluster during accumulation (C) or markup (D) phase"
}
```

**Rationale**:
- Fib time clusters indicate temporal pressure points
- Phase C (testing) + D (markup) are bullish phases
- Confluence of time + phase = higher probability setup

**Example**:
```python
# Scenario: Strong archetype signal at Fib(55) bars from Spring-A, in Phase D
fusion_score = 0.42  # Base fusion (above threshold)
fib_cluster_score = 0.75  # High temporal confluence
wyckoff_phase = 'D'  # Markup phase

# Adjustment:
fusion_score *= 1.10  # 0.42 → 0.462 (10% boost)
```

### Rule 2: Ranging Phase Suppression
**Condition**: Low temporal confluence + Wyckoff Phase B (ranging)

```json
{
  "rule": "ranging_phase_penalty",
  "condition": "fib_time_cluster_score < 0.2 AND wyckoff_phase_abc == 'B'",
  "adjustment": 0.95,
  "description": "5% penalty during ranging phase without temporal setup"
}
```

**Rationale**:
- Phase B (building cause) is choppy consolidation
- Low Fib cluster score = no temporal catalyst
- Slight penalty to favor cleaner setups

### Rule 3: Recent Buying/Selling Climax Caution
**Condition**: Very recent BC/SC (< 5 bars) + high PTI trap score

```json
{
  "rule": "recent_climax_caution",
  "condition": "bars_since_bc < 5 AND pti_score > 0.7",
  "adjustment": 1.05,
  "description": "5% boost if recent BC + high trap potential (fade the climax)"
}
```

**Rationale**:
- BC (buying climax) often reverses quickly
- High PTI = retail trapped long → smart money fades
- Small boost for counter-trend setups (conservative)

### Rule 4: Session Timing Adjustment (Asia Session)
**Condition**: Asian session (22:00-08:00 UTC) for crypto

```json
{
  "rule": "asia_session_penalty",
  "condition": "hour >= 22 OR hour < 8",
  "adjustment": 0.90,
  "description": "10% penalty during Asian session (lower volume, wider spreads)"
}
```

**Rationale**:
- Asian session has ~40% lower volume vs US/EU
- Wider bid-ask spreads, more slippage
- Slight penalty without killing signals outright

**Note**: This may already exist in global soft filters—avoid double-penalizing!

### Rule 5: Wyckoff Event Proximity Boost
**Condition**: Within 3 bars of LPS (Last Point of Support)

```json
{
  "rule": "lps_proximity_boost",
  "condition": "bars_since_lps <= 3 AND pti_score < 0.4",
  "adjustment": 1.08,
  "description": "8% boost near LPS (final test before markup) if no trap detected"
}
```

**Rationale**:
- LPS is the last shakeout before sustained uptrend
- Low PTI confirms genuine support, not trap
- Moderate boost for Wyckoff purists

---

## Implementation Pseudocode

```python
def apply_temporal_adjustments(
    fusion_score: float,
    row: pd.Series,
    config: Dict
) -> Tuple[float, Dict]:
    """
    Apply temporal context adjustments to fusion score.

    Args:
        fusion_score: Base fusion score from K2 or archetype-specific scoring
        row: Current bar data (with temporal features)
        config: Temporal fusion config with rules

    Returns:
        (adjusted_fusion_score, metadata)
    """
    original_score = fusion_score
    adjustments_applied = []

    # Extract temporal features
    fib_cluster_score = row.get('fib_time_cluster_score', 0.0)
    wyckoff_phase = row.get('wyckoff_phase_abc', 'neutral')
    pti_score = row.get('pti_score', 0.0)
    bars_since_bc = row.get('bars_since_bc', 999)
    bars_since_lps = row.get('bars_since_lps', 999)

    # Get hour for session timing (if timestamp available)
    hour = row.name.hour if hasattr(row.name, 'hour') else 12

    # Load rules from config (or use defaults)
    rules = config.get('rules', DEFAULT_RULES)

    # RULE 1: Fib cluster + Wyckoff phase boost
    if fib_cluster_score > 0.7 and wyckoff_phase in ['C', 'D']:
        adjustment = config.get('fib_cluster_boost', 1.10)
        fusion_score *= adjustment
        adjustments_applied.append({
            'rule': 'fib_cluster_wyckoff_boost',
            'multiplier': adjustment,
            'reason': f'Fib cluster {fib_cluster_score:.2f} in Phase {wyckoff_phase}'
        })

    # RULE 2: Ranging phase penalty
    elif fib_cluster_score < 0.2 and wyckoff_phase == 'B':
        adjustment = config.get('ranging_penalty', 0.95)
        fusion_score *= adjustment
        adjustments_applied.append({
            'rule': 'ranging_phase_penalty',
            'multiplier': adjustment,
            'reason': f'Phase B ranging, low Fib cluster {fib_cluster_score:.2f}'
        })

    # RULE 3: Recent BC + high PTI (fade the climax)
    if bars_since_bc < 5 and pti_score > 0.7:
        adjustment = config.get('climax_fade_boost', 1.05)
        fusion_score *= adjustment
        adjustments_applied.append({
            'rule': 'recent_climax_caution',
            'multiplier': adjustment,
            'reason': f'Recent BC ({bars_since_bc} bars) + PTI {pti_score:.2f}'
        })

    # RULE 4: Session timing (Asia penalty)
    # NOTE: Only apply if NOT already applied in global soft filters!
    if config.get('apply_session_penalty', False):
        if hour >= 22 or hour < 8:
            adjustment = config.get('asia_session_penalty', 0.90)
            fusion_score *= adjustment
            adjustments_applied.append({
                'rule': 'asia_session_penalty',
                'multiplier': adjustment,
                'reason': f'Asian session (hour={hour})'
            })

    # RULE 5: LPS proximity boost
    if bars_since_lps <= 3 and pti_score < 0.4:
        adjustment = config.get('lps_boost', 1.08)
        fusion_score *= adjustment
        adjustments_applied.append({
            'rule': 'lps_proximity_boost',
            'multiplier': adjustment,
            'reason': f'Near LPS ({bars_since_lps} bars), low PTI'
        })

    # Clip final score to [0, 1]
    fusion_score = max(0.0, min(1.0, fusion_score))

    # Build metadata
    metadata = {
        'original_score': original_score,
        'final_score': fusion_score,
        'total_adjustment': fusion_score / original_score if original_score > 0 else 1.0,
        'adjustments': adjustments_applied,
        'rules_triggered': len(adjustments_applied)
    }

    return fusion_score, metadata


# Default rules (if not in config)
DEFAULT_RULES = [
    {
        'name': 'fib_cluster_wyckoff_boost',
        'condition': 'fib_time_cluster_score > 0.7 AND wyckoff_phase_abc IN ["C", "D"]',
        'adjustment': 1.10,
        'description': 'Boost near Fib time cluster in accumulation/markup'
    },
    {
        'name': 'ranging_phase_penalty',
        'condition': 'fib_time_cluster_score < 0.2 AND wyckoff_phase_abc == "B"',
        'adjustment': 0.95,
        'description': 'Slight suppression in ranging phase without time confluence'
    },
    {
        'name': 'recent_climax_caution',
        'condition': 'bars_since_bc < 5 AND pti_score > 0.7',
        'adjustment': 1.05,
        'description': 'Small boost if recent BC + high trap potential'
    },
    {
        'name': 'lps_proximity_boost',
        'condition': 'bars_since_lps <= 3 AND pti_score < 0.4',
        'adjustment': 1.08,
        'description': 'Boost near LPS without trap signal'
    }
]
```

---

## Configuration Example

```json
{
  "temporal_fusion": {
    "enabled": true,
    "apply_session_penalty": false,

    "rules": [
      {
        "name": "fib_cluster_wyckoff_boost",
        "condition": "fib_time_cluster_score > 0.7 AND wyckoff_phase_abc IN ['C', 'D']",
        "adjustment": 1.10,
        "description": "Boost near Fib time cluster in accumulation/markup"
      },
      {
        "name": "ranging_phase_penalty",
        "condition": "fib_time_cluster_score < 0.2 AND wyckoff_phase_abc == 'B'",
        "adjustment": 0.95,
        "description": "Slight suppression in ranging phase without time confluence"
      },
      {
        "name": "recent_climax_caution",
        "condition": "bars_since_bc < 5 AND pti_score > 0.7",
        "adjustment": 1.05,
        "description": "Small boost if recent BC + high trap potential"
      },
      {
        "name": "lps_proximity_boost",
        "condition": "bars_since_lps <= 3 AND pti_score < 0.4",
        "adjustment": 1.08,
        "description": "Boost near LPS without trap signal"
      }
    ],

    "adjustments": {
      "fib_cluster_boost": 1.10,
      "ranging_penalty": 0.95,
      "climax_fade_boost": 1.05,
      "asia_session_penalty": 0.90,
      "lps_boost": 1.08
    }
  }
}
```

---

## Validation Plan

### Phase 1: Telemetry Logging
**Goal**: Understand adjustment frequency and magnitude

```python
# Add logging to apply_temporal_adjustments()
if len(adjustments_applied) > 0:
    logger.info(f"[TEMPORAL FUSION] {len(adjustments_applied)} rules triggered: "
                f"{original_score:.3f} → {fusion_score:.3f} "
                f"({(fusion_score/original_score - 1)*100:+.1f}%)")

    for adj in adjustments_applied:
        logger.debug(f"  - {adj['rule']}: {adj['multiplier']:.2f}x ({adj['reason']})")
```

**Metrics to Track**:
- Rule trigger frequency (% of bars where each rule fires)
- Average adjustment magnitude (mean boost/penalty per rule)
- Distribution of total adjustments (histogram of final multipliers)

### Phase 2: A/B Testing
**Baseline**: Archetypes WITHOUT temporal fusion (current state)
**Test**: Same archetypes WITH temporal fusion enabled

**Test Period**: 2022-2024 full dataset (bull + bear + chop)

**Metrics**:
| Metric | Without Temporal Fusion | With Temporal Fusion | Target Improvement |
|--------|------------------------|---------------------|-------------------|
| Profit Factor | X.XX | X.XX | +2-5% |
| Win Rate | XX% | XX% | +1-3% |
| Sharpe Ratio | X.XX | X.XX | +5-10% |
| Max Drawdown | -XX% | -XX% | -5-10% (smaller) |
| Trade Count | XXX | XXX | ±10% (no drastic change) |

**Success Criteria**:
- PF improvement >= +2%
- No trade count reduction > 15%
- No degradation in Sharpe ratio

### Phase 3: Edge Case Analysis
**Scenarios to Test**:
1. **Fib Cluster False Positives**: Random pivots creating fake clusters
   - Expected: Ranging phase penalty should offset
2. **Wyckoff Phase Misclassification**: Phase detector errors
   - Expected: Soft adjustments prevent catastrophic failures
3. **Session Timing Edge Cases**: DST transitions, holiday schedules
   - Expected: Minimal impact (adjustments are small)

### Phase 4: Ablation Study
**Goal**: Determine which rules contribute most to performance

**Method**: Disable rules one at a time and re-run backtest

| Rule Disabled | PF Change | Win Rate Change | Interpretation |
|--------------|-----------|-----------------|----------------|
| Fib cluster boost | -X% | -X% | Critical / Minor / Neutral |
| Ranging penalty | -X% | -X% | Critical / Minor / Neutral |
| Climax fade | -X% | -X% | Critical / Minor / Neutral |
| LPS boost | -X% | -X% | Critical / Minor / Neutral |

**Action**: Remove rules with minimal/negative impact to simplify logic.

---

## Implementation Checklist

- [ ] Create `engine/fusion/temporal.py` with `apply_temporal_adjustments()`
- [ ] Add integration hook in `engine/archetypes/logic_v2_adapter.py::detect()`
- [ ] Create config schema in `configs/temporal_fusion_example.json`
- [ ] Write unit tests (`tests/unit/test_temporal_fusion.py`)
  - Test multiplicative chaining
  - Test boundary clipping (0.0, 1.0)
  - Test rule condition parsing
- [ ] Add telemetry logging
- [ ] Run baseline backtest (2022-2024, temporal fusion OFF)
- [ ] Run A/B backtest (temporal fusion ON)
- [ ] Document results in `results/temporal_fusion_validation.md`
- [ ] Update `CHANGELOG.md` with Phase 2 completion

---

## Risk Mitigation

### Risk 1: Adjustment Accumulation
**Problem**: Too many rules triggered → fusion score inflated/deflated beyond reason

**Mitigation**:
- Hard clip to [0.85, 1.15] multiplier range
- Log warnings if > 3 rules trigger simultaneously
- Ablation testing to remove redundant rules

### Risk 2: Config Complexity
**Problem**: Too many knobs → overfitting / unmaintainable

**Mitigation**:
- Start with 4-5 core rules only
- Use defaults for all adjustment values
- Require config override for custom adjustments

### Risk 3: Feature Dependencies
**Problem**: Temporal features (Fib cluster, PTI) may not exist in feature store

**Mitigation**:
- Graceful fallbacks: `row.get('fib_time_cluster_score', 0.0)`
- Log warnings if expected features missing
- Disable temporal fusion if < 50% feature coverage

---

## Future Enhancements (Phase 3+)

### 1. Machine Learning Rule Discovery
- Train XGBoost model to learn optimal adjustments from backtest data
- Input: (fib_cluster_score, wyckoff_phase, pti_score, hour, ...)
- Output: Optimal fusion multiplier [0.85, 1.15]

### 2. Adaptive Adjustment Strength
- Adjust multiplier strength based on regime:
  - Bull market: Stronger boosts (1.15x max)
  - Bear market: Stronger penalties (0.85x min)
  - Ranging: Neutral (1.0x avg)

### 3. Time-of-Day Granularity
- Replace binary Asia/US/EU with continuous volume curve
- Use actual exchange volume data for precise weighting

### 4. Event-Driven Overrides
- NFP days: Disable temporal fusion (too volatile)
- Fed meetings: Boost Wyckoff phase signals (institutional repositioning)
- Options expiry: Boost temporal cluster signals (pinning effects)

---

## References

- **Fibonacci Time Clusters**: `docs/FIB_TIME_CLUSTER_SPEC.md`
- **Wyckoff Events**: `engine/wyckoff/events.py`
- **PTI (Trap Index)**: `engine/psychology/pti.py`
- **Global Soft Filters**: `engine/archetypes/logic_v2_adapter.py` lines 438-470
- **Feature Flags**: `engine/feature_flags.py` (BULL_SOFT_LIQUIDITY, BEAR_SOFT_REGIME, etc.)
