# Archetype Contract Migration Guide

**Version:** 2.0
**Status:** Complete
**Author:** System Architect (Claude Code)
**Date:** 2025-12-12

---

## Overview

This guide provides step-by-step instructions for migrating existing archetypes to the new **BaseArchetype contract** and **registry system**.

**Migration Strategy:**
- **Phase 1:** Migrate production archetypes (S1, S4, S5) - **PRIORITY**
- **Phase 2:** Migrate calibrated archetypes (H, B, K)
- **Phase 3:** Convert stub archetypes to proper placeholders (A, C)
- **Phase 4:** Deprecate old archetype system entirely

**Benefits After Migration:**
- Enforced interface consistency
- Automatic feature validation
- Centralized registry
- Better observability (diagnostics)
- Easier testing (mock RuntimeContext)

---

## Migration Checklist

For each archetype, follow this checklist:

- [ ] **Step 1:** Create archetype class inheriting from BaseArchetype
- [ ] **Step 2:** Set required class attributes (ID, NAME, MATURITY, etc.)
- [ ] **Step 3:** Implement required_features() method
- [ ] **Step 4:** Implement score() method
- [ ] **Step 5:** Implement veto() method
- [ ] **Step 6:** Implement entry() method
- [ ] **Step 7:** (Optional) Override exit() method if needed
- [ ] **Step 8:** (Optional) Override diagnostics() for better logging
- [ ] **Step 9:** Add to archetype_registry.yaml
- [ ] **Step 10:** Test with FeatureRealityGate
- [ ] **Step 11:** Integration test with backtest runner
- [ ] **Step 12:** Mark legacy code as deprecated

---

## Example: Migrating S1 (Liquidity Vacuum)

### Current State (Legacy)

**File:** `engine/strategies/archetypes/bear/liquidity_vacuum_runtime.py`

```python
class LiquidityVacuumRuntimeFeatures:
    """Runtime feature enrichment for S1"""

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... feature engineering
        return df
```

**Problems:**
- No standard interface
- No required_features() declaration
- No veto logic separate from scoring
- No diagnostics for debugging
- Not in registry

### New State (Migrated)

**File:** `engine/strategies/archetypes/bear/liquidity_vacuum_archetype.py`

```python
"""
Liquidity Vacuum Archetype - S1

Capitulation reversal during liquidity vacuum conditions.
Detects extreme orderbook drain + panic selling + wick rejection.
"""

from typing import List
from engine.archetypes.base_archetype import (
    BaseArchetype,
    ArchetypeScore,
    ArchetypeVeto,
    ArchetypeEntry,
    MaturityLevel,
    SignalType
)

class LiquidityVacuumArchetype(BaseArchetype):
    """
    S1: Liquidity Vacuum Reversal

    Pattern: Capitulation reversal during liquidity drain
    Direction: LONG (counter-trend)
    Regimes: risk_off, crisis
    Target: 10-15 trades/year, PF > 2.0
    """

    # ========================================================================
    # CLASS ATTRIBUTES
    # ========================================================================

    ARCHETYPE_ID = "S1"
    ARCHETYPE_NAME = "Liquidity Vacuum Reversal"
    MATURITY = MaturityLevel.PRODUCTION
    DIRECTION = SignalType.LONG
    REGIME_TAGS = ["risk_off", "crisis"]
    REQUIRES_ENGINES = ["liquidity", "wyckoff", "macro"]

    # ========================================================================
    # REQUIRED METHODS
    # ========================================================================

    def required_features(self) -> List[str]:
        """
        S1 requires liquidity drain features + volume/wick confirmation.
        """
        return [
            # Critical features (must have)
            'liquidity_score',
            'liquidity_drain_pct',
            'volume_zscore',
            'wick_lower_ratio',

            # Recommended features (degraded without)
            'VIX_Z',
            'DXY_Z',
            'funding_Z',
            'crisis_composite',

            # Optional features (fallback available)
            'capitulation_depth',
            'liquidity_velocity',
            'liquidity_persistence'
        ]

    def score(self, context) -> ArchetypeScore:
        """
        Score liquidity vacuum pattern.

        Components:
        1. Liquidity drain (-30% or more below 7d avg)
        2. Volume panic (z-score > 2.0)
        3. Wick rejection (lower wick > 30% of candle)
        4. Crisis context (VIX elevated, DXY strong)
        """
        row = context.row

        # Component 1: Liquidity drain (25% weight)
        liquidity_drain = row.get('liquidity_drain_pct', 0.0)
        drain_score = min(abs(liquidity_drain) / 0.30, 1.0) if liquidity_drain < 0 else 0.0

        # Component 2: Volume panic (20% weight)
        volume_z = row.get('volume_zscore', 0.0)
        volume_score = min((volume_z - 2.0) / 2.0, 1.0) if volume_z > 2.0 else 0.0

        # Component 3: Wick rejection (20% weight)
        wick_lower = row.get('wick_lower_ratio', 0.0)
        wick_score = min(wick_lower / 0.30, 1.0)

        # Component 4: Crisis context (15% weight)
        crisis = row.get('crisis_composite', 0.0)
        crisis_score = crisis

        # Weighted total
        total_score = (
            0.25 * drain_score +
            0.20 * volume_score +
            0.20 * wick_score +
            0.15 * crisis_score
        )

        # Generate reasons
        reasons = []
        if drain_score > 0.5:
            reasons.append(f"Liquidity drained {abs(liquidity_drain)*100:.1f}% below 7d avg")
        if volume_score > 0.5:
            reasons.append(f"Volume panic z-score: {volume_z:.1f}")
        if wick_score > 0.5:
            reasons.append(f"Deep lower wick: {wick_lower*100:.1f}% of candle")
        if crisis_score > 0.5:
            reasons.append(f"Crisis context elevated: {crisis:.2f}")

        return ArchetypeScore(
            total_score=total_score,
            component_scores={
                'liquidity_drain': drain_score,
                'volume_panic': volume_score,
                'wick_rejection': wick_score,
                'crisis_context': crisis_score
            },
            reasons=reasons,
            metadata={
                'pattern_quality': 'high' if total_score > 0.7 else 'medium' if total_score > 0.5 else 'low'
            }
        )

    def veto(self, context) -> ArchetypeVeto:
        """
        S1 veto logic.

        Hard blocks:
        1. Wrong regime (not risk_off or crisis)
        2. Missing critical features
        3. Uptrend (S1 is for downtrend reversals)
        """
        # Check regime
        if context.regime_label not in ['risk_off', 'crisis']:
            return ArchetypeVeto(
                is_vetoed=True,
                reason=f"S1 requires risk_off or crisis regime (got {context.regime_label})",
                veto_type='regime_mismatch'
            )

        # Check critical features
        all_present, missing = self.validate_features(context)
        if not all_present:
            return ArchetypeVeto(
                is_vetoed=True,
                reason=f"Missing critical features: {', '.join(missing)}",
                veto_type='feature_missing'
            )

        # Check trend (S1 is for downtrend reversals)
        trend = context.row.get('tf4h_external_trend', 'neutral')
        if trend == 'up':
            return ArchetypeVeto(
                is_vetoed=True,
                reason="S1 requires downtrend or neutral (not uptrend)",
                veto_type='hard_stop'
            )

        # All checks passed
        return ArchetypeVeto.no_veto()

    def entry(self, context) -> ArchetypeEntry:
        """
        Generate S1 entry signal.

        Entry specification:
        - Signal: LONG (counter-trend reversal)
        - Entry: Market order (capitalize on panic)
        - Stop: 2.5% below entry (tight stop, this is reversal)
        - Target: 8% profit (capitulation bounces are sharp but limited)
        - Max hold: 72 hours (3 days - reversals are quick)
        """
        # Get score for confidence
        score_result = self.score(context)
        confidence = score_result.total_score

        # Entry metadata
        metadata = {
            'stop_loss_pct': -0.025,  # 2.5% stop
            'take_profit_pct': 0.08,   # 8% target
            'position_size_mult': 1.0,
            'max_hold_bars': 72,
            'entry_reason': f"Liquidity vacuum reversal: {', '.join(score_result.reasons[:2])}",
            'pattern_components': score_result.component_scores,

            # Partial exits
            'partial_exit_1_pct': 0.04,  # Exit 50% at +4%
            'partial_exit_1_size': 0.5,
            'partial_exit_2_pct': 0.08,  # Exit remaining at +8%
            'partial_exit_2_size': 1.0,

            # Trailing stop (after +4%, trail by 2%)
            'trailing_stop_trigger_pct': 0.04,
            'trailing_stop_distance_pct': 0.02
        }

        return ArchetypeEntry(
            signal=SignalType.LONG,
            confidence=confidence,
            entry_price=None,  # Market order
            metadata=metadata
        )

    def exit(self, context) -> dict | None:
        """
        S1 exit override: Exit if liquidity recovers.

        Standard exits (max hold, stop loss) handled by position manager.
        This override catches early exit if reversal invalidated.
        """
        # Exit if liquidity recovers above baseline
        liquidity_drain = context.row.get('liquidity_drain_pct', 0.0)
        if liquidity_drain > 0.10:  # 10% above 7d avg = recovered
            return {
                'exit_signal': True,
                'exit_reason': 'Liquidity recovered above baseline',
                'exit_price': None  # Market exit
            }

        # Continue holding
        return None

    def diagnostics(self, context) -> dict:
        """S1 diagnostics for post-trade analysis"""
        base = super().diagnostics(context)

        # Add S1-specific diagnostics
        row = context.row
        base.update({
            'features_used': {
                'liquidity_score': row.get('liquidity_score'),
                'liquidity_drain_pct': row.get('liquidity_drain_pct'),
                'volume_zscore': row.get('volume_zscore'),
                'wick_lower_ratio': row.get('wick_lower_ratio'),
                'crisis_composite': row.get('crisis_composite')
            },
            'thresholds_applied': {
                'liquidity_drain_min': -0.30,
                'volume_zscore_min': 2.0,
                'wick_lower_min': 0.30
            }
        })

        return base
```

### Registry Entry

Add to `archetype_registry.yaml`:

```yaml
- id: S1
  name: "Liquidity Vacuum Reversal"
  slug: "liquidity_vacuum"
  class: "engine.strategies.archetypes.bear.LiquidityVacuumArchetype"
  maturity: production
  direction: long
  regime_tags:
    - risk_off
    - crisis
  requires_engines:
    - liquidity
    - wyckoff
    - macro
  requires_features:
    critical:
      - liquidity_score
      - liquidity_drain_pct
      - volume_zscore
      - wick_lower_ratio
    recommended:
      - VIX_Z
      - DXY_Z
      - funding_Z
      - crisis_composite
    optional:
      - capitulation_depth
      - liquidity_velocity
  enable_flag: enable_S1
  description: |
    Capitulation reversal during liquidity vacuum conditions.
  historical_performance:
    backtest_period: "2022-2024"
    total_trades: 34
    win_rate: 0.68
    profit_factor: 2.34
```

---

## Migration Templates

### Template 1: Simple Archetype (No Runtime Features)

Use for archetypes that only use existing feature store columns.

```python
from typing import List
from engine.archetypes.base_archetype import (
    BaseArchetype, ArchetypeScore, ArchetypeVeto, ArchetypeEntry,
    MaturityLevel, SignalType
)

class MyArchetype(BaseArchetype):
    ARCHETYPE_ID = "X"
    ARCHETYPE_NAME = "My Archetype"
    MATURITY = MaturityLevel.CALIBRATED
    DIRECTION = SignalType.LONG
    REGIME_TAGS = ["risk_on"]
    REQUIRES_ENGINES = ["smc"]

    def required_features(self) -> List[str]:
        return ['bos_detected', 'liquidity_score', 'volume_zscore']

    def score(self, context) -> ArchetypeScore:
        # Implement scoring logic
        row = context.row
        score = 0.0  # Calculate from features
        return ArchetypeScore(
            total_score=score,
            component_scores={},
            reasons=[],
            metadata={}
        )

    def veto(self, context) -> ArchetypeVeto:
        # Implement veto logic
        return ArchetypeVeto.no_veto()

    def entry(self, context) -> ArchetypeEntry:
        return ArchetypeEntry(
            signal=SignalType.LONG,
            confidence=0.5,
            entry_price=None,
            metadata={'stop_loss_pct': -0.02}
        )
```

### Template 2: Archetype With Runtime Features

Use for archetypes that compute derived features at runtime.

```python
from typing import List
import pandas as pd
from engine.archetypes.base_archetype import BaseArchetype, ...

class MyRuntimeArchetype(BaseArchetype):
    ARCHETYPE_ID = "Y"
    ARCHETYPE_NAME = "My Runtime Archetype"
    MATURITY = MaturityLevel.PRODUCTION
    DIRECTION = SignalType.SHORT
    REGIME_TAGS = ["risk_off"]
    REQUIRES_ENGINES = ["funding"]

    def __init__(self):
        super().__init__()
        # Initialize runtime feature calculator
        from engine.strategies.archetypes.bear.my_runtime_features import MyRuntimeFeatures
        self.runtime_features = MyRuntimeFeatures()

    def required_features(self) -> List[str]:
        # Return BASE features (runtime features computed on-demand)
        return ['funding_rate', 'oi_change', 'liquidity_score']

    def enrich_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich dataframe with runtime features BEFORE backtest.

        This is called by backtest runner during initialization.
        """
        return self.runtime_features.enrich_dataframe(df)

    def score(self, context) -> ArchetypeScore:
        # Use runtime-computed features
        row = context.row
        my_feature = row.get('my_runtime_feature', 0.0)

        # ... scoring logic
        return ArchetypeScore(...)

    # ... rest of methods
```

### Template 3: Stub Archetype

Use for archetypes not yet implemented.

```python
from engine.archetypes.base_archetype import (
    StubArchetype, MaturityLevel, SignalType
)

class MyStubArchetype(StubArchetype):
    ARCHETYPE_ID = "Z"
    ARCHETYPE_NAME = "My Stub Archetype"
    MATURITY = MaturityLevel.STUB
    DIRECTION = SignalType.LONG
    REGIME_TAGS = ["risk_on"]
    REQUIRES_ENGINES = ["wyckoff"]

    # No need to implement methods - StubArchetype raises NotImplementedError
```

---

## Integration With Backtest Runner

### Before Migration (Legacy)

```python
# Old way - archetypes scattered everywhere
from engine.archetypes.logic import ArchetypeLogic

archetype_logic = ArchetypeLogic(config)

for idx, row in df.iterrows():
    # Archetype logic hidden inside ArchetypeLogic class
    signal = archetype_logic.evaluate(row)
```

### After Migration (New Contract)

```python
# New way - centralized registry + feature validation
from engine.archetypes.registry_manager import get_registry
from engine.validation.feature_reality_gate import FeatureRealityGate

# 1. Load registry
registry = get_registry()
registry.log_status_report()

# 2. Get enabled archetypes
enabled_archetypes = registry.get_archetypes(
    maturity=['production', 'calibrated'],
    enabled_only=True,
    config=config
)

# 3. Validate features
gate = FeatureRealityGate(allow_degraded=True, fail_on_critical=True)
gate_report = gate.validate_all(enabled_archetypes, df)

# 4. Instantiate archetypes
archetype_instances = registry.instantiate_all(
    maturity=['production', 'calibrated'],
    enabled_only=True,
    config=config
)

# 5. Run backtest
for idx, row in df.iterrows():
    context = RuntimeContext(
        ts=idx,
        row=row,
        regime_probs=regime_probs[idx],
        regime_label=regime_labels[idx],
        adapted_params=adapted_params[idx],
        thresholds=thresholds[idx]
    )

    # Evaluate all archetypes
    for arch_id, archetype in archetype_instances.items():
        # Veto check
        veto = archetype.veto(context)
        if veto.is_vetoed:
            continue

        # Score
        score = archetype.score(context)
        if score.total_score < threshold:
            continue

        # Entry
        entry = archetype.entry(context)

        # Execute trade...
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/archetypes/test_liquidity_vacuum.py

import pytest
import pandas as pd
from engine.archetypes.base_archetype import SignalType
from engine.strategies.archetypes.bear.liquidity_vacuum_archetype import LiquidityVacuumArchetype
from engine.runtime.context import RuntimeContext

def test_s1_required_features():
    """S1 declares correct required features"""
    arch = LiquidityVacuumArchetype()
    features = arch.required_features()

    assert 'liquidity_score' in features
    assert 'liquidity_drain_pct' in features
    assert 'volume_zscore' in features
    assert 'wick_lower_ratio' in features

def test_s1_veto_wrong_regime():
    """S1 vetoes in risk_on regime"""
    arch = LiquidityVacuumArchetype()

    # Mock context with risk_on regime
    context = RuntimeContext(
        ts=pd.Timestamp('2024-01-01'),
        row=pd.Series({'close': 40000}),
        regime_probs={'risk_on': 0.8},
        regime_label='risk_on',
        adapted_params={},
        thresholds={}
    )

    veto = arch.veto(context)
    assert veto.is_vetoed
    assert 'regime' in veto.reason.lower()

def test_s1_score_liquidity_vacuum():
    """S1 scores high on liquidity vacuum pattern"""
    arch = LiquidityVacuumArchetype()

    # Mock context with liquidity vacuum
    context = RuntimeContext(
        ts=pd.Timestamp('2024-01-01'),
        row=pd.Series({
            'liquidity_drain_pct': -0.44,  # 44% drain
            'volume_zscore': 2.8,
            'wick_lower_ratio': 0.48,
            'crisis_composite': 0.65
        }),
        regime_probs={'risk_off': 0.7},
        regime_label='risk_off',
        adapted_params={},
        thresholds={}
    )

    score = arch.score(context)
    assert score.total_score > 0.6  # High conviction
    assert len(score.reasons) > 0

def test_s1_entry_signal():
    """S1 generates correct entry signal"""
    arch = LiquidityVacuumArchetype()

    context = RuntimeContext(
        ts=pd.Timestamp('2024-01-01'),
        row=pd.Series({
            'liquidity_drain_pct': -0.35,
            'volume_zscore': 2.5,
            'wick_lower_ratio': 0.35
        }),
        regime_probs={'crisis': 0.9},
        regime_label='crisis',
        adapted_params={},
        thresholds={}
    )

    entry = arch.entry(context)
    assert entry.signal == SignalType.LONG
    assert entry.confidence > 0.0
    assert entry.entry_price is None  # Market order
    assert 'stop_loss_pct' in entry.metadata
```

### Integration Tests

```python
# tests/integration/test_archetype_backtest.py

import pytest
from engine.archetypes.registry_manager import get_registry
from engine.validation.feature_reality_gate import FeatureRealityGate

def test_registry_loads_all_archetypes():
    """Registry successfully loads all archetypes"""
    registry = get_registry()
    status = registry.get_status_report()

    assert status['total_archetypes'] > 0
    assert status['loaded_classes_count'] > 0

def test_feature_gate_validates_archetypes(sample_df):
    """FeatureRealityGate validates archetypes against feature store"""
    registry = get_registry()
    archetypes = registry.get_archetypes(maturity=['production'])

    gate = FeatureRealityGate()
    report = gate.validate_all(archetypes, sample_df)

    # At least some archetypes should be runnable
    assert report.can_run_count > 0

def test_backtest_with_new_contract(sample_df, sample_config):
    """Full backtest using new archetype contract"""
    # ... full integration test
    pass
```

---

## Rollback Plan

If migration causes issues:

1. **Keep Legacy Code:** Don't delete old archetype code until migration is stable
2. **Feature Flag:** Add `use_new_archetype_contract` flag to config
3. **A/B Testing:** Run both old and new systems in parallel, compare results
4. **Gradual Migration:** Migrate one archetype at a time, validate each

**Rollback Steps:**
```python
# In backtest runner
if config.get('use_new_archetype_contract', False):
    # New system
    registry = get_registry()
    archetypes = registry.instantiate_all(...)
else:
    # Legacy system
    from engine.archetypes.logic import ArchetypeLogic
    archetype_logic = ArchetypeLogic(config)
```

---

## Timeline Estimate

**Phase 1: Production Archetypes** (3-4 days)
- Day 1: Migrate S1 (Liquidity Vacuum)
- Day 2: Migrate S4 (Funding Divergence)
- Day 3: Migrate S5 (Long Squeeze)
- Day 4: Integration testing + validation

**Phase 2: Calibrated Archetypes** (2-3 days)
- Day 5: Migrate H (Trap Within Trend)
- Day 6: Migrate B (Order Block Retest), K (Wick Trap)
- Day 7: Integration testing

**Phase 3: Stub Archetypes** (1 day)
- Day 8: Convert A, C to StubArchetype

**Phase 4: Cleanup** (1 day)
- Day 9: Deprecate legacy code, update docs

**Total: 9 days for full migration**

---

## Success Criteria

Migration is complete when:

- [ ] All production archetypes implement BaseArchetype
- [ ] All archetypes registered in archetype_registry.yaml
- [ ] FeatureRealityGate validates all archetypes successfully
- [ ] Backtest runner uses new contract system
- [ ] Unit tests pass for all migrated archetypes
- [ ] Integration tests show parity with legacy system
- [ ] Documentation updated
- [ ] Legacy code marked as deprecated

---

**End of Migration Guide**
