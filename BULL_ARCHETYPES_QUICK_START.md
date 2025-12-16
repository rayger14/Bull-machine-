# Bull Archetypes - Quick Start Guide

**5 Production-Ready Bull Archetypes (MVP)**
**Date:** 2025-12-12

---

## TL;DR

```python
from engine.strategies.archetypes.bull import SpringUTADArchetype

archetype = SpringUTADArchetype()
name, confidence, metadata = archetype.detect(bar, regime='risk_on')

if name:
    print(f"LONG signal: {name} @ {confidence:.2%}")
```

---

## Available Archetypes

| Code | Name | Pattern | Trades/Year | Win Rate |
|------|------|---------|-------------|----------|
| **A** | Spring/UTAD | Wyckoff spring reversals | 15-25 | 55-65% |
| **B** | Order Block Retest | SMC demand zone retests | 20-35 | 60-70% |
| **C** | BOS/CHOCH | Break of structure | 25-40 | 65-75% |
| **G** | Liquidity Sweep | Stop hunt reversals | 20-30 | 60-70% |
| **H** | Trap Within Trend | False breakdowns in uptrend | 25-40 | 65-75% |

---

## Quick Import

```python
from engine.strategies.archetypes.bull import (
    SpringUTADArchetype,           # A: Spring reversals
    OrderBlockRetestArchetype,     # B: OB retests
    BOSCHOCHReversalArchetype,    # C: Break of structure
    LiquiditySweepArchetype,       # G: Liquidity sweeps
    TrapWithinTrendArchetype       # H: Trend continuations
)
```

---

## Basic Usage

### Single Archetype

```python
# Initialize
archetype = SpringUTADArchetype()

# Detect on bar
name, confidence, metadata = archetype.detect(
    row=current_bar,        # pd.Series with features
    regime_label='risk_on'  # Current regime
)

# Check signal
if name:
    print(f"Signal: {name}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Entry: {current_bar['close']}")
    print(f"Stop: {current_bar['close'] - 2.5 * current_bar['atr_14']}")
```

### All Archetypes

```python
# Initialize all
archetypes = {
    'spring': SpringUTADArchetype(),
    'ob_retest': OrderBlockRetestArchetype(),
    'bos_choch': BOSCHOCHReversalArchetype(),
    'liq_sweep': LiquiditySweepArchetype(),
    'trap_trend': TrapWithinTrendArchetype()
}

# Scan bar
for key, arch in archetypes.items():
    name, conf, meta = arch.detect(bar, regime)
    if name:
        print(f"{key}: {conf:.2%} confidence")
```

---

## Configuration

### Load Custom Config

```python
import json

# Load config
with open('configs/archetypes/spring_utad_baseline.json') as f:
    config = json.load(f)

# Initialize with config
archetype = SpringUTADArchetype(config=config)
```

### Override Thresholds

```python
custom_config = {
    'thresholds': {
        'min_fusion_score': 0.45,  # More selective
        'min_wyckoff_confidence': 0.60,
        'wyckoff_weight': 0.35  # Increase Wyckoff importance
    }
}

archetype = SpringUTADArchetype(config=custom_config)
```

---

## Feature Requirements

### Minimal Required Features
```python
required = [
    'open', 'high', 'low', 'close', 'volume',
    'rsi_14', 'adx_14', 'volume_zscore',
    'tf4h_trend_direction', 'tf4h_fusion_score'
]
```

### Recommended Features (Wyckoff)
```python
wyckoff = [
    'wyckoff_spring_a', 'wyckoff_spring_a_confidence',
    'wyckoff_spring_b', 'wyckoff_spring_b_confidence',
    'wyckoff_lps', 'wyckoff_lps_confidence',
    'wyckoff_sos', 'wyckoff_sos_confidence',
    'wyckoff_phase_abc'
]
```

### Recommended Features (SMC)
```python
smc = [
    'smc_demand_zone', 'smc_liquidity_sweep', 'smc_choch',
    'tf1h_ob_bull_bottom', 'tf1h_ob_bull_top',
    'tf1h_bos_bullish', 'tf4h_bos_bullish',
    'tf1h_fvg_bull'
]
```

---

## Backtest Integration

```python
import pandas as pd

# Load data
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

# Initialize archetype
archetype = SpringUTADArchetype()

# Scan for signals
signals = []
for idx, row in df.iterrows():
    regime = row.get('regime_label', 'neutral')
    name, conf, meta = archetype.detect(row, regime)

    if name:
        signals.append({
            'timestamp': idx,
            'archetype': name,
            'confidence': conf,
            'entry': row['close'],
            'stop': row['close'] - 2.5 * row.get('atr_14', row['close'] * 0.02),
            'metadata': meta
        })

print(f"Found {len(signals)} signals")

# Convert to DataFrame
signals_df = pd.DataFrame(signals)
```

---

## Signal Structure

Each `detect()` call returns `(name, confidence, metadata)`:

```python
name: Optional[str]
    - Archetype name if signal detected
    - None if no signal

confidence: float
    - Fusion score (0.0-1.0)
    - Higher = stronger signal

metadata: Dict
    - Domain scores (wyckoff_score, smc_score, etc.)
    - Pattern type
    - Veto reason (if rejected)
```

**Example:**
```python
{
    'wyckoff_score': 0.80,
    'smc_score': 0.70,
    'price_action_score': 0.65,
    'momentum_score': 0.55,
    'regime_score': 1.0,
    'fusion_score': 0.72,
    'pattern_type': 'spring_long'
}
```

---

## Testing

### Run Unit Tests

```bash
# All tests
pytest tests/archetypes/test_bull_archetypes_mvp.py -v

# Single archetype
pytest tests/archetypes/test_bull_archetypes_mvp.py::TestSpringUTAD -v

# Quick smoke test
python3 tests/archetypes/test_bull_archetypes_mvp.py
```

### Quick Verification

```python
from engine.strategies.archetypes.bull import SpringUTADArchetype
import pandas as pd

# Create sample bar
bar = pd.Series({
    'open': 30000, 'high': 30500, 'low': 29800, 'close': 30400,
    'rsi_14': 55, 'adx_14': 25, 'volume_zscore': 1.5,
    'wyckoff_spring_a': True, 'wyckoff_spring_a_confidence': 0.7,
    'tf4h_trend_direction': 1, 'tf4h_fusion_score': 0.6,
    'smc_demand_zone': True
})

# Test
arch = SpringUTADArchetype()
name, conf, meta = arch.detect(bar, 'risk_on')

assert name == 'spring_utad', "Should detect spring"
print(f"✅ Detection working: {conf:.2%} confidence")
```

---

## Common Patterns

### Multi-Archetype Voting

```python
# Get signals from all archetypes
votes = []
for arch_name, archetype in archetypes.items():
    name, conf, meta = archetype.detect(bar, regime)
    if name and conf > 0.40:
        votes.append((arch_name, conf))

# Consensus entry
if len(votes) >= 2:
    avg_conf = sum(c for _, c in votes) / len(votes)
    print(f"STRONG SIGNAL: {len(votes)} archetypes agree ({avg_conf:.2%})")
```

### Regime-Aware Filtering

```python
# Adjust threshold by regime
regime_thresholds = {
    'risk_on': 0.35,
    'neutral': 0.40,
    'risk_off': 0.50,
    'crisis': 0.60
}

threshold = regime_thresholds.get(regime, 0.40)

name, conf, meta = archetype.detect(bar, regime)
if name and conf >= threshold:
    print(f"Signal accepted in {regime} regime")
```

### Confidence-Based Position Sizing

```python
if name:
    # Scale position by confidence
    base_size = 1000  # $1000 base
    confidence_mult = (conf - 0.35) / 0.65  # Normalize [0.35, 1.0] to [0, 1]
    position_size = base_size * (0.5 + 0.5 * confidence_mult)
    print(f"Position size: ${position_size:.0f} ({conf:.2%} confidence)")
```

---

## Performance Tuning

### Increase Selectivity

```python
config = {
    'thresholds': {
        'min_fusion_score': 0.50,  # Higher threshold
        'min_wyckoff_confidence': 0.70,
        'min_volume_zscore': 1.5
    }
}
```

### Increase Signal Count

```python
config = {
    'thresholds': {
        'min_fusion_score': 0.30,  # Lower threshold
        'min_wyckoff_confidence': 0.40,
        'min_volume_zscore': 0.8
    }
}
```

### Adjust Domain Weights

```python
config = {
    'thresholds': {
        'wyckoff_weight': 0.40,  # More Wyckoff influence
        'smc_weight': 0.20,
        'price_action_weight': 0.20,
        'momentum_weight': 0.15,
        'regime_weight': 0.05
    }
}
```

---

## Troubleshooting

### No Signals Detected

**Check:**
1. Feature availability: `'wyckoff_spring_a' in bar.index`
2. Threshold too high: Lower `min_fusion_score`
3. Regime filter: Check `allowed_regimes`
4. Veto triggers: Review `metadata['veto_reason']`

### Too Many Signals

**Solutions:**
1. Increase `min_fusion_score` (e.g., 0.35 → 0.45)
2. Increase domain thresholds
3. Add stricter regime filters
4. Require higher timeframe alignment

### Low Confidence Scores

**Possible causes:**
1. Missing domain features (Wyckoff, SMC)
2. Weak pattern (edge case detection)
3. Mixed signals (some domains conflict)
4. Wrong regime (regime_score low)

**Fix:**
- Ensure all domain features computed
- Check individual domain scores in metadata
- Verify feature quality

---

## Next Steps

1. **Backtest:** Run on 2022-2024 data to verify signal counts
2. **Optimize:** Use Optuna to find optimal thresholds
3. **Validate:** Walk-forward test on 2024 OOS data
4. **Deploy:** Integrate with production backtester

---

## Files Reference

**Implementation:**
- `/engine/strategies/archetypes/bull/spring_utad.py`
- `/engine/strategies/archetypes/bull/order_block_retest.py`
- `/engine/strategies/archetypes/bull/bos_choch_reversal.py`
- `/engine/strategies/archetypes/bull/liquidity_sweep.py`
- `/engine/strategies/archetypes/bull/trap_within_trend.py`

**Configs:**
- `/configs/archetypes/spring_utad_baseline.json`
- `/configs/archetypes/order_block_retest_baseline.json`
- `/configs/archetypes/bos_choch_reversal_baseline.json`
- `/configs/archetypes/liquidity_sweep_baseline.json`
- `/configs/archetypes/trap_within_trend_baseline.json`

**Tests:**
- `/tests/archetypes/test_bull_archetypes_mvp.py`

**Docs:**
- `/BULL_ARCHETYPES_MVP_REPORT.md` (detailed report)
- `/BULL_ARCHETYPES_QUICK_START.md` (this file)

---

**Status:** ✅ Production-ready MVP
**Date:** 2025-12-12
**Next:** Backtest on 2022-2024 data
