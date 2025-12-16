# Bull Machine - Current State Quick Reference

**Date:** 2025-12-03 | **Status:** 🟡 PARTIALLY FUNCTIONAL

---

## TL;DR: Where We Are

✅ **Working:**
- Feature store (167 columns, all S1 v2 features present)
- Baseline models (PF 3.17 achieved)
- New backtesting framework
- ArchetypeModel wrapper
- S4 (PF 2.22) and S5 (PF 1.86) configs

❌ **Broken:**
- Archetype comparison (0 trades)
- Runtime enrichment disconnected
- Regime routing incomplete

🎯 **#1 Blocker:** Regime hard-filter veto (S1 needs crisis/risk_off, gets neutral)

---

## The Gap

```
OLD BACKTESTER (bin/backtest_knowledge_v2.py):
┌─────────────┐
│ Load Data   │
└──────┬──────┘
       │
       v
┌─────────────────────────┐
│ apply_liquidity_vacuum  │  ← RUNTIME ENRICHMENT ✅
│ _enrichment()           │
└──────┬──────────────────┘
       │
       v
┌─────────────┐
│ Regime      │  ← GMM CLASSIFIER ✅
│ Classify    │
└──────┬──────┘
       │
       v
┌─────────────┐
│ Archetype   │  ← ROUTING ✅
│ Detect      │
└─────────────┘

NEW BACKTESTER (engine/backtesting/engine.py):
┌─────────────┐
│ Load Data   │  ← Expects enriched ❌
└──────┬──────┘
       │
       v
┌─────────────┐
│ model.      │  ← No enrichment call ❌
│ predict()   │
└──────┬──────┘
       │
       v
┌─────────────┐
│ Archetype   │  ← Regime = 'neutral' ❌
│ Wrapper     │     S1 blocked!
└─────────────┘
```

---

## Data Layer Status

### Main File
`data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet`
- **Shape:** 26,236 bars × 167 columns
- **Size:** 12 MB
- **Status:** ✅ COMPLETE

### Feature Categories
```
19  Base (OHLCV + indicators)
64  Multi-timeframe (tf1h_, tf4h_, tf1d_)
30  Macro (VIX, DXY, funding, OI)
27  Wyckoff events
27  Pattern & liquidity

Total: 167 columns ✅
```

### S1 v2 Features (CRITICAL)
```
✅ wick_lower_ratio
✅ liquidity_vacuum_score
✅ volume_panic
✅ crisis_context
✅ liquidity_vacuum_fusion
✅ liquidity_drain_pct      ← KEY: relative drain vs 7d avg
✅ liquidity_velocity
✅ liquidity_persistence
✅ capitulation_depth
✅ crisis_composite
✅ volume_climax_last_3b
✅ wick_exhaustion_last_3b

All 12 features PRESENT in file!
```

---

## Archetype Status

### Production Ready
```
S4 (Funding Divergence):  PF 2.22 ✅
S5 (Long Squeeze):        PF 1.86 ✅
K  (Trap Within Trend):   PF >2.0 ✅ (bull market)
B  (Order Block Retest):  PF >1.8 ✅ (bull market)
```

### Blocked/Disabled
```
S1 (Liquidity Vacuum):    ❌ Regime filter blocks (needs crisis)
S2 (Failed Rally):        ❌ BROKEN PATTERN (PF 0.48)
```

### Regime Requirements
```
S1: ['risk_off', 'crisis']        ← STRICT
S4: ['risk_off', 'neutral']       ← FLEXIBLE
S5: ['risk_on', 'neutral']        ← FLEXIBLE
K:  ['risk_on', 'neutral']        ← FLEXIBLE
```

---

## Comparison Results

### Baseline (WORKS)
```
Model                   Test PF   WR    Trades
Baseline-Conservative   3.17      42.9% 7
Baseline-Aggressive     2.10      33.3% 36
```

### Archetype (FAILS)
```
Model                   Test PF   WR    Trades  Issue
S1-LiquidityVacuum      N/A       N/A   0       Regime veto
S4-FundingDivergence    N/A       N/A   0       Config/regime?
```

---

## Why 0 Trades?

### Root Cause: Hard Regime Filter

```python
# engine/archetypes/logic_v2_adapter.py
def detect(self, context: RuntimeContext):
    regime_label = context.regime_label  # 'neutral'

    if self.enabled['S1']:
        allowed = ['risk_off', 'crisis']
        if regime_label not in allowed:
            return (None, 0.0, 0.0)  # HARD VETO ← 0 TRADES
```

### Why Regime = 'neutral'?

```python
# engine/models/archetype_model.py
def _build_runtime_context(self, bar):
    regime_label = bar.get('macro_regime', 'neutral')  # Fallback
    # Problem: No RegimeClassifier instantiated
    # Problem: macro_regime column may all be 'neutral'
```

---

## Quick Fixes

### Option 1: Force Regime (1 min)
```python
archetype_s1 = ArchetypeModel(...)
archetype_s1.set_regime('crisis')  # Force S1-friendly regime
```

### Option 2: Try S4/S5 (5 min)
```python
# S4/S5 allow 'neutral' regime - should work
archetype_s4 = ArchetypeModel(
    config_path='configs/mvp/mvp_bear_market_v1.json',
    archetype_name='funding_divergence'  # or 'long_squeeze'
)
```

### Option 3: Check Regime Column (5 min)
```bash
python3 -c "import pandas as pd; \
  df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'); \
  print(df['macro_regime'].value_counts())"
```

---

## Complete Fix (1 day)

### 1. Add RegimeClassifier (4 hours)
```python
# engine/models/archetype_model.py
from engine.context.regime_classifier import RegimeClassifier

def __init__(self, config_path, archetype_name):
    self.regime_classifier = RegimeClassifier(
        model_path='models/regime_classifier_gmm.pkl'
    )

def _build_runtime_context(self, bar):
    regime_label = self.regime_classifier.predict_single(bar)
    regime_probs = self.regime_classifier.predict_proba_single(bar)
    ...
```

### 2. Add Enrichment Hook (2 hours)
```python
# engine/models/base.py
class BaseModel(ABC):
    def requires_enrichment(self) -> bool:
        return False

    def enrich(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

# engine/models/archetype_model.py
def requires_enrichment(self) -> bool:
    return self.archetype_name == 'liquidity_vacuum'

def enrich(self, data: pd.DataFrame) -> pd.DataFrame:
    if self.requires_enrichment():
        from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import apply_liquidity_vacuum_enrichment
        return apply_liquidity_vacuum_enrichment(data)
    return data

# engine/backtesting/engine.py
def run(self, start, end):
    if self.model.requires_enrichment():
        self.data = self.model.enrich(self.data)
    ...
```

### 3. Verify S1 Config (2 hours)
- Check `configs/s1_v2_production.json` structure
- Ensure thresholds match ArchetypeModel expectations
- Test with forced regime first

---

## File Locations

### Data
```
data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet  ← Main
```

### Configs
```
configs/mvp/mvp_bull_market_v1.json   ← Bull (K, B enabled)
configs/mvp/mvp_bear_market_v1.json   ← Bear (S5 enabled)
configs/s1_v2_production.json         ← S1 standalone (needs verification)
```

### Old Backtester
```
bin/backtest_knowledge_v2.py          ← 39k lines, has enrichment ✅
```

### New Backtester
```
engine/backtesting/engine.py          ← Clean, missing enrichment ❌
engine/backtesting/comparison.py      ← Framework ✅
engine/models/archetype_model.py      ← Wrapper ✅ (needs regime)
```

### Runtime Enrichment
```
engine/strategies/archetypes/bear/liquidity_vacuum_runtime.py  ← S1
engine/strategies/archetypes/bear/funding_divergence_runtime.py ← S4
engine/strategies/archetypes/bear/long_squeeze_runtime.py      ← S5
```

---

## Next Actions

### Today (1 hour)
1. Check regime column values
2. Try S4 comparison (allows neutral)
3. Force regime test for S1

### This Week (1 day)
1. Integrate RegimeClassifier
2. Add enrichment hook
3. Re-run full comparison
4. Document results

### Next Week
1. Build config validator
2. Create regime debugger
3. Document enrichment pattern
4. Deploy production configs

---

## Success Criteria

### For S1 to Work
- [ ] Regime = crisis or risk_off (via classifier or force)
- [ ] S1 v2 features present (✅ already present)
- [ ] Config structure matches ArchetypeModel
- [ ] Fusion threshold reasonable for data

### For Comparison to Work
- [ ] Baselines generate trades (✅ already working)
- [ ] Archetypes generate trades (❌ blocked by regime)
- [ ] PF > 2.5 for production consideration
- [ ] Overfit < 1.0 (train-test gap)

### For Production
- [ ] Test PF > 2.5
- [ ] Win rate > 35%
- [ ] 5-15 trades/year
- [ ] Overfit < 1.0
- [ ] Walk-forward validated

---

**Quick Start: Get 1 Archetype Working**

```bash
# Terminal
cd /Users/raymondghandchi/Bull-machine-/Bull-machine-

# Python
python3 <<EOF
from engine.models import ArchetypeModel
import pandas as pd

# Load data
data = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
print(f"Regime distribution: {data['macro_regime'].value_counts()}")

# Try S4 (allows neutral regime)
s4 = ArchetypeModel(
    config_path='configs/mvp/mvp_bear_market_v1.json',
    archetype_name='long_squeeze',  # S5
    name='S5-LongSqueeze'
)

# Test predict on one bar
bar = data.iloc[1000]
signal = s4.predict(bar)
print(f"Signal: {signal.direction}, confidence: {signal.confidence:.2f}")

# If still 'hold', force regime
s4.set_regime('neutral')
signal = s4.predict(bar)
print(f"After regime force: {signal.direction}, confidence: {signal.confidence:.2f}")
EOF
```

---

**Full Report:** See `BULL_MACHINE_CURRENT_STATE_REPORT.md` for detailed analysis
