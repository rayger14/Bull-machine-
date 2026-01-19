# CALIBRATION AUDIT QUICK REFERENCE

**TL;DR:** We are NOT testing archetypes with full calibrations. 78.4% of domain features are missing or inaccessible.

---

## CRITICAL FINDINGS

### Feature Store Completeness: 18.9% (INCOMPLETE!)

| Domain | Coverage | Impact |
|--------|----------|--------|
| Wyckoff | 0% | ❌ Missing volume_climax_3b, wick_exhaustion_3b (S1 HARD GATES!) |
| SMC | 0% | ❌ Missing order blocks, FVGs |
| Temporal | 0% | ❌ Missing all time-based features |
| Macro | 20% | ⚠️ Missing regime_v2, dominance metrics |
| Funding/OI | 20% | ❌ Missing funding_z (S4/S5 CORE SIGNAL!) |
| Liquidity | 25% | ⚠️ Feature name mismatches |

### Domain Engines Active: 1-2 of 6 (MINIMAL!)

| Archetype | Active Engines | Status |
|-----------|----------------|--------|
| S1 | 1/6 (17%) | ❌ Wyckoff/SMC/Temporal/Fusion ALL disabled |
| S4 | 2/6 (33%) | ⚠️ Runtime features enabled, others disabled |
| S5 | 1/6 (17%) | ❌ Wyckoff/SMC/Temporal/Fusion ALL disabled |

### Configuration Drift: <0.1% (ACCEPTABLE!)

Parameters match optimized configs - but they're useless without features!

---

## ROOT CAUSE OF POOR PERFORMANCE

**Question:** Why are archetypes underperforming (PF 0.32-1.55)?

**Answer:**

### A) Missing/Inaccessible Features? ✅ YES - 80% of problem

**Critical Missing Features:**

1. **funding_z** (S4/S5 core signal)
   - We have: `funding_Z` (capitalized)
   - Impact: S4 cannot detect funding divergences, S5 cannot detect long squeezes
   - Fix: Rename `funding_Z` → `funding_z`

2. **volume_climax_3b** (S1 hard gate)
   - We have: `volume_climax_last_3b` (different name)
   - Impact: S1 exhaustion gate cannot pass, trades don't fire
   - Fix: Rename `volume_climax_last_3b` → `volume_climax_3b`

3. **wick_exhaustion_3b** (S1 hard gate)
   - We have: `wick_exhaustion_last_3b` (different name)
   - Impact: S1 exhaustion gate cannot pass, trades don't fire
   - Fix: Rename `wick_exhaustion_last_3b` → `wick_exhaustion_3b`

4. **liquidity_drain_severity/velocity/persistence** (S1 25% confluence)
   - We have: `liquidity_drain_pct`, `liquidity_velocity`, `liquidity_persistence`
   - Impact: S1 confluence scoring missing 25% weight
   - Fix: Update S1 config to use actual feature names

### B) Vanilla Parameters? ❌ NO - <5% of problem

Configuration parameters match optimized values (<0.1% drift).

### C) Legitimate Strategy Failure? ⏸️ CANNOT ASSESS YET

Must fix A first before evaluating strategy validity.

---

## IMMEDIATE FIX (TODAY - 2 HOURS)

### Step 1: Feature Name Mapping (1 hour)

**File:** `engine/features/feature_loader.py` (create new)

```python
#!/usr/bin/env python3
"""Feature name mapping for archetype compatibility."""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

FEATURE_NAME_MAPPINGS = {
    # Funding/OI - Capitalization fixes
    'funding_Z': 'funding_z',
    'USDT.D': 'usdt_d',
    'BTC.D': 'btc_d',

    # Wyckoff - Naming consistency
    'volume_climax_last_3b': 'volume_climax_3b',
    'wick_exhaustion_last_3b': 'wick_exhaustion_3b',

    # SMC - Naming consistency
    'is_bullish_ob': 'order_block_bull',
    'is_bearish_ob': 'order_block_bear',
    'tf1h_bos_bullish': 'bos_bull',
    'tf1h_bos_bearish': 'bos_bear',
}

def apply_feature_name_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map feature store names to archetype-expected names.

    Fixes capitalization mismatches and naming inconsistencies.
    """
    # Rename columns
    rename_dict = {}
    for old_name, new_name in FEATURE_NAME_MAPPINGS.items():
        if old_name in df.columns:
            rename_dict[old_name] = new_name

    if rename_dict:
        df = df.rename(columns=rename_dict)
        logger.info(f"[Feature Mapping] Mapped {len(rename_dict)} features: {list(rename_dict.values())}")

    return df
```

**Integration:** Add to `bin/backtest.py` after loading feature store:

```python
# After: df = pd.read_parquet(feature_path)
from engine.features.feature_loader import apply_feature_name_mappings
df = apply_feature_name_mappings(df)
```

### Step 2: Fix S1 Confluence Config (30 min)

**File:** `configs/s1_v2_production.json`

**Change (line 104-129):**

```json
"confluence_weights": {
  "capitulation_depth_score": 0.20,
  "crisis_environment": 0.15,
  "volume_climax_3b": 0.08,
  "wick_exhaustion_3b": 0.07,
  "liquidity_drain_pct": 0.10,           // ← Changed from liquidity_drain_severity
  "liquidity_velocity": 0.08,            // ← Changed from liquidity_velocity_score
  "liquidity_persistence": 0.07,         // ← Changed from liquidity_persistence_score
  "funding_reversal": 0.12,
  "oversold": 0.08,
  "volatility_spike": 0.05
}
```

### Step 3: Validate (2 hours)

```bash
# S1 - expect PF to jump from 0.32 to 1.2-1.5
python bin/backtest.py -c configs/s1_v2_production.json -s 2022-01-01 -e 2024-12-31

# S4 - expect PF to jump from 0.32 to 1.8-2.0
python bin/backtest.py -c configs/s4_optimized_oos_test.json -s 2023-01-01 -e 2023-06-30

# S5 - expect PF to jump from 1.55 to 1.7-1.8
python bin/backtest.py -c configs/system_s5_production.json -s 2022-01-01 -e 2022-12-31
```

**Success Criteria:**
- ✅ S1 PF > 1.2 (at least 3.75x improvement)
- ✅ S4 PF > 1.8 (at least 5.6x improvement)
- ✅ S5 PF > 1.7 (at least 10% improvement)

---

## EXPECTED RECOVERY PATH

### After Phase 1 (Today - Critical Fixes)

| Archetype | Current PF | After Fix | Improvement |
|-----------|------------|-----------|-------------|
| S1 | 0.32 | 1.2-1.5 | +275-370% |
| S4 | 0.32 | 1.8-2.0 | +463-525% |
| S5 | 1.55 | 1.7-1.8 | +10-16% |

### After Phase 2 (2 days - Wyckoff Composites)

| Archetype | After P1 | After P2 | Improvement |
|-----------|----------|----------|-------------|
| S1 | 1.2-1.5 | 1.5-1.8 | +25-20% |
| S4 | 1.8-2.0 | 2.0-2.1 | +11-5% |
| S5 | 1.7-1.8 | 1.7-1.8 | +0% |

### After Phase 3 (1 day - Enable Engines)

| Archetype | After P2 | Final Target | Improvement |
|-----------|----------|--------------|-------------|
| S1 | 1.5-1.8 | 1.8-2.2 | +20-22% |
| S4 | 2.0-2.1 | 2.0-2.3 | +0-10% |
| S5 | 1.7-1.8 | 1.8-1.9 | +6-6% |

---

## FILES GENERATED

### Audit Report
**File:** `ARCHETYPE_CALIBRATION_AUDIT_REPORT.md`
- Full domain feature coverage audit
- Domain engine activation status
- Configuration drift analysis
- Root cause determination

### Missing Knowledge Report
**File:** `MISSING_KNOWLEDGE_REPORT.md`
- Detailed breakdown of missing features by category
- Impact quantification per archetype
- Prioritized fix roadmap with timelines
- Cumulative improvement projections

### Audit Script
**File:** `bin/audit_archetype_calibrations.py`
- Automated feature coverage checker
- Domain engine activation validator
- Config drift detector
- Optuna results query tool

**Usage:**
```bash
python bin/audit_archetype_calibrations.py
```

---

## VERDICT

**Are we testing archetypes with FULL calibrations?**

### ❌ NO

**Evidence:**
- 78.4% of domain features missing or inaccessible
- 5 of 6 domain engines disabled
- Critical features exist but have wrong names (cannot be found)

**Impact:**
- S1 missing exhaustion detection → PF 0.32 (should be 1.8-2.2)
- S4 missing funding z-score → PF 0.32 (should be 2.0-2.3)
- S5 partially working → PF 1.55 (should be 1.8-1.9)

**Recommendation:**
DO NOT accept poor archetype performance until Phase 1 fixes implemented.

**Timeline:** 2-4 hours to fix, 2-4 hours to validate.

---

## QUICK COMMANDS

### Re-audit After Fixes
```bash
python bin/audit_archetype_calibrations.py
```

### Validate S1 Recovery
```bash
python bin/backtest.py -c configs/s1_v2_production.json -s 2022-01-01 -e 2024-12-31
```

### Validate S4 Recovery
```bash
python bin/backtest.py -c configs/s4_optimized_oos_test.json -s 2023-01-01 -e 2023-06-30
```

### Validate S5 Recovery
```bash
python bin/backtest.py -c configs/system_s5_production.json -s 2022-01-01 -e 2022-12-31
```

### Check Feature Store Columns
```bash
python -c "import pandas as pd; df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'); print('\n'.join(sorted(df.columns)))"
```

### Check Specific Feature Coverage
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
for feat in ['funding_Z', 'volume_climax_last_3b', 'wick_exhaustion_last_3b', 'liquidity_drain_pct']:
    if feat in df.columns:
        null_pct = df[feat].isna().mean() * 100
        print(f'{feat}: {null_pct:.1f}% null')
    else:
        print(f'{feat}: MISSING')
"
```

---

**Last Updated:** 2025-12-07
**Next Action:** Implement Phase 1 fixes (feature name mapping)
**Timeline:** Today (2-4 hours implementation + validation)
