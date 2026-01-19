# Post-Phase 1 Roadmap: Bear Archetype Integration

**Date:** 2025-11-19
**Status:** Phase 1 Complete ✅ (Infrastructure ready, archetypes need activation)

---

## Phase 1 Completion Summary

### ✅ What Was Completed
1. **liquidity_score backfilled** - 26,236 rows, 100% coverage, mean=0.450
2. **Macro features verified** - 14 features with 97-100% coverage
3. **OI features documented** - Graceful degradation implemented in S5
4. **Validation backtest executed** - 2022 bear market tested

### 🔍 Critical Finding: Archetypes Not Firing
**Validation Results (2022):**
- Total trades: 118
- **109 trades (92.4%)**: `tier1_market` (legacy bull logic)
- **8 trades (6.8%)**: `archetype_long_squeeze` (S5)
- **1 trade (0.8%)**: `phase4_reentry`
- **ZERO trades**: S1, S2, S3, S4, S6, S7, S8

**Why This Happened:**
1. Bear archetypes (S1-S8) exist in code but are **disabled by default**
2. Feature flags set to `enable_S2=True, enable_S5=True` but **no config file uses them**
3. Validation used legacy config → fell back to bull archetypes (A-M)
4. Routing system not directing to bear archetypes in bear regime

**Performance (with wrong archetypes):**
- Win Rate: 33.1% (expected in bear market with bull patterns)
- Profit Factor: 0.43
- Total R: -41.62
- Mean entry liquidity: 0.450 ✅ (working correctly)

---

## Priority 1: Activate Bear Archetypes in Config 🎯

**Goal:** Make S2 and S5 actually fire in bear market backtests

**Root Cause:**
The bear archetype logic exists and is correct (S5 funding logic fixed, S2 detection implemented), but **no config file enables them**. The validation used `configs/v150/assets/BTC.json` which has:
```json
{
  "archetypes": {
    "use_archetypes": true,
    "enable_A": true,
    "enable_B": true,
    // ... bull archetypes enabled ...
    // S1-S8 NOT PRESENT (defaults to False)
  }
}
```

**Actions:**

1. **Create bear market config** → `configs/bear_market_2022_test.json`
   ```json
   {
     "archetypes": {
       "use_archetypes": true,
       
       // DISABLE bull archetypes for isolation test
       "enable_A": false,
       "enable_B": false,
       "enable_C": false,
       "enable_D": false,
       "enable_E": false,
       "enable_F": false,
       "enable_G": false,
       "enable_H": false,
       "enable_K": false,
       "enable_L": false,
       "enable_M": false,
       
       // ENABLE bear archetypes (Phase 1: S2, S5 only)
       "enable_S1": false,  // Defer (needs liquidity tuning)
       "enable_S2": true,   // ✅ APPROVED - Failed Rally Rejection
       "enable_S3": false,  // Merged into S2
       "enable_S4": false,  // Defer (needs tightening)
       "enable_S5": true,   // ✅ APPROVED - Long Squeeze (corrected)
       "enable_S6": false,  // REJECTED
       "enable_S7": false,  // REJECTED
       "enable_S8": false,  // REJECTED
       
       "thresholds": {
         "failed_rally": {
           "fusion_threshold": 0.36,
           "wick_ratio_min": 2.0,
           "rsi_min": 70.0,
           "vol_z_max": 0.5,
           "use_runtime_features": false  // Start with basic logic
         },
         "long_squeeze": {
           "fusion_threshold": 0.35,
           "funding_z_min": 1.2,
           "rsi_min": 70,
           "liquidity_max": 0.25
         }
       },
       
       "routing": {
         "neutral": {
           "weights": {
             "failed_rally": 1.5,
             "long_squeeze": 1.5
           }
         },
         "risk_off": {
           "weights": {
             "failed_rally": 2.0,
             "long_squeeze": 2.2
           }
         },
         "crisis": {
           "weights": {
             "failed_rally": 2.5,
             "long_squeeze": 2.8
           }
         }
       }
     }
   }
   ```

2. **Run isolation test**
   ```bash
   python bin/backtest_knowledge_v2.py \
     --asset BTC \
     --start 2022-01-01 \
     --end 2022-12-31 \
     --config configs/bear_market_2022_test.json \
     --export-trades results/validation/bear_isolation_2022.csv
   ```

3. **Verify archetype activation**
   ```bash
   # Should see S2 and S5 trades, NOT tier1_market
   python3 << 'VERIFY'
   import pandas as pd
   df = pd.read_csv('results/validation/bear_isolation_2022.csv')
   print(df.filter(regex='archetype_|tier1_').sum())
   # Expected: archetype_failed_rally > 0, archetype_long_squeeze > 0
   VERIFY
   ```

**Files to Create/Modify:**
- `configs/bear_market_2022_test.json` (new)
- Update `bin/backtest_knowledge_v2.py` if needed to load config properly

**Expected Outcome:**
- See **15-30 trades** from S2 (Failed Rally Rejection)
- See **5-10 trades** from S5 (Long Squeeze) 
- **ZERO** `tier1_market` trades (bull archetypes disabled)
- Mean liquidity still ~0.45 (unchanged, already working)

**Success Criteria:**
- `archetype_failed_rally` trades > 0
- `archetype_long_squeeze` trades > 0
- No crashes or errors
- Logs show "S2 RUNTIME" or "S5 DEBUG" messages

---

## Priority 2: Diagnose Zero-Matches for S1-S4, S6-S8 🔬

**Goal:** Understand why 6 bear archetypes produced zero signals

**Hypothesis:**
1. **S1 (Breakdown)**: Likely firing but filtered by hard liquidity check
2. **S2 (Failed Rally)**: Should work, but needs config activation (Priority 1)
3. **S3 (Whipsaw)**: Merged into S2, expected zero
4. **S4 (Distribution)**: Thresholds too strict (vol_z > 1.5 rare in 1H data)
5. **S6/S7/S8**: REJECTED patterns, expected zero

**Actions:**

1. **Add archetype gate logging** to understand filtering
   
   Modify `engine/archetypes/logic_v2_adapter.py`:
   ```python
   # In _check_S1, _check_S2, etc., add structured logging
   def _check_S2(self, context):
       # ... existing logic ...
       
       # Log first rejection reason
       if not hasattr(self, '_s2_rejection_counts'):
           self._s2_rejection_counts = {}
       
       if not ob_retest_flag:
           reason = "no_ob_retest"
           self._s2_rejection_counts[reason] = self._s2_rejection_counts.get(reason, 0) + 1
           if self._s2_rejection_counts[reason] == 1:
               logger.info(f"[S2 GATE] First rejection: {reason}")
           return False, 0.0, {"reason": reason}
   ```

2. **Run diagnostic backtest** with ALL bear archetypes enabled but relaxed thresholds
   
   Create `configs/bear_diagnostic_2022.json`:
   ```json
   {
     "archetypes": {
       "enable_S1": true,
       "enable_S2": true,
       "enable_S3": true,
       "enable_S4": true,
       "enable_S5": true,
       "enable_S6": true,
       "enable_S7": true,
       "enable_S8": true,
       
       "thresholds": {
         "breakdown": {"fusion": 0.30, "liq_max": 0.30, "vol_z": 0.8},
         "failed_rally": {"fusion_threshold": 0.30, "wick_ratio_min": 1.5},
         "whipsaw": {"fusion": 0.30, "rsi_extreme": 65.0},
         "distribution": {"fusion": 0.30, "vol_climax": 1.2},
         "long_squeeze": {"fusion_threshold": 0.30, "funding_z_min": 0.8, "rsi_min": 60},
         "alt_rotation_down": {"fusion": 0.30},
         "curve_inversion": {"fusion": 0.30},
         "volume_fade_chop": {"fusion": 0.30}
       }
     }
   }
   ```

3. **Analyze gate logs**
   ```bash
   python bin/backtest_knowledge_v2.py \
     --asset BTC \
     --start 2022-01-01 \
     --end 2022-12-31 \
     --config configs/bear_diagnostic_2022.json \
     2>&1 | tee results/diagnostic_2022.log
   
   # Extract rejection reasons
   grep -E "\[S[1-8] GATE\]" results/diagnostic_2022.log | \
     awk '{print $3, $NF}' | sort | uniq -c | sort -rn
   ```

**Files to Modify:**
- `engine/archetypes/logic_v2_adapter.py` (add gate logging)
- `configs/bear_diagnostic_2022.json` (new diagnostic config)

**Expected Outcome:**
- Clear rejection reason counts for each archetype
- Example: "S1: 2341 rejected (liquidity_too_high), 45 rejected (vol_z_low)"
- Identify which thresholds are blocking signals

**Success Criteria:**
- Gate logs show **which condition** blocks each archetype
- Rejection counts help tune thresholds (e.g., if 95% fail liquidity, relax it)

---

## Priority 3: Implement Regime-Aware Routing 🌍

**Goal:** Make bear archetypes dominant in bear regimes, suppress bull archetypes

**Problem:**
Current regime routing is configured but **not being applied correctly**:
1. Code has routing logic (lines 546-569 in `logic_v2_adapter.py`)
2. Config has routing weights in validation summary
3. But 2022 validation showed 92% `tier1_market` (bull logic)

**Root Cause Investigation:**

Check if regime routing is active:
```python
# In logic_v2_adapter.py line 553-554
routing_config = self.config.get('routing', {})
regime_routing = routing_config.get(regime, {})
```

The issue: `self.config` is the **archetypes subdictionary**, so routing should be at:
`self.config['routing']` (NOT `self.config['archetypes']['routing']`)

**Actions:**

1. **Verify routing config structure**
   
   Add debug logging to `logic_v2_adapter.py`:
   ```python
   # Line 553, before reading routing
   if len(candidates) > 1:
       logger.info(f"[ROUTING DEBUG] self.config keys: {list(self.config.keys())}")
       logger.info(f"[ROUTING DEBUG] Has 'routing'? {'routing' in self.config}")
       logger.info(f"[ROUTING DEBUG] regime={regime}")
   ```

2. **Create properly structured config**
   
   Ensure routing is at correct level:
   ```json
   {
     "archetypes": {
       "use_archetypes": true,
       "enable_S2": true,
       "enable_S5": true,
       
       "routing": {
         "neutral": {
           "weights": {
             "failed_rally": 1.5,
             "long_squeeze": 1.5,
             "trap_within_trend": 0.3
           }
         },
         "risk_off": {
           "weights": {
             "failed_rally": 2.0,
             "long_squeeze": 2.2,
             "trap_within_trend": 0.1
           }
         }
       }
     }
   }
   ```

3. **Test routing in mixed archetype config**
   
   Create `configs/bear_routing_test.json` with S2+S5 enabled AND trap_within_trend enabled (the dominant bull archetype from 2022):
   ```json
   {
     "archetypes": {
       "enable_H": true,  // trap_within_trend (was 96% in 2022)
       "enable_S2": true, // failed_rally
       "enable_S5": true, // long_squeeze
       
       "routing": {
         "neutral": {
           "weights": {
             "trap_within_trend": 0.5,  // Suppress 50%
             "failed_rally": 1.5,
             "long_squeeze": 1.5
           }
         },
         "risk_off": {
           "weights": {
             "trap_within_trend": 0.1,  // Suppress 90%
             "failed_rally": 2.0,
             "long_squeeze": 2.2
           }
         }
       }
     }
   }
   ```

4. **Run routing validation backtest**
   ```bash
   python bin/backtest_knowledge_v2.py \
     --asset BTC \
     --start 2022-01-01 \
     --end 2022-12-31 \
     --config configs/bear_routing_test.json \
     --export-trades results/validation/routing_test_2022.csv \
     2>&1 | tee results/routing_test_2022.log
   
   # Check if routing applied
   grep -E "\[REGIME ROUTING\]" results/routing_test_2022.log
   ```

5. **Verify archetype distribution changed**
   ```python
   import pandas as pd
   df = pd.read_csv('results/validation/routing_test_2022.csv')
   
   # Count archetype usage
   arch_cols = [c for c in df.columns if 'archetype' in c]
   for col in arch_cols:
       count = df[col].sum()
       if count > 0:
           print(f"{col}: {count} ({count/len(df)*100:.1f}%)")
   
   # Expected: trap_within_trend << 96%, failed_rally + long_squeeze > 0
   ```

**Files to Modify:**
- `engine/archetypes/logic_v2_adapter.py` (add routing debug logs)
- `configs/bear_routing_test.json` (new test config)

**Expected Outcome:**
- `trap_within_trend` drops from 96% → <30% in 2022
- `failed_rally` + `long_squeeze` combined: 40-60% of trades
- Logs show "[REGIME ROUTING] applying weights: {...}"

**Success Criteria:**
- Routing logs appear in backtest output
- Archetype distribution shifts as expected
- No crashes from routing logic

---

## Priority 4: Threshold Tuning for Signal Detection 🎛️

**Goal:** Optimize S2/S5 thresholds to maximize trades while maintaining quality

**Prerequisites:**
- Priority 1 complete (archetypes firing)
- Priority 2 complete (know rejection reasons)

**Actions:**

1. **Analyze baseline rejection rates** (from Priority 2)
   
   Example output from gate logs:
   ```
   S2 rejections:
     2341 no_ob_retest (82% of checks)
     287 weak_wick (10%)
     156 score_below_threshold (5%)
   
   S5 rejections:
     1893 funding_not_extreme (89% of checks)
     234 rsi_not_overbought (8%)
     67 liquidity_not_thin (2%)
   ```

2. **Create threshold sweep configs**
   
   For S2 (Failed Rally):
   ```json
   // configs/tuning/s2_relaxed.json
   {
     "archetypes": {
       "enable_S2": true,
       "thresholds": {
         "failed_rally": {
           "fusion_threshold": 0.30,  // Relaxed from 0.36
           "wick_ratio_min": 1.5,     // Relaxed from 2.0
           "rsi_min": 65.0,           // Relaxed from 70.0
           "vol_z_max": 0.7           // Relaxed from 0.5
         }
       }
     }
   }
   
   // configs/tuning/s2_strict.json
   {
     "archetypes": {
       "enable_S2": true,
       "thresholds": {
         "failed_rally": {
           "fusion_threshold": 0.40,
           "wick_ratio_min": 2.5,
           "rsi_min": 72.0,
           "vol_z_max": 0.3
         }
       }
     }
   }
   ```

3. **Run threshold sweep**
   ```bash
   # Baseline
   python bin/backtest_knowledge_v2.py \
     --asset BTC --start 2022-01-01 --end 2022-12-31 \
     --config configs/tuning/s2_baseline.json \
     --export-trades results/tuning/s2_baseline.csv
   
   # Relaxed
   python bin/backtest_knowledge_v2.py \
     --asset BTC --start 2022-01-01 --end 2022-12-31 \
     --config configs/tuning/s2_relaxed.json \
     --export-trades results/tuning/s2_relaxed.csv
   
   # Strict
   python bin/backtest_knowledge_v2.py \
     --asset BTC --start 2022-01-01 --end 2022-12-31 \
     --config configs/tuning/s2_strict.json \
     --export-trades results/tuning/s2_strict.csv
   ```

4. **Compare results**
   ```python
   import pandas as pd
   
   configs = ['baseline', 'relaxed', 'strict']
   results = []
   
   for cfg in configs:
       df = pd.read_csv(f'results/tuning/s2_{cfg}.csv')
       trades = len(df)
       wins = df['trade_won'].sum()
       wr = wins / trades if trades > 0 else 0
       avg_r = df['r_multiple'].mean()
       total_r = df['r_multiple'].sum()
       
       win_r = df[df['trade_won']==1]['r_multiple'].sum()
       loss_r = abs(df[df['trade_won']==0]['r_multiple'].sum())
       pf = win_r / loss_r if loss_r > 0 else 0
       
       results.append({
           'config': cfg,
           'trades': trades,
           'win_rate': wr,
           'avg_r': avg_r,
           'total_r': total_r,
           'pf': pf
       })
   
   results_df = pd.DataFrame(results)
   print(results_df.to_string(index=False))
   
   # Target: 15-30 trades, WR > 45%, PF > 1.3
   ```

5. **Select optimal thresholds**
   
   Criteria:
   - Trade count: 15-30 (too few = missing opportunities, too many = noise)
   - Win rate: >45% (bear market is hard, don't expect 60%)
   - Profit factor: >1.3 (minimum for viable pattern)
   - Avg R: >-0.2 (don't bleed equity)

**Files to Create:**
- `configs/tuning/s2_baseline.json`
- `configs/tuning/s2_relaxed.json`
- `configs/tuning/s2_strict.json`
- `configs/tuning/s5_baseline.json` (repeat for S5)
- `configs/tuning/s5_relaxed.json`
- `configs/tuning/s5_strict.json`

**Expected Outcome:**
- Find "Goldilocks zone" for each archetype
- Example: S2 with `wick_ratio=1.8, rsi=68` gives 22 trades, 48% WR, PF 1.45

**Success Criteria:**
- At least one config per archetype meets criteria (15-30 trades, WR>45%, PF>1.3)
- Clear trade-off between trade count and quality documented

---

## Priority 5: Full Period Validation (2022-2024) 📊

**Goal:** Validate bear archetypes work in BOTH bear (2022) and bull (2024) markets

**Prerequisites:**
- Priorities 1-4 complete
- Optimal thresholds selected
- Routing weights tuned

**Actions:**

1. **Create production bear config** with best thresholds from Priority 4
   
   `configs/production_bear_archetypes_v1.json`:
   ```json
   {
     "archetypes": {
       "use_archetypes": true,
       
       // Bull archetypes (existing, proven)
       "enable_A": true,
       "enable_B": true,
       "enable_H": true,  // trap_within_trend
       "enable_L": true,  // volume_exhaustion
       
       // Bear archetypes (new, validated)
       "enable_S2": true,  // failed_rally
       "enable_S5": true,  // long_squeeze
       
       "thresholds": {
         // Use optimized thresholds from Priority 4
         "failed_rally": {
           "fusion_threshold": 0.36,
           "wick_ratio_min": 1.8,  // Tuned
           "rsi_min": 68.0         // Tuned
         },
         "long_squeeze": {
           "fusion_threshold": 0.35,
           "funding_z_min": 1.0,   // Tuned
           "rsi_min": 65.0         // Tuned
         }
       },
       
       "routing": {
         "risk_on": {
           "weights": {
             "trap_within_trend": 1.2,
             "failed_rally": 0.3,
             "long_squeeze": 0.2
           }
         },
         "neutral": {
           "weights": {
             "trap_within_trend": 0.8,
             "failed_rally": 1.0,
             "long_squeeze": 1.0
           }
         },
         "risk_off": {
           "weights": {
             "trap_within_trend": 0.2,
             "failed_rally": 2.0,
             "long_squeeze": 2.2
           }
         }
       }
     }
   }
   ```

2. **Run validation on three periods**
   
   ```bash
   # 2022 (bear market)
   python bin/backtest_knowledge_v2.py \
     --asset BTC --start 2022-01-01 --end 2022-12-31 \
     --config configs/production_bear_archetypes_v1.json \
     --export-trades results/final/bear_2022.csv
   
   # 2023 (transition)
   python bin/backtest_knowledge_v2.py \
     --asset BTC --start 2023-01-01 --end 2023-12-31 \
     --config configs/production_bear_archetypes_v1.json \
     --export-trades results/final/transition_2023.csv
   
   # 2024 (bull market)
   python bin/backtest_knowledge_v2.py \
     --asset BTC --start 2024-01-01 --end 2024-09-30 \
     --config configs/production_bear_archetypes_v1.json \
     --export-trades results/final/bull_2024.csv
   
   # Full period (2022-2024)
   python bin/backtest_knowledge_v2.py \
     --asset BTC --start 2022-01-01 --end 2024-09-30 \
     --config configs/production_bear_archetypes_v1.json \
     --export-trades results/final/full_period_2022_2024.csv
   ```

3. **Analyze regime-specific performance**
   
   ```python
   import pandas as pd
   
   periods = {
       '2022 Bear': 'results/final/bear_2022.csv',
       '2023 Transition': 'results/final/transition_2023.csv',
       '2024 Bull': 'results/final/bull_2024.csv',
       'Full Period': 'results/final/full_period_2022_2024.csv'
   }
   
   for name, path in periods.items():
       df = pd.read_csv(path)
       
       # Overall stats
       trades = len(df)
       wr = df['trade_won'].sum() / trades if trades > 0 else 0
       total_r = df['r_multiple'].sum()
       
       win_r = df[df['trade_won']==1]['r_multiple'].sum()
       loss_r = abs(df[df['trade_won']==0]['r_multiple'].sum())
       pf = win_r / loss_r if loss_r > 0 else 0
       
       print(f"\n{name}:")
       print(f"  Trades: {trades}")
       print(f"  Win Rate: {wr:.1%}")
       print(f"  Total R: {total_r:.2f}")
       print(f"  PF: {pf:.2f}")
       
       # Archetype breakdown
       arch_cols = [c for c in df.columns if 'archetype' in c or 'tier1' in c]
       active_archetypes = {}
       for col in arch_cols:
           count = df[col].sum()
           if count > 0:
               active_archetypes[col] = count
       
       print(f"  Archetypes:")
       for arch, count in sorted(active_archetypes.items(), key=lambda x: -x[1]):
           pct = count / trades * 100
           subset = df[df[arch] == 1]
           arch_wr = subset['trade_won'].sum() / len(subset) if len(subset) > 0 else 0
           print(f"    {arch}: {count} ({pct:.1f}%), WR={arch_wr:.1%}")
   ```

4. **Validate success criteria**
   
   **2022 Bear Market:**
   - ✅ Trades: 40-80 (diversified signal generation)
   - ✅ PF: >1.2 (profitable in bear market)
   - ✅ WR: >40% (realistic for shorts in bear)
   - ✅ Bear archetype %: >30% of trades
   
   **2024 Bull Market:**
   - ✅ Trades: 40-80
   - ✅ PF: >1.5 (bull archetypes perform well)
   - ✅ WR: >50% (easier to long in bull)
   - ✅ Bull archetype %: >60% of trades (routing working)
   
   **Full Period (2022-2024):**
   - ✅ Trades: 120-200
   - ✅ PF: >1.4
   - ✅ WR: >45%
   - ✅ Max DD: <8% (risk-managed)
   - ✅ Sharpe: >0.8 (consistent returns)

5. **Document regime adaptation**
   
   Create `REGIME_ADAPTATION_REPORT.md`:
   ```markdown
   # Regime Adaptation Validation
   
   ## Archetype Usage by Market Regime
   
   | Archetype | 2022 Bear | 2023 Mixed | 2024 Bull | Routing Works? |
   |-----------|-----------|------------|-----------|----------------|
   | trap_within_trend | 15% | 45% | 68% | ✅ Suppressed in bear |
   | failed_rally | 42% | 28% | 8% | ✅ Amplified in bear |
   | long_squeeze | 23% | 12% | 3% | ✅ Amplified in bear |
   | volume_exhaustion | 12% | 10% | 15% | ✅ Regime-neutral |
   
   ## Performance by Regime
   
   | Metric | 2022 Bear | 2023 Mixed | 2024 Bull | Target | Pass? |
   |--------|-----------|------------|-----------|--------|-------|
   | Trades | 62 | 54 | 71 | 40-80 | ✅ |
   | Win Rate | 43% | 48% | 56% | >40% | ✅ |
   | PF | 1.35 | 1.52 | 1.87 | >1.2 | ✅ |
   | Sharpe | 0.62 | 0.91 | 1.24 | >0.5 | ✅ |
   ```

**Files to Create:**
- `configs/production_bear_archetypes_v1.json`
- `results/final/bear_2022.csv`
- `results/final/transition_2023.csv`
- `results/final/bull_2024.csv`
- `results/final/full_period_2022_2024.csv`
- `REGIME_ADAPTATION_REPORT.md`

**Expected Outcome:**
- Profitable across full market cycle (bear + bull)
- Clear regime adaptation (bear patterns in 2022, bull patterns in 2024)
- Production-ready config with validated thresholds

**Success Criteria:**
- All validation criteria met (see #4 above)
- No degradation vs gold standard in 2024 (PF still >1.5)
- Improved 2022 performance (PF >1.2 vs baseline 0.43)

---

## Validation Milestones

### Phase 1 Complete ✅
- [x] liquidity_score backfilled
- [x] Macro features verified
- [x] OI graceful degradation
- [x] S5 funding logic corrected
- [x] S2 detection logic implemented

### Phase 2: Activation (Week 1)
- [ ] Bear config created with S2+S5 enabled
- [ ] Isolation test passes (S2/S5 fire, tier1_market doesn't)
- [ ] Gate logging added

### Phase 3: Diagnosis (Week 1-2)
- [ ] Rejection reasons logged for all archetypes
- [ ] Threshold bottlenecks identified
- [ ] Routing debug logs confirm weights applied

### Phase 4: Tuning (Week 2)
- [ ] Threshold sweep complete
- [ ] Optimal thresholds selected
- [ ] Trade count 15-30, WR >45%, PF >1.3 achieved

### Phase 5: Production (Week 3)
- [ ] Full period validation complete (2022-2024)
- [ ] Regime adaptation confirmed
- [ ] Production config deployed

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Archetypes still don't fire after config fix | HIGH | LOW | Add verbose logging, verify enable flags |
| Routing weights not applied | MEDIUM | MEDIUM | Add routing debug logs, verify config structure |
| S2/S5 overfit to 2022 | MEDIUM | MEDIUM | Validate on 2023+2024, accept lower PF if consistent |
| Threshold tuning takes >1 week | LOW | MEDIUM | Use grid search script, parallelize |

### Data Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| OI still broken (all zeros) | MEDIUM | HIGH | S5 already has graceful degradation, works without OI |
| liquidity_score coverage drops | LOW | LOW | Already validated 100% coverage |
| Feature drift in 2024 data | LOW | LOW | Full validation will catch this |

### Performance Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 2024 bull performance degrades | HIGH | LOW | Keep bull archetypes enabled, routing should preserve |
| 2022 still unprofitable | MEDIUM | MEDIUM | Acceptable if PF >1.0 and 2024 strong (cycle alpha) |
| Too few trades (<40 total) | MEDIUM | MEDIUM | Relax thresholds, accept lower quality if needed |

---

## Definition of Done

### Priority 1 (Activation)
- Bear config exists at `configs/bear_market_2022_test.json`
- Backtest runs without errors
- Logs show "archetype_failed_rally" or "archetype_long_squeeze" (not tier1_market)
- Trade CSV has S2/S5 columns with values > 0

### Priority 2 (Diagnosis)
- Gate logs show rejection counts for S1, S2, S4, S5
- Clear bottleneck identified (e.g., "95% of S2 rejections = weak_wick")
- Routing debug logs confirm weights being read from config

### Priority 3 (Routing)
- Logs show "[REGIME ROUTING] applying weights: {...}"
- trap_within_trend drops from 96% → <40% in 2022 mixed test
- failed_rally + long_squeeze combined >30% in bear routing test

### Priority 4 (Tuning)
- 3 configs per archetype tested (baseline, relaxed, strict)
- Results table shows trade-off (trades vs quality)
- At least 1 config per archetype meets criteria

### Priority 5 (Production)
- 4 backtests complete (2022, 2023, 2024, full)
- All success criteria met (trades, WR, PF per period)
- Regime adaptation report shows expected distribution
- Production config committed to git

---

## Timeline

**Week 1:**
- Priority 1: 1 day (config + test)
- Priority 2: 2 days (logging + diagnosis)
- Priority 3: 2 days (routing debug + validation)

**Week 2:**
- Priority 4: 3-4 days (threshold sweep + analysis)

**Week 3:**
- Priority 5: 3-4 days (full validation + documentation)

**Total: 2-3 weeks to production**

---

## Next Steps (Immediate)

1. **Create bear config** → `configs/bear_market_2022_test.json`
2. **Run isolation test** → Verify S2/S5 fire
3. **Check logs** → Confirm no crashes, see archetype labels
4. **Report findings** → Share trade counts and archetypes detected

**Expected Time:** 2-4 hours

**Command to run:**
```bash
# Step 1: Create config (see Priority 1, Action 1)
# Step 2: Run backtest
python bin/backtest_knowledge_v2.py \
  --asset BTC \
  --start 2022-01-01 \
  --end 2022-12-31 \
  --config configs/bear_market_2022_test.json \
  --export-trades results/validation/activation_test.csv

# Step 3: Check results
python3 << 'CHECK'
import pandas as pd
df = pd.read_csv('results/validation/activation_test.csv')
print(f"Total trades: {len(df)}")
print("\nArchetype distribution:")
for col in df.columns:
    if 'archetype' in col or 'tier1' in col:
        count = df[col].sum()
        if count > 0:
            print(f"  {col}: {count}")
CHECK
```

---

**End of Roadmap**
