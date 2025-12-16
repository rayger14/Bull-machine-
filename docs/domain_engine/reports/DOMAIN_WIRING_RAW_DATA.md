# Domain Engine Wiring Verification - Raw Data
## Test Execution: 2025-12-10
## Dataset: BTC 2022-01-01 to 2022-12-31 (1 year bear market)

---

## Test Configuration

**Variants Tested:**
1. **Core**: Minimal domain engines (baseline)
2. **Full**: All 6 domain engines enabled

**Domain Engines:**
- Wyckoff (structural market phases)
- SMC (Smart Money Concepts)
- Temporal (time-based confluence)
- HOB (Hidden Order Blocks)
- Fusion (meta-learning layer)
- Macro (regime context)

---

## Raw Performance Data

### S1 (Liquidity Vacuum)

**S1_core (Wyckoff only):**
```
Archetype Telemetry:
  Total Checks: 7,331
  Total Matches: 0
  Match Rate: 0.0%

Trade Results:
  Total Trades: 110
  Trade Types: 110x tier1_market (100% legacy fusion)

Performance:
  Total PNL: -$3,652.60
  Profit Factor: 0.32
  Win Rate: 31.8%
  Sharpe Ratio: -0.70
  Max Drawdown: 37.9%
  Final Equity: $6,347.40
```

**S1_full (6 engines):**
```
Archetype Telemetry:
  Total Checks: 7,331
  Total Matches: 0
  Match Rate: 0.0%

Trade Results:
  Total Trades: 110
  Trade Types: 110x tier1_market (100% legacy fusion)

Performance:
  Total PNL: -$3,652.60
  Profit Factor: 0.32
  Win Rate: 31.8%
  Sharpe Ratio: -0.70
  Max Drawdown: 37.9%
  Final Equity: $6,347.40
```

**Comparison:**
- Matches: 0 vs 0 (IDENTICAL)
- Trades: 110 vs 110 (IDENTICAL)
- PF: 0.32 vs 0.32 (IDENTICAL)
- **Result: S1 archetype NOT firing, domain engines never invoked**

---

### S4 (Funding Divergence)

**S4_core (Funding only):**
```
Archetype Telemetry:
  Total Checks: 7,331
  Total Matches: 16
  Match Rate: 0.2%

Trade Results:
  Total Trades: 122
  Trade Types:
    - 12x archetype_funding_divergence
    - 110x tier1_market (legacy fusion)

Performance:
  Total PNL: -$3,279.29
  Profit Factor: 0.36
  Win Rate: 34.4%
  Sharpe Ratio: -0.59
  Max Drawdown: 36.8%
  Final Equity: $6,720.71
```

**S4_full (+ Macro regime routing):**
```
Archetype Telemetry:
  Total Checks: 7,331
  Total Matches: 16
  Match Rate: 0.2%

Trade Results:
  Total Trades: 122
  Trade Types:
    - 12x archetype_funding_divergence
    - 110x tier1_market (legacy fusion)

Performance:
  Total PNL: -$3,279.29
  Profit Factor: 0.36
  Win Rate: 34.4%
  Sharpe Ratio: -0.59
  Max Drawdown: 36.8%
  Final Equity: $6,720.71
```

**Comparison:**
- Matches: 16 vs 16 (IDENTICAL)
- Archetype Trades: 12 vs 12 (IDENTICAL)
- PF: 0.36 vs 0.36 (IDENTICAL)
- **Result: S4 archetype IS firing, but macro has ZERO impact**

**Runtime Feature Enrichment:**
Both variants show: `[S4] ✓ Enriched 8718 bars with runtime features`
- Proves runtime features ARE being calculated
- But they don't affect final results

---

### S5 (Long Squeeze)

**S5_core (Funding + RSI only):**
```
Archetype Telemetry:
  Total Checks: 7,331
  Total Matches: 24
  Match Rate: 0.3%

Trade Results:
  Total Trades: 134
  Trade Types:
    - 24x archetype_long_squeeze
    - 110x tier1_market (legacy fusion)

Performance:
  Total PNL: -$3,452.67
  Profit Factor: 0.34
  Win Rate: 34.3%
  Sharpe Ratio: -0.57
  Max Drawdown: 38.5%
  Final Equity: $6,547.33
```

**S5_full (6 engines):**
```
Archetype Telemetry:
  Total Checks: 7,331
  Total Matches: 24
  Match Rate: 0.3%

Trade Results:
  Total Trades: 115
  Trade Types:
    - 5x archetype_long_squeeze (⚠️ blocked 19/24)
    - 110x tier1_market (legacy fusion)

Performance:
  Total PNL: -$3,652.60
  Profit Factor: 0.32
  Win Rate: 31.3%
  Sharpe Ratio: -0.67
  Max Drawdown: 38.1%
  Final Equity: $6,347.40
```

**Comparison:**
- Matches: 24 vs 24 (SAME)
- Archetype Trades: 24 vs 5 (⚠️ **-19 trades blocked**)
- Total Trades: 134 vs 115 (**-19 total**)
- PF: 0.34 vs 0.32 (**-5.9% degradation**)
- Win Rate: 34.3% vs 31.3% (**-3.0% worse**)
- **Result: Domain engines ARE wired, but DEGRADE performance**

**Trade Blocking Analysis:**
- S5_core detected 24 long squeeze opportunities
- S5_full also detected same 24 opportunities
- But domain engines REJECTED 19 of them (79% rejection rate)
- Only 5/24 passed domain engine filters
- Those 5 trades performed WORSE than the full 24

---

## Key Findings

### 1. Wiring Status by Archetype

| Archetype | Archetype Fires? | Domain Engines Used? | Performance Impact |
|-----------|------------------|---------------------|-------------------|
| S1 | ❌ NO (0 matches) | N/A | N/A (not firing) |
| S4 | ✅ YES (16 matches) | ❌ NO (identical results) | ZERO |
| S5 | ✅ YES (24 matches) | ✅ YES (filters trades) | ⚠️ NEGATIVE (-5.9%) |

### 2. Trade Distribution

**S1:**
- Core: 0 archetype + 110 legacy = 110 total
- Full: 0 archetype + 110 legacy = 110 total
- **Issue**: Archetype completely broken in 2022

**S4:**
- Core: 12 archetype + 110 legacy = 122 total
- Full: 12 archetype + 110 legacy = 122 total
- **Issue**: Macro domain engine has no filtering effect

**S5:**
- Core: 24 archetype + 110 legacy = 134 total
- Full: 5 archetype + 110 legacy = 115 total
- **Issue**: Domain engines block 79% of archetype trades, performance drops

### 3. Legacy Fusion Contamination

**All variants heavily contaminated by legacy "tier1_market" trades:**
- S1: 110/110 trades (100%) from legacy fusion
- S4: 110/122 trades (90%) from legacy fusion
- S5_core: 110/134 trades (82%) from legacy fusion
- S5_full: 110/115 trades (96%) from legacy fusion

**Problem**: Legacy fusion system dominates results
- Makes it hard to isolate domain engine impact
- PF 0.32-0.36 driven by poor legacy trades (-36% PNL)
- Even "good" archetype trades diluted by legacy noise

---

## Configuration Validation

### Feature Flags Verified

**S1_core:**
```json
{
  "enable_wyckoff": true,
  "enable_smc": false,
  "enable_temporal": false,
  "enable_hob": false,
  "enable_fusion": false,
  "enable_macro": false
}
```

**S1_full:**
```json
{
  "enable_wyckoff": true,
  "enable_smc": true,
  "enable_temporal": true,
  "enable_hob": true,
  "enable_fusion": true,
  "enable_macro": true
}
```

**S4_core:**
```json
{
  "enable_macro": false,
  "use_macro_regime": false
}
```

**S4_full:**
```json
{
  "enable_macro": true,
  "use_macro_regime": true
}
```

**S5_core:**
```json
{
  "enable_wyckoff": false,
  "enable_smc": false,
  "enable_temporal": false,
  "enable_hob": false,
  "enable_fusion": false,
  "enable_macro": false
}
```

**S5_full:**
```json
{
  "enable_wyckoff": true,
  "enable_smc": true,
  "enable_temporal": true,
  "enable_hob": true,
  "enable_fusion": true,
  "enable_macro": true
}
```

**Validation: ✅ All feature flags set correctly**

---

## Diagnostic Questions

### Why S1 Has Zero Matches?

**Possible causes:**
1. **Thresholds too strict for 2022:**
   - `liquidity_max: 0.172` (very low)
   - `volume_z_min: 1.967` (high)
   - `wick_lower_min: 0.338` (specific)
   - `confluence_threshold: 0.65` (high)
   - `confluence_min_conditions: 3` (restrictive)

2. **Regime filter blocking:**
   - Core has `use_regime_filter: false` but still 0 matches
   - Full has `use_regime_filter: true` and 0 matches
   - Suggests issue is NOT regime filter

3. **V2 multi-bar logic broken:**
   - `use_v2_logic: true` in both variants
   - May have bugs in 3-bar capitulation detection
   - Need to test V1 logic as control

4. **Data quality:**
   - 2022 extreme bear (-64%)
   - Maybe insufficient "liquidity vacuum" conditions
   - Need to verify on known-working 2024 data

### Why S4 Has No Macro Impact?

**Possible causes:**
1. **Regime routing not applied:**
   - Config has regime weights: `risk_on: 0.5, crisis: 1.5`
   - But identical results suggest weights not used
   - Check if archetype selection code reads routing config

2. **2022 regime homogeneity:**
   - If entire 2022 is "crisis" regime
   - Then all 16 matches get same weight (1.5x)
   - No discrimination between trades
   - Need mixed regime period to test

3. **Macro features not in scoring:**
   - Runtime features calculated but not used
   - Check if fusion_score includes macro features
   - Verify macro gates are actually applied

### Why S5 Degrades with Domain Engines?

**Critical finding: 19/24 trades blocked, PF drops**

**Hypotheses:**
1. **Domain engines filtering winners:**
   - If blocked 19 trades were WINNERS
   - Would explain -5.9% PF drop
   - Need to compare PNL of blocked vs. accepted trades

2. **Engine scoring inverted:**
   - Domain features may have negative correlation
   - "Good" S5 setups score LOW on domain metrics
   - "Bad" S5 setups score HIGH
   - Filters keep bad, reject good

3. **Bear market incompatibility:**
   - Domain engines tuned for bull/mixed markets
   - Wyckoff phases don't map cleanly to bear
   - SMC concepts fail in constant downtrend

4. **Threshold mismatch:**
   - `fusion_threshold: 0.45` may be wrong level
   - If good trades score 0.40-0.44
   - And bad trades score 0.45-0.50
   - Filter keeps exactly the wrong trades

---

## Recommended Actions

### Immediate (Next 24h)

1. **Re-test S1 on working period:**
   ```bash
   python3 bin/backtest_knowledge_v2.py \
     --asset BTC \
     --config configs/variants/s1_core.json \
     --start 2024-01-01 \
     --end 2024-09-30
   ```
   Expected: 17 trades, PF 6.17 (known gold standard)
   If still 0 matches → V2 logic is broken

2. **Analyze S5 blocked trades:**
   ```python
   # Extract S5 trade logs
   s5_core_trades = parse_trades("S5_core_output.txt")
   s5_full_trades = parse_trades("S5_full_output.txt")

   # Find blocked trades
   blocked = [t for t in s5_core_trades
              if t not in s5_full_trades and
              t.type == "archetype_long_squeeze"]

   # Compare PNL
   blocked_pnl = sum(t.pnl for t in blocked)
   accepted_pnl = sum(t.pnl for t in s5_full_trades)
   ```
   If blocked_pnl > 0 → engines filtering winners

3. **Test S4 on mixed regime:**
   ```bash
   # 2023-2024: regimes vary
   python3 bin/backtest_knowledge_v2.py \
     --asset BTC \
     --config configs/variants/s4_core.json \
     --start 2023-01-01 \
     --end 2024-12-31
   ```
   Expected: Core ≠ Full if regime routing works

### Research (Next Week)

4. **Isolate domain engine impact:**
   - Create S5 variants with ONE engine at a time
   - S5 + Wyckoff only
   - S5 + SMC only
   - S5 + Temporal only
   - Find which engine blocks trades

5. **Disable legacy fusion:**
   - Set `fusion.entry_threshold_confidence: 0.99`
   - Eliminates 110 tier1_market trades
   - Clean comparison of archetype-only performance

6. **Grid search domain thresholds:**
   - Test S5 `fusion_threshold` 0.2 to 0.8
   - Find optimal point where Full > Core

---

## Conclusion

**Wiring Status: ✅ PARTIALLY WORKING**
- S4, S5 archetypes fire and invoke domain engines
- Runtime feature enrichment confirmed
- Feature flags control behavior

**Performance Status: ❌ NOT WORKING**
- S1: Broken (0 matches)
- S4: No impact (identical results)
- S5: Negative impact (-5.9% PF)

**Root Cause: UNDETERMINED**
- Could be implementation bugs
- Could be 2022 dataset edge case
- Could be threshold misconfiguration
- **Need more testing to isolate**

**Recommendation: PAUSE RE-OPTIMIZATION**
Do NOT proceed until domain engines show positive value on ANY dataset.
