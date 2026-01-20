# Dual-System Architecture: B0 + Archetypes

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** ACTIVE - Two Independent Systems Running in Parallel
**Strategy:** Run both systems, evaluate, then decide on integration path

---

## Executive Summary

Bull Machine currently operates TWO independent trading systems in parallel:

1. **System B0 (Baseline-Conservative):** Simple drawdown-based strategy with PF 3.17
2. **System ARCHETYPES (S4/S5/S1):** Complex pattern recognition with PF 2.2/1.86

**Why Two Systems?**
- B0 provides a HIGH-PERFORMANCE baseline that's simple and proven
- Archetypes provide SPECIALIZED PATTERNS that work in specific regimes
- Running both allows us to evaluate if complexity adds value
- Parallel operation reduces risk while we gather live data

**Key Insight:** These systems are NOT competitors - they're complementary strategies with different strengths.

---

## System Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BULL MACHINE v2                              │
│                     DUAL-SYSTEM ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐  ┌──────────────────────────────┐
│       SYSTEM B0                  │  │     SYSTEM ARCHETYPES        │
│   (Baseline-Conservative)        │  │      (S4/S5/S1)              │
├──────────────────────────────────┤  ├──────────────────────────────┤
│ Framework: New (v2)              │  │ Framework: Original (v1)     │
│ PF: 3.17 (test 2023)             │  │ PF: 2.2 (S4), 1.86 (S5)     │
│ WR: 42.9%                        │  │ WR: 55.7% (S4), varies       │
│ Trades: ~7/year                  │  │ Trades: 12-14/year (regime)  │
│ Complexity: LOW                  │  │ Complexity: HIGH             │
│ Regime: All-weather              │  │ Regime: Specialists          │
└──────────────────────────────────┘  └──────────────────────────────┘
            │                                      │
            └──────────────┬───────────────────────┘
                           ▼
                 ┌─────────────────────┐
                 │  CAPITAL ROUTER     │
                 │  (Independent)      │
                 └─────────────────────┘
                           │
                           ▼
                 ┌─────────────────────┐
                 │   MARKET (BTC)      │
                 └─────────────────────┘
```

---

## System B0: Baseline-Conservative

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   SYSTEM B0                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  [Market Data] → [Feature Store v1]                │
│                        ↓                            │
│                  [BuyHoldSellClassifier]            │
│                        ↓                            │
│              Entry Logic (Simple):                  │
│              - 30-day drawdown < -15%               │
│              - No volume requirement                │
│              - 2.5x ATR stop loss                   │
│                        ↓                            │
│              Exit Logic (Simple):                   │
│              - +8% profit target                    │
│              - Stop loss hit                        │
│                        ↓                            │
│                [Backtest Engine v2]                 │
│                        ↓                            │
│                [Performance Metrics]                │
│                                                     │
└─────────────────────────────────────────────────────┘

Data Flow:
1. Load OHLCV + basic indicators (ATR, drawdown)
2. Calculate 30-day rolling max drawdown
3. Signal when drawdown < -15%
4. Exit at +8% profit or stop loss
5. NO feature enrichment needed
6. NO regime classification needed
```

### Key Characteristics

**Strengths:**
- Simple logic, easy to debug
- Production-ready (PF 3.17 > 2.5 threshold)
- All-weather (works in bull, bear, neutral)
- Excellent generalization (negative overfit: -1.89)
- Low maintenance overhead

**Limitations:**
- Low frequency (7 trades/year)
- Doesn't use advanced features
- No pattern recognition
- No regime awareness
- Misses specialized opportunities

**Tech Stack:**
- Framework: `engine/backtesting/engine.py` (new v2)
- Model: `engine/models/simple_classifier.py` → `BuyHoldSellClassifier`
- Data: `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet` (116 columns, only uses ~10)
- Config: Hardcoded in model class (no external JSON)

---

## System ARCHETYPES: S4/S5/S1

### Architecture

```
┌─────────────────────────────────────────────────────┐
│              SYSTEM ARCHETYPES                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  [Market Data] → [Feature Store v1]                │
│                        ↓                            │
│              [Runtime Enrichment]                   │
│              - S1: liquidity_vacuum_enrichment()    │
│              - S4: funding_divergence_runtime()     │
│              - S5: long_squeeze_runtime()           │
│                        ↓                            │
│              [Regime Classifier (GMM)]              │
│              - risk_on / neutral / risk_off / crisis│
│                        ↓                            │
│              [Archetype Detector (logic_v2)]        │
│              - S1 allowed: [crisis, risk_off]       │
│              - S4 allowed: [risk_off, neutral]      │
│              - S5 allowed: [risk_on, neutral]       │
│                        ↓                            │
│              [Fusion Score Calculation]             │
│              - Component weights                    │
│              - Fusion threshold (0.5-0.7)           │
│                        ↓                            │
│              [Signal Generation]                    │
│              - Long entry if fusion > threshold     │
│              - Stop loss: entry - (ATR * mult)      │
│                        ↓                            │
│              [Old Backtest Engine (v1)]             │
│              - bin/backtest_knowledge_v2.py         │
│                        ↓                            │
│              [Performance Metrics]                  │
│                                                     │
└─────────────────────────────────────────────────────┘

Data Flow:
1. Load feature store (116 columns)
2. Apply runtime enrichment (archetype-specific)
3. Classify regime (GMM on macro features)
4. Check regime gating (hard filter)
5. Calculate fusion score (weighted components)
6. Generate signal if fusion > threshold AND regime OK
7. Complex exit logic (trailing stops, confluence)
```

### Key Characteristics

**Strengths:**
- Specialized patterns for specific regimes
- Higher trade frequency (12-14/year per archetype)
- Uses advanced features (Wyckoff, liquidity, funding)
- Proven in bear markets (S4 PF 2.22)
- Regime-aware (abstains when inappropriate)

**Limitations:**
- High complexity (hard to debug)
- Requires complete feature store
- Regime dependency (S1 needs crisis)
- Old framework (bin/backtest_knowledge_v2.py)
- Wrapper issues (0 trades in new framework)

**Tech Stack:**
- Framework: `bin/backtest_knowledge_v2.py` (old v1, 39k lines)
- Wrapper: `engine/models/archetype_model.py` (NEW, bridges to v2)
- Logic: `engine/archetypes/logic_v2_adapter.py`
- Runtime: `engine/strategies/archetypes/bear/*.py`
- Configs:
  - `configs/s1_v2_production.json` (S1 Liquidity Vacuum)
  - `configs/s4_optimized_oos_2024.json` (S4 Funding Divergence)
  - `configs/mvp/mvp_bear_market_v1.json` (S5 Long Squeeze)
- Data: Same parquet (but uses 80+ columns)

---

## System Comparison

| Dimension | System B0 | System ARCHETYPES |
|-----------|-----------|-------------------|
| **Framework** | New v2 (clean) | Old v1 (legacy) |
| **Code Location** | `engine/backtesting/engine.py` | `bin/backtest_knowledge_v2.py` |
| **Model Interface** | `BuyHoldSellClassifier` | `ArchetypeModel` (wrapper) |
| **Lines of Code** | ~500 | ~39,000 |
| **Performance (PF)** | 3.17 | 2.2 (S4), 1.86 (S5) |
| **Win Rate** | 42.9% | 55.7% (S4), varies |
| **Trade Frequency** | 7/year | 12-14/year (per archetype) |
| **Complexity** | LOW | HIGH |
| **Features Used** | ~10 (OHLCV + basic) | ~80 (advanced) |
| **Regime Awareness** | None | Full (GMM classifier) |
| **Runtime Enrichment** | None | Required (per archetype) |
| **Maintenance Cost** | Low | High |
| **Debugging Ease** | Easy | Hard |
| **Production Status** | ✅ Ready | ⚠️ Needs validation |
| **Best Use Case** | All-weather baseline | Regime specialists |

---

## Why Two Systems?

### Strategic Rationale

**1. Risk Mitigation**
- B0 provides proven baseline (PF 3.17 validated)
- Archetypes add upside potential (2.2-1.86 PF per pattern)
- If archetypes fail, B0 keeps us profitable
- Parallel operation allows comparison on live data

**2. Complementary Strengths**
- B0: All-weather, simple, reliable
- Archetypes: Specialists, higher frequency, regime-aware
- B0 catches broad dips (-15% drawdown)
- Archetypes catch specific patterns (funding divergence, squeezes)
- Different signal sources reduce correlation

**3. Evaluation Period**
- Need live data to determine if archetypes beat B0
- Current test (2023): B0 wins (PF 3.17 > 2.2)
- But archetypes excel in specific regimes (2022 bear: S4 PF 2.22)
- Running both allows regime-specific performance analysis

**4. Framework Migration**
- B0 uses clean new framework (v2)
- Archetypes stuck on legacy framework (v1)
- Wrapper exists but has issues (0 trades due to regime gating)
- Parallel operation allows gradual migration

**5. Optionality**
- Keep both (capital router)
- Fix wrapper and merge
- Build meta-system
- Deploy best performer only

---

## Data Flow Comparison

### System B0 Data Flow

```
Market Data (OHLCV 1H)
    ↓
Feature Store v1 (116 columns)
    ↓
Extract minimal features:
  - close
  - high
  - low
  - atr_14
    ↓
Calculate drawdown_30d = (close - high_30d) / high_30d
    ↓
IF drawdown_30d < -0.15:
    entry_signal = True
    entry_price = close
    stop_loss = close - (2.5 * atr_14)
    take_profit = close * 1.08
    ↓
Hold position until:
  - Take profit hit (+8%)
  - Stop loss hit
    ↓
Exit and wait for next signal
```

**Characteristics:**
- Stateless (no memory between bars)
- Deterministic (same input → same output)
- Fast (minimal computation)
- No external dependencies

### System ARCHETYPES Data Flow

```
Market Data (OHLCV 1H)
    ↓
Feature Store v1 (116 columns)
    ↓
Runtime Enrichment (archetype-specific):
  S1: apply_liquidity_vacuum_enrichment()
      → liquidity_drain_pct, liquidity_velocity, crisis_composite
  S4: funding_divergence_runtime()
      → funding_z, funding_negative, price_resilience
  S5: long_squeeze_runtime()
      → funding_z (positive), rsi_overbought, liquidity_score
    ↓
Regime Classification (GMM):
  macro_features = [vix, dxy, funding_rate, oi_change]
      ↓
  regime_label = gmm.predict(macro_features)
  regime_probs = gmm.predict_proba(macro_features)
    ↓
Build RuntimeContext:
  - bar data (all 116+ columns)
  - regime_label (risk_on/neutral/risk_off/crisis)
  - regime_probs (distribution)
    ↓
Archetype Detection (logic_v2_adapter):
  FOR each enabled archetype:
      IF regime_label NOT IN allowed_regimes:
          SKIP (hard veto)

      Calculate fusion_score:
          S1: liquidity_vacuum_fusion = weighted_avg(
              liquidity_drain_pct, crisis_composite, volume_climax, ...
          )
          S4: funding_divergence_fusion = weighted_avg(
              funding_negative, price_resilience, liquidity_score, ...
          )
          S5: long_squeeze_fusion = weighted_avg(
              funding_z, rsi_overbought, liquidity_score, ...
          )

      IF fusion_score > fusion_threshold:
          entry_signal = True
          confidence = fusion_score
          stop_loss = entry - (atr_14 * atr_stop_multiplier)
    ↓
Complex Exit Logic:
  - Trailing stops (regime-dependent)
  - Confluence checks
  - Time-based exits
  - Fusion score decay
    ↓
Exit and cooldown period before next signal
```

**Characteristics:**
- Stateful (runtime enrichment, regime memory)
- Complex dependencies (GMM model, enrichment functions)
- Slow (many feature calculations)
- Regime-gated (can block all signals)

---

## Critical Differences

### 1. Framework Version

**B0 (v2):**
- Clean architecture (`engine/backtesting/engine.py`)
- Model-agnostic interface (`BaseModel`)
- Easy to extend
- Well-tested

**Archetypes (v1):**
- Monolithic script (`bin/backtest_knowledge_v2.py`, 39k lines)
- Archetype-specific hardcoding
- Difficult to extend
- Legacy tech debt

**Issue:** Wrapper bridge (`ArchetypeModel`) exists but has bugs (regime veto causing 0 trades).

### 2. Signal Generation Philosophy

**B0:**
- Simple heuristic: "Buy deep dips"
- No machine learning
- No pattern recognition
- Works because markets mean-revert

**Archetypes:**
- Complex pattern recognition
- Multi-component fusion scores
- Regime awareness
- Works because patterns repeat in specific conditions

**Question:** Does pattern complexity justify the maintenance cost?

### 3. Regime Dependency

**B0:**
- Regime-agnostic (fires in bull, bear, neutral)
- Always active
- Consistent frequency

**Archetypes:**
- Regime-dependent (hard filters)
  - S1: ONLY [crisis, risk_off] → Often 0 trades
  - S4: ONLY [risk_off, neutral] → Idle in bull markets
  - S5: ONLY [risk_on, neutral] → Idle in bear markets
- Activity varies by regime
- Can go months without trades

**Issue:** If regime classifier is wrong, archetype blocked entirely.

### 4. Feature Dependencies

**B0:**
- Minimal features (OHLCV, ATR, drawdown)
- No external data needed
- Self-contained

**Archetypes:**
- Heavy feature dependencies:
  - S1: 12 custom features (liquidity_drain_pct, crisis_composite, etc.)
  - S4: Funding rate data (may have gaps)
  - S5: OI data (missing for 2022-2023)
- Requires complete feature store
- Breaks if features missing

**Issue:** OI data gaps cause S5 to degrade (uses 75% signal strength fallback).

### 5. Performance Attribution

**B0:**
- PF 3.17 from simple logic
- Easy to understand WHY it works
- Can explain each trade

**Archetypes:**
- PF 2.2/1.86 from complex fusion
- Hard to attribute performance to specific components
- Black box debugging

**Trade-off:** Explainability vs sophistication.

---

## Current Blockers

### System B0
- ✅ No blockers - production ready
- Ready for paper trading immediately
- Config can be externalized if needed

### System ARCHETYPES
1. **Regime Gating Bug** (CRITICAL)
   - `ArchetypeModel` wrapper returns 0 trades
   - Root cause: `macro_regime` column all `neutral`
   - S1 needs `crisis` or `risk_off` → hard veto
   - S4/S5 should work but config may be too strict

2. **Feature Store Gaps** (MEDIUM)
   - OI data missing for 2022-2023 (only 2024)
   - `liquidity_score` column missing (all features need it)
   - Some derived features not present

3. **Framework Migration** (MEDIUM)
   - Old backtester works, new wrapper doesn't
   - Need to fix `ArchetypeModel._build_runtime_context()`
   - Need to add `RegimeClassifier` instantiation
   - Need enrichment hooks in `BacktestEngine`

4. **Config Validation** (LOW)
   - Unknown if thresholds translate correctly
   - S1/S4/S5 configs may need relaxation for test data

---

## Separation of Concerns

### Independent Operation

**B0 responsibilities:**
- Monitor all-market conditions
- Generate signals on deep drawdowns
- Execute simple profit-taking
- Provide baseline performance floor

**Archetypes responsibilities:**
- Monitor regime-specific conditions
- Generate signals on pattern matches
- Execute complex exits
- Provide upside via specialization

**Capital Router responsibilities:**
- Allocate capital between systems
- Rebalance based on performance
- Risk management (per-system limits)
- Conflict resolution (if both signal at once)

### No Cross-Dependencies

- B0 doesn't need archetypes to work
- Archetypes don't need B0 to work
- Each system can be disabled independently
- Each system has separate configs
- Each system logs separately
- Each system backtests separately

**Benefit:** Operational independence reduces risk of total failure.

---

## Future Integration Paths

### Path 1: Keep Separate (Capital Router)
```
┌────────────┐     ┌────────────┐
│   B0       │     │ Archetypes │
│  System    │     │   System   │
└─────┬──────┘     └──────┬─────┘
      │                   │
      └─────────┬─────────┘
                ▼
        ┌───────────────┐
        │ Capital Router│
        │ (Smart Alloc) │
        └───────────────┘
```
- Simplest approach
- Keep both systems intact
- Router decides allocation dynamically
- Minimal code changes

### Path 2: Fix Wrapper (Unified Framework)
```
┌────────────────────────────────┐
│     New Framework (v2)         │
├────────────────────────────────┤
│  BuyHoldSellClassifier (B0)    │
│  ArchetypeModel (S1/S4/S5)     │ ← Fixed wrapper
│  ...                           │
└────────────────────────────────┘
```
- Fix `ArchetypeModel` wrapper bugs
- Migrate all archetypes to new framework
- Deprecate old backtester
- Maximum code reuse

### Path 3: Build Meta-System (Ensemble)
```
┌──────────────────────────────────┐
│       Meta-System                │
├──────────────────────────────────┤
│  Input: B0 signal + S4/S5/S1     │
│  ML Model: XGBoost / RF          │
│  Output: Weighted ensemble       │
└──────────────────────────────────┘
```
- Treat B0 and archetypes as features
- Train ML model to combine signals
- Learns optimal weights dynamically
- Most sophisticated but highest complexity

**Decision:** Defer until we have live performance data from both systems.

---

## Why This Architecture Exists

### Historical Context

1. **Phase 1 (2022-2023):** Build archetype system
   - Implemented S1-S7 bear archetypes
   - Used old framework (bin/backtest_knowledge_v2.py)
   - Optimized thresholds (S4 PF 2.22, S5 PF 1.86)

2. **Phase 2 (2024):** Build new framework
   - Clean architecture (`engine/backtesting/`)
   - Model-agnostic interface (`BaseModel`)
   - Easy to compare models

3. **Phase 3 (2025 Q4):** Benchmark baselines
   - Created simple drawdown classifier
   - Achieved PF 3.17 (beats archetypes!)
   - Raises question: "Do we need complexity?"

4. **Current State (2025-12-03):**
   - B0 ready, archetypes need work
   - Running both to gather data
   - Will decide on integration after evaluation

### Design Philosophy

**Pragmatic Approach:**
- Don't discard working systems
- Don't optimize prematurely
- Gather data before deciding
- Keep options open

**Risk Management:**
- B0 provides safety net
- Archetypes provide upside
- Parallel operation allows evaluation
- Can abort archetype path if needed

---

## Performance Summary

### System B0 (Baseline-Conservative)

**Test Period:** 2023 (Recovery Market)
```
Profit Factor:     3.17
Win Rate:          42.9%
Trades:            7
Net PnL:           +$XXX (10k initial capital)
Max Drawdown:      -X.X%
Overfit:           -1.89 (EXCELLENT - test better than train)
```

**Why it works:**
- Patience (wait for -15% dips)
- Mean reversion (markets bounce)
- Simple exits (+8% target)
- No overtrading (7 trades/year)

### System S4 (Funding Divergence)

**Training Period:** 2022 (Bear Market)
```
Profit Factor:     2.22
Win Rate:          55.7%
Trades:            12
Regime:            Bear/Volatile
```

**OOS Period:** 2024 Q1-Q2 (Volatility)
```
Profit Factor:     2.32
Win Rate:          42.9%
Trades:            7 (annualized: 14)
Regime:            Volatile
```

**Why it works:**
- Specialists in bear/volatile markets
- Funding rate anomalies signal squeezes
- Abstains in bull markets (correct behavior)

### System S5 (Long Squeeze)

**Training Period:** 2022 (Bear Market)
```
Profit Factor:     1.86
Win Rate:          ~60%
Trades:            9
Regime:            Risk_on/Crisis
```

**Why it works:**
- Shorts overleveraged longs
- Positive funding rate signals crowding
- Works in bull pullbacks and crises

---

## Next Steps

### Immediate (Week 1)
1. Fix `ArchetypeModel` wrapper regime bug
2. Run 4-model comparison (B0 vs B0-Agg vs S4 vs S5)
3. Answer: "Do archetypes beat B0?"
4. Document decision in `PHASE0_DECISION.md`

### Short-Term (Month 1)
1. If archetypes win: Optimize and deploy both systems
2. If archetypes lose: Deploy B0 only, investigate why complexity fails
3. Paper trading: Start with winner(s)
4. Build capital allocation strategy

### Medium-Term (Quarter 1)
1. Collect live performance data
2. Regime-specific analysis (which system wins in what conditions)
3. Decide on integration path (router / wrapper / meta)
4. Build production infrastructure

---

## File Locations

### System B0
- **Model:** `engine/models/simple_classifier.py`
- **Engine:** `engine/backtesting/engine.py`
- **Example:** `examples/baseline_vs_archetype_comparison.py`

### System ARCHETYPES
- **Old Engine:** `bin/backtest_knowledge_v2.py`
- **Wrapper:** `engine/models/archetype_model.py`
- **Logic:** `engine/archetypes/logic_v2_adapter.py`
- **Runtime:**
  - `engine/strategies/archetypes/bear/liquidity_vacuum_runtime.py` (S1)
  - `engine/strategies/archetypes/bear/funding_divergence_runtime.py` (S4)
  - `engine/strategies/archetypes/bear/long_squeeze_runtime.py` (S5)
- **Configs:**
  - `configs/s1_v2_production.json`
  - `configs/s4_optimized_oos_2024.json`
  - `configs/mvp/mvp_bear_market_v1.json` (S5)

### Shared Data
- **Feature Store:** `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet`

---

## Key Takeaways

1. **Two systems, one goal:** Maximize risk-adjusted returns
2. **B0 is simple and works:** PF 3.17, production-ready
3. **Archetypes are complex but specialized:** PF 2.2/1.86 in target regimes
4. **Not a competition:** Complementary strategies
5. **Parallel operation:** Allows data-driven decision
6. **Integration is future work:** Focus on making both systems work first

**Strategy:** Run both, gather data, decide later. Pragmatic beats perfect.

---

**Document Owner:** System Architecture Team
**Last Updated:** 2025-12-03
**Next Review:** After Phase 0 comparison complete
