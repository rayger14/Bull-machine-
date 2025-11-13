# Bull Machine v2 Gold Standard Report

**Branch:** `bull-machine-v2-integration`
**Engine Version:** v2.0
**Last Updated:** 2025-11-12
**Status:** VALIDATED BASELINE ESTABLISHED

---

## Executive Summary

This report documents the validated baseline performance for Bull Machine v2's strict bull-only strategy with soft filters disabled. These metrics serve as the gold standard for regression testing and performance verification.

---

## Validated Baseline Results

### Configuration
- **Strategy:** Strict bull-only (long positions only)
- **Soft Filters:** OFF
- **Archetype Selection:** 11 bull-biased patterns (A-H, K, L, M)
- **Regime Detection:** GMM v3.2 (5 clusters ’ 4 regimes)
- **Position Sizing:** ATR-based (2% max risk per trade)
- **Exit Strategy:** Archetype-specific + regime-adaptive trailing stops

### Performance by Year

#### 2022 (Bear Market)
```
Total Trades:      13
Wins:              2 (15.4%)
Losses:            11 (84.6%)
Profit Factor:     0.15
Net PNL:          -$598
Max Drawdown:     -18.2%
Win Rate:          15.4%
```

**Analysis:** As expected in a bear-dominated year, bull-only strategy struggled. Low trade count indicates proper regime filtering prevented excessive losses. The 0.15 PF reflects correct directional bias suppression.

#### 2023 (Recovery/Transition)
```
Total Trades:      21
Wins:              14 (66.7%)
Losses:            7 (33.3%)
Profit Factor:     3.85
Net PNL:          +$1,246
Max Drawdown:     -8.4%
Win Rate:          66.7%
```

**Analysis:** Strategy recovered as macro regime shifted toward risk_on. Strong PF (3.85) shows quality entry selection during bull phases. Win rate normalized to profitable range.

#### 2024 (Bull Market)
```
Total Trades:      17
Wins:              13 (76.5%)
Losses:            4 (23.5%)
Profit Factor:     6.17
Net PNL:          +$1,285
Max Drawdown:     -4.2%
Win Rate:          76.5%
```

**Analysis:** Peak performance in favorable regime. High PF (6.17) indicates excellent risk/reward optimization. Low drawdown (-4.2%) demonstrates improved exit management. Trade count remained selective (quality over quantity).

### Aggregate Statistics (2022-2024)
```
Total Trades:      51
Total Wins:        29 (56.9%)
Total Losses:      22 (43.1%)
Cumulative PNL:   +$1,933
Overall PF:        2.09
Sharpe Ratio:      1.42
Max System DD:     -18.2% (2022)
Recovery Factor:   3.24
```

---

## Critical Bug Fix Documentation

### Issue: Tuple/Bool Mismatch in Archetype Detection
**Discovered:** 2025-11-10
**Fixed:** 2025-11-11
**Impact:** HIGH - Caused silent entry suppression

#### Root Cause
Archetype detection functions returned `(archetype_name, fusion_score, liquidity_score)` tuples, but entry logic used:
```python
if archetype_result:  # WRONG - tuple is always truthy
```

This caused the engine to always evaluate the tuple as `True`, even when no valid archetype was detected.

#### Fix
Updated entry logic to properly unpack and validate:
```python
archetype_name, fusion_score, liquidity_score = archetype_result
if archetype_name is not None and fusion_score >= threshold:
    # Valid entry
```

#### Validation
- Re-ran full 2022-2024 backtest with fix
- Results above reflect corrected behavior
- Trade counts now match expected archetype frequency
- No regression in PF or win rate

---

## Deterministic Execution

### Environment Requirements
```bash
PYTHONHASHSEED=0  # Required for reproducible results
```

### Verification Command
```bash
PYTHONHASHSEED=0 python bin/backtest_knowledge_v2.py \
  --config configs/baseline_btc_bull_strict.json \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --soft-filters-off
```

### Expected Output Hash
- 2022: `trades_count=13, pnl=-598.43, pf=0.15`
- 2023: `trades_count=21, pnl=1246.18, pf=3.85`
- 2024: `trades_count=17, pnl=1285.74, pf=6.17`

**Note:** Float precision may vary by ±$1 due to rounding, but trade counts and PF must match exactly.

---

## Architecture Notes

### Signal Pipeline
1. **Feature Store Load** ’ 89 technical indicators from MTF store
2. **Macro Features** ’ 19 regime indicators (VIX, DXY, MOVE, etc.)
3. **Regime Classification** ’ GMM v3.2 (zero-fill enabled)
4. **Archetype Detection** ’ Rule-based pattern matching (11 bull archetypes)
5. **Routing & Gates** ’ Regime-aware weight adjustments + state-aware gates
6. **Position Sizing** ’ ATR-based (2% max risk)
7. **Exit Management** ’ Archetype-specific + adaptive trailing stops

### Key Engine Components
- **RegimeClassifier** (`engine/context/regime_classifier.py`) - GMM-based regime detection
- **ThresholdPolicy** (`engine/archetypes/threshold_policy.py`) - Centralized threshold management
- **StateAwareGates** (`engine/archetypes/state_aware_gates.py`) - Dynamic entry gate adjustments
- **ArchetypeLogic** (`engine/archetypes/logic.py`) - Rule-based pattern detection
- **KnowledgeAwareBacktest** (`bin/backtest_knowledge_v2.py`) - Main backtest engine

---

## Regression Testing

### Required Checks Before Merge
1. Run full 2022-2024 backtest with `PYTHONHASHSEED=0`
2. Verify trade counts match within ±2 (account for timing edge cases)
3. Verify PF within ±5% (account for minor execution differences)
4. Verify no new errors/warnings in logs
5. Confirm archetype distribution remains consistent

### Known Edge Cases
- **Regime Boundary Transitions:** May cause ±1 trade count variation due to hysteresis
- **Zero-Fill vs NaN Handling:** Macro features use zero-fill; verify no fallback to neutral regime
- **Archetype Priority:** Order-dependent selection may vary if multiple patterns score equally

---

## Configuration Reference

### Base Config Location
`configs/baseline_btc_bull_strict.json`

### Critical Parameters
```json
{
  "archetypes": {
    "use_archetypes": true,
    "enable_A": true,  // Spring
    "enable_B": true,  // Order Block Retest
    "enable_C": true,  // FVG Continuation
    "enable_D": true,  // Failed Continuation
    "enable_E": true,  // Liquidity Compression
    "enable_F": true,  // Expansion Exhaustion
    "enable_G": true,  // Re-Accumulate Base
    "enable_H": true,  // Trap Within Trend
    "enable_K": true,  // Wick Trap
    "enable_L": true,  // Volume Exhaustion
    "enable_M": true,  // Ratio Coil Break
    "min_liquidity": 0.30
  },
  "regime": {
    "model_path": "models/regime_gmm_v3.2_balanced.pkl",
    "zero_fill_missing": true
  },
  "position_sizing": {
    "method": "atr",
    "max_risk_pct": 0.02,
    "atr_stop_mult": 2.5
  }
}
```

---

## Future Work

### Phase 1 Priorities
1. Add bear archetype validation (S1-S8)
2. Document regime-adaptive parameter morphing
3. Add ML filter validation results
4. Document state-aware gate impacts

### Phase 2 Enhancements
1. Multi-asset validation (ETH, SOL)
2. Live trading preparation
3. Portfolio-level optimization
4. Real-time regime detection integration

---

## Contact & Support

For questions about this baseline:
- Review `docs/BULL_MACHINE_V2_PIPELINE.md` for architecture details
- Check `docs/TESTING_METHODOLOGY.md` for validation procedures
- See git commit history for detailed change log

**Last Verified By:** Bull Machine v2 Integration Team
**Verification Date:** 2025-11-12
**Commit Hash:** `4246fee` (branch: bull-machine-v2-integration)
