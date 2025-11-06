# Router v10 - Specification

## Purpose

Intelligent config selector that switches between bull/bear configs based on regime + confidence + events.

**Goal**: Maximize all-weather performance by using the right config at the right time.

## Architecture

### Inputs (per bar)

1. **Regime Label**: `risk_on`, `risk_off`, `crisis`, `neutral`
   - Source: GMM v3.1 regime classifier
   - Features: VIX, DXY, yields, funding, vol, correlations

2. **Regime Confidence**: 0.0 to 1.0
   - Probability of assigned regime
   - Low confidence = uncertain regime

3. **Event Window Flag**: boolean
   - True during macro events (CPI, FOMC, NFP, etc.)
   - Window: T-12h to T+2h around scheduled release

### Output (per bar)

**Active Config Path**: Which config to use for this bar
- `configs/v10_bases/btc_bull_v10_best.json`
- `configs/v10_bases/btc_bear_v10_best.json`
- `CASH` (no new entries, manage existing only)

### Decision Logic

```python
def select_config(regime_label, confidence, event_flag):
    """Router v10 decision logic."""

    # Rule 1: Confidence veto (stand down if uncertain)
    if confidence < 0.60:
        return 'CASH'  # No new entries

    # Rule 2: Event suppression (stand down during macro events)
    if event_flag:
        return 'CASH'  # No new entries during CPI/FOMC/etc

    # Rule 3: Regime-based selection
    if regime_label in ['risk_off', 'crisis']:
        return 'configs/v10_bases/btc_bear_v10_best.json'  # Defensive
    else:  # risk_on or neutral
        return 'configs/v10_bases/btc_bull_v10_best.json'   # Aggressive
```

## Config Differences

### Bull v10 Best (Aggressive)
- `final_fusion_floor`: 0.30 (low bar - more entries)
- `min_liquidity`: 0.16 (more aggressive)
- `trail_atr_mult`: 1.2 (tight stops - capture quick moves)
- `size_max`: 1.35 (larger positions)
- **Use when**: Risk-on regime, high confidence

### Bear v10 Best (Defensive)
- `final_fusion_floor`: 0.56 (high bar - selective entries)
- `min_liquidity`: 0.20 (more selective)
- `trail_atr_mult`: 1.9 (wide stops - ride trends)
- `size_max`: 0.95 (smaller positions)
- **Use when**: Risk-off/crisis regime OR low confidence OR events

## Implementation Plan

### Phase 1: Regime Detection (24h)

**Input**: Macro features (VIX, DXY, yields, funding, etc.)
**Output**: `(regime_label, confidence)`

**Components**:
1. `engine/regime_detector.py` - GMM v3.1 wrapper
2. Feature normalization (z-scores)
3. Regime mapping (0→crisis, 1→risk_off, 2→neutral, 3→risk_on)
4. Confidence = max cluster probability

**Validation**: Run on 2022-2024 data, verify regime assignments match reality
- 2022 Q1-Q2: Should detect risk_off/crisis
- 2024 Q1-Q4: Should detect risk_on
- Transitions: Should show low confidence during regime shifts

### Phase 2: Event Calendar (12h)

**Input**: Timestamp
**Output**: `event_flag` (boolean)

**Components**:
1. `engine/event_calendar.py` - Event schedule lookup
2. Hard-coded major events:
   - CPI: First Thursday/Friday of month, 8:30 AM ET
   - FOMC: ~8x/year, 2:00 PM ET decisions
   - NFP: First Friday of month, 8:30 AM ET
3. Window: T-12h to T+2h (suppress entries)

**Validation**: Verify suppression windows on 2024 data
- Count suppressed bars
- Verify no catastrophic moves missed

### Phase 3: Router Core (12h)

**Input**: `(regime_label, confidence, event_flag)`
**Output**: `active_config_path` or `'CASH'`

**Components**:
1. `engine/router_v10.py` - Decision logic (as above)
2. Config loader (read JSON on demand or cache)
3. Telemetry: Log every decision with timestamp + reason

**Validation**: Unit tests
- Low confidence → CASH
- Event window → CASH
- Risk-off + high conf → bear config
- Risk-on + high conf → bull config

### Phase 4: Backtest Integration (24h)

**Modification**: `bin/backtest_knowledge_v2.py`

**Changes**:
1. Add `--use-router` flag
2. Per-bar router call:
   ```python
   active_cfg = router.select_config(regime, conf, event_flag)
   if active_cfg == 'CASH':
       # Skip new entries, manage existing positions only
       continue
   else:
       # Use active_cfg for this bar
   ```
3. Track regime switches and cash periods

**Validation**:
- Router backtest 2022-2024 should blend bull/bear performance
- Expected: PF > 1.8, DD ≤ 6%, smoother equity curve

## Acceptance Criteria

### Backtest Performance (2022-2024)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Total PNL | > $1,000 | Positive expectancy |
| Profit Factor | > 1.8 | Blend of bull (4.2) + bear (0.5-1.0) |
| Max Drawdown | ≤ 6% | Better than bear-only (~6%) |
| Trades | 100-150 | ~50 trades/year after filtering |
| Confidence Stand-Downs | 20-40% | Significant filtering |
| Event Suppressions | ~48 events | 2 years × 24 major events/year |

### Regime Detection Quality

- **Crisis periods (2022 Q1-Q2)**: ≥ 80% risk_off/crisis classification
- **Bull periods (2024 Q1-Q4)**: ≥ 70% risk_on classification
- **Low-confidence periods**: ≤ 30% of total bars (most bars should be decisive)

### Config Switching Behavior

- **Bear config usage**: 40-60% of bars (defensive default)
- **Bull config usage**: 30-50% of bars (when clearly risk-on)
- **Cash periods**: 10-20% of bars (confidence veto + events)
- **Churn**: ≤ 2 regime switches per week (avoid whipsaw)

## Guardrails (Hard-Coded)

1. **min_liquidity ≤ 0.22** (enforced in both configs)
   - Catastrophic failure above this level
   - Pre-trade validation check

2. **Confidence floor = 0.60**
   - Can increase to 0.70 if early paper trades look noisy
   - Cannot decrease below 0.60 (safety)

3. **Event suppression windows**
   - CPI: 8:30 AM ET ± 12h
   - FOMC: 2:00 PM ET ± 12h
   - NFP: 8:30 AM ET ± 12h
   - No overrides allowed

4. **Circuit breaker**: If router DD > 5% on any backtest fold → halt and review

## Telemetry & Monitoring

### Per-Bar Logs
```json
{
  "timestamp": "2024-01-15T14:00:00Z",
  "regime_label": "risk_on",
  "regime_confidence": 0.73,
  "event_flag": false,
  "decision": "bull_v10_best",
  "reason": "risk_on_high_confidence"
}
```

### Daily Summary
- Active config distribution (% bull, % bear, % cash)
- Regime switches count
- Confidence distribution (histogram)
- Event suppressions count

### Weekly Review
- PnL by regime (risk_on vs risk_off)
- DD by config (bull trades vs bear trades)
- False regimes (if manual review says wrong)

## Risk Mitigation

### What Could Go Wrong

1. **Regime whipsaw**: Switching too frequently
   - **Fix**: Add hysteresis (require 2-3 consecutive bars before switch)

2. **Low confidence = always cash**: Missing all opportunities
   - **Fix**: Lower threshold to 0.55 if >50% of bars are cash

3. **Events miss big moves**: Suppress window too wide
   - **Fix**: Narrow to T-6h to T+1h if missing clear setups

4. **Wrong regime classification**: GMM mislabels regime
   - **Fix**: Add override file for known periods (e.g., force crisis for FTX week)

### Fallback Strategy

If router underperforms single-config approach:
- **Fallback A**: Just use Bear v10 Best everywhere (proven all-weather)
- **Fallback B**: Confidence-only filter (no regime switching, just veto low-conf)
- **Fallback C**: Manual regime labels (human-in-loop for major events)

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Regime Detection | 24h | Regime classifier working on 2022-2024 |
| Phase 2: Event Calendar | 12h | Event flag generation + validation |
| Phase 3: Router Core | 12h | Decision logic + unit tests |
| Phase 4: Backtest Integration | 24h | Full 2022-2024 backtest with router |
| **Total** | **72h** | **Router v10 ready for paper trading** |

## Next Steps After Router

1. **Paper Trade (2 weeks)**
   - Live regime detection
   - Monitor config switches
   - Track confidence distribution

2. **Go/No-Go Decision**
   - If PF ≥ 1.6, DD ≤ 3%: Move to small live sizing
   - Else: Tune confidence threshold or add hysteresis

3. **Scale to Multi-Asset**
   - Build feature stores for ETH, SOL, SPY
   - Run frontiers for each asset
   - Per-asset configs in router

---

**Status**: Specification complete, ready for implementation
**Next**: Build Phase 1 (Regime Detection)
