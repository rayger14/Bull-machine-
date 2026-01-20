# Bull Machine v2: State-Aware Evolution - Implementation Plan

## Mission
Evolve the Bull Machine to handle 2024 bull market conditions without simplifying any layer. Fix overtrading (643 → 150 trades) and profitability (PF 0.95 → 1.8+) through state-aware gates, ML retraining, and cost-aware optimization.

---

## Current Status (2024 with GMM v3.2)

| Metric | Value | Status |
|--------|-------|--------|
| **Trades** | 643 | ❌ 4× target |
| **Profit Factor** | 0.95 | ❌ Losing |
| **Win Rate** | 47.0% | ⚠️ Marginal |
| **trap_within_trend** | ~95% of entries | ❌ Dominates |
| **GMM v3.2** | Correctly detects neutral/risk_on | ✅ Fixed |

**Root Cause**: Static fusion thresholds (B: 0.359, C: 0.494) optimized on 2022-2023 don't adapt to 2024's chop/overfunded/thin conditions.

---

## Implementation Strategy

### Phase 1: State-Aware Gates ✅ **COMPLETED**

**File Created**: `engine/archetypes/state_aware_gates.py`

**What It Does**:
- Computes dynamic gates based on:
  - ADX (trend strength): +6% penalty when ADX < 18
  - ATR percentile (vol regime): +5% penalty when ATR_pctile < 25
  - Funding z-score (late-long risk): +5% penalty when funding_z > 1.0
  - 4H trend alignment: +3% penalty when misaligned
  - Archetype-specific logic (trap extra strict in chop, OB checks strength, VE checks volume)
  - Regime modulation (+8% crisis penalty for bull archetypes, -2% risk_on bonus)

**Key Features**:
- Clamps: Max ±15% adjustment, gates stay in [0.25, 0.75] range
- Logging: Auto-logs when |adjustment| > 5%
- Backward compatible: Returns base_gate if module=None

**Traders' Logic Alignment**:
- Moneytaur: "No fuel = no trade" (funding + chop = skip) ✅
- Zeroika: "Clarity = signal + context" (gates adapt) ✅
- Wyckoff: "Preserve capital in distribution" (high-cost bars penalized) ✅

---

### Phase 2: Integration into Archetype Logic (NEXT)

**Files to Modify**:
1. `engine/archetypes/logic_v2_adapter.py` - Add state gate calls
2. `bin/backtest_knowledge_v2.py` - Initialize StateAwareGates module
3. Configs - Add `state_aware_gates` section

**Integration Pattern** (example for trap_within_trend):

```python
# BEFORE (static):
fusion_th = ctx.get_threshold('trap_within_trend', 'fusion_threshold', 0.35)
if score < fusion_th:
    return False, score, {...}

# AFTER (state-aware):
base_fusion_th = ctx.get_threshold('trap_within_trend', 'fusion_threshold', 0.35)
fusion_th = apply_state_aware_gate(
    'trap_within_trend',
    base_fusion_th,
    ctx,
    self.state_gate_module,
    log_components=False
)
if score < fusion_th:
    return False, score, {"reason": "score_below_adaptive_threshold", ...}
```

**Config Addition**:

```json
{
  "state_aware_gates": {
    "enable": true,
    "weights": {
      "adx_weak_penalty": 0.06,
      "atr_low_penalty": 0.05,
      "funding_high_penalty": 0.05,
      "tf4h_misalign_penalty": 0.03,
      "adx_strong_bonus": -0.03,
      "funding_low_bonus": -0.02
    },
    "thresholds": {
      "adx_weak": 18.0,
      "adx_strong": 30.0,
      "atr_pctile_low": 25.0,
      "atr_pctile_good": 60.0,
      "funding_z_high": 1.0,
      "funding_z_low": 0.0
    },
    "max_adjustment": 0.15,
    "min_gate": 0.25,
    "max_gate": 0.75
  }
}
```

---

### Phase 3: Anti-Churn Exit Logic (NEXT)

**File to Modify**: `engine/exits/macro_echo.py` or new `engine/exits/anti_churn.py`

**Features**:
1. **Zone-Based Re-Entry Lockout**
   - After signal_neutralized/macro_flip, block re-entry in same price zone
   - Zone = last OB/FVG level ± 0.5% price buffer
   - Lockout duration = max(6 bars, 1×ATR time equivalent)

2. **Min-Hold for Traps**
   - trap_within_trend in risk_on: min 2-3 bars
   - Prevents 1-bar scratches that immediately re-enter

3. **Archetype-Specific Trailing**
   - Wider trails for OB in trend (trail_atr * (1 + ADX/50))
   - Tighter trails for VE in chop (trail_atr * (1 - 0.3*is_chop))

**Implementation Sketch**:

```python
class AntiChurnExitManager:
    def __init__(self, config):
        self.zone_buffer_pct = 0.005  # 0.5%
        self.min_lockout_bars = 6
        self.atr_lockout_mult = 1.0
        self.recent_exits = []  # (timestamp, price, archetype, reason)

    def check_lockout(self, timestamp, price, archetype):
        """Returns True if entry should be blocked due to recent churn."""
        for exit_ts, exit_price, exit_arch, exit_reason in self.recent_exits:
            # Same zone?
            if abs(price - exit_price) / exit_price < self.zone_buffer_pct:
                # Recent enough?
                bars_since = (timestamp - exit_ts).total_seconds() / 3600
                if bars_since < self.min_lockout_bars:
                    return True, f"zone_lockout_{bars_since:.1f}h"
        return False, None
```

---

### Phase 4: Opportunity Budget System (NEXT)

**File to Create**: `engine/opportunity_budget.py`

**Concept**: Not hard caps - dynamic "cost" per entry based on conditions.

**Implementation**:

```python
class OpportunityBudget:
    def __init__(self, config):
        self.daily_budget = config.get('daily_budget', 1.0)
        self.refill_rate = config.get('refill_rate', 0.05)  # +5% per hour
        self.current_budget = self.daily_budget

    def compute_entry_cost(self, ctx: RuntimeContext):
        """
        Cost = f(ATR_z, ADX, spread, recent_churn).
        High cost = noisy/thin/choppy conditions.
        """
        adx = ctx.row.get('adx', 0.0)
        atr_pctile = ctx.row.get('atr_percentile', 0.5) * 100
        spread_proxy = ctx.row.get('spread', 0.0)  # Or compute from bid/ask if available

        cost = 0.2  # Baseline cost

        # Penalty for low ADX (chop)
        if adx < 18:
            cost += 0.2

        # Penalty for low ATR (tight ranges)
        if atr_pctile < 25:
            cost += 0.2

        # Penalty for high spread (thin liquidity)
        if spread_proxy > 0.15:
            cost += 0.2

        # Penalty for recent churn (already traded this zone)
        if self.recent_churn_count > 2:
            cost += 0.2

        return cost

    def can_afford_entry(self, cost: float) -> bool:
        """Returns True if budget allows entry."""
        return self.current_budget >= cost

    def consume(self, cost: float):
        """Deduct cost from budget."""
        self.current_budget -= cost

    def refill(self, hours: float):
        """Refill budget over time."""
        self.current_budget = min(
            self.daily_budget,
            self.current_budget + self.refill_rate * hours
        )
```

**Integration**:
- Initialize in backtest_knowledge_v2.py
- Check before every archetype entry
- Log "budget spent / remaining" in trade metadata

---

### Phase 5: ML Quality v2 Retraining (CRITICAL)

**File to Create**: `bin/train_ml_quality_v2.py`

**Training Data**:
- Period: 2022-2024 (full cycle)
- Target: Realized R-multiple (or binary win/loss)
- Features (per signal):
  - Archetype one-hot (11 archetypes)
  - Archetype-specific scores (fusion, components)
  - Regime (GMM label + probas)
  - State: ADX, ATR_z, funding_z, vol_term, boms_strength, wick_metrics
  - MTF: tf4h_trend, tf1d_trend, alignment
  - Liquidity: liquidity_score, liquidity_z
  - Macro: VIX_Z, DXY_Z, YC_Z, PTI

**Model**:
- XGBoost or LightGBM (gradient boosting)
- 80/20 train/val split by time
- Calibrate outputs (Platt or isotonic) for well-calibrated probabilities
- Holdout: 2024 Q4 for final test

**Output**: `models/ml_quality_v2.onnx` + scalers

**Integration**:
- Replace `ml_filter.model_path` in configs
- Use calibrated probability as quality multiplier on archetype score

---

### Phase 6: Cost-Aware Objective (CRITICAL)

**File to Modify**: `bin/backtest_knowledge_v2.py` (optimization loop)

**Current Objective**: PF, Sharpe, max_bars, etc.

**New Objective**:
```
net_edge = (gross_pf - 1.0) - cost_drag

cost_drag = (fees + slippage) per trade × trade_count

Optimize:
  median(net_edge across folds)
  - 0.5 * max_dd_pct
  - 0.3 * trade_count_penalty  (if > 200 trades)
```

**Implementation**:
```python
def compute_net_objective(trades, config):
    gross_pf = trades['pf']
    trade_count = len(trades)
    avg_edge = trades['avg_pnl_pct']

    # Cost model
    fees_bps = config.get('fees_bps', 10.0)  # 0.10%
    slippage_bps = config.get('slippage_bps', 5.0)  # 0.05%
    cost_per_trade_pct = (fees_bps + slippage_bps) / 10000

    # Net edge after costs
    net_pf = (gross_pf - 1.0) - (cost_per_trade_pct * trade_count / initial_capital)

    # Penalties
    dd_penalty = 0.5 * trades['max_dd_pct'] / 100
    overtrade_penalty = 0.3 * max(0, (trade_count - 200) / 200)

    objective = net_pf - dd_penalty - overtrade_penalty
    return objective
```

---

### Phase 7: Walk-Forward Tuning (POST-IMPLEMENTATION)

**Objective**: Ensure gates + ML + thresholds generalize.

**Strategy**:
1. Train: 2022-2023 → Validate: 2024
2. Train: 2023-2024 → Validate: 2022 (reverse robustness check)
3. Final: Train 2022-2024 Q1-Q3 → Validate: 2024 Q4

**Targets**:
- 2024: 120-200 trades, PF ≥ 1.8, DD ≤ 15%
- 2022: PF ≥ 1.0 (don't bleed in bear), trap < 30% of entries
- 2023: PF ≥ 1.2 (recovery period)

**Per-Archetype Constraints**:
- No archetype with ≥20 trades and PF < 0.9
- trap_within_trend: < 15% of total entries

---

## Implementation Sequence (Recommended)

### Week 1: Core Gating + Anti-Churn
1. ✅ Create state_aware_gates.py
2. Integrate into logic_v2_adapter.py (trap, OB, VE first)
3. Add anti-churn exit logic
4. Test on 2024: expect 400-500 trades (improvement but not final)

### Week 2: Budget + ML v2
5. Implement opportunity budget
6. Train ML quality v2 on 2022-2024
7. Test on 2024: expect 200-300 trades

### Week 3: Cost-Aware Optimization
8. Implement cost-aware objective
9. Run walk-forward tuning
10. Validate 2022-2024 full cycle

### Week 4: Production Hardening
11. Add comprehensive logging (ParamEcho, RegimeEcho, CostTelemetry)
12. Stress test on edge cases (2022 Q1 crash, 2024 Q4 rally)
13. Document final parameter surfaces

---

## Success Criteria

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **2024 Trades** | 643 | 120-200 | 🎯 Target |
| **2024 PF** | 0.95 | ≥ 1.8 | 🎯 Target |
| **2024 trap %** | 95% | < 15% | 🎯 Target |
| **2022 PF** | 1.01 | ≥ 1.0 | 🎯 Maintain |
| **GMM Working** | ✅ | ✅ | ✅ Done |

---

## Files Reference

### Created
- ✅ `engine/archetypes/state_aware_gates.py` - State-aware gate surfaces
- ✅ `bin/train_gmm_v3.2_balanced.py` - Balanced GMM trainer
- ✅ `models/regime_gmm_v3.2_balanced.pkl` - Fixed GMM model

### To Create
- `engine/exits/anti_churn.py` - Zone lockout + min-hold logic
- `engine/opportunity_budget.py` - Dynamic entry cost system
- `bin/train_ml_quality_v2.py` - ML filter retraining script
- `bin/walk_forward_optimizer.py` - Cross-validation tuner

### To Modify
- `engine/archetypes/logic_v2_adapter.py` - Integrate state gates
- `bin/backtest_knowledge_v2.py` - Add budget, ML v2, cost objective
- `configs/baseline_btc_bear_archetypes_adaptive_v3.2.json` - Add state_aware_gates config

---

## Next Immediate Step

**ACTION**: Integrate state_aware_gates into logic_v2_adapter.py for trap_within_trend, order_block_retest, and volume_exhaustion archetypes.

**Expected Impact**: 643 → ~400 trades (38% reduction) with improved quality (PF 0.95 → 1.1+).

**Command to Test**:
```bash
python3 bin/backtest_knowledge_v2.py \
  --asset BTC \
  --config configs/baseline_btc_bear_archetypes_adaptive_v3.2_state_gates.json \
  --start 2024-01-01 \
  --end 2024-12-31
```

---

## Architecture Principles Maintained

✅ **GMM v3.2**: Regime detection working correctly
✅ **11 Archetypes**: All active, routing intact
✅ **ML Filter**: Will be retrained (not removed)
✅ **Multi-TF**: MTF alignment preserved
✅ **Exits**: Macro echo, liquidity trap, neutralization all intact
✅ **No Simplification**: Full Bull Machine stack preserved

**We're not dumbing down - we're making it smarter.**
