# Phase 1 Progress Update - Router v10 Optimization

**Date**: 2025-11-05
**Status**: Classical Optimization In Progress
**Current Session**: Continuing from Phase 0 completion

---

## 🚀 Currently Running

### Trap-Within-Trend Optimization (Active)
- **Process**: PID 24820 - Running
- **Progress**: ~25 minutes CPU time (0/200 trials visible in log)
- **Status**: First trial still executing (verbose backtest logs)
- **Expected**: ~4 hours total runtime
- **Monitor**: `tail -f /tmp/optuna_trap_full.log`

**What It's Optimizing**:
```python
trap_quality_threshold: [0.45, 0.65]  # Currently 0.35 (too lenient)
trap_confirmation_bars: [2, 5]        # Need more confirmation
trap_volume_ratio: [1.5, 2.5]         # Currently 1.2 (too low)
trap_stop_multiplier: [0.8, 1.5]      # Currently 2.5 (CRITICAL - too wide)
```

**Expected Output**:
- `results/optuna_trap_v10_full/best_params.json`
- `results/optuna_trap_v10_full/trials.csv`
- `results/optuna_trap_v10_full/trap_optimized_bull.json`
- `results/optuna_trap_v10_full/trap_optimized_bear.json`

---

## ✅ Completed This Session

### 1. Trap Optimization Script Created & Tested
- **File**: `bin/optuna_trap_v10.py`
- **Architecture**: RouterAwareBacktest integration
- **Test Results** (2 trials):
  - Training: PF 0.88, WR 41.4%, -$222 PNL
  - Validation: PF 3.37, WR 66.0%, +$1,479 PNL
  - **Improvement**: +$117/year (+8.6% vs baseline)

### 2. Bear Optimization Script Exists
- **File**: `bin/optuna_bear_v10.py`
- **Status**: Needs architecture update for RouterV10 integration
- **Note**: Currently uses subprocess approach (inconsistent with trap script)
- **Next**: Test and update to match trap optimization pattern

---

## 📋 Remaining Phase 1 Tasks

### High Priority (After Trap Optimization)
1. **Analyze Trap Results** (~30 min)
   - Review best parameters from 200 trials
   - Validate on full 2022-2024 period
   - Compare to baseline (+$1,140, 125 trades)
   - Acceptance: Must show improvement in PF, WR, and R:R

2. **Bear Config Optimization** (~3 hours)
   - Update script to use RouterAwareBacktest
   - Run on 2022 data only (H1 train, H2 validate)
   - Target: Reduce loss from -$965 to -$400 max
   - Expected gain: +$565 total

3. **Order Block Retest Scaling** (~2-4 hours)
   - **Challenge**: Detection thresholds are hardcoded in logic
   - **Current**: boms_str >= 0.30, wyckoff >= 0.35, fusion >= 0.374
   - **Options**:
     a. Modify archetype logic to read from config
     b. Use threshold_policy overrides (fusion only)
     c. Defer to Phase 2 (PyTorch quality classifiers)
   - **Recommended**: Option (a) - Add configurable thresholds

### Medium Priority
4. **Combined Validation** (~1 hour)
   - Test all Phase 1 optimizations together
   - Run full 2022-2024 backtest with:
     - Trap-optimized configs
     - Bear-optimized configs
     - Combined improvement measurement
   - Acceptance gate: +$1,000/year minimum

---

## 🎯 Phase 1 Goals vs Progress

| Optimization | Target | Status | Expected Gain |
|--------------|--------|--------|---------------|
| Trap-within-trend fix | WR 55%+, PF 1.5+ | 🚧 **Running** | +$400-600/year |
| Bear config (2022) | Loss < -$400 | ⏳ Script ready | +$565 total |
| OB retest scaling | 10 trades/year, WR 70%+ | ⏳ Needs code changes | +$800/year |
| **Phase 1 Total** | - | **30% complete** | **+$1,000-1,500/year** |

---

## 🔧 Technical Insights

### Trap Optimization Architecture (Working)
The trap optimization successfully integrates with RouterV10:

```python
class TrapOptunaOptimizer:
    def _create_config_with_params(self, trial):
        """Sample hyperparameters and inject into config."""
        # Sample from Optuna
        trap_quality = trial.suggest_float('trap_quality_threshold', 0.45, 0.65, step=0.05)
        trap_stop_mult = trial.suggest_float('trap_stop_multiplier', 0.8, 1.5, step=0.1)

        # Apply to config
        config['archetypes']['trap_within_trend'].update({
            'quality_threshold': trap_quality,
            'stop_multiplier': trap_stop_mult
        })
        return bull_config, bear_config

    def _run_backtest(self, bull_config, bear_config, df):
        """Run RouterAwareBacktest with temp config files."""
        # Save to temp files (RouterV10 requires paths, not objects)
        router = RouterV10(
            bull_config_path=bull_config_path,
            bear_config_path=bear_config_path
        )

        backtest = RouterAwareBacktest(
            df=df,
            params=KnowledgeParams(),  # Empty params
            bull_config=bull_config,    # Configs passed separately
            bear_config=bear_config,
            router=router,
            regime_detector=self.regime_detector,
            event_calendar=self.event_calendar,
            starting_capital=self.capital,
            asset='BTC'
        )

        return backtest.run()
```

**Key Learnings**:
1. RouterV10 expects config paths, not objects
2. KnowledgeParams() takes no arguments (empty container)
3. Configs are passed separately to RouterAwareBacktest
4. Temp files are created/cleaned up per trial

### OB Scaling Challenge (Blocked)
The order_block_retest archetype has hardcoded thresholds:

```python
# In engine/archetypes/logic_v2_adapter.py
def _check_B(self, ctx: RuntimeContext) -> bool:
    """Archetype B: Order Block Retest."""
    fusion_th = ctx.get_threshold('order_block_retest', 'fusion', 0.374)

    # These are HARDCODED:
    boms_str = self.g(ctx.row, "boms_strength", 0.0)
    wyckoff = self.g(ctx.row, "wyckoff_score", 0.0)

    return (bos_bullish and
            boms_str >= 0.30 and   # HARDCODED
            wyckoff >= 0.35 and    # HARDCODED
            fusion >= fusion_th)   # Configurable via threshold_policy
```

**To scale OB detection**, we need to:
1. Add archetype-specific config reading (like trap)
2. Make boms_str and wyckoff thresholds configurable
3. Re-run optimization with relaxed ranges

**Options**:
- **Quick fix**: Only optimize fusion threshold via threshold_policy
- **Proper fix**: Add config reading to archetype logic
- **Future**: PyTorch quality classifiers (Phase 2)

---

## 📈 Expected Timeline

### Today (Remaining):
- ⏳ Wait for trap optimization completion (~3-4 hours)
- ✅ Create this progress document
- ⏳ Monitor trap optimization progress

### Tomorrow:
- Analyze trap optimization results
- Validate trap-optimized config on full period
- If successful, commit and tag

### This Week:
- Update bear optimization script
- Run bear optimization (2022 focus)
- Decide on OB approach (code changes vs defer to Phase 2)
- Combined validation of Phase 1 improvements

---

## 🔍 Monitoring Commands

### Check Trap Optimization Status:
```bash
# Process status
ps aux | grep optuna_trap_v10

# View recent log output
tail -100 /tmp/optuna_trap_full.log

# Count completed trials (when trials start finishing)
grep -c "Trial.*finished" /tmp/optuna_trap_full.log

# Check best value (when available)
grep "Best trial:" /tmp/optuna_trap_full.log | tail -1
```

### When Trap Optimization Completes:
```bash
# View best parameters
cat results/optuna_trap_v10_full/best_params.json

# Analyze trials
python3 -c "
import pandas as pd
df = pd.read_csv('results/optuna_trap_v10_full/trials.csv')
print(df.describe())
print('\nTop 10 trials:')
print(df.nlargest(10, 'value'))
"

# Validate optimized config on full period
python3 bin/backtest_router_v10_full.py \
  --asset BTC \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --bull-config results/optuna_trap_v10_full/trap_optimized_bull.json \
  --bear-config results/optuna_trap_v10_full/trap_optimized_bear.json \
  --output results/router_v10_trap_optimized_validation
```

---

## 📊 Baseline Performance (Reference)

### 3-Year Combined (2022-2024)
- **Total PNL**: +$1,140.18 (+11.40%)
- **Total Trades**: 125
- **Win Rate**: 50.4%
- **Profit Factor**: 1.42
- **Max Drawdown**: 7.8%

### By Year
| Year | PNL | Trades | WR | PF | Status |
|------|-----|--------|-----|-----|--------|
| 2022 | -$965 | 32 | 25% | 0.52 | 🔥 Crisis |
| 2023 | +$744 | 38 | 55.3% | 2.41 | ✅ Recovery |
| 2024 | +$1,362 | 55 | 61.8% | 2.65 | ✅ Bull |

### By Archetype
| Archetype | Trades | WR | Avg Win | Avg Loss | Total PNL | Status |
|-----------|--------|-----|---------|----------|-----------|--------|
| order_block_retest | 10 | 90% | +$152 | -$25 | +$1,518 | 🏆 GOLDMINE |
| trap_within_trend | 104 | 46% | +$43 | -$78 | -$353 | 🔥 BROKEN |
| liquidity_compression | 5 | 60% | +$82 | -$105 | +$35 | ⚠️ Small sample |
| others | 6 | 50% | - | - | -$60 | ⚠️ Inactive |

---

## 🎯 Success Criteria for Phase 1

### Trap Optimization Success:
- [ ] Win Rate improves from 46% to 55%+
- [ ] Profit Factor improves from 0.88 to 1.5+
- [ ] Average loss reduces from -$78 to < -$50
- [ ] R:R inverts from 0.55:1 to 1.5:1+
- [ ] Full validation shows +$400-600/year gain

### Bear Optimization Success:
- [ ] 2022 loss reduces from -$965 to < -$400
- [ ] Win Rate improves from 25% to 35%+
- [ ] Drawdown stays < 15%
- [ ] Maintains minimum 20 trades for statistical validity

### OB Scaling Success:
- [ ] Trade frequency increases from 3.3 to 10/year
- [ ] Win Rate maintains > 70%
- [ ] Average win stays > $100
- [ ] Adds +$800/year in expected value

### Combined Phase 1 Gate:
- [ ] Total improvement ≥ +$1,000/year vs baseline
- [ ] No degradation in max drawdown
- [ ] All individual optimizations validate successfully
- [ ] Ready to proceed to archetype re-enablement

---

**Generated**: 2025-11-05 (continuation session)
**Phase**: 1 - Classical Optimization (30% complete)
**Next Milestone**: Trap optimization completion + validation
**Branch**: pr6a-archetype-expansion
**Tag**: v10_baseline_corrected (4246fee)
