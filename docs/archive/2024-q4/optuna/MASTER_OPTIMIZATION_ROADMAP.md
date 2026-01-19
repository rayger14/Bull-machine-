# Master Optimization Roadmap - Bull Machine Evolution
## From Rule-Based Optimizer → Learning Engine

**Created**: 2025-11-06
**Status**: Phase 1A in progress (trap optimization running)
**Vision**: Transform Bull Machine into a context-aware, self-learning system

---

# 🎯 EXECUTIVE SUMMARY

## Current State
- **Baseline**: +$1,140 over 3 years (125 trades, 50.4% WR, PF 1.42)
- **Problem**: Trap archetype (83% of trades) has net loss -$353
- **Phase 1A**: Optuna trap optimization running (114/200 trials, ETA ~1.5h)

## Target State (12 Months)
- **Phase 1-2**: Classical optimization → +$1,500-2,000/year
- **Phase 3-4**: ML integration → +$800-1,200/year additional
- **Phase 5-6**: Temporal + RL → +$1,000-1,500/year additional
- **Combined**: $380/year baseline → $3,500-5,000/year

## Architecture Evolution
```
Current:  Rules + Thresholds + Manual Tuning
    ↓
Phase 1:  Rules + Optuna-Tuned Thresholds
    ↓
Phase 2:  Rules + Optuna + XGBoost Surrogate
    ↓
Phase 3:  Rules + MLP Meta-Fusion (Quality Multiplier)
    ↓
Phase 4:  Rules + MLP + Transformer (Temporal Context)
    ↓
Phase 5:  Rules + MLP + Transformer + RL (Specter)
    ↓
Final:    Fully Adaptive Learning Engine
```

---

# 📋 PHASE 1: CLASSICAL OPTIMIZATION (Current)

## Phase 1A: Trap Optimization ✅ IN PROGRESS

### Status
- [🚧] Run 200-trial Optuna optimization (114/200 complete, ETA 1.5h)
- [ ] Execute validation plan (7 phases)
- [ ] Analyze results and document findings
- [ ] Accept/reject optimized parameters

### When Complete (Actions)
1. **Immediate Analysis** (30 min)
   ```bash
   # Run validation script
   python3 bin/validate_optuna_results.py \
     --study-dir results/optuna_trap_v10_full \
     --validation-plan OPTUNA_VALIDATION_PLAN.md
   ```

2. **Fixed-Size Validation** (CRITICAL - 1 hour)
   - Re-run winner with fixed 0.8% risk (no Kelly, no archetype sizing)
   - Compare to baseline with same fixed sizing
   - **Decision gate**: If no improvement → REJECT, re-run v2

3. **Rolling OOS Validation** (2 hours)
   - Test on 22H1→22H2, 22→23, 22-23H1→23H2, 22-23→24
   - Require median PF > 1.3, min PF > 1.0

4. **Regime Stratification** (1 hour)
   - Break down by RISK_ON/OFF/NEUTRAL/CRISIS/TRANSITIONAL
   - Ensure PF > 1.0 in 4/5 regimes

5. **Trade-Level Diagnostics** (1 hour)
   - Compare optimized vs baseline trade-by-trade
   - Identify where improvements came from

6. **Session Analysis** (30 min)
   - Check ASIA/EUROPE/US breakdown
   - Flag if one session is catastrophic

7. **Slippage Sensitivity** (30 min)
   - Add 3bp per trade + 1bp stop slippage
   - Verify improvement persists

### Deliverables
- [ ] `TRAP_OPTIMIZATION_RESULTS.md` - Full analysis
- [ ] `results/trap_validation/` - All validation outputs
- [ ] Decision: ACCEPT / REJECT / CONDITIONAL

---

## Phase 1B: Improved Trap Optimizer v2 ⏳ NEXT

**Trigger**: If Phase 1A fails fixed-size validation OR shows weak robustness

### Improvements Over v1
1. **Fixed Position Sizing**
   ```python
   'position_sizing': {
       'mode': 'fixed_fractional',
       'base_risk_per_trade_pct': 0.8,  # Fixed 0.8% risk
       'confidence_scaling': False,
       'archetype_quality_weight': 0.0  # DISABLE
   }
   ```

2. **Better Objective Function**
   ```python
   def objective(trial):
       # Risk-adjusted expectancy
       expectancy_R = total_pnl / (total_trades * risk_per_trade)
       stability = 1.0 / (1.0 + np.std(R_per_trade))

       # Soft penalties
       dd_penalty = 5.0 * max(0, dd - 0.10)
       trade_penalty = 2.0 * max(20 - total_trades, 0)

       score = (expectancy_R * np.sqrt(total_trades)) * stability
       score -= dd_penalty
       score -= trade_penalty

       return score
   ```

3. **Rolling Windows Built-In**
   ```python
   splits = [
       ('2022-01-01', '2022-06-30', '2022-07-01', '2022-12-31'),
       ('2022-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
       ('2022-01-01', '2023-06-30', '2023-07-01', '2023-12-31'),
       ('2022-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
   ]

   scores = []
   for train_start, train_end, test_start, test_end in splits:
       result = run_backtest(...)
       scores.append(result['score'])

   return np.median(scores)  # Optimize for median robustness
   ```

4. **Regime-Stratified Scoring**
   ```python
   # Get per-regime PF
   regime_pfs = {}
   for regime in ['RISK_ON', 'RISK_OFF', 'NEUTRAL', 'CRISIS']:
       regime_trades = trades[trades['regime'] == regime]
       regime_pfs[regime] = calculate_pf(regime_trades)

   # Minimize worst-case or average
   score = np.min(list(regime_pfs.values()))  # Robustness
   # OR
   score = np.mean(list(regime_pfs.values()))  # Balanced
   ```

5. **Pruning & Speed**
   ```python
   study = optuna.create_study(
       direction='maximize',
       sampler=optuna.samplers.TPESampler(
           multivariate=True,  # Learn param interactions
           group=True,
           seed=42
       ),
       pruner=optuna.pruners.MedianPruner(
           n_startup_trials=20,
           n_warmup_steps=5
       )
   )
   ```

6. **Feature Caching**
   ```python
   # Pre-compute once
   df = pd.read_parquet('data/features_mtf/BTC_1H_2022-2024.parquet')
   df['regime_label'] = regime_detector.classify(df)
   df.to_parquet('data/features_with_regime_cached.parquet')

   # In trials, just load
   df = pd.read_parquet('data/features_with_regime_cached.parquet')
   ```

7. **Slippage Model**
   ```python
   # Add realistic costs
   trade_cost = entry_price * size * 0.0003  # 3bp
   stop_slippage = stop_price * size * 0.0001  # 1bp
   adjusted_pnl = raw_pnl - trade_cost - stop_slippage
   ```

### Tasks
- [ ] Create `bin/optuna_trap_v10_v2.py` with all improvements
- [ ] Add `bin/cache_features_with_regime.py` for pre-computation
- [ ] Add `bin/validate_optuna_results.py` automation script
- [ ] Run 100-trial optimization (faster with caching + pruning)
- [ ] Execute full validation suite
- [ ] Document improvements vs v1

### Timeline
- **Code**: 2-3 hours
- **Run**: 4-5 hours (faster with pruning)
- **Validation**: 4-6 hours
- **Total**: 1-2 days

---

## Phase 1C: Order Block Retest Optimization 🔥 HIGH PRIORITY

**Goal**: Scale from 3.3 to 10 trades/year while maintaining 90% WR

### Parameters to Optimize
```python
params = {
    'ob_quality_threshold': [0.55, 0.85],        # Currently hardcoded 0.6
    'disp_z_min': [0.8, 2.0],                     # Displacement requirement
    'mitigation_age_max_bars': [0, 200],          # How old can OB be
    'eq_distance_max_atr': [0.5, 3.0],           # How far from equilibrium
    'retest_window_bars': [1, 12],                # Window for retest
    'stop_multiplier': [0.9, 1.6],                # Stop distance
    'wyckoff_min': [0.25, 0.45],                  # Wyckoff requirement
    'boms_strength_min': [0.20, 0.40]             # BOMS requirement
}
```

### Challenge
Currently, OB thresholds are **hardcoded** in `engine/archetypes/logic_v2_adapter.py:362-374`:

```python
def _check_B(self, ctx: RuntimeContext) -> bool:
    """Archetype B: Order Block Retest."""
    fusion_th = ctx.get_threshold('order_block_retest', 'fusion', 0.374)

    # HARDCODED THRESHOLDS:
    boms_str = self.g(ctx.row, "boms_strength", 0.0)
    wyckoff = self.g(ctx.row, "wyckoff_score", 0.0)

    return (bos_bullish and
            boms_str >= 0.30 and   # HARDCODED
            wyckoff >= 0.35 and    # HARDCODED
            fusion >= fusion_th)
```

### Solution Options

**Option A: Make Configurable (Recommended)**
```python
# In config
'archetypes': {
    'order_block_retest': {
        'boms_strength_min': 0.30,
        'wyckoff_min': 0.35,
        'ob_quality_threshold': 0.6,
        'disp_z_min': 1.0,
        'stop_multiplier': 1.2
    }
}

# In logic
def _check_B(self, ctx: RuntimeContext) -> bool:
    fusion_th = ctx.get_threshold('order_block_retest', 'fusion', 0.374)

    # READ FROM CONFIG
    boms_min = ctx.config.get('archetypes', {}).get('order_block_retest', {}).get('boms_strength_min', 0.30)
    wyckoff_min = ctx.config.get('archetypes', {}).get('order_block_retest', {}).get('wyckoff_min', 0.35)

    boms_str = self.g(ctx.row, "boms_strength", 0.0)
    wyckoff = self.g(ctx.row, "wyckoff_score", 0.0)

    return (bos_bullish and
            boms_str >= boms_min and
            wyckoff >= wyckoff_min and
            fusion >= fusion_th)
```

**Option B: Use Threshold Policy Only (Quick Fix)**
- Only optimize `fusion_threshold` via existing threshold_policy
- Leave boms/wyckoff hardcoded
- **Pro**: No code changes
- **Con**: Limited optimization scope

**Option C: Defer to Phase 4 (PyTorch)**
- Use ML quality classifier to replace hardcoded checks
- **Pro**: More powerful
- **Con**: Delays gains by months

### Tasks
- [ ] **Code Changes** (2-3 hours)
  - [ ] Modify `engine/archetypes/logic_v2_adapter.py` to read OB params from config
  - [ ] Add OB parameter schema to config validation
  - [ ] Test with baseline config (ensure no regression)

- [ ] **Create Optimizer** (2 hours)
  - [ ] Create `bin/optuna_ob_retest_v10.py`
  - [ ] Use improved v2 patterns (fixed sizing, rolling OOS, etc.)
  - [ ] Objective: Maximize (trade_count × WR) subject to WR > 70%

- [ ] **Run & Validate** (8-10 hours)
  - [ ] 100-trial optimization
  - [ ] Full validation suite
  - [ ] Verify no degradation in WR

### Expected Gain
- **Current**: 3.3 trades/year, 90% WR, $152 avg win
- **Target**: 10 trades/year, 75%+ WR, $120+ avg win
- **Impact**: +$800-1,000/year

### Timeline
- **Code**: 1 day
- **Optimization**: 1 day
- **Validation**: 1 day
- **Total**: 3 days

---

## Phase 1D: Bear Market Optimization ⚠️ MEDIUM PRIORITY

**Goal**: Reduce 2022 loss from -$965 to -$400 max

### Parameters to Optimize
```python
params = {
    # Risk parameters (capital preservation focus)
    'base_risk_per_trade_pct': [0.4, 1.0],
    'initial_stop_atr_multiplier': [1.5, 2.5],
    'trailing_atr_multiplier': [1.2, 2.0],

    # Quality gates (be more selective)
    'entry_threshold_confidence': [0.75, 0.85],
    'min_structural_quality': [0.65, 0.80],
    'min_timing_quality': [0.60, 0.75],

    # Fusion weights (may need liquidity emphasis in bear)
    'weight_wyckoff': [0.20, 0.30],
    'weight_liquidity': [0.25, 0.35],
    'weight_momentum': [0.20, 0.30],
    'weight_smc': [0.15, 0.25]
}
```

### Objective Function
```python
def objective(trial):
    # Bear market specific: minimize loss + improve WR
    pnl = results['total_pnl']
    wr = results['win_rate']
    pf = results['profit_factor']
    dd = results['max_drawdown']
    trades = results['total_trades']

    # Constraints
    if dd > 0.15:  # More lenient for bear
        return float('-inf')
    if trades < 10:
        return float('-inf')

    # Objective: Minimize absolute loss + reward higher WR
    # Bear will lose money, just lose less
    score = (-pnl / 100) + (wr * 10)

    # Bonus for PF approaching 1.0
    if pf > 0.8:
        score += (pf - 0.8) * 5

    return score
```

### Existing Script
- `bin/optuna_bear_v10.py` exists but uses subprocess approach
- Needs updating to RouterAwareBacktest pattern

### Tasks
- [ ] Update `bin/optuna_bear_v10.py` to match trap v2 architecture
- [ ] Use 2022-only data (H1 train, H2 validate)
- [ ] Add bear-specific objective function
- [ ] Run 60-80 trials
- [ ] Validate on full 2022 + spot check 2023

### Expected Gain
- **Current**: 2022 loss -$965
- **Target**: 2022 loss -$400 max
- **Impact**: +$565 total improvement

### Timeline
- **Update script**: 2-3 hours
- **Run**: 3-4 hours
- **Validate**: 2 hours
- **Total**: 1 day

---

## Phase 1E: Vacuum→Grab (S) Optimization 💎 HIGH UPSIDE

**Goal**: Re-enable silent archetype with optimized parameters

### Parameters to Optimize
```python
params = {
    'void_score_min': [0.55, 0.85],              # Liquidity void strength
    'sweep_quality_min': [0.5, 0.9],             # Sweep quality
    'reclaim_bars_max': [1, 6],                  # How fast must reclaim
    'followthrough_disp_z': [0.8, 1.8],          # Displacement after grab
    'stop_multiplier': [0.9, 1.5],
    'fusion_threshold': [0.40, 0.70]
}
```

### Current State
- Archetype **disabled** (part of silent archetypes)
- No baseline performance data
- High theoretical edge (liquidity grab patterns)

### Approach
1. **Enable in baseline config** with conservative defaults
2. **Run baseline backtest** to establish performance
3. **If promising** (WR > 50%, PF > 1.2), optimize parameters
4. **If weak**, analyze why and adjust detection logic

### Tasks
- [ ] Enable vacuum_grab in config with defaults
- [ ] Run baseline 2022-2024 backtest
- [ ] **Decision gate**: If WR < 45% → defer to Phase 4 (ML quality filter)
- [ ] If promising → create optimizer
- [ ] Run 80-100 trials
- [ ] Full validation

### Expected Gain
- **Estimate**: 15-20 trades/year, 55% WR
- **Impact**: +$400-600/year (if successful)

### Timeline
- **Enable + baseline**: 1 day
- **Optimization** (if promising): 1-2 days
- **Total**: 2-3 days

---

## Phase 1F: Wick Lies (K) & False Volume Break (L) 🎲 MEDIUM UPSIDE

Similar approach to Vacuum→Grab. Enable, baseline, optimize if promising.

### Timeline Each
- 2-3 days per archetype

---

# 📋 PHASE 2: META-OPTIMIZATION (Router, Fusion, Exits)

## Phase 2A: Fusion & Router Parameters 🎯 HIGH IMPACT

**Goal**: Optimize global decision-making, not just individual archetypes

### Parameters to Optimize

#### Router-Level
```python
params = {
    'enter_threshold': [0.45, 0.75],             # Global fusion cutoff
    'conflict_resolution': ['priority', 'max_score', 'ensemble'],
    'max_share_per_archetype': [0.35, 0.70],     # Diversity cap
    'cooldown_bars_per_archetype': [0, 50],      # Prevent spam

    # Archetype weights (relative preferences)
    'archetype_weight_H': [0.5, 1.5],  # OB retest
    'archetype_weight_A': [0.5, 1.5],  # Trap
    'archetype_weight_K': [0.5, 1.5],  # Wick lies
    'archetype_weight_S': [0.5, 1.5],  # Vacuum grab
}
```

#### Fusion Weights
```python
params = {
    'weight_wyckoff': [0.15, 0.35],
    'weight_liquidity': [0.20, 0.40],
    'weight_momentum': [0.15, 0.35],
    'weight_smc': [0.10, 0.30],

    # Alignment requirements
    'mtf_align_required_count': [1, 3],          # How many TFs must agree
    'momentum_band_z': [0.3, 1.5],               # Momentum threshold
    'liquidity_confluence_min': [0.4, 0.8]       # Liquidity agreement
}
```

### Approach
1. **Lock archetype params first** (from Phase 1)
2. **Optimize router globally** across all archetypes
3. **Use multi-objective** (Pareto front):
   - Maximize: Total PNL
   - Maximize: Sharpe Ratio
   - Minimize: Max Drawdown
   - Maximize: Trade Count (subject to quality)

### Tasks
- [ ] Create `bin/optuna_router_v10.py`
- [ ] Use locked archetype configs as base
- [ ] Implement multi-objective optimization
- [ ] Run 150-200 trials
- [ ] Analyze Pareto front
- [ ] Select best compromise

### Expected Gain
- **Impact**: +$300-500/year from better global decisions

### Timeline
- **Code**: 2 days
- **Run**: 1-2 days
- **Analysis**: 1 day
- **Total**: 4-5 days

---

## Phase 2B: Exit Optimization 💰 MASSIVE IMPACT

**Goal**: Optimize TP levels, partials, breakeven, trailing

### Parameters to Optimize
```python
params = {
    # Take profit levels (in R)
    'tp1_R': [0.6, 1.2],
    'tp2_R': [1.2, 2.5],
    'tp3_R': [2.0, 4.0],

    # Partial sizes
    'tp1_size_pct': [0.25, 0.60],  # How much to take at TP1
    'tp2_size_pct': [0.20, 0.50],  # TP2 partial
    # tp3 = remainder

    # Breakeven logic
    'move_to_breakeven_trigger': ['on_tp1', 'after_1R', 'after_bars'],
    'breakeven_buffer_atr': [0.0, 0.3],  # Small buffer above BE

    # Trailing
    'trail_type': ['none', 'swing_low', 'atr', 'hybrid'],
    'atr_trail_multiplier': [1.0, 3.0],
    'swing_lookback_bars': [5, 20]
}
```

### Why This Matters
- **Current**: Fixed TP levels may leave money on table
- **Opportunity**: Dynamic exits could improve R:R from 1.5:1 to 2.5:1
- **Impact**: Even 0.2R improvement per trade × 125 trades/year = +$2,500/year

### Approach
1. **Use locked entry params**
2. **Optimize exits independently**
3. **Test across regime types**
4. **Ensure no degradation in win rate**

### Tasks
- [ ] Create `bin/optuna_exits_v10.py`
- [ ] Modify backtest to support dynamic exits
- [ ] Run 200 trials
- [ ] Regime-stratified validation
- [ ] Compare equity curves

### Expected Gain
- **Impact**: +$400-800/year from better exit timing

### Timeline
- **Code**: 3-4 days (exit logic more complex)
- **Run**: 1-2 days
- **Validation**: 2 days
- **Total**: 6-8 days

---

## Phase 2C: Position Sizing Optimization ⚖️ FINAL TUNING

**Goal**: Optimize Kelly fraction, max risk, concurrent limits

### Parameters to Optimize
```python
params = {
    'kelly_fraction_cap': [0.1, 0.6],
    'base_risk_per_trade_R': [0.3, 1.2],
    'max_risk_per_trade_R': [0.8, 2.0],
    'max_simultaneous_trades': [1, 5],
    'session_exposure_cap_R': [0.5, 1.5],  # Sum of live R

    # Confidence scaling
    'confidence_scaling_enabled': [True, False],
    'archetype_quality_weight': [0.0, 0.5]
}
```

### Critical
- **This comes LAST** after entries and exits are locked
- **Why**: Sizing can mask poor entries/exits
- **Approach**: Use locked entry/exit configs, only tune sizing

### Tasks
- [ ] Create `bin/optuna_sizing_v10.py`
- [ ] Use locked entry + exit configs
- [ ] Optimize sizing in isolation
- [ ] Validate on multiple market conditions
- [ ] Stress test with worst-case scenarios

### Expected Gain
- **Impact**: +$200-400/year from better risk management

### Timeline
- **Code**: 2 days
- **Run**: 1 day
- **Validation**: 2 days
- **Total**: 5 days

---

# 📋 PHASE 3: ML INTEGRATION - MLP META-FUSION

## Phase 3A: MLP Quality Multiplier 🧠 GAME CHANGER

**Goal**: Replace rule-based fusion with learned quality scoring

### Architecture
```python
class MetaFusionMLP(nn.Module):
    """
    Lightweight MLP for fusion quality scoring.
    Input: Per-bar feature vector
    Output: Quality score [0, 1]
    """
    def __init__(self, input_dim=49, hidden_dims=[128, 64, 32]):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.2),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(0.2),

            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dims[2]),

            nn.Linear(hidden_dims[2], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

### Integration
```python
# In backtest
rule_based_fusion = calculate_fusion_score(...)

if meta_fusion_enabled:
    features = extract_features(row)
    quality_score = meta_fusion_mlp(features)

    # Quality multiplier
    alpha = 0.3  # Tunable
    multiplier = np.clip(
        1 + alpha * (quality_score - 0.5),
        0.75,  # Floor
        1.25   # Ceiling
    )

    final_fusion = rule_based_fusion * multiplier
else:
    final_fusion = rule_based_fusion
```

### Training Data
```python
# From historical trades
features = [
    'wyckoff_score', 'liquidity_score', 'momentum_score', 'smc_score',
    'ob_quality', 'fvg_quality', 'sweep_depth', 'displacement_z',
    'regime_label', 'volatility_regime', 'funding_Z',
    'volume_profile_poc', 'delta_divergence',
    # ... 49 total features
]

# Label: Trade outcome
target = {
    'profit_R': 1.5,  # Regression target: R achieved
    'success': 1      # Classification: hit TP1 (yes/no)
}

# Or binary classification
target = 1 if trade_pnl > 0 else 0
```

### Tasks

#### 3A.1: Data Preparation (2-3 days)
- [ ] Create `bin/build_ml_training_dataset.py`
- [ ] Extract features from 2020-2023 trade logs
- [ ] Create labels (profit_R, success, etc.)
- [ ] Split: Train (2020-2022), Val (2023), Test (2024)
- [ ] Save to `data/ml_training/meta_fusion_dataset.parquet`

#### 3A.2: Model Training (2-3 days)
- [ ] Create `models/meta_fusion_mlp.py`
- [ ] Implement training loop with:
  - Loss: MSE for regression OR BCE for classification
  - Optimizer: AdamW with cosine schedule
  - Early stopping on val loss
  - L2 regularization + dropout
- [ ] Train multiple model sizes (32/64/128 hidden)
- [ ] Select best on validation set
- [ ] Test on 2024 (held-out)

#### 3A.3: Integration (2 days)
- [ ] Add MLP to `engine/runtime/context.py`
- [ ] Add `meta_fusion_enabled` config flag
- [ ] Modify fusion calculation to use MLP
- [ ] Add inference caching (don't recompute every bar)

#### 3A.4: Backtesting (1-2 days)
- [ ] Run 2022-2024 backtest with MLP enabled
- [ ] Compare to rule-based baseline
- [ ] Analyze improvement sources
- [ ] Validate no degradation in any regime

#### 3A.5: Hyperparameter Tuning (1 day)
- [ ] Tune `alpha` (quality multiplier strength) via Optuna
- [ ] Tune `multiplier_bounds` [floor, ceiling]
- [ ] Tune `min_confidence` threshold

### Expected Gain
- **Impact**: +$400-600/year from better setup selection
- **Bonus**: Improves ALL archetypes simultaneously

### Timeline
- **Total**: 10-12 days

---

## Phase 3B: Archetype-Specific Quality Classifiers 🎯 PRECISION

**Goal**: Train per-archetype MLPs to filter false positives

### Models to Train
```python
# One classifier per archetype
models = {
    'order_block_retest': OrderBlockQualityNet(input_dim=25),
    'trap_within_trend': TrapQualityClassifier(input_dim=30),
    'vacuum_grab': VacuumQualityNet(input_dim=28),
    'wick_lies': WickQualityClassifier(input_dim=22)
}
```

### Architecture (Example)
```python
class OrderBlockQualityNet(nn.Module):
    def __init__(self, input_dim=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

### Features (Per-Archetype)
```python
# Order Block Retest
ob_features = [
    'ob_body_size_atr', 'ob_volume_ratio', 'ob_age_bars',
    'mitigation_count', 'displacement_before_z', 'displacement_after_z',
    'retest_precision_atr', 'htf_alignment_score',
    'wyckoff_score', 'momentum_at_retest', 'regime_label',
    # ... 25 total
]

# Trap Within Trend
trap_features = [
    'trap_depth_atr', 'volume_spike_ratio', 'reclaim_speed_bars',
    'trend_strength_adx', 'fvg_quality', 'liquidity_sweep_depth',
    'confirmation_bars_actual', 'rejection_wick_ratio',
    # ... 30 total
]
```

### Integration
```python
# In archetype detection
if archetype == 'order_block_retest':
    rule_based_score = calculate_ob_score(...)

    if use_ml_filter:
        features = extract_ob_features(row)
        ml_quality = ob_quality_net(features)

        if ml_quality < 0.6:  # Tunable threshold
            return False  # Reject setup

    return rule_based_score > threshold
```

### Tasks

#### 3B.1: Per-Archetype Datasets (3-4 days)
- [ ] Extract archetype-specific features from trade logs
- [ ] Label with outcome (profit_R, success)
- [ ] Balance datasets (equal pos/neg examples)
- [ ] Save to `data/ml_training/{archetype}_dataset.parquet`

#### 3B.2: Train Classifiers (4-5 days)
- [ ] Train 4 separate models (one per archetype)
- [ ] Use focal loss or weighted BCE (handle imbalance)
- [ ] Cross-validate on different market periods
- [ ] Save best models

#### 3B.3: Integration (2 days)
- [ ] Add ML quality filters to archetype logic
- [ ] Add config flags per archetype
- [ ] Implement inference pipeline

#### 3B.4: Validation (2-3 days)
- [ ] Backtest with ML filters enabled
- [ ] Compare to rule-based
- [ ] Measure precision/recall tradeoff
- [ ] Tune thresholds

### Expected Gain
- **Impact**: +$300-400/year from filtering false setups

### Timeline
- **Total**: 11-14 days

---

# 📋 PHASE 4: TRANSFORMER TEMPORAL LAYER

## Phase 4A: Temporal Context Encoder 🔮 NEXT LEVEL

**Goal**: Learn temporal patterns, phase detection, liquidity rhythm

### Architecture
```python
class MarketTransformer(nn.Module):
    """
    Transformer encoder for temporal market context.
    Input: Sequence of bars (e.g., last 240 bars = 10 days on 1H)
    Output: Context embedding + predictions
    """
    def __init__(self, n_features=49, d_model=64, n_heads=4, n_layers=3):
        super().__init__()
        self.feature_encoder = nn.Linear(n_features, d_model)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=128,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=n_layers
        )

        # Multiple prediction heads
        self.phase_head = nn.Linear(d_model, 4)  # Accumulation/Markup/Distribution/Markdown
        self.regime_head = nn.Linear(d_model, 5)  # RISK_ON/OFF/NEUTRAL/CRISIS/TRANS
        self.return_head = nn.Linear(d_model, 1)  # Next-bar return
        self.volatility_head = nn.Linear(d_model, 1)  # Volatility forecast

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        x = self.feature_encoder(x)

        # Transformer expects [seq_len, batch, d_model]
        x = x.transpose(0, 1)
        context = self.transformer(x)

        # Use last timestep for predictions
        final_context = context[-1]

        return {
            'context': final_context,
            'phase': self.phase_head(final_context),
            'regime': self.regime_head(final_context),
            'return': self.return_head(final_context),
            'volatility': self.volatility_head(final_context)
        }
```

### What It Learns
1. **Wyckoff Phases**: Accumulation → Markup → Distribution → Markdown
2. **Liquidity Rhythm**: When sweeps precede reversals vs continuations
3. **Volatility Clusters**: When range compression leads to expansion
4. **Regime Transitions**: Early detection of RISK_ON → RISK_OFF shifts
5. **Multi-Asset Context**: BTC + DXY + ETH correlation patterns

### Integration
```python
# In backtest
if temporal_context_enabled:
    # Get last 240 bars
    window = df.iloc[i-240:i]
    features = extract_features(window)

    # Transformer inference
    context_output = market_transformer(features)

    # Use predictions to adjust signals
    phase = context_output['phase'].argmax()
    regime_pred = context_output['regime'].argmax()

    # Phase-aware adjustments
    if phase == DISTRIBUTION and signal == LONG:
        signal_strength *= 0.7  # Be cautious in distribution

    if phase == MARKUP and signal == LONG:
        signal_strength *= 1.2  # Aggressive in markup
```

### Tasks

#### 4A.1: Sequence Dataset (1 week)
- [ ] Create `bin/build_sequence_dataset.py`
- [ ] Generate rolling windows (e.g., 240-bar sequences)
- [ ] Label with:
  - Current Wyckoff phase (manual labels on key periods)
  - Future returns (1H, 4H, 1D ahead)
  - Regime transitions
- [ ] Save to `data/ml_training/sequences/`

#### 4A.2: Model Training (1-2 weeks)
- [ ] Implement MarketTransformer
- [ ] Multi-task learning (phase + regime + return simultaneously)
- [ ] Train with gradient accumulation (sequences are large)
- [ ] Validate on held-out periods
- [ ] Analyze attention weights for interpretability

#### 4A.3: Integration (3-4 days)
- [ ] Add to runtime context
- [ ] Implement inference pipeline (batch sequences efficiently)
- [ ] Add phase-aware logic to signal generation

#### 4A.4: Backtesting (3-4 days)
- [ ] Run with temporal context enabled
- [ ] Compare to non-temporal baseline
- [ ] Analyze improvement sources

### Expected Gain
- **Impact**: +$600-900/year from temporal awareness

### Timeline
- **Total**: 4-5 weeks

---

## Phase 4B: Multi-Asset Context 🌍 CORRELATION AWARENESS

**Goal**: Learn cross-asset patterns (BTC, ETH, DXY, TOTAL, etc.)

### Architecture Extension
```python
class MultiAssetTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.btc_encoder = AssetEncoder(n_features=49)
        self.eth_encoder = AssetEncoder(n_features=49)
        self.dxy_encoder = AssetEncoder(n_features=20)
        self.total_encoder = AssetEncoder(n_features=30)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4
        )

        self.fusion_head = nn.Linear(256, 64)  # Combine all assets
```

### Tasks
- **Timeline**: 3-4 weeks

---

# 📋 PHASE 5: REINFORCEMENT LEARNING (SPECTER)

## Phase 5A: Self-Learning Position Manager 🤖 THE DREAM

**Goal**: Learn to adapt entries, exits, sizing from realized PNL

### Architecture
```python
class SpecterRL(nn.Module):
    """
    Reinforcement learning agent for trade management.
    State: Current market features + open position state
    Action: Entry size, exit timing, stop adjustment
    Reward: Realized R-multiple
    """
    def __init__(self):
        super().__init__()
        self.policy_net = nn.Sequential(...)
        self.value_net = nn.Sequential(...)
```

### Approach
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Environment**: Custom TradingEnv wrapping backtest
- **Reward**: Sharpe-weighted R-multiple
- **Training**: Online learning from live trades

### Tasks
- **Timeline**: 2-3 months (complex)

---

# 🗓️ EXECUTION TIMELINE

## Immediate (While Trap Opt Runs) - 1.5 Hours
- [🚧] Wait for trap optimization to complete
- [🚧] Monitor progress
- [✅] Create this TODO
- [ ] Start coding improved optimizer v2

## Week 1
- [ ] Complete trap validation (Phase 1A)
- [ ] Start trap v2 if needed (Phase 1B)
- [ ] Start OB retest code changes (Phase 1C)

## Week 2
- [ ] Complete OB retest optimization (Phase 1C)
- [ ] Start bear optimization (Phase 1D)
- [ ] Begin vacuum_grab baseline (Phase 1E)

## Week 3-4
- [ ] Complete bear optimization
- [ ] Complete vacuum_grab optimization (if promising)
- [ ] Start router optimization (Phase 2A)

## Month 2
- [ ] Complete router + fusion optimization (Phase 2A)
- [ ] Complete exit optimization (Phase 2B)
- [ ] Start MLP meta-fusion (Phase 3A)

## Month 3
- [ ] Complete MLP meta-fusion
- [ ] Start archetype classifiers (Phase 3B)
- [ ] Begin transformer planning (Phase 4A)

## Month 4-6
- [ ] Complete transformer temporal layer
- [ ] Multi-asset context
- [ ] Begin RL planning

---

# 📊 SUCCESS METRICS

## Phase 1 Gates
- Trap optimization passes fixed-size validation
- OB scaling achieves 8+ trades/year with WR > 70%
- Bear optimization reduces 2022 loss to < -$500
- Combined improvement > +$1,000/year

## Phase 2 Gates
- Router optimization improves Sharpe by 15%+
- Exit optimization improves avg R-multiple by 0.3+
- Sizing optimization reduces DD by 10%+

## Phase 3 Gates
- MLP meta-fusion improves WR by 3-5%
- Archetype classifiers reduce false positives by 20%+
- Combined ML impact > +$700/year

## Phase 4 Gates
- Transformer improves phase detection accuracy > 70%
- Temporal context reduces drawdown duration by 15%+
- Multi-asset awareness improves regime detection

## Phase 5 Gates
- RL agent achieves positive learning curve
- Self-adaptation improves on static config by 10%+

---

# 🎯 DECISION GATES

## Gate 1: After Trap Optimization
- **IF** fixed-size validation passes → Proceed to OB
- **IF** fails → Re-run trap v2 first
- **Criteria**: >10% PF improvement with fixed sizing

## Gate 2: After Classical Optimization (Phase 1)
- **IF** combined gain > $1,000/year → Proceed to Phase 2
- **IF** gain < $500/year → Revisit approach
- **Criteria**: Robust across all regimes

## Gate 3: Before ML Integration (Phase 3)
- **IF** Phase 1-2 gains validated → Build ML datasets
- **IF** unstable → Stabilize classical first
- **Criteria**: Baseline must be solid

## Gate 4: Before Transformer (Phase 4)
- **IF** MLP meta-fusion shows clear value → Invest in transformer
- **IF** marginal → Defer to later
- **Criteria**: >$400/year gain from Phase 3

## Gate 5: Before RL (Phase 5)
- **IF** Phases 1-4 stable and profitable → Experiment with RL
- **IF** any phase unstable → Fix foundations
- **Criteria**: System must be production-ready

---

# 📝 IMPLEMENTATION NOTES

## Code Organization
```
bin/
  optuna_trap_v10_v2.py          # Improved trap optimizer
  optuna_ob_retest_v10.py        # OB optimization
  optuna_bear_v10_v2.py          # Updated bear optimizer
  optuna_router_v10.py           # Router/fusion optimization
  optuna_exits_v10.py            # Exit optimization
  optuna_sizing_v10.py           # Sizing optimization

  build_ml_training_dataset.py  # ML dataset builder
  train_meta_fusion_mlp.py       # MLP trainer
  train_archetype_classifiers.py # Per-archetype MLPs
  train_temporal_transformer.py  # Transformer trainer

  validate_optuna_results.py     # Automated validation
  cache_features_with_regime.py  # Feature caching

models/
  meta_fusion_mlp.py             # MLP model definition
  archetype_classifiers.py       # Per-archetype models
  temporal_transformer.py        # Transformer model
  specter_rl.py                  # RL agent

data/ml_training/
  meta_fusion_dataset.parquet
  ob_retest_dataset.parquet
  trap_dataset.parquet
  sequences/
    btc_sequences_2020_2023.parquet
```

## Development Principles
1. **Modularity**: Each phase independent
2. **Validation-First**: Gate every change
3. **Reproducibility**: Seed everything, save metadata
4. **Interpretability**: Always explain why improvements work
5. **Incremental**: Ship value at each phase

---

**Generated**: 2025-11-06
**Current Phase**: 1A (trap optimization 114/200 trials)
**Next Action**: Wait for completion, execute validation, then proceed to next optimizer
**Timeline**: 12 months to full learning engine
