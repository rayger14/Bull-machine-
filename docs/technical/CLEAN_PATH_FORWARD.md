# Clean Path Forward: Fixing Overfitting Without Chaos

**Date:** 2026-01-17
**Philosophy:** Fix optimization, not the engine. Freeze the truth, then iterate carefully.
**Goal:** Get 1-2 archetypes passing walk-forward validation (<30% degradation, WFE >60%)

---

## The Situation

### ✅ What We Know Works

1. **Production engine is correct** (after archetype isolation fix)
   - Walk-forward validation harness works
   - Archetypes properly isolated
   - Fees, slippage, regime gating all applied

2. **Three archetypes show promise** (H, B, S1)
   - Positive OOS returns (+7% to +10%)
   - Real signal exists, just overfit parameters
   - Worth salvaging through re-optimization

3. **Bug fix process was successful**
   - Found critical bug (archetype isolation)
   - Fixed and verified with smoke test
   - Re-ran validation, got valid results

### ❌ What We Know Is Broken

1. **All archetypes severely overfit**
   - OOS degradation: 54-104% (target: <20%)
   - Walk-Forward Efficiency: 17-46% (target: >60%)
   - Profitable windows: 13-42% (target: >60%)

2. **Root cause: Unconstrained TPE optimization**
   - Wide parameter search spaces (6+ params)
   - No regularization penalties
   - Single training window (2018-2021)
   - No statistical validation

3. **Some archetypes have critical issues**
   - K: Loses money OOS (-5.1% return)
   - S4: Only 11 trades in 6 years
   - S5: Can't validate with walk-forward (crisis-only)

---

## Step 0: Freeze the Truth (DONE)

**Goal:** Lock in tests so we never regress on bugs

**Actions Completed:**
1. ✅ Created `tests/test_production_validation_gates.py`
   - Gate 1: Archetype isolation test
   - Gate 2: Walk-forward production harness test
   - Gate 3: Sanity checks (fees, regime gating, trade counts)

2. ✅ Permanent smoke test: `bin/test_archetype_isolation_fix.py`
   - Verifies H, K, S5 produce different results
   - Would have caught original bug

**Next Time Before Deployment:**
```bash
# Run gates before ANY production changes
pytest tests/test_production_validation_gates.py -v
```

---

## Step 1: Fix Overfitting by Changing Optimization, Not the Engine

**Goal:** Re-optimize H, B, S1 with constraints to reduce degradation from 66-83% → <30%

### A) Reduce Degrees of Freedom (Fastest Win)

**Problem:** 6+ parameters per archetype = overfitting

**Solution:** Cap to 3-5 parameters, lock rest to domain defaults

**Implementation:**
```python
# OLD (Unconstrained)
H_params = {
    'fusion_threshold': (0.35, 0.85),  # Wide
    'trend_weight': (0.2, 0.7),        # Wide
    'liquidity_weight': (0.1, 0.6),    # Wide
    'wick_weight': (0.1, 0.5),         # Wide
    'volume_weight': (0.1, 0.5),       # Wide
    'smc_weight': (0.1, 0.5),          # Wide
}
# 6 parameters × wide ranges = overfitting

# NEW (Constrained)
H_params_reduced = {
    'fusion_threshold': (0.55, 0.70),  # Narrowed
    'trend_weight': (0.35, 0.55),       # Narrowed
    'liquidity_weight': (0.25, 0.45),   # Narrowed
    # Lock wick_weight, volume_weight, smc_weight to defaults
}
# 3 parameters × narrow ranges = less overfitting
```

**File:** `bin/optimize_constrained_cpcv.py` (CREATED)

### B) Add Generalization Penalties Directly Into Objective

**Problem:** TPE optimizes for max Sortino, ignores generalization

**Solution:** Penalize patterns that indicate overfitting

**Penalties Added:**
1. **Too many trades** (trade count explosion)
   - If avg_trades > 200 per fold: penalty
2. **Too few trades** (insufficient sample)
   - If avg_trades < 5 per fold: penalty
3. **Performance concentration** (one great window, many bad)
   - If std(Sortino across folds) > 0.5: penalty
4. **Excessive drawdown**
   - If max_dd > 25%: penalty

**Implementation:**
```python
# Objective function
objective = avg_sortino_across_folds
objective -= trade_count_penalty
objective -= concentration_penalty
objective -= drawdown_penalty
objective += sharpe_bonus  # Encourage balanced returns
```

### C) Validate Inside the Loop (CPCV Instead of Single Split)

**Problem:** Single train/test split (2018-2021 train, 2022-2024 test) → parameters fit specific regime

**Solution:** Combinatorial Purged Cross-Validation (CPCV)

**How CPCV Works:**
```
OLD (Single Split):
|--- Train 2018-2021 ---|--- Test 2022-2024 ---|
         ↓
   Optimize here         Validate here (too late)

NEW (CPCV - 5 Folds):
|--- Train ---|[embargo]|--- Test 1 ---|
         ↓
     Optimize across all 5 test windows

|--- Train --------------|[embargo]|--- Test 2 ---|
                              ↓
                      Optimize across all 5

... continues for 5 folds ...

Objective = average(Sortino across all 5 test windows) - Penalties
```

**Benefits:**
- Parameters must work across multiple time periods
- Harder to overfit (can't memorize one regime)
- Detects overfitting during optimization, not after

**File:** `bin/optimize_constrained_cpcv.py` (CREATED)

---

## Step 2: Re-Optimize H, B, S1 with Constraints

### Execution Plan

**Phase 1: H (Trap Within Trend)**
```bash
# Run constrained CPCV optimization
python3 bin/optimize_constrained_cpcv.py --archetype H --trials 50 --folds 5

# Expected outcome:
# - Reduced params: 3 (fusion_threshold, trend_weight, liquidity_weight)
# - CPCV objective: ~0.6-0.8 (vs original 1.29 in-sample)
# - Output: results/optimization_constrained_cpcv/H_constrained_cpcv.json
```

**Phase 2: Validate H with Walk-Forward**
```bash
# Re-run walk-forward with new params
python3 bin/walk_forward_production_engine.py --archetype H --config results/optimization_constrained_cpcv/H_constrained_cpcv.json

# Target metrics:
# - OOS degradation: <30% (vs original 66%)
# - Profitable windows: >60% (vs original 25%)
# - OOS Sortino: >0.5 (vs original 0.44)
```

**Phase 3: Repeat for B and S1**
```bash
# B (Order Block Retest)
python3 bin/optimize_constrained_cpcv.py --archetype B --trials 50 --folds 5
python3 bin/walk_forward_production_engine.py --archetype B --config results/optimization_constrained_cpcv/B_constrained_cpcv.json

# S1 (Liquidity Vacuum)
python3 bin/optimize_constrained_cpcv.py --archetype S1 --trials 50 --folds 5
python3 bin/walk_forward_production_engine.py --archetype S1 --config results/optimization_constrained_cpcv/S1_constrained_cpcv.json
```

### Success Criteria

**Minimum Bar (1-2 archetypes pass):**
- OOS degradation < 30%
- OOS Sortino > 0.5
- Profitable windows > 60%
- Walk-Forward Efficiency (WFE) > 60%

**If All 3 Pass:**
- Move to Step 3 (regime upgrades + portfolio)

**If 0-1 Pass:**
- Tighten constraints further
- Consider alternative approaches (see Step 4)

---

## Step 3: Triage the Other Archetypes

### K (Wick Trap Moneytaur) - INVESTIGATE, DON'T ABANDON YET

**Current Status:**
- -5.1% OOS return (loses money)
- 104% degradation (worse than random)
- 20.8% profitable windows

**Possible Causes:**
1. Lookahead bias in features (regime labels, SMC features)
2. Inverse signal (parameters inversely correlated with edge)
3. Regime-dependent edge (works in bull, fails in bear)

**Investigation Steps:**
1. ✅ **Audit features for lookahead**
   ```bash
   # Check if wick_trap features use future data
   grep -r "shift\|rolling\|expanding" engine/strategies/archetypes/bull/wick_trap_moneytaur.py

   # Check regime label generation
   grep -r "lookahead\|future" engine/context/regime_service.py
   ```

2. ✅ **Test on streaming data**
   ```bash
   # Re-run with strict streaming mode (bar-by-bar feature generation)
   # If performance improves, lookahead bias confirmed
   ```

3. ✅ **Regime-stratified analysis**
   ```bash
   # Analyze K performance by regime:
   # - Bull markets (2019, 2021, 2024): Does K work?
   # - Bear markets (2018, 2022): Does K fail?
   # - Sideways (2023): Does K work?

   # If K only works in bull, add regime gating
   ```

**Decision Point:**
- If lookahead found → Fix features, re-optimize
- If inverse signal → Abandon archetype
- If regime-dependent → Add regime gating, re-optimize

### S4 (Funding Divergence) - REDESIGN OR MERGE

**Current Status:**
- Only 11 trades in 6 years (1.8/year)
- 54% degradation (lowest, but insufficient sample)
- 12.5% profitable windows

**Possible Causes:**
1. Funding divergence is too rare an event
2. Entry thresholds too strict
3. Feature engineering incorrect

**Investigation Steps:**
1. ✅ **Analyze funding divergence frequency**
   ```python
   # How often does funding divergence setup occur?
   # Check funding_z_max threshold: currently -2.5 to -1.5
   # Maybe should be -2.0 to -1.0 (more permissive)
   ```

2. ✅ **Check if S4 logic is correct**
   ```bash
   # Read S4 implementation
   cat engine/strategies/archetypes/bear/funding_divergence.py

   # Verify:
   # - Does it check for negative funding + bullish price action?
   # - Are thresholds reasonable?
   # - Is logic inverted?
   ```

**Decision Point:**
- If logic is correct, thresholds too strict → Loosen, re-optimize
- If logic is incorrect → Fix, re-optimize
- If funding divergence is too rare → Merge with S1 or abandon

### S5 (Long Squeeze Cascade) - ALTERNATIVE VALIDATION

**Current Status:**
- 0 trades in walk-forward (EXPECTED - crisis-only)
- Can't validate with standard walk-forward

**Investigation Steps:**
1. ✅ **Crisis event replay**
   ```bash
   # Test on known crisis periods:
   # - March 2020: COVID crash
   # - May 2022: LUNA/UST collapse
   # - November 2022: FTX collapse

   python3 bin/validate_s5_crisis_events.py
   ```

2. ✅ **Measure precision/recall, not returns**
   ```python
   # For crisis-only archetype:
   # - Precision: When S5 trades, is it actually a crisis?
   # - Recall: When there's a crisis, does S5 trade?
   # - False positives: Does S5 trade in non-crisis?

   # Success = Precision >80%, Recall >60%, FP <5%
   ```

---

## Step 4: Alternative Approaches (If Constrained CPCV Fails)

### Option A: No-Fitting Philosophy (Rob Carver)

**Concept:** Don't optimize parameters at all. Use fixed params based on theory.

**Implementation:**
```python
# Instead of optimizing fusion_threshold (0.35-0.85)
# Set it to theoretical value: 0.60 (60% confidence)

H_params_fixed = {
    'fusion_threshold': 0.60,  # Fixed (theory)
    'trend_weight': 0.40,       # Fixed (theory)
    'liquidity_weight': 0.30,   # Fixed (theory)
    # No optimization, just backtest
}
```

**Pros:**
- No overfitting possible
- Parameters stable across regimes
- Simpler to explain and debug

**Cons:**
- May leave edge on the table
- Requires strong domain knowledge

**When to Use:**
- If constrained CPCV still shows 40-50% degradation
- If optimization consistently fails

### Option B: Ensemble of Simple Strategies

**Concept:** Instead of 6 optimized archetypes, use 20+ simple unoptimized strategies

**Implementation:**
```python
# Example: Simple momentum strategies
strategies = [
    {'name': 'MA_cross_10_50', 'params': {'fast': 10, 'slow': 50}},  # Fixed
    {'name': 'MA_cross_20_100', 'params': {'fast': 20, 'slow': 100}},  # Fixed
    {'name': 'RSI_oversold', 'params': {'threshold': 30}},  # Fixed
    # ... 17 more simple strategies
]

# Equal weight ensemble
# Trades when 3+ strategies agree
```

**Pros:**
- Diversification across strategies reduces overfitting
- No parameter optimization required
- Robust to regime changes

**Cons:**
- May have lower Sortino than optimized single archetype
- More complex to manage

**When to Use:**
- If individual archetypes can't pass validation
- If looking for robustness over peak performance

### Option C: Feature Importance (Not Parameter Optimization)

**Concept:** Focus on which signals work, not which parameters fit

**Implementation:**
```python
# Instead of optimizing parameters:
# 1. Generate all archetype signals (raw, not filtered)
# 2. Use ML to predict which signals will be profitable
# 3. Trade only high-quality signals

# Example:
signal_features = {
    'archetype_confidence': 0.65,
    'regime_alignment': 0.80,
    'liquidity_score': 0.45,
    'trend_strength': 0.72,
}

# Train XGBoost: predict P(profitable | signal_features)
# Trade only if P(profitable) > 0.60
```

**Pros:**
- Learns to combine signals, not overfit parameters
- Can discover non-obvious patterns
- Adapts to regime changes

**Cons:**
- More complex (ML layer)
- Requires more data for training
- Risk of overfitting the meta-model

**When to Use:**
- If you have archetype signals but parameters don't work
- If looking to build meta-model layer

---

## Step 5: Don't Touch Regime Detection Yet

**Rationale:**
- Regime upgrades can mask signal weakness
- Get archetypes working first, THEN enhance regime detection

**Wrong Sequencing:**
```
1. Upgrade regime detection to perfection
2. Archetypes suddenly "work better"
3. Deploy to production
4. Regime detection fails in new regime → archetypes fail

Problem: Archetypes have no standalone edge
```

**Right Sequencing:**
```
1. Get 1-2 archetypes passing validation (standalone edge)
2. Archetypes work even with basic regime detection
3. Upgrade regime detection → improves sizing/filtering
4. Archetypes now work BETTER, but had edge before

Benefit: Archetypes have real edge, regime detection is enhancement
```

**Timeline:**
- Week 1-2: Constrained CPCV re-optimization (H, B, S1)
- Week 3: Walk-forward validation of re-optimized params
- Week 4: Regime detection upgrades (only if 1-2 pass validation)

---

## Using the awesome-systematic-trading GitHub Repo

**What It Is:**
- Curated list of battle-tested libraries for systematic trading
- Organized by category: backtesting, metrics, optimization, ML, etc.
- Not a drop-in solution, but a reference for proven components

**How to Use It (Targeted Upgrades, Not Full Rebuild):**

### 1. Validation Methodology References
**Look For:**
- Libraries implementing purged CV / embargo concepts
- López de Prado style workflows (CPCV, barriers, etc.)
- Walk-forward frameworks

**Copy Patterns Into:**
- Our existing `bin/optimize_constrained_cpcv.py`
- Our existing `bin/walk_forward_production_engine.py`

**Examples:**
- How do they structure CPCV folds?
- What embargo periods do they use?
- How do they calculate WFE?

### 2. Reporting & Attribution
**Look For:**
- Mature metrics/reporting packages
- Equity curve plotting
- Drawdown analysis
- Regime-tagged breakdowns

**Add To:**
- Our existing backtest engine outputs
- Walk-forward validation reports

**Benefit:**
- Reduce "missing stats" errors
- Professional-quality reports
- Easier to debug failures

### 3. Optional: QLib as Research Harness
**Look For:**
- Microsoft QLib (AI-powered quant research platform)
- Already includes SOTA alpha factors

**Use As:**
- Sanity check: "If I express this idea in QLib, do I see same fragility?"
- Feature engineering reference
- Alternative to Bull Machine for simple ideas

**NOT:**
- Replacement for Bull Machine
- Production trading engine

### 4. Specific Libraries to Check
```
Backtest/Live Frameworks:
- vectorbt: Vectorized backtesting (fast parameter sweeps)
- pysystemtrade: Rob Carver's implementation
- backtrader: Event-driven framework

Metrics/Risk:
- empyrical: Quantopian's risk metrics (Sharpe, Sortino, etc.)
- pyfolio: Tear sheet generation
- quantstats: HTML reports

Optimization:
- Optuna: Already using (good choice)
- scikit-optimize: Bayesian optimization alternative

Time-Series:
- statsmodels: Time series analysis
- arch: GARCH models for volatility

ML/AI:
- Microsoft QLib: Full quant research platform
- ta-lib: Technical analysis library
```

---

## Timeline & Milestones

### Week 1: Constrained CPCV Re-Optimization
- ✅ Step 0: Freeze truth with permanent tests (DONE)
- ✅ Step 1: Create constrained CPCV framework (DONE)
- ⏳ Step 2: Re-optimize H with CPCV (50 trials, 5 folds)
- ⏳ Step 2: Re-optimize B with CPCV (50 trials, 5 folds)
- ⏳ Step 2: Re-optimize S1 with CPCV (50 trials, 5 folds)

**Deliverable:** 3 JSON files with constrained params

### Week 2: Walk-Forward Validation
- ⏳ Run walk-forward on H (new params)
- ⏳ Run walk-forward on B (new params)
- ⏳ Run walk-forward on S1 (new params)
- ⏳ Analyze results: How many passed?

**Deliverable:** Walk-forward results showing <30% degradation (target: 1-2 pass)

### Week 3: Triage & Investigation
- ⏳ If H/B/S1 pass → Move to regime upgrades
- ⏳ If 0 pass → Try alternative approaches (no-fitting, ensemble)
- ⏳ Investigate K (lookahead audit)
- ⏳ Investigate S4 (feature analysis)
- ⏳ Validate S5 on crisis events

**Deliverable:** Decision on each archetype (keep/fix/abandon)

### Week 4: Regime Detection & Portfolio (If 1-2 Pass)
- ⏳ Upgrade regime detection (only if archetypes have standalone edge)
- ⏳ Implement HRP portfolio allocation
- ⏳ Paper trading setup
- ⏳ Production deployment plan

**Deliverable:** Production-ready system (if successful)

---

## Success Metrics

### Minimum Success (Proceed to Production)
- ✅ 1-2 archetypes pass walk-forward (<30% degradation)
- ✅ OOS Sortino > 0.5
- ✅ Profitable windows > 60%
- ✅ WFE > 60%
- ✅ No lookahead bias detected
- ✅ All validation gates pass

### Full Success (Strong Production System)
- ✅ All 3 archetypes (H, B, S1) pass walk-forward
- ✅ Combined portfolio Sortino > 1.0
- ✅ Regime-aware allocation working
- ✅ K investigation complete (fix or abandon)
- ✅ S4 investigation complete (fix or abandon)
- ✅ S5 validated on crisis events

---

## Key Principles

1. **Freeze the truth first** - Tests prevent regression
2. **Fix optimization, not engine** - Engine is working correctly
3. **Reduce degrees of freedom** - 3-5 params max
4. **Validate inside the loop** - CPCV, not single split
5. **Don't abandon yet** - Moving parts, could be other bugs
6. **Sequence correctly** - Archetypes first, regime upgrades second
7. **Use industry standards** - WFE >60%, degradation <30%
8. **Reference battle-tested tools** - awesome-systematic-trading repo

---

## Files Created

1. **tests/test_production_validation_gates.py** - Permanent test suite (3 gates)
2. **bin/optimize_constrained_cpcv.py** - Constrained optimization with CPCV
3. **CLEAN_PATH_FORWARD.md** - This roadmap

---

## Next Action

**Immediate:**
```bash
# Run constrained CPCV optimization for H
python3 bin/optimize_constrained_cpcv.py --archetype H --trials 50 --folds 5

# Expected runtime: 30-60 minutes (50 trials × 5 folds = 250 backtests)
# Expected output: results/optimization_constrained_cpcv/H_constrained_cpcv.json
```

**After H Completes:**
- Analyze results: Did CPCV objective improve?
- Run walk-forward validation with new params
- Check: Did degradation improve from 66% → <30%?
- If yes: Repeat for B and S1
- If no: Tighten constraints or try alternative approaches

---

**Created By:** Claude Code (Sonnet 4.5)
**Date:** 2026-01-17 21:00
**Status:** Ready to execute Week 1 (constrained CPCV re-optimization)
