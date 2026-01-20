# Quant Skills Implementation Guide - Bull Machine

**Date**: 2026-01-19
**Purpose**: Define custom skills for institutional-grade quant development
**Philosophy**: Tools are plumbing, not soul - preserve RegimeService → Archetypes → Wyckoff/SMC/Temporal flow

---

## Executive Summary

This guide defines 5 custom skills to build the Bull Machine like a real quant team. These skills integrate institutional-grade tools (NautilusTrader, hftbacktest, qf-lib, FinRL, Qlib) while preserving the engine's "soul":

**Soul (Preserved)**:
- RegimeService as brainstem
- Archetype pattern detection
- Wyckoff structural grammar
- Plus-One conviction stacking
- Soft gating + circuit breaker

**Plumbing (Enhanced)**:
- Backtest execution kernel
- Orderbook fill simulation
- Risk analytics layer
- ML optimization layer
- Alpha discovery platform

---

## 🎯 Skills Overview

| Skill | Priority | Tool | Purpose | Integration Time |
|-------|----------|------|---------|-----------------|
| `nautilus-integration` | 1 | NautilusTrader | Production backtest + live execution | 2-3 weeks |
| `orderbook-analysis` | 1 | hftbacktest | Orderbook microstructure validation | 1 week |
| `risk-analytics` | 1 | qf-lib | Advanced risk metrics + attribution | 1 week |
| `alpha-discovery` | 2 | Qlib | ML-based alpha discovery | 2-3 weeks |
| `rl-optimizer` | 2 | FinRL | RL-based position sizing | 3-4 weeks |

---

## 1️⃣ SKILL: `nautilus-integration` ⭐⭐⭐⭐⭐

### Purpose
Integrate NautilusTrader as production-grade backtest + live execution chassis while preserving Bull Machine decision intelligence.

### Philosophy
**NautilusTrader = Execution chassis (plumbing)**
**Bull Machine = Decision intelligence (soul)**

NautilusTrader handles:
- Event loop (time advancement)
- Order management system (OMS)
- Fill simulation (realistic slippage)
- Exchange adapters (live trading)
- Position tracking
- Telemetry + logging

Bull Machine provides:
- Regime detection (RegimeService)
- Archetype signals (pattern recognition)
- Domain evidence (Wyckoff, SMC, temporal)
- Plus-One conviction (multiplicative stacking)
- Risk logic (soft gating, circuit breaker concepts)

### Integration Approach

#### Phase A: Harness Only (Validation, 1 week)
**Goal**: Verify NautilusTrader can reproduce current backtest results

**Tasks**:
1. Install NautilusTrader (`pip install nautilus_trader`)
2. Convert historical data to Nautilus format (parquet → Nautilus Data)
3. Write simple strategy that calls RegimeService + Archetypes
4. Run parallel backtest (current engine vs Nautilus)
5. Compare metrics (PnL, Sharpe, trades, regime transitions)

**Acceptance**: <5% difference in key metrics

#### Phase B: Adapter Layer (Integration, 1 week)
**Goal**: Wrap Bull Machine logic in Nautilus Strategy interface

**Tasks**:
1. Create `BullMachineStrategy` class inheriting from `Strategy`
2. Map `on_bar()` → RegimeService.get_regime() + archetype evaluation
3. Convert archetype signals → Nautilus Order objects
4. Integrate soft gating (position sizing logic)
5. Integrate circuit breaker (risk limits)

**Code Skeleton**:
```python
from nautilus_trader.trading.strategy import Strategy
from engine.context.regime_service import RegimeService
from engine.archetypes.logic_v2_adapter import LogicV2Adapter

class BullMachineStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.regime_service = RegimeService(mode='HYBRID', enable_hysteresis=True)
        self.archetype_logic = LogicV2Adapter()

    def on_bar(self, bar):
        # 1. Get regime state (BRAINSTEM)
        features = self._extract_features(bar)
        regime_result = self.regime_service.get_regime(features, bar.ts)

        # 2. Evaluate archetypes (PATTERN RECOGNITION)
        context = self._build_runtime_context(bar, regime_result)
        archetype_signals = self.archetype_logic.evaluate_all(context)

        # 3. Fusion + Plus-One (CONVICTION STACKING)
        final_score = self._apply_domain_boosts(archetype_signals, context)

        # 4. Risk / Sizing (SOFT GATING + CIRCUIT BREAKER)
        position_size = self._compute_position_size(final_score, regime_result)

        # 5. Execution (NAUTILUS HANDLES)
        if position_size > 0:
            self.submit_order(
                instrument_id=bar.instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity(position_size),
                order_type=OrderType.MARKET
            )
```

**Acceptance**: Same logic as current engine, Nautilus handles execution

#### Phase C: Live Trading (Production, ongoing)
**Goal**: Deploy same strategy code to live exchanges

**Tasks**:
1. Configure exchange adapters (Binance, Coinbase)
2. Set up paper trading environment
3. Monitor live vs backtest performance
4. Add live-specific risk controls

**Acceptance**: Ready for paper trading

### Expected Benefits
- **10-100x faster backtesting** (Cython/Rust core)
- **Production-ready execution** (OMS, exchange adapters)
- **Same code backtest → live** (no translation bugs)
- **Institutional-grade telemetry** (trade logging, metrics)

### Risks & Mitigations
**Risk**: Semantic changes (timestamps, fills, accounting)
**Mitigation**: Phase A validation (parallel backtests until parity)

**Risk**: Over-engineering (complexity creep)
**Mitigation**: Start simple (Phase A), add complexity only when needed

**Risk**: Losing the soul (bull machine logic gets diluted)
**Mitigation**: Adapter pattern (Nautilus calls our logic, we don't rewrite for Nautilus)

---

## 2️⃣ SKILL: `orderbook-analysis` ⭐⭐⭐⭐

### Purpose
Use hftbacktest to validate fill assumptions and orderbook microstructure for liquidity-sensitive archetypes.

### Philosophy
**hftbacktest = Reality check (validation layer)**
**NOT a replacement** for main backtest engine

hftbacktest provides:
- Level 2 orderbook snapshots
- Queue position modeling (your order's place in book)
- Latency simulation (network + exchange + execution)
- Market impact modeling

Use cases:
- Validate Liquidity Vacuum (S1) fill assumptions
- Test Order Block Retest (B) slippage
- Analyze wick traps (K) in thin liquidity
- Quantify market impact for large positions

### Integration Approach

#### Step 1: Orderbook Data Pipeline (3 days)
**Tasks**:
1. Install hftbacktest (`pip install hftbacktest`)
2. Download Level 2 orderbook snapshots (Binance API)
3. Convert aggTrades → orderbook reconstruction
4. Store in efficient format (HDF5 or Parquet)

#### Step 2: Fill Validation (2 days)
**Tasks**:
1. Run current backtest → extract entry/exit timestamps
2. Replay orderbook at those timestamps
3. Simulate fills with queue position model
4. Compare assumed fill price vs realistic fill price

**Metrics**:
- Average slippage (bps)
- Fill probability (% of orders filled)
- Market impact (large vs small positions)

#### Step 3: Archetype-Specific Analysis (2 days)
**Tasks**:
1. **Liquidity Vacuum (S1)**: Does orderbook confirm vacuum? (bid-ask spread >1%, depth imbalance >70%)
2. **Order Block Retest (B)**: Are OB zones filled with resting orders? (high density at OB levels)
3. **Wick Trap (K)**: Does wick correspond to liquidity sweep? (orderbook cleared at wick extreme)

**Output**: Archetype validation report (which signals have orderbook evidence)

### Expected Benefits
- **Realistic slippage estimates** (vs current fixed assumptions)
- **Orderbook microstructure insights** (validate liquidity patterns)
- **Fill probability quantification** (especially for large sizes)
- **Archetype validation** (orderbook confirms structural logic)

### Risks & Mitigations
**Risk**: Data cost (Level 2 orderbook can be expensive)
**Mitigation**: Start with free Binance data, sample key periods

**Risk**: Complexity (orderbook reconstruction is non-trivial)
**Mitigation**: Use hftbacktest's built-in reconstruction, validate on known events

---

## 3️⃣ SKILL: `risk-analytics` ⭐⭐⭐⭐

### Purpose
Use qf-lib to compute advanced risk metrics, regime-conditioned attribution, and portfolio optimization.

### Philosophy
**qf-lib = Post-trade intelligence (analytics layer)**
**NOT part of execution** - pure analysis

qf-lib provides:
- Risk-adjusted metrics (Sharpe, Sortino, Calmar, Omega)
- Drawdown analytics (max DD, avg duration, recovery time)
- Factor attribution (which archetypes drive returns)
- Portfolio optimization (mean-variance, risk parity)

Use cases:
- Regime-conditioned performance (crisis vs risk_on Sharpe)
- Archetype attribution (which signals contribute most)
- Soft gating validation (empirical edge vs theoretical)
- Portfolio weight optimization (archetype allocation)

### Integration Approach

#### Step 1: Metrics Integration (2 days)
**Tasks**:
1. Install qf-lib (`pip install qf-lib`)
2. Wrap backtest results in qf-lib Portfolio object
3. Compute risk metrics (Sharpe, Sortino, Calmar, Omega, VaR, CVaR)
4. Generate performance report with regime labels

**Code Skeleton**:
```python
from qf_lib.backtesting import Portfolio
from qf_lib.common.utils.returns.max_drawdown import max_drawdown

# Wrap backtest results
portfolio = Portfolio(
    dates=backtest_results['dates'],
    returns=backtest_results['returns'],
    regime_labels=backtest_results['regime_labels']
)

# Compute metrics
sharpe = portfolio.sharpe_ratio()
sortino = portfolio.sortino_ratio()
calmar = portfolio.calmar_ratio()
max_dd = max_drawdown(portfolio.returns)

# Regime-conditioned metrics
crisis_sharpe = portfolio.returns[portfolio.regime_labels == 'crisis'].sharpe_ratio()
risk_on_sharpe = portfolio.returns[portfolio.regime_labels == 'risk_on'].sharpe_ratio()
```

#### Step 2: Regime-Conditioned Attribution (2 days)
**Tasks**:
1. Decompose returns by regime (crisis, risk_off, neutral, risk_on)
2. Compute per-regime Sharpe, max DD, win rate
3. Analyze regime transition costs (PnL lost during transitions)
4. Quantify hysteresis effectiveness (transitions reduced vs PnL)

**Output**: Regime performance matrix (4 regimes × key metrics)

#### Step 3: Archetype Attribution (2 days)
**Tasks**:
1. Tag each trade with archetype ID + regime
2. Compute per-archetype PnL, Sharpe, win rate
3. Analyze archetype × regime interaction (which archetypes work in which regimes)
4. Domain engine attribution (Wyckoff, SMC, temporal contribution)

**Output**: Archetype performance heatmap (13 archetypes × 4 regimes)

#### Step 4: Portfolio Optimization (1 day)
**Tasks**:
1. Use qf-lib's mean-variance optimizer
2. Optimize archetype weights (allocation per archetype)
3. Compare optimized vs current soft gating weights
4. Validate out-of-sample

**Output**: Optimized allocation weights

### Expected Benefits
- **Comprehensive risk analytics** (publication-quality metrics)
- **Regime-conditioned insights** (which regimes we excel/struggle in)
- **Archetype attribution** (data-driven archetype selection)
- **Portfolio optimization** (optimal archetype weights)

### Risks & Mitigations
**Risk**: Overfitting (optimizer fits to noise)
**Mitigation**: Out-of-sample validation, walk-forward testing

**Risk**: Ignoring structural logic (optimizer suggests "turn off Wyckoff")
**Mitigation**: Use as guidance, not gospel - preserve structural intuition

---

## 4️⃣ SKILL: `alpha-discovery` ⭐⭐⭐ (Optional)

### Purpose
Use Qlib to discover new alphas, train ML models, and explore feature engineering enhancements.

### Philosophy
**Qlib = Research platform (alpha discovery layer)**
**Use for**: Generating archetype ideas, not replacing archetypes

Qlib provides:
- SOTA ML models (LightGBM, XGBoost, Transformer)
- Alpha discovery pipeline
- Advanced feature engineering
- Experiment tracking

Use cases:
- Discover new archetype candidates (alpha mining)
- Train ML models for regime features
- Compare Qlib alphas vs our archetypes
- Feature importance analysis (which features matter most)

### Integration Approach

#### Step 1: Data Conversion (3 days)
**Tasks**:
1. Install Qlib (`pip install pyqlib`)
2. Convert parquet data to Qlib format
3. Define feature universe (100+ technical indicators)
4. Create Qlib dataset

#### Step 2: Alpha Discovery (1 week)
**Tasks**:
1. Run Qlib's alpha discovery pipeline
2. Train LightGBM/XGBoost on our features
3. Extract top alphas (rank by IC, Sharpe)
4. Validate alphas on our crypto data

**Output**: Top 10 Qlib alphas (potential new archetypes)

#### Step 3: Model Comparison (3 days)
**Tasks**:
1. Compare Qlib models vs our archetypes
2. Ensemble Qlib + archetypes (meta-model)
3. Validate out-of-sample

**Output**: Qlib vs archetype comparison report

### Expected Benefits
- **New archetype ideas** (data-driven alpha discovery)
- **ML model zoo** (access to SOTA models)
- **Feature engineering** (automated feature generation)
- **Experiment tracking** (reproducible research)

### Risks & Mitigations
**Risk**: Stock-focused (Qlib designed for equities)
**Mitigation**: Adapt for crypto (different features, faster timeframes)

**Risk**: Black-box alphas (hard to explain)
**Mitigation**: Use for ideas, validate with structural logic (Wyckoff)

**Risk**: Overfitting (ML models overfit to backtest)
**Mitigation**: Strict out-of-sample validation, walk-forward testing

---

## 5️⃣ SKILL: `rl-optimizer` ⭐⭐⭐ (Optional)

### Purpose
Use FinRL to train reinforcement learning agents for position sizing and archetype allocation.

### Philosophy
**FinRL = Data-driven optimizer (learning layer)**
**Use for**: Learning allocation weights, not replacing archetypes

FinRL provides:
- Deep RL algorithms (PPO, A2C, SAC, TD3)
- Trading environment wrappers
- Portfolio rebalancing agents
- Meta-strategy learning

Use cases:
- Learn optimal position sizing (vs fixed Kelly)
- Learn archetype allocation weights (vs Bayesian shrinkage)
- Learn adaptive circuit breaker thresholds (vs fixed rules)
- Meta-strategy (learn when to use which archetype)

### Integration Approach

#### Step 1: Environment Definition (1 week)
**Tasks**:
1. Install FinRL (`pip install finrl`)
2. Define RL state space:
   - Regime (crisis, risk_off, neutral, risk_on)
   - Archetype scores (13 archetypes)
   - Portfolio state (equity, positions, drawdown)
   - Macro features (RV_7, funding_Z, etc.)
3. Define action space:
   - Position sizing (0-100% per archetype)
   - Archetype weights (allocation per archetype)
4. Define reward function:
   - Sharpe ratio (risk-adjusted returns)
   - Sortino ratio (downside risk)
   - Regime stability (minimize transitions)

**Code Skeleton**:
```python
from finrl.agents.stablebaselines3 import PPO
from gym import Env

class BullMachineEnv(Env):
    def __init__(self):
        super().__init__()
        self.regime_service = RegimeService()
        self.archetype_logic = LogicV2Adapter()

    def step(self, action):
        # action = [pos_size_A, pos_size_B, ..., pos_size_S8]

        # Execute trades with RL-determined sizes
        pnl = self._execute_trades(action)

        # Compute reward (Sharpe + regime stability)
        reward = self._compute_reward(pnl)

        # Advance to next bar
        self.current_bar += 1
        obs = self._get_observation()

        return obs, reward, done, info

    def _compute_reward(self, pnl):
        # Sharpe component
        sharpe = pnl.mean() / (pnl.std() + 1e-8)

        # Regime stability component (penalize excessive transitions)
        transitions = len(self.regime_transitions)
        stability_bonus = -0.01 * transitions  # Penalize transitions

        return sharpe + stability_bonus
```

#### Step 2: Agent Training (2 weeks)
**Tasks**:
1. Train PPO agent on historical data (2018-2024)
2. Validate on out-of-sample data (2024)
3. Compare RL vs rule-based allocation
4. Tune hyperparameters (learning rate, batch size, etc.)

**Output**: Trained RL agent checkpoint

#### Step 3: Validation (1 week)
**Tasks**:
1. Walk-forward validation (train on 2018-2022, test on 2023-2024)
2. Compare RL vs Bayesian shrinkage vs equal weight
3. Analyze RL learned policy (which archetypes it favors in which regimes)

**Output**: RL vs rule-based comparison report

### Expected Benefits
- **Data-driven position sizing** (learns from history)
- **Adaptive allocation** (responds to regime changes)
- **Non-linear optimization** (captures complex interactions)
- **Meta-learning** (learns across regimes)

### Risks & Mitigations
**Risk**: Sample inefficiency (RL needs lots of data)
**Mitigation**: Use historical data (2018-2024), data augmentation

**Risk**: Overfitting (RL agents overfit to backtest)
**Mitigation**: Strict out-of-sample validation, conservative rewards

**Risk**: Black-box (hard to interpret RL policy)
**Mitigation**: Policy visualization, feature importance analysis

**Risk**: Instability (RL can be unstable)
**Mitigation**: Multiple random seeds, ensemble agents

---

## 📋 Implementation Roadmap

### Phase 1: Production-Critical (Weeks 1-4) ⭐⭐⭐⭐⭐

**Priority**: MUST-HAVE for production deployment

**Week 1**: `nautilus-integration` Phase A (validation)
- Install NautilusTrader
- Convert data to Nautilus format
- Run parallel backtest
- Validate metrics (<5% difference)

**Week 2**: `nautilus-integration` Phase B (adapter)
- Create BullMachineStrategy wrapper
- Integrate RegimeService + Archetypes
- Test soft gating + circuit breaker
- Validate signal generation

**Week 3**: `orderbook-analysis`
- Install hftbacktest
- Download Level 2 orderbook data
- Run fill validation for Liquidity Vacuum
- Quantify slippage assumptions

**Week 4**: `risk-analytics`
- Install qf-lib
- Integrate risk metrics
- Regime-conditioned attribution
- Archetype performance heatmap

**Deliverables**:
- NautilusTrader integration (10-100x faster backtesting)
- Orderbook validation report (realistic slippage)
- Risk analytics dashboard (Sharpe, Sortino, attribution)

---

### Phase 2: ML Enhancement (Weeks 5-8) ⭐⭐⭐ (Optional)

**Priority**: SHOULD-HAVE for research/optimization

**Week 5-6**: `alpha-discovery`
- Install Qlib
- Convert data to Qlib format
- Run alpha discovery
- Compare Qlib vs archetypes

**Week 7-8**: `rl-optimizer`
- Install FinRL
- Define RL environment
- Train PPO agent
- Validate RL vs rule-based

**Deliverables**:
- Top 10 Qlib alphas (new archetype candidates)
- RL agent checkpoint (learned allocation policy)
- ML vs rule-based comparison report

---

## 🎯 Success Criteria

### Nautilus Integration
- ✅ Backtest parity: <5% difference in PnL, Sharpe, trades
- ✅ Strategy adapter: Clean separation (Nautilus = plumbing, Bull Machine = soul)
- ✅ Live trading ready: Paper trading deployment successful

### Orderbook Analysis
- ✅ Slippage validation: Realistic estimates (bps) per archetype
- ✅ Liquidity validation: Orderbook confirms Liquidity Vacuum signals
- ✅ Fill probability: Quantified for different position sizes

### Risk Analytics
- ✅ Regime-conditioned metrics: Per-regime Sharpe, max DD, win rate
- ✅ Archetype attribution: 13 archetypes × 4 regimes heatmap
- ✅ Portfolio optimization: Optimized allocation weights validated OOS

### Alpha Discovery (Optional)
- ✅ New archetype ideas: Top 10 Qlib alphas with structural logic
- ✅ ML models: LightGBM/XGBoost trained and validated
- ✅ Feature importance: Key features identified

### RL Optimizer (Optional)
- ✅ RL agent trained: Converged PPO agent
- ✅ OOS validation: RL beats rule-based on 2024 data
- ✅ Policy interpretation: Understand RL learned strategy

---

## 🛡️ Preserving the Soul - Integration Checklist

Before integrating ANY tool, verify:

### ✅ RegimeService Remains Brainstem
- [ ] RegimeService.get_regime() is still the single entry point
- [ ] Regime state flows downstream immutably (RuntimeContext frozen)
- [ ] Hysteresis logic unchanged (dual thresholds, min dwell times)

### ✅ Archetypes Remain Pattern Detectors
- [ ] Archetype logic unchanged (pattern recognition)
- [ ] Regime routing enforced (allowed regimes per archetype)
- [ ] Soft penalties applied (regime-based score scaling)

### ✅ Wyckoff Remains Structural Grammar
- [ ] Wyckoff lives inside archetypes (domain evidence)
- [ ] Wyckoff multipliers unchanged (2.50x spring, 1.80x SOS)
- [ ] Wyckoff is NOT control flow (multiplicative, not binary)

### ✅ Plus-One Stacking Remains Multiplicative
- [ ] Domain engines stack: domain_boost = ∏ multipliers
- [ ] Caps enforced: [0.0, 5.0]
- [ ] Metadata tracked: domain_signals list

### ✅ Soft Gating Remains Regime-Conditioned
- [ ] Regime budgets unchanged: crisis 30%, risk_off 50%, neutral 70%, risk_on 80%
- [ ] Bayesian shrinkage formula unchanged
- [ ] Guardrails enforced: min 1%, max 20% negative edge

### ✅ Circuit Breaker Remains Regime-Aware
- [ ] 4-tier escalation unchanged: INSTANT_HALT, SOFT_HALT, WARNING, INFO
- [ ] Regime-aware thresholds: tighter in crisis, looser in risk_on
- [ ] Event logging unchanged

---

## 📝 Usage Examples

### Skill Invocation (Command Line)

```bash
# Priority 1 Skills (Production-Critical)
/nautilus-integration --phase validation --compare-metrics
/orderbook-analysis --archetype S1 --period 2024-01-01:2024-12-31
/risk-analytics --regime-conditioned --archetype-attribution

# Priority 2 Skills (Optional)
/alpha-discovery --top-alphas 10 --validate-oos
/rl-optimizer --train --agent PPO --timesteps 1000000
```

### Python API

```python
# Nautilus Integration
from skills.nautilus_integration import BullMachineStrategy
strategy = BullMachineStrategy(regime_service, archetype_logic)
backtest_engine.run(strategy, data)

# Orderbook Analysis
from skills.orderbook_analysis import validate_fills
slippage_report = validate_fills(
    archetype='S1',
    entry_timestamps=backtest_results['entries'],
    orderbook_data=orderbook_snapshots
)

# Risk Analytics
from skills.risk_analytics import compute_regime_attribution
attribution = compute_regime_attribution(
    returns=backtest_results['returns'],
    regime_labels=backtest_results['regime_labels'],
    archetype_ids=backtest_results['archetype_ids']
)

# Alpha Discovery
from skills.alpha_discovery import run_qlib_pipeline
qlib_alphas = run_qlib_pipeline(
    data=parquet_df,
    top_k=10,
    validate_oos=True
)

# RL Optimizer
from skills.rl_optimizer import train_ppo_agent
rl_agent = train_ppo_agent(
    env=BullMachineEnv(),
    total_timesteps=1000000,
    reward_fn='sharpe_with_stability'
)
```

---

## 🏁 Conclusion

These 5 skills transform Bull Machine from research-grade to institutional-grade infrastructure:

**Must-Have (Phase 1)**:
1. ✅ `nautilus-integration` - Production backtest + live execution (10-100x faster)
2. ✅ `orderbook-analysis` - Realistic fill simulation (validate liquidity assumptions)
3. ✅ `risk-analytics` - Comprehensive risk metrics (Sharpe, attribution, optimization)

**Should-Have (Phase 2)**:
4. 🔄 `alpha-discovery` - ML-based alpha discovery (new archetype ideas)
5. 🔄 `rl-optimizer` - RL-based position sizing (data-driven optimization)

**Philosophy**: Tools enhance plumbing, soul remains intact.

**Next Steps**:
1. Implement Phase 1 skills (4 weeks)
2. Validate preservation of soul (checklist above)
3. Deploy to production (paper trading → live)
4. Optionally: Phase 2 skills (ML enhancement)

**The quant team is ready. The engine's soul is protected. Time to build.**

---

**Document Date**: 2026-01-19
**Author**: Claude Code (System Architect)
**Status**: Ready for implementation
**Timeline**: Phase 1 = 4 weeks, Phase 2 = 4 weeks (optional)
