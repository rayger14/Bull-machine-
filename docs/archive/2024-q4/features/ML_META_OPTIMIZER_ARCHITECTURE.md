# ML Meta-Optimizer Architecture

**Branch**: `feature/ml-meta-optimizer`
**Date**: 2025-10-16
**Goal**: Use PyTorch to learn optimal configurations across all Bull Machine layers

---

## 🎯 Vision

Instead of manually tuning thresholds, weights, and parameters, train a neural network to:
1. **Understand the entire system** - All domain engines, fusion logic, macro context, risk management
2. **Learn optimal configurations** - Threshold values, domain weights, risk parameters
3. **Adapt to market regimes** - Different configs for bull/bear/sideways/volatile conditions
4. **Meta-optimize** - Find parameter combinations that maximize Sharpe, minimize drawdown

---

## 📚 Knowledge Layers to Encode

### Layer 1: Domain Engines (Feature Extractors)
Each domain engine produces signals that the ML model needs to understand:

#### **Wyckoff Engine**
- **Phases**: accumulation, spring, markup, distribution, upthrust, markdown, reaccumulation, redistribution
- **Confidence**: 0-1 score
- **CRT (Creek Crossing)**: Boolean flag
- **Key Insight**: Institutional accumulation/distribution patterns

#### **SMC (Smart Money Concepts)**
- **BOS** (Break of Structure): Trend continuation signal
- **CHOCH** (Change of Character): Trend reversal signal
- **FVG** (Fair Value Gap): Inefficiency zones (reversion targets)
- **Order Blocks**: Institutional supply/demand zones
- **Liquidity Sweeps**: Stop hunts before reversals
- **Confluence Score**: 0-1 (how many patterns align)

#### **HOB/Liquidity Engine**
- **Volume Surge**: Current vol / mean vol ratio
- **Wick Analysis**: Absorption patterns (institutional activity)
- **Quality**: retail vs institutional
- **Order Flow**: Buy/sell pressure at key levels

#### **Momentum Engine**
- **RSI**: 0-100 (oversold/overbought)
- **MACD**: Normalized -1 to 1
- **Divergences**: Price vs indicator disagreements
- **Delta**: Rate of change

#### **Fibonacci Engine** (v1.8.5)
- **Negative Fibs**: -0.272, -0.618, -1.0 levels (underground support)
- **Proximity Bonus**: Distance to key fib levels
- **Confluence**: Multiple fibs at same price

#### **Fourier Noise Filter** (v1.8.5)
- **Signal/Noise Ratio**: FFT-based filtering
- **Multiplier**: 0-1 (suppress noisy signals)

#### **Temporal/Gann Engine** (v1.8.6)
- **Gann Cycles**: 30/60/90 day ACF vibrations
- **Square of 9**: Price-time geometry
- **Thermo Floor**: Mining cost floor (BTC)
- **LPPLS Blowoff**: Bubble detection

### Layer 2: Multi-Timeframe (MTF) Analysis
- **1H Trend**: up/down/neutral (SMA20 deviation)
- **4H Trend**: up/down/neutral
- **1D Trend**: up/down/neutral
- **Alignment Score**: 0-1 (how well timeframes agree)
- **Nested Structures**: 1H pullback in 4H trend (healthy)

### Layer 3: Macro Context
- **VIX**: Fear gauge (0-100)
- **DXY**: Dollar strength
- **Yields**: 10Y Treasury (risk-on/off)
- **Funding Rate**: Perpetual futures cost
- **Open Interest**: Leverage buildup
- **Total Crypto Market Cap**: Liquidity proxy
- **BTC Dominance**: Risk appetite

### Layer 4: Risk Management
- **ATR Percentile**: Volatility filter (10th-90th percentile)
- **Loss Streak**: Consecutive losses (cool-off trigger)
- **Position Sizing**: Dynamic risk based on confidence
- **Stop Loss**: ATR-based, regime-aware
- **Take Profit**: Multi-target (50%@1R, 50%@2R)

### Layer 5: Fusion Logic
- **Domain Weights**: wyckoff=0.30, liquidity=0.25, momentum=0.30, smc=0.15
- **Tie-Breaker Thresholds**: 0.52 bullish, 0.48 bearish
- **MTF Penalty**: 20% score reduction if not aligned
- **Event Vetos**: Conference dates, high leverage warnings
- **Narrative Traps**: HODL trap detection

---

## 🧠 PyTorch Model Architecture

### Input Features (State Vector)
```python
state = {
    # Domain Scores (4)
    'wyckoff_score': float,      # 0-1
    'smc_score': float,          # 0-1
    'hob_score': float,          # 0-1
    'momentum_score': float,     # 0-1

    # Domain Directions (4, one-hot encoded = 12)
    'wyckoff_dir': ['long', 'short', 'neutral'],
    'smc_dir': ['long', 'short', 'neutral'],
    'hob_dir': ['long', 'short', 'neutral'],
    'momentum_dir': ['long', 'short', 'neutral'],

    # Domain Metadata (8)
    'wyckoff_phase': str,        # Encoded as embedding
    'wyckoff_confidence': float,
    'wyckoff_crt_active': bool,
    'smc_confluence': float,
    'hob_vol_surge': float,
    'hob_quality': str,          # retail/institutional
    'momentum_rsi': float,       # 0-100
    'momentum_macd': float,      # -1 to 1

    # MTF Features (6)
    'trend_1h': ['up', 'down', 'neutral'],
    'trend_4h': ['up', 'down', 'neutral'],
    'trend_1d': ['up', 'down', 'neutral'],
    'mtf_aligned': bool,
    'mtf_confidence': float,
    'mtf_nested': bool,

    # Macro Features (8)
    'vix': float,
    'dxy': float,
    'yields_10y': float,
    'funding_rate': float,
    'open_interest_delta': float,
    'total_mcap': float,
    'btc_dominance': float,
    'macro_veto_strength': float,

    # Risk State (5)
    'atr_percentile': float,     # 0-1
    'loss_streak': int,          # 0-10
    'current_drawdown': float,   # 0-1
    'time_since_last_trade': int,
    'position_size_pct': float,  # 0-1

    # Market Context (4)
    'price': float,              # Normalized
    'volume_ratio': float,       # Current/mean
    'volatility_regime': str,    # low/medium/high
    'market_regime': str,        # bull/bear/sideways

    # Total: ~50 features
}
```

### Model Architecture
```python
class BullMachineMetaOptimizer(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature embeddings
        self.phase_embed = nn.Embedding(8, 16)      # Wyckoff phases
        self.regime_embed = nn.Embedding(4, 8)      # Market regimes
        self.quality_embed = nn.Embedding(2, 4)     # HOB quality

        # Feature encoder (compress to latent space)
        self.encoder = nn.Sequential(
            nn.Linear(50, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # Policy head (what action to take)
        self.policy = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # [long, short, hold]
        )

        # Value head (expected reward)
        self.value = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Configuration head (optimal parameters)
        self.config_optimizer = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 20)  # 20 tunable parameters
        )

    def forward(self, state):
        # Encode features
        x = self.encoder(state)

        # Generate outputs
        policy_logits = self.policy(x)
        value = self.value(x)
        config = self.config_optimizer(x)

        return policy_logits, value, config

# Tunable Parameters Output (20)
config_params = [
    'wyckoff_weight',           # 0-1
    'smc_weight',               # 0-1
    'hob_weight',               # 0-1
    'momentum_weight',          # 0-1
    'entry_threshold',          # 0-1
    'mtf_penalty_factor',       # 0-1
    'bullish_bias_threshold',   # 0-1
    'bearish_bias_threshold',   # 0-1
    'atr_floor_percentile',     # 0-1
    'atr_cap_percentile',       # 0-1
    'loss_streak_threshold',    # 1-10
    'position_size_base',       # 0-0.1
    'stop_loss_atr_mult',       # 0-5
    'take_profit_mult_1',       # 0-10
    'take_profit_mult_2',       # 0-10
    'wyckoff_min_conf',         # 0-1
    'hob_vol_surge_thresh',     # 1-3
    'hob_wick_ratio_thresh',    # 0-1
    'mtf_align_conf_thresh',    # 0-1
    'fourier_enabled',          # 0/1 (binary)
]
```

---

## 🏋️ Training Strategy

### 1. Supervised Pre-Training
- **Input**: Historical data + domain engine outputs
- **Labels**: Known profitable trades (from backtest winners)
- **Loss**: Cross-entropy (action classification) + MSE (value estimation)
- **Goal**: Learn basic patterns from historical optimal configs

### 2. Reinforcement Learning (PPO/A2C)
- **Environment**: Bull Machine backtest environment
- **State**: Current market state + domain outputs
- **Action**: [long, short, hold] + config adjustments
- **Reward**:
  - Immediate: PnL of trade
  - Delayed: Sharpe ratio over rolling window
  - Penalty: Drawdown, loss streaks
- **Episodes**: Each episode = 1 backtest run (e.g., 3 months)

### 3. Meta-Learning (MAML/Reptile)
- **Task Distribution**: Different market regimes (bull 2021, bear 2022, sideways 2023, volatile 2024)
- **Goal**: Learn to quickly adapt to new regimes with few samples
- **Outer Loop**: Optimize for fast adaptation
- **Inner Loop**: Regime-specific fine-tuning

### Reward Function
```python
def calculate_reward(trades, equity_curve, config):
    # Profit component
    total_pnl = sum(t['pnl'] for t in trades)

    # Risk-adjusted return (Sharpe)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-9)

    # Drawdown penalty
    max_dd = calculate_max_drawdown(equity_curve)
    dd_penalty = -max_dd * 2  # 2x weight on drawdowns

    # Loss streak penalty
    loss_streak = count_consecutive_losses(trades)
    streak_penalty = -loss_streak * 0.1

    # Win rate bonus
    win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
    wr_bonus = win_rate * 0.5

    # Final reward
    reward = (
        total_pnl * 0.3 +          # 30% weight on profit
        sharpe * 100 * 0.4 +       # 40% weight on risk-adjusted return
        dd_penalty * 0.2 +         # 20% penalty on drawdown
        streak_penalty * 0.05 +    # 5% penalty on streaks
        wr_bonus * 0.05            # 5% bonus on win rate
    )

    return reward
```

---

## 📂 Implementation Plan

### Phase 1: Data Collection & Feature Engineering ✅
- [x] Extract all domain engine outputs to structured format
- [ ] Create feature store (HDF5/Parquet) with:
  - Raw OHLCV data (1H, 4H, 1D)
  - Domain scores at each timestamp
  - Macro snapshots aligned by time
  - Trade outcomes (labels)
- [ ] Build feature pipeline: `raw_data → domain_engines → feature_vector`

### Phase 2: Model Development
- [ ] Implement `BullMachineMetaOptimizer` in PyTorch
- [ ] Create training dataset loader
- [ ] Implement reward function
- [ ] Build backtest environment (Gym-compatible)

### Phase 3: Training
- [ ] Supervised pre-training (2 epochs)
- [ ] RL fine-tuning (PPO, 100 episodes)
- [ ] Meta-learning across regimes (MAML, 50 tasks)
- [ ] Hyperparameter search (Optuna)

### Phase 4: Validation
- [ ] Out-of-sample testing (Q3 2024)
- [ ] Walk-forward validation (2020-2024)
- [ ] Regime transfer test (train 2020-2022, test 2023-2024)
- [ ] Compare to manual configs (v1.9 baseline)

### Phase 5: Production Integration
- [ ] Export trained model to ONNX
- [ ] Integrate with `hybrid_runner.py`
- [ ] Add config override: `"use_ml_optimizer": true`
- [ ] Real-time inference (<10ms per prediction)

---

## 🔬 Experiments to Run

### Experiment 1: Domain Weight Learning
**Hypothesis**: Optimal domain weights vary by market regime
**Method**: Train model to predict weights for each regime
**Baseline**: Fixed weights (wyckoff=0.30, liquidity=0.25, momentum=0.30, smc=0.15)
**Success Metric**: >10% Sharpe improvement

### Experiment 2: Dynamic Threshold Adaptation
**Hypothesis**: Entry threshold should be higher in high volatility, lower in low volatility
**Method**: Model outputs threshold as function of ATR percentile
**Baseline**: Fixed threshold=0.70
**Success Metric**: >5% drawdown reduction

### Experiment 3: Regime-Specific Configs
**Hypothesis**: Bull markets need aggressive momentum weights, bear markets need conservative Wyckoff weights
**Method**: Cluster market states, learn config per cluster
**Baseline**: Single config for all regimes
**Success Metric**: >15% total return improvement

### Experiment 4: Multi-Asset Transfer Learning
**Hypothesis**: Patterns learned on BTC transfer to ETH/SOL with fine-tuning
**Method**: Pre-train on BTC, fine-tune on ETH with 10% data
**Baseline**: Train ETH from scratch
**Success Metric**: 80% of full-data performance with 10% data

---

## 🛠️ Technology Stack

- **ML Framework**: PyTorch 2.0 (for GPU acceleration)
- **RL Library**: Stable-Baselines3 (PPO/A2C implementations)
- **Feature Store**: Pandas + Parquet (lightweight) or HDF5 (large scale)
- **Experiment Tracking**: Weights & Biases or TensorBoard
- **Hyperparameter Tuning**: Optuna
- **Model Export**: ONNX (for production inference)

---

## 📊 Expected Outcomes

### Best Case (90th Percentile)
- **Sharpe Ratio**: 3.0+ (vs 1.5 baseline)
- **Max Drawdown**: <15% (vs 25% baseline)
- **Win Rate**: 65% (vs 55% baseline)
- **Adaptability**: <5% performance drop on new regimes

### Realistic Case (50th Percentile)
- **Sharpe Ratio**: 2.0 (33% improvement)
- **Max Drawdown**: <20% (20% improvement)
- **Win Rate**: 60% (9% improvement)
- **Adaptability**: <10% performance drop on new regimes

### Worst Case (10th Percentile)
- **Sharpe Ratio**: 1.5 (no improvement)
- **Max Drawdown**: <25% (no improvement)
- **Win Rate**: 55% (no improvement)
- **Lesson Learned**: Manual tuning remains superior, but we gain insights

---

## 🚧 Risks & Mitigations

### Risk 1: Overfitting to Historical Data
**Mitigation**:
- Walk-forward validation
- Dropout regularization (p=0.2)
- Early stopping on validation loss
- Regime-diverse training set

### Risk 2: Unstable RL Training
**Mitigation**:
- Supervised pre-training first
- Conservative PPO hyperparams (clip=0.2, lr=1e-4)
- Reward shaping (multiple objectives)
- Gradient clipping

### Risk 3: Computational Cost
**Mitigation**:
- Feature pre-computation (offline)
- Batch training on GPU
- Model quantization for inference
- ONNX export for production

### Risk 4: Model Interpretability
**Mitigation**:
- SHAP values for feature importance
- Attention visualization (if using transformers)
- Config parameter ablation studies
- Compare learned configs to manual configs

---

## 📝 Next Steps

1. **Immediate** (This Session):
   - [ ] Create feature extraction pipeline
   - [ ] Build feature store from historical data
   - [ ] Implement PyTorch model skeleton

2. **Short-Term** (Next Week):
   - [ ] Supervised pre-training
   - [ ] Baseline RL training (simple PPO)
   - [ ] Initial validation results

3. **Medium-Term** (Next Month):
   - [ ] Meta-learning implementation
   - [ ] Multi-asset experiments
   - [ ] Production integration

4. **Long-Term** (Next Quarter):
   - [ ] Live paper trading with ML configs
   - [ ] Continuous learning pipeline
   - [ ] Automated retraining on new data

---

**Status**: Ready to begin implementation
**Branch**: `feature/ml-meta-optimizer`
**Estimated Development Time**: 2-3 weeks for MVP
