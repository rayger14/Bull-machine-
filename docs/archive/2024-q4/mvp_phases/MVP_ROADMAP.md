# Bull Machine MVP Build Roadmap

**Status**: Knowledge v2.0 testing infrastructure complete
**Next Phase**: Multi-asset feature stores → Fast optimize → Fast backtest → Live shadow

---

## Current State (✅ Complete)

### Knowledge v2.0 Testing Infrastructure
- ✅ A/B/C comparison tool (`bin/compare_knowledge_v2_abc.py`)
- ✅ Feature store builder v2.0 (`bin/build_feature_store_v2.py`)
- ✅ Test configurations (baseline/shadow/active)
- ✅ Complete Week 1-4 features (104 columns)
- ✅ Documentation (TESTING_KNOWLEDGE_V2.md, KNOWLEDGE_V2_TESTING_STATUS.md)
- ✅ ETH Q3 2024 feature store built (310KB, 2,166 bars)

### Git Status
- Branch: `feature/phase2-regime-classifier`
- Commits: 7 commits documenting full testing workflow
- Ready to merge after validation

---

## MVP Deliverables (Per Brief)

### Phase 1: Multi-Timeframe Feature Stores
**Goal**: Build reproducible 1H feature stores with proper MTF down-casting

#### Assets & Periods
```
Crypto (1H):   BTC, ETH → Jan 1, 2024 – Present
Equities (1H): SPY, TSLA → Jan 1, 2024 – Present (RTH only 09:30-16:00 ET)
```

#### Feature Hierarchy Implementation
```
1D Features (HTF "Governor")
├─ Wyckoff phase, BOMS, Trend regime
├─ Range state, FRVP zones
├─ PTI, Fakeout intensity
├─ Macro regime (DXY/Yields/Oil/VIX)
└─ Forward-fill → 24 hours

4H Features (MTF "Structure")
├─ BOS/CHoCH, Internal vs External
├─ Squiggle 1-2-3, Range outcomes
├─ FRVP, Liquidity sweeps
├─ OB/FVG zones
└─ Forward-fill → 4 bars

1H Features (LTF "Execution")
├─ Micro structure
├─ Entry window triggers
├─ Kelly inputs
└─ Native resolution

Cross-TF Derived
├─ MTF alignment flags
├─ Conflict scores
├─ Knowledge v2.0 deltas
└─ Final fusion hints
```

#### Column Naming Convention
```python
# TF-prefixed
tf1d_wyckoff_phase, tf1d_boms_strength, tf1d_frvp_poc
tf4h_squiggle_stage, tf4h_choch_flag, tf4h_internal_vs_external
tf1h_pti_score, tf1h_kelly_hint, tf1h_entry_window

# Cross-TF
mtf_alignment_ok, mtf_conflict_score

# Knowledge v2.0
k2_threshold_delta, k2_score_delta, k2_risk_multiplier

# Macro
macro_regime, macro_dxy_trend, macro_vix_regime

# Labels (for later PyTorch)
label_forward_R, label_hit_1R, label_drawdown_R, label_outcome_cls
```

#### Implementation Steps

**1. Extend `bin/build_feature_store_v2.py`**
```python
# Add timeframe-specific computation modules
def compute_tf1d_features(df_1d, df_4h, df_1h, macro_data, config):
    """Compute 1D pack: Wyckoff, BOMS, Range, FRVP, PTI, Macro Echo"""
    ...

def compute_tf4h_features(df_4h, df_1h, config):
    """Compute 4H pack: Structure, Squiggle, OB/FVG, Internal/External"""
    ...

def compute_tf1h_features(df_1h, config):
    """Compute 1H pack: Micro PTI, FRVP local, Kelly inputs"""
    ...

def downcast_to_1h(df_1d_features, df_4h_features, df_1h_index):
    """Forward-fill 1D/4H features to 1H resolution"""
    df_1h = pd.DataFrame(index=df_1h_index)

    # 1D: forward-fill for 24 hours
    df_1h = df_1h.merge(df_1d_features, left_index=True, right_index=True, how='left')
    df_1h = df_1h.ffill()

    # 4H: forward-fill for 4 bars
    df_1h = df_1h.merge(df_4h_features, left_index=True, right_index=True, how='left')
    df_1h = df_1h.ffill()

    return df_1h

def compute_cross_tf_features(df_merged, config):
    """Compute MTF alignment, conflicts, K2 deltas"""
    df_merged['mtf_alignment_ok'] = (
        (df_merged['tf1d_trend_regime'] == df_merged['tf4h_trend_health']) &
        (df_merged['tf4h_bos_flag'] == df_merged['tf1h_micro_break_quality'] > 0.5)
    )
    ...
```

**2. Add Asset-Specific Handlers**
```python
def filter_rth_only(df, asset):
    """Filter equity data to RTH (09:30-16:00 ET)"""
    if asset in ['SPY', 'TSLA']:
        return df.between_time('09:30', '16:00')
    return df

def add_asset_specific_features(df, asset, config):
    """Add crypto-only or equity-only features"""
    if asset in ['BTC', 'ETH']:
        # Crypto: funding rate, OI, long/short ratio
        df['tf1h_funding_rate_z'] = ...
        df['tf1h_oi_change_pct'] = ...
    elif asset in ['SPY', 'TSLA']:
        # Equity: VIX regime, earnings windows, gaps
        df['tf1d_vix_regime'] = ...
        df['tf1d_earnings_window_d'] = ...
        df['tf1h_rth_mask'] = True
    return df
```

**3. Add Label Generation (for PyTorch later)**
```python
def compute_labels(df, lookforward_bars=8):
    """Compute forward-looking labels for supervised learning"""
    df['label_forward_R'] = (df['close'].shift(-lookforward_bars) - df['close']) / df['atr_14']
    df['label_hit_1R'] = df['label_forward_R'] >= 1.0
    df['label_drawdown_R'] = df['low'].rolling(lookforward_bars).min() / df['close'] - 1
    df['label_outcome_cls'] = pd.cut(
        df['label_forward_R'],
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=['loss', 'neutral', 'win']
    )
    return df
```

**4. Schema Validation & Audit**
```python
def validate_schema_v2(df, asset, output_dir):
    """Validate all 104+ columns present, generate audit report"""
    required_cols = {
        # TF1D pack
        'tf1d_wyckoff_phase', 'tf1d_boms_strength', ...,
        # TF4H pack
        'tf4h_squiggle_stage', 'tf4h_choch_flag', ...,
        # TF1H pack
        'tf1h_pti_score', 'tf1h_kelly_hint', ...,
        # Cross-TF
        'mtf_alignment_ok', 'mtf_conflict_score', ...,
        # Labels
        'label_forward_R', 'label_hit_1R', ...
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Generate schema report
    report = {
        'asset': asset,
        'schema_version': '2.0',
        'total_columns': len(df.columns),
        'total_rows': len(df),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'date_range': [str(df.index.min()), str(df.index.max())]
    }

    output_file = output_dir / f'{asset}_schema_report.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    return report
```

#### Expected Outputs
```
data/features_v2/
  ├── BTC_1H_20240101_to_20241215.parquet
  ├── ETH_1H_20240101_to_20241215.parquet
  ├── SPY_1H_20240101_to_20241215.parquet
  └── TSLA_1H_20240101_to_20241215.parquet

reports/features_v2/
  ├── BTC_schema_report.json
  ├── ETH_schema_report.json
  ├── SPY_schema_report.json
  └── TSLA_schema_report.json
```

---

### Phase 2: Fast Optimizer (Cached Features)
**Goal**: Maximize returns & trade count using Bayesian optimization on cached features

#### Script: `bin/optimize_v2_cached.py`

```python
#!/usr/bin/env python3
"""
Fast Optimizer v2.0 - Cached Feature Bayesian Optimization

Uses pre-built feature stores for 30-60× speedup vs on-the-fly computation.
"""

import optuna
from pathlib import Path
import pandas as pd
import numpy as np

def objective_function(trial, features_df, config_template, costs):
    """
    Optuna objective: maximize multi-objective (PF × Sharpe × trade_count_factor)

    Search space:
    - Fusion weights (sum=1.0)
    - Entry threshold (0.55-0.75)
    - Knowledge hooks (on/off flags)
    - Exit profile (R-ladder, structural, time-based)
    """
    # Sample configuration
    config = config_template.copy()

    # Fusion weights (Dirichlet-like constraint: sum=1)
    w_wyckoff = trial.suggest_float('wyckoff_weight', 0.15, 0.45)
    w_smc = trial.suggest_float('smc_weight', 0.05, 0.25)
    w_hob = trial.suggest_float('hob_weight', 0.15, 0.35)
    w_momentum = 1.0 - (w_wyckoff + w_smc + w_hob)

    config['fusion']['weights'] = {
        'wyckoff': w_wyckoff,
        'smc': w_smc,
        'liquidity': w_hob,
        'momentum': w_momentum
    }

    # Entry threshold
    config['fusion']['entry_threshold_confidence'] = trial.suggest_float('threshold', 0.55, 0.75)

    # Knowledge hooks (binary flags)
    config['knowledge_v2']['hooks'] = {
        'pti': trial.suggest_categorical('hook_pti', [True, False]),
        'fakeout': trial.suggest_categorical('hook_fakeout', [True, False]),
        'frvp': trial.suggest_categorical('hook_frvp', [True, False]),
        'boms': trial.suggest_categorical('hook_boms', [True, False]),
        'range': trial.suggest_categorical('hook_range', [True, False]),
        'squiggle': trial.suggest_categorical('hook_squiggle', [True, False]),
        'macro': trial.suggest_categorical('hook_macro', [True, False]),
    }

    # Exit profile
    config['exits']['r_ladder_enabled'] = trial.suggest_categorical('r_ladder', [True, False])
    config['exits']['structural_enabled'] = trial.suggest_categorical('structural', [True, False])
    config['exits']['time_bars'] = trial.suggest_int('time_exit_bars', 8, 48, step=4)

    # Run vectorized backtest
    metrics = run_fast_backtest(features_df, config, costs)

    # Guards: reject if PF < 1.05 or MaxDD > 15%
    if metrics['profit_factor'] < 1.05 or metrics['max_drawdown'] > 0.15:
        return -999.0

    # Multi-objective: PF × Sharpe × trade_count_factor
    trade_factor = min(metrics['total_trades'] / 40.0, 2.0)  # Cap at 2× bonus
    score = metrics['profit_factor'] * metrics['sharpe_ratio'] * trade_factor

    return score

def run_optimization(features_path, objective_type, min_trades_qtr, costs, n_trials=200):
    """
    Run Bayesian optimization using Optuna
    """
    features_df = pd.read_parquet(features_path)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        lambda trial: objective_function(trial, features_df, {}, costs),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Extract top 50 configs
    top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:50]

    return top_trials

# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--objective', default='multi', choices=['pf', 'sharpe', 'returns', 'multi'])
    parser.add_argument('--min-trades-per-qtr', type=int, default=40)
    parser.add_argument('--costs', default='slippage=bp2,fees=bp1')
    parser.add_argument('--bayes', action='store_true')
    parser.add_argument('--n-trials', type=int, default=200)

    args = parser.parse_args()

    # Parse costs
    costs = {k: float(v.replace('bp', '')) / 10000 for k,v in
             [c.split('=') for c in args.costs.split(',')]}

    # Run optimization
    results = run_optimization(args.features, args.objective, args.min_trades_per_qtr, costs, args.n_trials)

    # Save outputs
    asset = Path(args.features).stem.split('_')[0]
    output_dir = Path('reports/opt')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f'{asset}_grid_summary.json', 'w') as f:
        json.dump([{**t.params, 'score': t.value} for t in results], f, indent=2)
```

#### Expected Outputs
```
reports/opt/
  ├── BTC_grid_summary.json       # Top 50 configs
  ├── BTC_best_by_pf.json
  ├── BTC_best_by_sharpe.json
  ├── BTC_best_by_trades.json
  └── ... (repeat for ETH, SPY, TSLA)
```

---

### Phase 3: Fast Backtest (Vectorized)
**Goal**: 30-60× speedup vs bar-by-bar hybrid runner

#### Script: `bin/fast_backtest_v2.py`

```python
#!/usr/bin/env python3
"""
Fast Backtest v2.0 - Vectorized execution using cached features

No feature recomputation - pure vectorized fusion + exits.
"""

def run_fast_backtest(features_df, config, costs):
    """
    Vectorized backtest using pre-computed features

    Returns: metrics dict with PF, Sharpe, MaxDD, trade count
    """
    # Apply fusion threshold vectorized
    features_df['fusion_pass'] = (
        features_df['wyckoff'] * config['fusion']['weights']['wyckoff'] +
        features_df['smc'] * config['fusion']['weights']['smc'] +
        features_df['hob'] * config['fusion']['weights']['liquidity'] +
        features_df['momentum'] * config['fusion']['weights']['momentum']
    ) >= config['fusion']['entry_threshold_confidence']

    # Apply knowledge hooks vectorized
    if config.get('knowledge_v2', {}).get('enabled'):
        features_df['k2_threshold_delta'] = 0.0
        features_df['k2_score_delta'] = 0.0

        if config['knowledge_v2']['hooks'].get('pti'):
            features_df['k2_threshold_delta'] += features_df['tf1h_pti_score'] * 0.05

        if config['knowledge_v2']['hooks'].get('boms'):
            features_df.loc[features_df['tf4h_boms_flag'], 'k2_score_delta'] += 0.15

        # ... apply other hooks

        features_df['adjusted_threshold'] = (
            config['fusion']['entry_threshold_confidence'] +
            features_df['k2_threshold_delta'].clip(-0.10, 0.10)
        )

        features_df['fusion_pass'] = features_df['fusion_score'] >= features_df['adjusted_threshold']

    # Generate signals vectorized
    features_df['signal'] = 0
    features_df.loc[features_df['fusion_pass'] & features_df['mtf_alignment_ok'], 'signal'] = 1

    # Compute trades & P&L vectorized
    trades = []
    position = None

    for idx, row in features_df.iterrows():
        if row['signal'] == 1 and position is None:
            # Open position
            entry_price = row['close'] * (1 + costs['slippage'])
            stop_loss = entry_price - 2 * row['atr_14']
            take_profit = entry_price + 3 * row['atr_14']
            position = {'entry': entry_price, 'stop': stop_loss, 'tp': take_profit, 'entry_time': idx}

        elif position is not None:
            # Check exits
            if row['low'] <= position['stop']:
                # Stop hit
                pnl = (position['stop'] - position['entry']) / position['entry']
                trades.append({'pnl': pnl, 'R': -2.0, 'exit_reason': 'stop'})
                position = None
            elif row['high'] >= position['tp']:
                # TP hit
                pnl = (position['tp'] - position['entry']) / position['entry']
                trades.append({'pnl': pnl, 'R': 3.0, 'exit_reason': 'tp'})
                position = None

    # Calculate metrics
    if not trades:
        return {'profit_factor': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'total_trades': 0}

    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [abs(t['pnl']) for t in trades if t['pnl'] < 0]

    profit_factor = sum(wins) / sum(losses) if losses else 999.0
    sharpe_ratio = np.mean([t['pnl'] for t in trades]) / np.std([t['pnl'] for t in trades])

    # Max drawdown
    equity_curve = np.cumsum([t['pnl'] for t in trades])
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    return {
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades),
        'win_rate': len(wins) / len(trades),
        'total_return': equity_curve[-1]
    }
```

---

### Phase 4: Live Shadow Runner
**Goal**: Real-time decision logging without trading

#### Script: `bin/live_runner.py`

```python
#!/usr/bin/env python3
"""
Live Shadow Runner - Real-time feature computation + decision logging

No orders placed - pure shadow mode for parity validation.
"""

def run_live_shadow(asset, tf, lookback_days, config_path):
    """
    Pull latest bars, compute delta features, log decisions
    """
    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Pull last N bars from exchange
    df_1h = fetch_latest_bars(asset, tf='1H', lookback_hours=lookback_days*24)
    df_4h = fetch_latest_bars(asset, tf='4H', lookback_hours=lookback_days*24)
    df_1d = fetch_latest_bars(asset, tf='1D', lookback_days=lookback_days)

    # Compute features on latest bar only (delta)
    latest_features = compute_latest_features(df_1h, df_4h, df_1d, config)

    # Apply fusion + knowledge hooks
    fusion_score = compute_fusion(latest_features, config)
    k2_deltas = apply_knowledge_hooks(latest_features, config)

    # Log decision
    decision = {
        'timestamp': df_1h.index[-1].isoformat(),
        'asset': asset,
        'fusion_score': fusion_score,
        'adjusted_threshold': config['fusion']['entry_threshold'] + k2_deltas['threshold_delta'],
        'risk_mult': k2_deltas['risk_multiplier'],
        'reasons': k2_deltas['reasons'],
        'vetoes': latest_features.get('macro_veto', False),
        'would_trade': fusion_score >= (config['fusion']['entry_threshold'] + k2_deltas['threshold_delta'])
    }

    # Write to log
    log_dir = Path(f'logs/live/{asset}/{datetime.now().strftime("%Y-%m")}')
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / f'{datetime.now().strftime("%Y-%m-%d")}.jsonl', 'a') as f:
        f.write(json.dumps(decision) + '\n')

    return decision
```

---

## Acceptance Gates (Per Asset)

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| PF Uplift | ≥ +0.10 | Meaningful edge over baseline |
| Sharpe Uplift | ≥ +0.10 | Risk-adjusted improvement |
| Max DD | ≤ Baseline | No degradation in worst case |
| Trades/Qtr | ≥ 40 | Sufficient sample size |
| Fast BT Speed | ≥ 30× | Optimizer viability |
| Live Shadow Parity | ≥ 70% | Deployment readiness |

---

## Timeline (4 Weeks)

### Week 1: Multi-TF Feature Stores
- [ ] Extend build_feature_store_v2.py with TF hierarchy
- [ ] Add RTH filtering for equities
- [ ] Build BTC, ETH, SPY, TSLA stores (Jan 2024 → Present)
- [ ] Generate schema reports
- [ ] Validate 104+ columns present

### Week 2: Fast Optimizer
- [ ] Implement optimize_v2_cached.py with Optuna
- [ ] Define search space (weights, thresholds, hooks, exits)
- [ ] Run 200-trial Bayesian sweep per asset
- [ ] Extract top 50 configs per objective
- [ ] Validate PF > 1.05, MaxDD < 15% guards

### Week 3: Fast Backtest + Parity
- [ ] Implement fast_backtest_v2.py (vectorized)
- [ ] Prove 30-60× speedup vs hybrid runner
- [ ] Run parity test: fast_bt vs hybrid (±1-2% tolerance)
- [ ] Generate trade CSVs for audit

### Week 4: Live Shadow
- [ ] Implement live_runner.py
- [ ] Wire exchange API connectors
- [ ] Deploy 4-asset shadow runners (1H cron)
- [ ] Daily rollup + parity checks
- [ ] Paper trading toggle (if parity ≥70%)

---

## Next Immediate Steps

1. **Validate Knowledge v2.0 testing**
   - Run `./RUN_TESTS.sh` to completion
   - Analyze comparison report
   - If gates pass (≥3/4), merge to main

2. **Implement Multi-TF feature store builder**
   - Copy build_feature_store_v2.py → build_feature_store_mtf.py
   - Add TF hierarchy computation
   - Add down-casting logic
   - Add asset-specific handlers

3. **Build first complete feature store**
   - Run for BTC Jan 2024 → Present
   - Validate schema report
   - Benchmark build time (<10 min target)

4. **Create optimizer MVP**
   - Implement optimize_v2_cached.py skeleton
   - Wire Optuna framework
   - Test on BTC feature store
   - Validate speedup vs current approach

---

## Dependencies & Prerequisites

### Python Packages
```
optuna>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0
```

### Data Feeds
- TradingView CSV exports (current)
- Future: Bybit/Binance WebSocket (for funding/OI)
- Equity: NYSE calendar for RTH filtering

### Compute
- Laptop: Feature store build (2-10 min per asset)
- Optimizer: ~30 min per 200-trial sweep
- Fast backtest: ~10 sec per config

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Feature store build time | N/A | <10 min per asset |
| Optimizer runtime | N/A | <30 min for 200 trials |
| Fast backtest speedup | 1× (hybrid) | ≥30× |
| Assets covered | 1 (ETH partial) | 4 (BTC/ETH/SPY/TSLA) |
| Trade frequency | Variable | ≥40/quarter |
| Live shadow parity | N/A | ≥70% decision match |

---

**End of Roadmap**

This document serves as the execution blueprint for the Bull Machine MVP. All code templates are provided inline - ready for implementation.
