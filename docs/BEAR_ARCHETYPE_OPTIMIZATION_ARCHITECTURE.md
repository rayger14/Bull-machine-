# Bear Archetype Threshold Optimization Architecture
## Comprehensive Design for S2/S5 Parameter Tuning with Frontier Mapping

**Document Version:** 1.0
**Date:** 2025-11-19
**Author:** Performance Engineering Team
**Status:** Design Complete - Ready for Implementation

---

## Executive Summary

This document defines the complete optimization architecture for tuning bear archetype thresholds (S2 Failed Rally, S5 Long Squeeze) to address the critical overtrading problem (751 → 25-40 trades/year) while maximizing profit factor (0.56 → 1.3+).

**Key Innovation:** Multi-objective Bayesian optimization with Pareto frontier mapping to visualize the PF vs trade count vs win rate tradeoff surface, enabling informed parameter selection across different risk profiles.

**Deliverables:**
1. Optuna-based optimization framework with temporal cross-validation
2. 3D Pareto frontier visualization (PF × Trade Count × Win Rate)
3. 3-5 production-ready configurations (conservative → aggressive)
4. Overfitting risk mitigation via train/val split + walk-forward testing

---

## 1. Problem Statement

### Current Performance (Baseline)

**S2 (Failed Rally Rejection):**
- Baseline PF: 0.38 (2022), 0.56 (optimized)
- Trade Count: 335 (baseline), 444 (optimized) → **SEVERE OVERTRADING**
- Win Rate: 38.5% (baseline), 42.6% (optimized)
- Root Cause: Threshold too loose (fusion=0.36, wick_min=0.4 too low)

**S5 (Long Squeeze Cascade):**
- Baseline PF: 0.11 (2022), improved post-funding fix
- Trade Count: 13 (too few - missing signals)
- Win Rate: 23.1%
- Root Cause: Thresholds too strict (funding_z=1.5, rsi=75) + OI data broken

**System-Wide:**
- Combined trades: 751 trades/year (2022) → unsustainable transaction costs
- Target: 25-40 high-conviction trades/year
- Required PF: >1.3 to overcome friction

### Optimization Objectives

**Primary Goal:** Maximize profit factor while constraining trade frequency

**Multi-Objective Function:**
```
maximize: PF × selectivity_bonus
where:
    selectivity_bonus = 1.2 if 20 ≤ trades ≤ 50 else 0.8
    constraints:
        - PF ≥ 1.3 (breakeven after fees + slippage)
        - Win Rate ≥ 45% (psychological requirement)
        - Trade Count: 20-50/year (avoid under/over-trading)
```

**Secondary Goals:**
1. Maximize Sharpe ratio (risk-adjusted returns)
2. Minimize correlation with bull archetypes (H, B, L)
3. Maximize regime adaptability (works in 2022 bear + 2024 bull)

---

## 2. Parameter Search Space

### 2.1 S2 (Failed Rally Rejection) Parameters

**Tunable Parameters (6 total):**

| Parameter | Type | Current | Range | Step | Rationale |
|-----------|------|---------|-------|------|-----------|
| `fusion_threshold` | float | 0.36 | [0.40, 0.65] | 0.01 | Primary selectivity gate |
| `wick_ratio_min` | float | 2.0 | [1.5, 3.0] | 0.1 | Rejection strength (40%-60% of candle range) |
| `rsi_min` | float | 70 | [65, 75] | 1.0 | Overbought threshold |
| `volume_fade_z_max` | float | 0.5 | [0.3, 0.8] | 0.05 | Volume exhaustion signal |
| `ob_retest_tolerance` | float | 0.02 | [0.01, 0.05] | 0.005 | OB resistance proximity (%) |
| `use_multi_confluence` | bool | False | {False, True} | - | Enable 8-factor confluence filter |

**Conditional Parameters (if multi_confluence=True):**

| Parameter | Type | Current | Range | Step | Rationale |
|-----------|------|---------|-------|------|-----------|
| `min_confluence` | int | 6 | [5, 8] | 1 | Minimum factors required (out of 8) |
| `vix_z_max` | float | 1.5 | [1.0, 2.0] | 0.1 | Crisis veto threshold |

**Total Search Space:** 6 continuous + 1 boolean + 2 conditional = **~15,000 combinations** (discretized)

### 2.2 S5 (Long Squeeze Cascade) Parameters

**Tunable Parameters (4 total):**

| Parameter | Type | Current | Range | Step | Rationale |
|-----------|------|---------|-------|------|-----------|
| `fusion_threshold` | float | 0.35 | [0.40, 0.60] | 0.01 | Primary selectivity gate |
| `funding_z_min` | float | 1.2 | [1.0, 2.0] | 0.1 | Funding extremity (std deviations) |
| `rsi_min` | float | 70 | [65, 75] | 1.0 | Exhaustion threshold |
| `liquidity_max` | float | 0.25 | [0.20, 0.35] | 0.01 | Thin book threshold (cascade amplifier) |

**Total Search Space:** 4 continuous = **~5,000 combinations** (discretized)

### 2.3 Combined Search Space

**Joint Optimization:**
- Total parameters: 10 continuous + 1 boolean + 2 conditional = **~75M combinations**
- Pruning strategy: Bayesian optimization (TPE sampler) + early stopping
- Expected trials: 200-500 to reach convergence

---

## 3. Objective Function Design

### 3.1 Primary Objective: Profit Factor with Selectivity Bonus

```python
def compute_objective(trades: pd.DataFrame, params: dict) -> float:
    """
    Multi-objective function optimizing PF with trade count constraints.

    Returns:
        objective: float, higher is better
    """
    # Extract metrics
    pf = trades['pnl'].sum() / abs(trades[trades['pnl'] < 0]['pnl'].sum())
    trade_count = len(trades)
    win_rate = (trades['pnl'] > 0).mean()
    sharpe = compute_sharpe_ratio(trades)  # Annualized

    # Hard constraints (return -inf if violated)
    if pf < 1.3:
        return -np.inf  # Below breakeven threshold
    if win_rate < 0.45:
        return -np.inf  # Psychological barrier
    if trade_count < 10:
        return -np.inf  # Too few signals (underfit)
    if trade_count > 100:
        return -np.inf  # Too many signals (noise)

    # Selectivity bonus (encourage sweet spot)
    if 20 <= trade_count <= 50:
        selectivity_bonus = 1.2
    elif 10 <= trade_count < 20 or 50 < trade_count <= 70:
        selectivity_bonus = 1.0
    else:
        selectivity_bonus = 0.8

    # Weighted objective (70% PF, 30% Sharpe)
    base_objective = 0.7 * pf + 0.3 * sharpe

    # Apply selectivity bonus
    objective = base_objective * selectivity_bonus

    return objective
```

### 3.2 Secondary Objectives (Tracked, Not Optimized)

**Correlation Penalty:**
```python
def compute_correlation_penalty(s2_trades, s5_trades, bull_trades) -> float:
    """
    Penalize high correlation with existing archetypes.

    Returns:
        penalty: float in [0, 1], lower is better
    """
    # Correlation between S2 and S5 (want diversity)
    s2_s5_corr = compute_signal_correlation(s2_trades, s5_trades)

    # Correlation with bull archetypes H, B, L (want orthogonality)
    bear_bull_corr = max([
        compute_signal_correlation(s2_trades + s5_trades, bull_trades['H']),
        compute_signal_correlation(s2_trades + s5_trades, bull_trades['B']),
        compute_signal_correlation(s2_trades + s5_trades, bull_trades['L']),
    ])

    # Combined penalty (want low correlation)
    penalty = 0.3 * s2_s5_corr + 0.7 * bear_bull_corr

    # Log for analysis (not directly optimized)
    logging.info(f"Correlation penalty: {penalty:.3f} (s2_s5={s2_s5_corr:.2f}, bear_bull={bear_bull_corr:.2f})")

    return penalty
```

**Regime Robustness:**
```python
def compute_regime_robustness(trades: pd.DataFrame) -> dict:
    """
    Measure performance consistency across market regimes.

    Returns:
        metrics: dict with per-regime PF and trade counts
    """
    regime_performance = {}

    for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
        regime_trades = trades[trades['regime'] == regime]

        if len(regime_trades) > 0:
            pf = compute_pf(regime_trades)
            count = len(regime_trades)
        else:
            pf = np.nan
            count = 0

        regime_performance[regime] = {
            'pf': pf,
            'trade_count': count,
            'win_rate': (regime_trades['pnl'] > 0).mean() if count > 0 else np.nan
        }

    # Robustness score: std deviation of PF across regimes (lower is better)
    valid_pfs = [v['pf'] for v in regime_performance.values() if not np.isnan(v['pf'])]
    robustness = 1.0 / (np.std(valid_pfs) + 1e-6) if len(valid_pfs) > 1 else 0.0

    return {
        'regime_performance': regime_performance,
        'robustness_score': robustness
    }
```

---

## 4. Validation Strategy

### 4.1 Temporal Cross-Validation (No Data Leakage)

**Data Split:**

| Split | Period | Bars | Regime | Purpose |
|-------|--------|------|--------|---------|
| Train | 2022-01-01 to 2022-12-31 | 8,760 | Bear market | Parameter optimization |
| Validation | 2023-01-01 to 2023-12-31 | 8,760 | Recovery/mixed | Out-of-sample test |
| Test | 2024-01-01 to 2024-09-30 | 6,570 | Bull market | Regime robustness |

**Rationale:**
- Train on pure bear market (2022) to learn bear archetype thresholds
- Validate on mixed regime (2023) to test generalization
- Test on bull market (2024) to ensure thresholds don't break in different regime

### 4.2 Walk-Forward Testing (Production Validation)

**Rolling Window Approach:**

```python
def walk_forward_test(data: pd.DataFrame, n_splits: int = 6) -> list:
    """
    Walk-forward optimization with expanding window.

    Returns:
        results: list of dicts with train/test metrics per fold
    """
    results = []

    # Split data into 6-month chunks
    splits = split_data_by_months(data, n_splits=n_splits)

    for i in range(2, n_splits):  # Start after 2 folds (need history)
        # Train on ALL data up to current point
        train_data = pd.concat([splits[j] for j in range(i)])

        # Test on next fold
        test_data = splits[i]

        # Optimize on train
        best_params = optuna_optimize(train_data)

        # Evaluate on test
        test_trades = backtest_with_params(test_data, best_params)
        test_metrics = compute_metrics(test_trades)

        results.append({
            'fold': i,
            'train_period': f"{train_data.index[0]} to {train_data.index[-1]}",
            'test_period': f"{test_data.index[0]} to {test_data.index[-1]}",
            'params': best_params,
            'test_pf': test_metrics['pf'],
            'test_trades': len(test_trades),
            'test_wr': test_metrics['win_rate']
        })

    return results
```

**Stability Check:**
- Parameter drift: Monitor how optimal parameters change across folds
- If params drift >20%, consider regime-adaptive thresholds
- If params stable, use median values for production

### 4.3 Overfitting Detection

**Statistical Tests:**

1. **Permutation Test:**
   ```python
   def permutation_test(train_pf: float, val_pf: float, n_permutations: int = 1000) -> float:
       """
       Test if validation PF drop is due to chance or overfitting.

       Returns:
           p_value: probability that PF drop is random
       """
       pf_drops = []

       for _ in range(n_permutations):
           # Shuffle train/val split randomly
           shuffled_train_pf, shuffled_val_pf = random_split_and_backtest()
           pf_drop = shuffled_train_pf - shuffled_val_pf
           pf_drops.append(pf_drop)

       # Compare actual PF drop to null distribution
       actual_drop = train_pf - val_pf
       p_value = (np.array(pf_drops) >= actual_drop).mean()

       return p_value
   ```

2. **PF Degradation Threshold:**
   - Accept if `val_pf >= 0.85 * train_pf` (15% tolerance)
   - Warn if `val_pf >= 0.70 * train_pf` (30% drop)
   - Reject if `val_pf < 0.70 * train_pf` (severe overfit)

3. **Trade Count Stability:**
   - Accept if `abs(val_trades - train_trades) / train_trades <= 0.25` (25% variance)
   - Reject if variance >50% (unstable parameters)

---

## 5. Frontier Mapping Methodology

### 5.1 Pareto Frontier Definition

**3D Optimization Surface:**
- **X-axis:** Trade Count (frequency)
- **Y-axis:** Profit Factor (returns per risk)
- **Z-axis:** Win Rate (psychological comfort)

**Pareto Optimality:**
A parameter set is **Pareto optimal** if no other configuration improves one metric without degrading another.

**Example:**
```
Config A: PF=1.5, Trades=30, WR=50% ← Pareto optimal
Config B: PF=1.4, Trades=25, WR=55% ← Pareto optimal (higher WR, lower PF)
Config C: PF=1.3, Trades=35, WR=48% ← Dominated by A (worse on all metrics)
```

### 5.2 Frontier Generation Algorithm

```python
def generate_pareto_frontier(trials: List[OptunaTrialResult]) -> pd.DataFrame:
    """
    Extract Pareto-optimal configurations from Optuna trials.

    Returns:
        frontier_df: DataFrame with Pareto-optimal configs and metrics
    """
    # Convert trials to DataFrame
    trial_data = []
    for trial in trials:
        trial_data.append({
            'trial_id': trial.number,
            'pf': trial.user_attrs['pf'],
            'trade_count': trial.user_attrs['trade_count'],
            'win_rate': trial.user_attrs['win_rate'],
            'sharpe': trial.user_attrs['sharpe'],
            'params': trial.params,
            'objective': trial.value
        })

    df = pd.DataFrame(trial_data)

    # Filter to feasible region (meets hard constraints)
    feasible = df[
        (df['pf'] >= 1.3) &
        (df['win_rate'] >= 0.45) &
        (df['trade_count'] >= 20) &
        (df['trade_count'] <= 50)
    ].copy()

    # Compute Pareto frontier (3D)
    pareto_mask = np.ones(len(feasible), dtype=bool)

    for i, row_i in feasible.iterrows():
        # Check if dominated by any other point
        dominated = (
            (feasible['pf'] >= row_i['pf']) &
            (feasible['trade_count'] <= row_i['trade_count']) &  # Fewer trades is better
            (feasible['win_rate'] >= row_i['win_rate']) &
            (
                (feasible['pf'] > row_i['pf']) |
                (feasible['trade_count'] < row_i['trade_count']) |
                (feasible['win_rate'] > row_i['win_rate'])
            )
        )

        if dominated.any():
            pareto_mask[i] = False

    frontier = feasible[pareto_mask].copy()

    # Sort by PF descending
    frontier = frontier.sort_values('pf', ascending=False).reset_index(drop=True)

    return frontier
```

### 5.3 Visualization Strategy

**3D Interactive Plot (Plotly):**

```python
import plotly.graph_objects as go

def plot_pareto_frontier_3d(all_trials: pd.DataFrame, frontier: pd.DataFrame):
    """
    Create interactive 3D scatter plot with Pareto frontier highlighted.
    """
    fig = go.Figure()

    # All feasible trials (gray)
    fig.add_trace(go.Scatter3d(
        x=all_trials['trade_count'],
        y=all_trials['pf'],
        z=all_trials['win_rate'] * 100,
        mode='markers',
        marker=dict(size=4, color='lightgray', opacity=0.5),
        name='All Trials',
        hovertemplate='<b>Trial %{text}</b><br>' +
                      'Trades: %{x}<br>' +
                      'PF: %{y:.2f}<br>' +
                      'WR: %{z:.1f}%',
        text=all_trials['trial_id']
    ))

    # Pareto frontier (red)
    fig.add_trace(go.Scatter3d(
        x=frontier['trade_count'],
        y=frontier['pf'],
        z=frontier['win_rate'] * 100,
        mode='markers+lines',
        marker=dict(size=8, color='red', symbol='diamond'),
        line=dict(color='red', width=2),
        name='Pareto Frontier',
        hovertemplate='<b>Config %{text}</b><br>' +
                      'Trades: %{x}<br>' +
                      'PF: %{y:.2f}<br>' +
                      'WR: %{z:.1f}%<br>' +
                      'Sharpe: %{customdata:.2f}',
        text=frontier.index,
        customdata=frontier['sharpe']
    ))

    # Layout
    fig.update_layout(
        title='Bear Archetype Optimization: Pareto Frontier (3D)',
        scene=dict(
            xaxis_title='Trade Count (per year)',
            yaxis_title='Profit Factor',
            zaxis_title='Win Rate (%)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        width=1200,
        height=800
    )

    fig.write_html('results/pareto_frontier_3d.html')
    fig.show()
```

**2D Projections (Matplotlib):**

```python
import matplotlib.pyplot as plt

def plot_pareto_frontier_2d(all_trials: pd.DataFrame, frontier: pd.DataFrame):
    """
    Create 2D projection plots for each metric pair.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: PF vs Trade Count
    axes[0].scatter(all_trials['trade_count'], all_trials['pf'],
                    alpha=0.3, s=20, c='gray', label='All Trials')
    axes[0].scatter(frontier['trade_count'], frontier['pf'],
                    alpha=1.0, s=100, c='red', marker='D',
                    edgecolors='black', linewidths=2, label='Pareto Frontier')
    axes[0].set_xlabel('Trade Count (per year)', fontsize=12)
    axes[0].set_ylabel('Profit Factor', fontsize=12)
    axes[0].set_title('PF vs Trade Count', fontsize=14, fontweight='bold')
    axes[0].axhline(1.3, color='green', linestyle='--', alpha=0.5, label='Min PF=1.3')
    axes[0].axvline(40, color='blue', linestyle='--', alpha=0.5, label='Target Trades=40')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: PF vs Win Rate
    axes[1].scatter(all_trials['win_rate'] * 100, all_trials['pf'],
                    alpha=0.3, s=20, c='gray')
    axes[1].scatter(frontier['win_rate'] * 100, frontier['pf'],
                    alpha=1.0, s=100, c='red', marker='D',
                    edgecolors='black', linewidths=2)
    axes[1].set_xlabel('Win Rate (%)', fontsize=12)
    axes[1].set_ylabel('Profit Factor', fontsize=12)
    axes[1].set_title('PF vs Win Rate', fontsize=14, fontweight='bold')
    axes[1].axhline(1.3, color='green', linestyle='--', alpha=0.5)
    axes[1].axvline(45, color='blue', linestyle='--', alpha=0.5, label='Min WR=45%')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Plot 3: Win Rate vs Trade Count
    axes[2].scatter(all_trials['trade_count'], all_trials['win_rate'] * 100,
                    alpha=0.3, s=20, c='gray')
    axes[2].scatter(frontier['trade_count'], frontier['win_rate'] * 100,
                    alpha=1.0, s=100, c='red', marker='D',
                    edgecolors='black', linewidths=2)
    axes[2].set_xlabel('Trade Count (per year)', fontsize=12)
    axes[2].set_ylabel('Win Rate (%)', fontsize=12)
    axes[2].set_title('Win Rate vs Trade Count', fontsize=14, fontweight='bold')
    axes[2].axhline(45, color='blue', linestyle='--', alpha=0.5)
    axes[2].axvline(40, color='blue', linestyle='--', alpha=0.5)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/pareto_frontier_2d.png', dpi=150)
    plt.show()
```

### 5.4 Configuration Selection Strategy

**Extract 5 Representative Configs from Frontier:**

| Profile | Target | Selection Criteria |
|---------|--------|-------------------|
| Conservative | Maximize WR, accept lower PF | `max(wr)` where `pf >= 1.3` |
| Balanced | Optimize PF/trade ratio | `max(pf / sqrt(trades))` |
| Aggressive | Maximize PF, accept lower WR | `max(pf)` where `wr >= 0.45` |
| High-Frequency | More trades, moderate PF | `trades in [40, 50]`, `max(pf)` |
| Sniper | Few trades, very high PF | `trades in [20, 30]`, `max(pf)` |

**Example Output:**

```python
{
    "conservative": {
        "pf": 1.45,
        "trade_count": 28,
        "win_rate": 0.54,
        "params": {
            "s2_fusion_threshold": 0.48,
            "s2_wick_ratio_min": 2.3,
            "s2_rsi_min": 72,
            "s5_fusion_threshold": 0.52,
            "s5_funding_z_min": 1.4,
            "s5_rsi_min": 72
        },
        "use_case": "Risk-averse traders, prioritize consistency"
    },

    "balanced": {
        "pf": 1.62,
        "trade_count": 35,
        "win_rate": 0.49,
        "params": {
            "s2_fusion_threshold": 0.44,
            "s2_wick_ratio_min": 2.0,
            "s2_rsi_min": 70,
            "s5_fusion_threshold": 0.48,
            "s5_funding_z_min": 1.2,
            "s5_rsi_min": 70
        },
        "use_case": "Default production config, best risk-adjusted returns"
    },

    "aggressive": {
        "pf": 1.88,
        "trade_count": 42,
        "win_rate": 0.46,
        "params": {
            "s2_fusion_threshold": 0.42,
            "s2_wick_ratio_min": 1.8,
            "s2_rsi_min": 68,
            "s5_fusion_threshold": 0.45,
            "s5_funding_z_min": 1.1,
            "s5_rsi_min": 68
        },
        "use_case": "High-conviction traders, maximize absolute returns"
    }
}
```

---

## 6. Implementation Architecture

### 6.1 Optimization Framework (Optuna)

**Main Optimization Loop:**

```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for bear archetype optimization.

    Args:
        trial: Optuna trial object

    Returns:
        objective_value: float, higher is better
    """
    # Sample S2 parameters
    s2_params = {
        'fusion_threshold': trial.suggest_float('s2_fusion_threshold', 0.40, 0.65, step=0.01),
        'wick_ratio_min': trial.suggest_float('s2_wick_ratio_min', 1.5, 3.0, step=0.1),
        'rsi_min': trial.suggest_float('s2_rsi_min', 65, 75, step=1.0),
        'volume_fade_z_max': trial.suggest_float('s2_volume_fade_z_max', 0.3, 0.8, step=0.05),
        'ob_retest_tolerance': trial.suggest_float('s2_ob_retest_tolerance', 0.01, 0.05, step=0.005),
        'use_multi_confluence': trial.suggest_categorical('s2_use_multi_confluence', [False, True])
    }

    # Conditional parameters
    if s2_params['use_multi_confluence']:
        s2_params['min_confluence'] = trial.suggest_int('s2_min_confluence', 5, 8)
        s2_params['vix_z_max'] = trial.suggest_float('s2_vix_z_max', 1.0, 2.0, step=0.1)

    # Sample S5 parameters
    s5_params = {
        'fusion_threshold': trial.suggest_float('s5_fusion_threshold', 0.40, 0.60, step=0.01),
        'funding_z_min': trial.suggest_float('s5_funding_z_min', 1.0, 2.0, step=0.1),
        'rsi_min': trial.suggest_float('s5_rsi_min', 65, 75, step=1.0),
        'liquidity_max': trial.suggest_float('s5_liquidity_max', 0.20, 0.35, step=0.01)
    }

    # Run backtest with sampled parameters
    trades = run_backtest_with_params(s2_params, s5_params, data=TRAIN_DATA)

    # Compute metrics
    metrics = compute_metrics(trades)

    # Store metrics for frontier analysis
    trial.set_user_attr('pf', metrics['pf'])
    trial.set_user_attr('trade_count', metrics['trade_count'])
    trial.set_user_attr('win_rate', metrics['win_rate'])
    trial.set_user_attr('sharpe', metrics['sharpe'])
    trial.set_user_attr('max_dd', metrics['max_dd'])

    # Early pruning if hard constraints violated
    if metrics['pf'] < 1.3 or metrics['win_rate'] < 0.45:
        raise optuna.TrialPruned()

    # Compute objective
    objective_value = compute_objective(trades, s2_params | s5_params)

    return objective_value


def run_optimization(n_trials: int = 500, timeout: int = 7200):
    """
    Execute Optuna optimization study.

    Args:
        n_trials: Number of trials (default 500)
        timeout: Max runtime in seconds (default 2 hours)

    Returns:
        study: Completed Optuna study object
    """
    # Create study
    study = optuna.create_study(
        study_name='bear_archetype_optimization',
        direction='maximize',
        sampler=TPESampler(seed=42, n_startup_trials=50),
        pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=5)
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=4,  # Parallel trials
        show_progress_bar=True
    )

    # Save study
    study.trials_dataframe().to_csv('results/optimization_trials.csv')

    # Generate Pareto frontier
    frontier = generate_pareto_frontier(study.trials)
    frontier.to_csv('results/pareto_frontier.csv', index=False)

    # Visualize
    plot_pareto_frontier_3d(study.trials_dataframe(), frontier)
    plot_pareto_frontier_2d(study.trials_dataframe(), frontier)

    # Extract production configs
    configs = extract_production_configs(frontier)

    with open('results/production_configs.json', 'w') as f:
        json.dump(configs, f, indent=2)

    return study
```

### 6.2 Backtest Integration

**Modified Backtest Engine:**

```python
def run_backtest_with_params(
    s2_params: dict,
    s5_params: dict,
    data: pd.DataFrame,
    regime_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Run backtest with custom archetype parameters.

    Args:
        s2_params: S2 threshold dictionary
        s5_params: S5 threshold dictionary
        data: Feature store DataFrame
        regime_data: Regime labels DataFrame

    Returns:
        trades: DataFrame with trade results
    """
    # Build archetype config
    archetype_config = {
        'use_archetypes': True,
        'enable_S2': True,
        'enable_S5': True,
        'enable_A': False,  # Disable bull archetypes during bear optimization
        'enable_B': False,
        'enable_C': False,
        'enable_D': False,
        'enable_E': False,
        'enable_F': False,
        'enable_G': False,
        'enable_H': False,
        'enable_K': False,
        'enable_L': False,
        'enable_M': False,

        'thresholds': {
            'failed_rally': s2_params,  # S2 canonical name
            'long_squeeze': s5_params   # S5 canonical name
        },

        'routing': {
            'risk_off': {
                'weights': {
                    'failed_rally': 2.0,
                    'long_squeeze': 2.2
                }
            },
            'crisis': {
                'weights': {
                    'failed_rally': 2.5,
                    'long_squeeze': 2.8
                }
            }
        }
    }

    # Initialize archetype logic
    archetype_logic = ArchetypeLogic(archetype_config)

    # Run backtest (simplified)
    trades = []

    for i in range(len(data)):
        row = data.iloc[i]
        regime_row = regime_data.iloc[i]

        # Create RuntimeContext
        ctx = RuntimeContext(
            ts=row.name,
            row=row,
            regime_probs=regime_row['regime_probs'],
            regime_label=regime_row['regime_label'],
            adapted_params={},
            thresholds=archetype_config['thresholds']
        )

        # Detect archetype
        archetype, fusion, liquidity = archetype_logic.detect(ctx)

        if archetype in ['failed_rally', 'long_squeeze']:
            # Simulate trade
            entry_price = row['close']
            atr = row['atr_20']
            stop_loss = entry_price + 1.5 * atr  # Short position

            # Find exit (simplified - use vectorized forward lookup in production)
            exit_idx = i + 1
            exit_price = data.iloc[exit_idx]['close'] if exit_idx < len(data) else entry_price

            pnl_pct = (entry_price - exit_price) / entry_price

            trades.append({
                'entry_time': row.name,
                'archetype': archetype,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'regime': regime_row['regime_label']
            })

    return pd.DataFrame(trades)
```

### 6.3 Performance Benchmarks

**Expected Compute Time:**

| Component | Time (sec) | Notes |
|-----------|------------|-------|
| Single trial backtest | 0.5 | 8,760 bars (2022) |
| Optuna trial (with pruning) | 0.8 | Includes metric computation |
| 500 trials (serial) | 400 | 6.7 minutes |
| 500 trials (4 parallel) | 120 | 2 minutes |
| Frontier generation | 2 | Post-processing |
| Visualization | 5 | Plotly + matplotlib |
| **Total (500 trials)** | **~130 sec** | **2.2 minutes** |

**Scaling to 2000 trials:**
- Serial: 26 minutes
- 4 parallel: 7 minutes
- 8 parallel: 4 minutes

**Memory Requirements:**
- Feature store (2022): ~500 MB
- Trial history (500 trials): ~10 MB
- Peak memory: ~800 MB (acceptable for laptop)

---

## 7. Risk Mitigation Strategies

### 7.1 Overfitting Prevention

**1. Parameter Regularization:**
```python
def apply_parameter_regularization(params: dict, reference_params: dict) -> float:
    """
    Penalize parameters that drift too far from baseline.

    Returns:
        penalty: float, higher means more drift
    """
    drift_penalty = 0.0

    for key, value in params.items():
        if key in reference_params:
            ref_value = reference_params[key]

            # Compute normalized drift (% change)
            drift = abs(value - ref_value) / abs(ref_value + 1e-6)

            # Penalize if drift >30%
            if drift > 0.30:
                drift_penalty += (drift - 0.30) ** 2

    return drift_penalty
```

**2. Ensemble Validation:**
- Train 5 models on different random splits of 2022 data
- Production config = median parameters across 5 models
- Reject if parameter variance >25%

**3. Out-of-Sample Testing:**
- Never tune on 2023 or 2024 data
- Use 2023 as pure validation (one-time test)
- Use 2024 as regime robustness check (one-time test)

### 7.2 Regime Change Adaptation

**Problem:** Parameters optimized on 2022 bear market may fail in 2024 bull market.

**Solution 1: Regime-Conditional Thresholds**

```json
{
  "archetypes": {
    "failed_rally": {
      "base_thresholds": {
        "fusion_threshold": 0.44,
        "wick_ratio_min": 2.0,
        "rsi_min": 70
      },
      "regime_adjustments": {
        "risk_on": {
          "fusion_threshold": 0.50,
          "wick_ratio_min": 2.5
        },
        "risk_off": {
          "fusion_threshold": 0.40,
          "wick_ratio_min": 1.8
        }
      }
    }
  }
}
```

**Solution 2: Regime Routing Weights (Already Implemented)**

- Use fixed thresholds but adjust archetype weights by regime
- Risk-off: Amplify bear archetypes 2x
- Risk-on: Suppress bear archetypes 0.3x

**Recommendation:** Use Solution 2 (routing weights) to avoid parameter explosion.

### 7.3 Transaction Cost Sensitivity

**Problem:** Optimizing on gross PF may ignore transaction costs.

**Solution: Net PF Calculation**

```python
def compute_net_pf(trades: pd.DataFrame, fees_bps: float = 6.0, slippage_bps: float = 2.0) -> float:
    """
    Compute profit factor after fees and slippage.

    Args:
        trades: Trade results DataFrame
        fees_bps: Trading fees in basis points (default 0.06%)
        slippage_bps: Slippage in basis points (default 0.02%)

    Returns:
        net_pf: Profit factor after costs
    """
    total_costs_bps = fees_bps + slippage_bps  # 8 bps round-trip

    # Adjust PnL for costs (charge on entry + exit)
    trades['net_pnl'] = trades['pnl_pct'] - 2 * (total_costs_bps / 10000)

    gross_profit = trades[trades['net_pnl'] > 0]['net_pnl'].sum()
    gross_loss = abs(trades[trades['net_pnl'] < 0]['net_pnl'].sum())

    net_pf = gross_profit / (gross_loss + 1e-9)

    return net_pf
```

**Sensitivity Analysis:**

| Fee Level | Target Net PF | Implied Gross PF |
|-----------|---------------|------------------|
| Low (4 bps) | 1.3 | 1.36 |
| Medium (8 bps) | 1.3 | 1.42 |
| High (12 bps) | 1.3 | 1.48 |

**Recommendation:** Optimize for **gross PF ≥ 1.42** to ensure net PF ≥ 1.3 under 8 bps friction.

---

## 8. Deliverables and Timeline

### 8.1 Code Deliverables

**1. Optimization Engine:** `/bin/optimize_bear_archetypes.py`
```python
# Main optimization script with Optuna integration
# Features:
# - TPE sampler with early pruning
# - Temporal cross-validation
# - Walk-forward testing
# - Pareto frontier generation
```

**2. Frontier Visualization:** `/bin/visualize_pareto_frontier.py`
```python
# Standalone visualization script
# Outputs:
# - 3D interactive HTML (plotly)
# - 2D projection plots (matplotlib)
# - Parameter sensitivity heatmaps
```

**3. Configuration Generator:** `/bin/generate_production_configs.py`
```python
# Extract 5 production configs from Pareto frontier
# Outputs:
# - JSON configs for each risk profile
# - Validation reports (train/val/test metrics)
# - Deployment recommendations
```

**4. Validation Suite:** `/bin/validate_bear_configs.py`
```python
# Test production configs on unseen data
# Features:
# - Walk-forward backtesting
# - Regime robustness analysis
# - Correlation matrix with bull archetypes
# - Monte Carlo permutation tests
```

### 8.2 Documentation Deliverables

**1. This Document:** `docs/BEAR_ARCHETYPE_OPTIMIZATION_ARCHITECTURE.md`
- Design specification (COMPLETE)
- Mathematical formulation
- Implementation guide

**2. Results Report:** `results/OPTIMIZATION_RESULTS_REPORT.md`
- Pareto frontier analysis
- Recommended production configs
- Performance comparison (baseline vs optimized)
- Deployment checklist

**3. Parameter Sensitivity Analysis:** `results/PARAMETER_SENSITIVITY.md`
- Which parameters matter most?
- Interaction effects (fusion_threshold × wick_ratio)
- Regime-specific sensitivities

### 8.3 Timeline

| Phase | Duration | Deliverables | Status |
|-------|----------|--------------|--------|
| **Phase 1: Design** | 1 day | This architecture document | ✅ COMPLETE |
| **Phase 2: Implementation** | 2 days | Optimization engine + visualization | ⏳ Ready to start |
| **Phase 3: Execution** | 4 hours | Run 500-trial optimization | ⏳ Pending Phase 2 |
| **Phase 4: Validation** | 1 day | Walk-forward testing + regime analysis | ⏳ Pending Phase 3 |
| **Phase 5: Deployment** | 0.5 days | Generate production configs + docs | ⏳ Pending Phase 4 |
| **Total** | **4.5 days** | Full optimization pipeline | **20% complete** |

---

## 9. Success Criteria

### 9.1 Quantitative Metrics

**Hard Requirements (Must Pass):**
- ✅ Train PF ≥ 1.3 (2022 bear market)
- ✅ Validation PF ≥ 1.1 (2023 mixed regime) - allows 15% degradation
- ✅ Test PF ≥ 0.8 (2024 bull market) - bear patterns may underperform in bull
- ✅ Trade Count: 20-50 per year (2022)
- ✅ Win Rate ≥ 45% (all periods)

**Soft Goals (Nice to Have):**
- 🎯 Train PF ≥ 1.5 (aggressive target)
- 🎯 Validation PF ≥ 1.3 (minimal degradation)
- 🎯 Sharpe ≥ 0.5 (2022 bear market)
- 🎯 Max DD ≤ 5% (2022)
- 🎯 Correlation with bull archetypes ≤ 0.3

### 9.2 Qualitative Criteria

**Interpretability:**
- Parameters should be within reasonable ranges (no extreme values)
- Parameter stability across walk-forward folds (variance <25%)
- Clear regime-specific behavior (bear archetypes amplified in risk-off)

**Robustness:**
- Performance degrades gracefully in adverse regimes
- No catastrophic failures (single trade >10% loss)
- Consistent behavior across different market conditions

**Production Readiness:**
- All required features available in production pipeline
- No manual intervention needed
- Configs can be deployed via JSON files

---

## 10. Monitoring and Maintenance

### 10.1 Post-Deployment Monitoring

**Weekly Metrics:**
- Live PF vs backtest PF (detect regime drift)
- Trade count vs expected (detect parameter staleness)
- Win rate tracking (psychological check)

**Monthly Reviews:**
- Rerun validation on trailing 3 months
- Check if parameters need retuning (if PF drops >20%)
- Update regime routing weights if regime classifier drifts

### 10.2 Reoptimization Triggers

**When to Retune:**
1. **Regime Shift:** If regime classifier changes behavior (e.g., new crisis)
2. **Performance Degradation:** If live PF <80% of backtest PF for 2 months
3. **Market Structure Change:** If volatility regime shifts (ATR doubles)
4. **Data Pipeline Updates:** If new features added (e.g., OI_CHANGE fixed)

**How to Retune:**
1. Append new data to training set (expand window)
2. Rerun optimization with updated data
3. Compare new vs old configs (if >20% drift, investigate)
4. A/B test new config vs old config on paper trading (1 month)
5. Deploy if new config wins by >10% PF

---

## 11. Conclusion

This architecture provides a complete, production-ready framework for optimizing bear archetype thresholds with the following key features:

**Strengths:**
1. ✅ Multi-objective optimization (PF × selectivity)
2. ✅ Rigorous validation (temporal CV + walk-forward)
3. ✅ Pareto frontier visualization (informed config selection)
4. ✅ Overfitting mitigation (regularization + ensemble + out-of-sample)
5. ✅ Regime robustness (adaptive routing weights)
6. ✅ Fast execution (~2 minutes for 500 trials)

**Limitations:**
1. ⚠️ Assumes feature store completeness (OI_CHANGE needs fixing)
2. ⚠️ Bear archetypes may underperform in bull markets (expected)
3. ⚠️ Requires periodic retuning as market structure evolves

**Next Steps:**
1. Implement optimization engine (`bin/optimize_bear_archetypes.py`)
2. Run 500-trial optimization on 2022 train data
3. Extract Pareto frontier and 5 production configs
4. Validate on 2023/2024 out-of-sample data
5. Deploy best config to production

**Expected Outcome:**
- Reduce trade count from 751 → 30-40 (95% reduction)
- Increase PF from 0.56 → 1.4+ (2.5x improvement)
- Maintain win rate ~50% (psychological comfort)
- Enable profitable bear market trading for first time

---

**Document Status:** ✅ APPROVED FOR IMPLEMENTATION
**Review Date:** 2025-11-19
**Next Review:** After Phase 3 execution
