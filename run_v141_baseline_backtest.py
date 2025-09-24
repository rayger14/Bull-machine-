#!/usr/bin/env python3
"""
v1.4.1 Baseline Backtest - For comparison with v1.4.2 improvements
Run the same backtest without the hotfix improvements
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from bull_machine.scoring.fusion import FusionEngineV141

def generate_realistic_layer_scores(df: pd.DataFrame, enabled_modules: list) -> pd.DataFrame:
    """Generate identical layer scores as v1.4.2 for fair comparison."""
    np.random.seed(42)  # Same seed for identical layer scores
    scores_df = pd.DataFrame(index=df.index)

    # Calculate price volatility for more realistic scoring
    if 'close' in df.columns:
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std()
        trend_strength = abs(returns.rolling(10).mean()) * 100
    else:
        volatility = pd.Series(0.02, index=df.index)
        trend_strength = pd.Series(1.0, index=df.index)

    for module in enabled_modules:
        if module == 'wyckoff':
            base_score = 0.5 + (trend_strength * 0.3).clip(0, 0.4)
            noise = np.random.normal(0, 0.1, len(df))
            scores_df['wyckoff'] = (base_score + noise).clip(0.2, 0.9)

        elif module == 'liquidity':
            volume_changes = df['volume'].pct_change().abs()
            base_score = 0.45 + (volume_changes * 2).clip(0, 0.3)
            noise = np.random.normal(0, 0.08, len(df))
            scores_df['liquidity'] = (base_score + noise).clip(0.1, 0.85)

        elif module == 'structure':
            base_score = 0.4 + 0.3 * np.sin(np.arange(len(df)) * 0.1)
            noise = np.random.normal(0, 0.1, len(df))
            scores_df['structure'] = (base_score + noise).clip(0.2, 0.8)

        elif module == 'momentum':
            momentum = returns.rolling(5).mean().abs()
            base_score = 0.35 + (momentum * 20).clip(0, 0.35)
            noise = np.random.normal(0, 0.12, len(df))
            scores_df['momentum'] = (base_score + noise).clip(0.15, 0.85)

        elif module == 'volume':
            vol_sma = df['volume'].rolling(20).mean()
            vol_ratio = (df['volume'] / vol_sma).clip(0.5, 3.0)
            base_score = 0.3 + (vol_ratio - 1) * 0.2
            noise = np.random.normal(0, 0.1, len(df))
            scores_df['volume'] = (base_score + noise).clip(0.1, 0.8)

        elif module == 'context':
            base_score = 0.4
            spikes = np.random.random(len(df)) < 0.05
            context_scores = np.full(len(df), base_score)
            context_scores[spikes] *= 0.3
            noise = np.random.normal(0, 0.05, len(df))
            scores_df['context'] = (context_scores + noise).clip(0.1, 0.7)

        elif module == 'mtf':
            base_score = 0.55 + 0.2 * np.cos(np.arange(len(df)) * 0.02)
            noise = np.random.normal(0, 0.1, len(df))
            scores_df['mtf'] = (base_score + noise).clip(0.3, 0.85)

        elif module == 'bojan':
            base_score = 0.6 + 0.3 * np.random.normal(0, 0.2, len(df))
            scores_df['bojan'] = base_score.clip(0.2, 1.0)

    return scores_df

def simulate_trades_v141_baseline(df: pd.DataFrame, fusion_df: pd.DataFrame,
                                 enter_threshold: float, config: dict) -> list:
    """Simulate trades with v1.4.1 baseline (no hotfix improvements)."""
    trades = []
    in_position = False
    entry_idx = None
    entry_price = None

    # v1.4.1 baseline: fixed 1.5x stops for all phases
    base_atr_mult = 1.5

    # Calculate ATR for stop calculation
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()

    for i in range(50, len(df) - 10):

        if not in_position:
            # Entry logic - v1.4.1 baseline includes stricter regime filter
            if (fusion_df.iloc[i]['weighted_score'] >= enter_threshold and
                not fusion_df.iloc[i]['global_veto']):

                # v1.4.1 regime filter: veto ALL A/C phases (no high-vol override)
                # We'll simulate this by being more restrictive on some entries
                wyckoff_score = 0.7  # Assume decent Wyckoff score

                # Simulate A/C phase detection
                recent_high = df.iloc[i-20:i]['high'].max()
                recent_low = df.iloc[i-20:i]['low'].min()
                range_pos = (df.iloc[i]['close'] - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5

                # Simulate phase A/C detection (consolidation phases)
                if 0.3 < range_pos < 0.7:  # Mid-range = consolidation
                    vol_sma = df.iloc[i-20:i]['volume'].mean()
                    vol_ratio = df.iloc[i]['volume'] / vol_sma if vol_sma > 0 else 1.0

                    # v1.4.1: strict veto for A/C phases regardless of volume
                    if vol_ratio < 1.2 and wyckoff_score < 0.85:
                        continue  # Skip this entry (regime veto)

                in_position = True
                entry_idx = i
                entry_price = df.iloc[i]['close']

                # v1.4.1: Fixed 1.5x stops for all phases
                atr_value = df.iloc[i]['atr'] if not np.isnan(df.iloc[i]['atr']) else entry_price * 0.02
                stop_distance = atr_value * base_atr_mult
                stop_price = entry_price - stop_distance

        else:
            # Exit logic
            current_price = df.iloc[i]['close']
            bars_held = i - entry_idx

            # Stop loss
            if current_price <= stop_price:
                pnl_pct = (current_price - entry_price) / entry_price
                trades.append({
                    'entry_time': df.index[entry_idx],
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'bars_held': bars_held,
                    'exit_reason': 'stop_loss',
                    'atr_multiplier': base_atr_mult
                })
                in_position = False
                continue

            # v1.4.1: Earlier distribution exit (1.5x volume threshold)
            vol_sma = df.iloc[i-20:i]['volume'].mean()
            vol_ratio = df.iloc[i]['volume'] / vol_sma if vol_sma > 0 else 1.0

            if vol_ratio >= 1.5 and bars_held >= 5:  # v1.4.1: 1.5x threshold
                # Early partial exit in v1.4.1
                pnl_pct = (current_price - entry_price) / entry_price
                trades.append({
                    'entry_time': df.index[entry_idx],
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'bars_held': bars_held,
                    'exit_reason': 'distribution_exit',
                    'atr_multiplier': base_atr_mult
                })
                in_position = False
                continue

            # Time-based exit
            max_bars = 36
            if bars_held >= max_bars:
                pnl_pct = (current_price - entry_price) / entry_price
                trades.append({
                    'entry_time': df.index[entry_idx],
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'bars_held': bars_held,
                    'exit_reason': 'time_stop',
                    'atr_multiplier': base_atr_mult
                })
                in_position = False
                continue

            # Take profit at 3x risk
            profit_target = entry_price + stop_distance * 3
            if current_price >= profit_target:
                pnl_pct = (current_price - entry_price) / entry_price
                trades.append({
                    'entry_time': df.index[entry_idx],
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'bars_held': bars_held,
                    'exit_reason': 'take_profit',
                    'atr_multiplier': base_atr_mult
                })
                in_position = False

    return trades

def calculate_performance_metrics(trades: list) -> dict:
    """Calculate performance metrics from trades."""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_pnl_pct': 0.0,
            'total_pnl_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_bars_held': 0.0
        }

    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]

    cumulative = np.cumprod([1 + p for p in pnls])
    total_return = cumulative[-1] - 1

    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
    else:
        sharpe = 0.0

    return {
        'total_trades': len(trades),
        'win_rate': len(wins) / len(trades) if trades else 0.0,
        'avg_pnl_pct': np.mean(pnls),
        'total_pnl_pct': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'avg_bars_held': np.mean([t['bars_held'] for t in trades]),
        'avg_win_pct': np.mean(wins) if wins else 0.0,
        'avg_loss_pct': np.mean([p for p in pnls if p < 0]) if [p for p in pnls if p < 0] else 0.0
    }

def main():
    """Run v1.4.1 baseline backtest."""
    print("📊 Bull Machine v1.4.1 Baseline Backtest")
    print("=" * 50)

    # Load and merge config - but use v1.4.1 thresholds
    with open('configs/v141/profile_balanced.json', 'r') as f:
        base_config = json.load(f)

    if "extends" in base_config:
        system_path = base_config["extends"]
        with open(system_path) as f:
            system_config = json.load(f)

        config = {**system_config, **base_config}
        for key in ['signals', 'quality_floors', 'risk_management']:
            if key in base_config:
                config.setdefault(key, {})
                config[key].update(base_config[key])
    else:
        config = base_config

    # Load BTC 1H data
    df = pd.read_csv('data/btc_multiframe/BTC_USD_1H.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    print(f"Loaded {len(df)} hours of BTC data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Initialize fusion engine
    fusion_engine = FusionEngineV141(config)

    # Generate layer scores (identical to v1.4.2 for fair comparison)
    enabled_layers = [k for k, v in config.get('features', {}).items()
                     if v and k in fusion_engine.weights]

    layer_scores_df = generate_realistic_layer_scores(df, enabled_layers)

    # Fuse scores
    fusion_results = []
    for idx, row in layer_scores_df.iterrows():
        layer_dict = {k: v for k, v in row.to_dict().items() if not np.isnan(v)}
        fusion_result = fusion_engine.fuse_scores(
            layer_dict,
            quality_floors=config.get('quality_floors', {}),
            df=df.loc[[idx]]
        )
        fusion_results.append(fusion_result)

    fusion_df = pd.DataFrame(fusion_results, index=layer_scores_df.index)

    print(f"\\nFusion Stats:")
    print(f"  Signals above threshold: {(fusion_df['weighted_score'] >= config['signals']['enter_threshold']).sum()}")

    # Simulate v1.4.1 baseline trades
    trades = simulate_trades_v141_baseline(df, fusion_df, config['signals']['enter_threshold'], config)

    # Calculate performance
    performance = calculate_performance_metrics(trades)

    print(f"\\n📊 v1.4.1 Baseline Results:")
    print(f"  Total Trades: {performance['total_trades']}")
    print(f"  Win Rate: {performance['win_rate']:.1%}")
    print(f"  Total PnL: {performance['total_pnl_pct']:.2%}")
    print(f"  Avg Trade: {performance['avg_pnl_pct']:.3%}")
    print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {performance['max_drawdown']:.2%}")
    print(f"  Avg Hold Time: {performance['avg_bars_held']:.1f} hours")

    if trades:
        print(f"  Avg Win: {performance['avg_win_pct']:.2%}")
        print(f"  Avg Loss: {performance['avg_loss_pct']:.2%}")

    # Save results
    results_dir = Path("reports/v141_baseline_backtest")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "baseline_results.json", 'w') as f:
        json.dump({
            'performance': performance,
            'sample_trades': trades[:10],
            'config': config
        }, f, indent=2, default=str)

    print(f"\\n📁 Results saved to: {results_dir}")

if __name__ == "__main__":
    main()