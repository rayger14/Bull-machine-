#!/usr/bin/env python3
"""
v1.4.2 Demo Backtest - Show PnL improvement potential
Create a realistic backtest showing the v1.4.2 improvements
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from bull_machine.scoring.fusion import FusionEngineV141


def generate_realistic_layer_scores(df: pd.DataFrame, enabled_modules: list) -> pd.DataFrame:
    """Generate more realistic layer scores that will create some trades."""
    np.random.seed(42)  # For reproducible results
    scores_df = pd.DataFrame(index=df.index)

    # Calculate price volatility for more realistic scoring
    if "close" in df.columns:
        returns = df["close"].pct_change()
        volatility = returns.rolling(20).std()
        trend_strength = abs(returns.rolling(10).mean()) * 100  # Convert to %
    else:
        volatility = pd.Series(0.02, index=df.index)
        trend_strength = pd.Series(1.0, index=df.index)

    for module in enabled_modules:
        if module == "wyckoff":
            # Higher scores during trending periods
            base_score = 0.5 + (trend_strength * 0.3).clip(0, 0.4)
            noise = np.random.normal(0, 0.1, len(df))
            scores_df["wyckoff"] = (base_score + noise).clip(0.2, 0.9)

        elif module == "liquidity":
            # Correlate with volume changes
            volume_changes = df["volume"].pct_change().abs()
            base_score = 0.45 + (volume_changes * 2).clip(0, 0.3)
            noise = np.random.normal(0, 0.08, len(df))
            scores_df["liquidity"] = (base_score + noise).clip(0.1, 0.85)

        elif module == "structure":
            # Some periodic higher scores
            base_score = 0.4 + 0.3 * np.sin(np.arange(len(df)) * 0.1)
            noise = np.random.normal(0, 0.1, len(df))
            scores_df["structure"] = (base_score + noise).clip(0.2, 0.8)

        elif module == "momentum":
            # Based on recent price movement
            momentum = returns.rolling(5).mean().abs()
            base_score = 0.35 + (momentum * 20).clip(0, 0.35)
            noise = np.random.normal(0, 0.12, len(df))
            scores_df["momentum"] = (base_score + noise).clip(0.15, 0.85)

        elif module == "volume":
            # Volume-based scoring
            vol_sma = df["volume"].rolling(20).mean()
            vol_ratio = (df["volume"] / vol_sma).clip(0.5, 3.0)
            base_score = 0.3 + (vol_ratio - 1) * 0.2
            noise = np.random.normal(0, 0.1, len(df))
            scores_df["volume"] = (base_score + noise).clip(0.1, 0.8)

        elif module == "context":
            # Macro context - generally stable with occasional spikes
            base_score = 0.4
            spikes = np.random.random(len(df)) < 0.05  # 5% chance of spike
            context_scores = np.full(len(df), base_score)
            context_scores[spikes] *= 0.3  # Context stress
            noise = np.random.normal(0, 0.05, len(df))
            scores_df["context"] = (context_scores + noise).clip(0.1, 0.7)

        elif module == "mtf":
            # Multi-timeframe alignment
            base_score = 0.55 + 0.2 * np.cos(np.arange(len(df)) * 0.02)
            noise = np.random.normal(0, 0.1, len(df))
            scores_df["mtf"] = (base_score + noise).clip(0.3, 0.85)

        elif module == "bojan":
            # Bojan layer - capped at 0.6 in fusion
            base_score = 0.6 + 0.3 * np.random.normal(0, 0.2, len(df))
            scores_df["bojan"] = base_score.clip(0.2, 1.0)

    return scores_df


def simulate_trades_v142(df: pd.DataFrame, fusion_df: pd.DataFrame, enter_threshold: float, config: dict) -> list:
    """Simulate trades with v1.4.2 improvements."""
    trades = []
    in_position = False
    entry_idx = None
    entry_price = None

    stop_loss_config = config.get("stop_loss", {})
    base_atr_mult = stop_loss_config.get("base_atr_multipliers", {}).get("default", 1.5)
    markup_atr_mult = stop_loss_config.get("base_atr_multipliers", {}).get("markup_phases", 2.0)

    # Calculate ATR for stop calculation
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))),
    )
    df["atr"] = df["tr"].rolling(14).mean()

    for i in range(50, len(df) - 10):  # Skip first 50 bars for indicators
        if not in_position:
            # Entry logic
            if fusion_df.iloc[i]["weighted_score"] >= enter_threshold and not fusion_df.iloc[i]["global_veto"]:
                in_position = True
                entry_idx = i
                entry_price = df.iloc[i]["close"]

                # Determine phase for stop calculation
                recent_high = df.iloc[i - 20 : i]["high"].max()
                recent_low = df.iloc[i - 20 : i]["low"].min()
                range_pos = (entry_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5

                # v1.4.2 improvement: wider stops in trending phases
                if range_pos > 0.7 or range_pos < 0.3:  # Trending phases
                    atr_multiplier = markup_atr_mult  # 2.0x for v1.4.2
                else:
                    atr_multiplier = base_atr_mult  # 1.5x

                atr_value = df.iloc[i]["atr"] if not np.isnan(df.iloc[i]["atr"]) else entry_price * 0.02
                stop_distance = atr_value * atr_multiplier
                stop_price = entry_price - stop_distance  # Long position

        else:
            # Exit logic
            current_price = df.iloc[i]["close"]
            bars_held = i - entry_idx

            # Stop loss
            if current_price <= stop_price:
                pnl_pct = (current_price - entry_price) / entry_price
                trades.append(
                    {
                        "entry_time": df.index[entry_idx],
                        "exit_time": df.index[i],
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "pnl_pct": pnl_pct,
                        "bars_held": bars_held,
                        "exit_reason": "stop_loss",
                        "atr_multiplier": atr_multiplier,
                    }
                )
                in_position = False
                continue

            # Time-based exit
            max_bars = 36  # 1.5 days for hourly data
            if bars_held >= max_bars:
                pnl_pct = (current_price - entry_price) / entry_price
                trades.append(
                    {
                        "entry_time": df.index[entry_idx],
                        "exit_time": df.index[i],
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "pnl_pct": pnl_pct,
                        "bars_held": bars_held,
                        "exit_reason": "time_stop",
                        "atr_multiplier": atr_multiplier,
                    }
                )
                in_position = False
                continue

            # Take profit at 3x risk
            profit_target = entry_price + stop_distance * 3
            if current_price >= profit_target:
                pnl_pct = (current_price - entry_price) / entry_price
                trades.append(
                    {
                        "entry_time": df.index[entry_idx],
                        "exit_time": df.index[i],
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "pnl_pct": pnl_pct,
                        "bars_held": bars_held,
                        "exit_reason": "take_profit",
                        "atr_multiplier": atr_multiplier,
                    }
                )
                in_position = False

    return trades


def calculate_performance_metrics(trades: list) -> dict:
    """Calculate performance metrics from trades."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "total_pnl_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "avg_bars_held": 0.0,
        }

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]

    # Calculate cumulative performance
    cumulative = np.cumprod([1 + p for p in pnls])
    total_return = cumulative[-1] - 1

    # Calculate drawdown
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

    # Sharpe calculation
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)  # Annualized
    else:
        sharpe = 0.0

    return {
        "total_trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "avg_pnl_pct": np.mean(pnls),
        "total_pnl_pct": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "avg_bars_held": np.mean([t["bars_held"] for t in trades]),
        "avg_win_pct": np.mean(wins) if wins else 0.0,
        "avg_loss_pct": np.mean([p for p in pnls if p < 0]) if [p for p in pnls if p < 0] else 0.0,
    }


def main():
    """Run v1.4.2 demo backtest."""
    print("🚀 Bull Machine v1.4.2 Demo Backtest")
    print("=" * 50)

    # Load and merge config
    with open("configs/v141/profile_balanced.json", "r") as f:
        base_config = json.load(f)

    if "extends" in base_config:
        system_path = base_config["extends"]
        with open(system_path) as f:
            system_config = json.load(f)

        # Merge configs
        config = {**system_config, **base_config}
        for key in ["signals", "quality_floors", "risk_management"]:
            if key in base_config:
                config.setdefault(key, {})
                config[key].update(base_config[key])
    else:
        config = base_config

    # Load BTC 1H data
    df = pd.read_csv("data/btc_multiframe/BTC_USD_1H.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    print(f"Loaded {len(df)} hours of BTC data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['low'].min():,.0f} - ${df['high'].max():,.0f}")

    # Initialize fusion engine
    fusion_engine = FusionEngineV141(config)

    # Generate layer scores
    enabled_layers = [k for k, v in config.get("features", {}).items() if v and k in fusion_engine.weights]
    print(f"Enabled layers: {enabled_layers}")

    layer_scores_df = generate_realistic_layer_scores(df, enabled_layers)

    # Fuse scores
    fusion_results = []
    for idx, row in layer_scores_df.iterrows():
        layer_dict = {k: v for k, v in row.to_dict().items() if not np.isnan(v)}
        fusion_result = fusion_engine.fuse_scores(
            layer_dict,
            quality_floors=config.get("quality_floors", {}),
            df=df.loc[[idx]],  # Pass single row for regime filter
        )
        fusion_results.append(fusion_result)

    fusion_df = pd.DataFrame(fusion_results, index=layer_scores_df.index)

    print(f"\\nFusion Stats:")
    print(f"  Avg weighted score: {fusion_df['weighted_score'].mean():.3f}")
    print(f"  Avg aggregate: {fusion_df['aggregate'].mean():.3f}")
    print(f"  Veto rate: {fusion_df['global_veto'].mean():.1%}")
    print(f"  Signals above threshold: {(fusion_df['weighted_score'] >= config['signals']['enter_threshold']).sum()}")

    # Simulate trades
    trades = simulate_trades_v142(df, fusion_df, config["signals"]["enter_threshold"], config)

    # Calculate performance
    performance = calculate_performance_metrics(trades)

    print(f"\\n📊 v1.4.2 Backtest Results:")
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

        # Show breakdown by stop type (v1.4.2 feature)
        wide_stop_trades = [t for t in trades if t["atr_multiplier"] > 1.8]
        narrow_stop_trades = [t for t in trades if t["atr_multiplier"] <= 1.8]

        print(f"\\n🎯 v1.4.2 Stop Analysis:")
        print(f"  Wide stops (2.0x): {len(wide_stop_trades)} trades")
        if wide_stop_trades:
            wide_pnl = np.mean([t["pnl_pct"] for t in wide_stop_trades])
            print(f"    Avg PnL: {wide_pnl:.3%}")

        print(f"  Narrow stops (1.5x): {len(narrow_stop_trades)} trades")
        if narrow_stop_trades:
            narrow_pnl = np.mean([t["pnl_pct"] for t in narrow_stop_trades])
            print(f"    Avg PnL: {narrow_pnl:.3%}")

    # Save results
    results_dir = Path("reports/v142_demo_backtest")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "demo_results.json", "w") as f:
        json.dump(
            {
                "performance": performance,
                "sample_trades": trades[:10],  # Save first 10 trades
                "fusion_stats": {
                    "avg_weighted_score": float(fusion_df["weighted_score"].mean()),
                    "avg_aggregate": float(fusion_df["aggregate"].mean()),
                    "veto_rate": float(fusion_df["global_veto"].mean()),
                    "layers_enabled": enabled_layers,
                },
                "config": config,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\\n📁 Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
