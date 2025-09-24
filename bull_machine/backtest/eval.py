"""
Backtest Evaluation & Ablation Framework
Bull Machine v1.4.1 - Enhanced analysis with layer contribution studies
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from bull_machine.scoring.fusion import FusionEngineV141


def calculate_sharpe(trades: List[Dict]) -> float:
    """Calculate Sharpe ratio from trade list."""
    if not trades:
        return 0.0

    returns = [t["pnl_pct"] for t in trades if "pnl_pct" in t]
    if not returns:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    # Annualized Sharpe (assuming ~250 trades per year)
    return (mean_return * np.sqrt(250)) / std_return


def calculate_mar(trades: List[Dict]) -> float:
    """Calculate MAR (Mean Annual Return / Max Drawdown)."""
    if not trades:
        return 0.0

    returns = [t["pnl_pct"] for t in trades if "pnl_pct" in t]
    if not returns:
        return 0.0

    # Calculate cumulative returns
    cumulative = np.cumprod([1 + r for r in returns])
    total_return = cumulative[-1] - 1

    # Calculate max drawdown
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_dd = abs(min(drawdown))

    if max_dd == 0:
        return float("inf") if total_return > 0 else 0.0

    # Annualize return
    annual_return = total_return * (250 / len(trades))

    return annual_return / max_dd


def calculate_max_dd(trades: List[Dict]) -> float:
    """Calculate maximum drawdown percentage."""
    if not trades:
        return 0.0

    returns = [t["pnl_pct"] for t in trades if "pnl_pct" in t]
    if not returns:
        return 0.0

    cumulative = np.cumprod([1 + r for r in returns])
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak

    return abs(min(drawdown))


def calculate_win_rate(trades: List[Dict]) -> float:
    """Calculate win rate percentage."""
    if not trades:
        return 0.0

    wins = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
    return wins / len(trades)


def compute_mock_layer_scores(df: pd.DataFrame, enabled_modules: List[str], config: Dict) -> pd.DataFrame:
    """
    Compute mock layer scores for ablation study.
    Replace with actual layer computation in full implementation.
    """

    scores_df = pd.DataFrame(index=df.index)

    # Mock scoring - replace with actual layer computations
    for module in enabled_modules:
        if module == "wyckoff":
            # Mock Wyckoff scoring
            scores_df["wyckoff"] = 0.70 + 0.1 * np.sin(np.arange(len(df)) * 0.1)
        elif module == "liquidity":
            # Mock liquidity scoring
            scores_df["liquidity"] = 0.65 + 0.15 * np.random.normal(0, 0.1, len(df))
        elif module == "structure":
            # Mock structure scoring
            scores_df["structure"] = 0.60 + 0.1 * np.cos(np.arange(len(df)) * 0.05)
        elif module == "momentum":
            # Mock momentum scoring
            scores_df["momentum"] = 0.55 + 0.2 * np.random.normal(0, 0.1, len(df))
        elif module == "volume":
            # Mock volume scoring
            scores_df["volume"] = 0.50 + 0.15 * np.random.normal(0, 0.1, len(df))
        elif module == "context":
            # Mock context scoring
            scores_df["context"] = 0.45 + 0.1 * np.random.normal(0, 0.1, len(df))
        elif module == "mtf":
            # Mock MTF scoring
            scores_df["mtf"] = 0.60 + 0.2 * np.random.normal(0, 0.1, len(df))
        elif module == "bojan":
            # Mock Bojan scoring (will be capped)
            scores_df["bojan"] = 0.75 + 0.2 * np.random.normal(0, 0.1, len(df))

    # Clip all scores to [0, 1]
    scores_df = scores_df.clip(0, 1)

    return scores_df


def simulate_trades(df: pd.DataFrame, fusion_scores: pd.DataFrame, enter_threshold: float, config: Dict) -> List[Dict]:
    """
    Simulate trades based on fusion scores.
    Replace with actual trade simulation in full implementation.
    """

    trades = []
    in_position = False
    entry_price = 0
    entry_idx = 0

    for i, (idx, row) in enumerate(fusion_scores.iterrows()):
        if i < 20:  # Skip first 20 bars
            continue

        current_price = df.loc[idx, "close"]

        if not in_position:
            # Check for entry
            if row.get("weighted_score", 0) >= enter_threshold and not row.get("global_veto", True):
                in_position = True
                entry_price = current_price
                entry_idx = i

        else:
            # Check for exit (simplified)
            bars_held = i - entry_idx
            pnl_pct = (current_price - entry_price) / entry_price

            # Simple exit conditions
            if (
                bars_held >= 30  # Time stop
                or pnl_pct >= 0.03  # 3% profit
                or pnl_pct <= -0.015  # 1.5% loss
                or row.get("global_veto", False)
            ):  # Global veto
                trades.append(
                    {
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "pnl_pct": pnl_pct,
                        "bars_held": bars_held,
                        "entry_score": fusion_scores.iloc[entry_idx].get("weighted_score", 0),
                    }
                )

                in_position = False

    return trades


def run_ablation(df: pd.DataFrame, config: Dict) -> Dict:
    """
    Run ablation study to measure individual layer contributions.
    """

    fusion_engine = FusionEngineV141(config)

    # Define layer sets for ablation
    layer_sets = [
        ["wyckoff"],
        ["wyckoff", "liquidity"],
        ["wyckoff", "liquidity", "structure"],
        ["wyckoff", "liquidity", "structure", "momentum"],
        ["wyckoff", "liquidity", "structure", "momentum", "volume"],
        ["wyckoff", "liquidity", "structure", "momentum", "volume", "context"],
        ["wyckoff", "liquidity", "structure", "momentum", "volume", "context", "mtf"],
    ]

    # Add Bojan if enabled
    if config.get("features", {}).get("bojan", False):
        layer_sets.append(["wyckoff", "liquidity", "structure", "momentum", "volume", "context", "mtf", "bojan"])

    results = {}

    for layer_set in layer_sets:
        logging.info(f"Running ablation for layers: {'+'.join(layer_set)}")

        # Compute layer scores
        layer_scores_df = compute_mock_layer_scores(df, layer_set, config)

        # Fuse scores
        fusion_results = []
        for idx, row in layer_scores_df.iterrows():
            layer_dict = row.to_dict()
            fusion_result = fusion_engine.fuse_scores(layer_dict)
            fusion_results.append(fusion_result)

        fusion_df = pd.DataFrame(fusion_results, index=layer_scores_df.index)

        # Simulate trades
        trades = simulate_trades(df, fusion_df, config["signals"]["enter_threshold"], config)

        # Calculate metrics
        set_name = "+".join(layer_set)
        results[set_name] = {
            "sharpe": calculate_sharpe(trades),
            "mar": calculate_mar(trades),
            "max_dd": calculate_max_dd(trades),
            "win_rate": calculate_win_rate(trades),
            "total_trades": len(trades),
            "avg_score": fusion_df["weighted_score"].mean(),
            "layer_count": len(layer_set),
        }

        logging.info(
            f"  {set_name}: Sharpe={results[set_name]['sharpe']:.2f}, "
            f"MAR={results[set_name]['mar']:.2f}, Trades={len(trades)}"
        )

    return results


def run_backtest(df: pd.DataFrame, config: Dict, output_dir: str) -> Dict:
    """
    Run full backtest with specified config.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize fusion engine
    fusion_engine = FusionEngineV141(config)

    # Compute all layer scores
    enabled_layers = [k for k, v in config.get("features", {}).items() if v and k in fusion_engine.weights]
    layer_scores_df = compute_mock_layer_scores(df, enabled_layers, config)

    # Fuse scores
    fusion_results = []
    for idx, row in layer_scores_df.iterrows():
        layer_dict = row.to_dict()
        fusion_result = fusion_engine.fuse_scores(layer_dict)
        fusion_results.append(fusion_result)

    fusion_df = pd.DataFrame(fusion_results, index=layer_scores_df.index)

    # Simulate trades
    trades = simulate_trades(df, fusion_df, config["signals"]["enter_threshold"], config)

    # Calculate performance metrics
    performance = {
        "total_trades": len(trades),
        "win_rate": calculate_win_rate(trades),
        "sharpe_ratio": calculate_sharpe(trades),
        "mar_ratio": calculate_mar(trades),
        "max_drawdown": calculate_max_dd(trades),
        "avg_pnl_pct": np.mean([t["pnl_pct"] for t in trades]) if trades else 0,
        "avg_bars_held": np.mean([t["bars_held"] for t in trades]) if trades else 0,
    }

    # Generate summary
    summary = {
        "config": config,
        "performance": performance,
        "trades": trades[:100],  # Save first 100 trades
        "fusion_stats": {
            "avg_weighted_score": fusion_df["weighted_score"].mean(),
            "avg_aggregate": fusion_df["aggregate"].mean(),
            "veto_rate": fusion_df["global_veto"].mean(),
            "layers_enabled": enabled_layers,
        },
    }

    # Save results
    with open(output_path / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save telemetry files (mock)
    telemetry_files = [
        "exits_applied.json",
        "parameter_usage.json",
        "layer_masks.json",
        "exit_counts.json",
    ]
    for filename in telemetry_files:
        mock_data = {
            "mock": True,
            "trades": len(trades),
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        with open(output_path / filename, "w") as f:
            json.dump(mock_data, f, indent=2)

    logging.info(f"Backtest complete: {len(trades)} trades, Sharpe={performance['sharpe_ratio']:.2f}")

    return summary


def main():
    """CLI entry point for backtest evaluation."""

    parser = argparse.ArgumentParser(description="Bull Machine v1.4.1 Backtest Evaluation")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--maxbars", type=int, default=None, help="Limit bars for testing")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    # Load data
    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    if args.maxbars:
        df = df.tail(args.maxbars)

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Loaded {len(df)} bars from {args.data}")

    if args.ablation:
        # Run ablation study
        results = run_ablation(df, config)

        # Save ablation results
        output_file = Path(args.out)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Ablation results saved to {output_file}")

    else:
        # Run full backtest
        summary = run_backtest(df, config, args.out)
        print(f"Backtest complete. Results in {args.out}")


if __name__ == "__main__":
    main()
