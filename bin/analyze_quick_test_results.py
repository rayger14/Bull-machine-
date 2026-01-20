#!/usr/bin/env python3
"""
Analyze Quick Test Results (Phase 1)

Parses Phase 1 backtest logs, extracts metrics, and creates visualizations
to recommend the best starting point for Phase 2 optimization.

Usage:
    python3 bin/analyze_quick_test_results.py [--input-dir DIR]

Outputs:
    - Summary table (console + CSV)
    - Trade count vs PF scatter plot
    - Recommendations for Phase 2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import re
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Phase 1 Quick Test Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input-dir",
        default="results/phase1_quick_validation",
        help="Directory containing backtest logs"
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase1_quick_validation",
        help="Output directory for analysis"
    )
    parser.add_argument(
        "--target-trades-min",
        type=int,
        default=25,
        help="Minimum trades for target range"
    )
    parser.add_argument(
        "--target-trades-max",
        type=int,
        default=40,
        help="Maximum trades for target range"
    )
    return parser.parse_args()


def extract_metrics_from_log(log_path: Path) -> Dict:
    """
    Extract performance metrics from backtest log file.

    Returns:
        dict with metrics: trades, pf, wr, dd, sharpe, return, etc.
    """
    metrics = {
        "config": log_path.stem.replace("_2022_bear_market", ""),
        "trades": 0,
        "pf": 0.0,
        "wr": 0.0,
        "dd": 0.0,
        "sharpe": 0.0,
        "return": 0.0,
        "wins": 0,
        "losses": 0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "max_win": 0.0,
        "max_loss": 0.0,
        "status": "unknown"
    }

    if not log_path.exists():
        metrics["status"] = "missing"
        return metrics

    try:
        with open(log_path, 'r') as f:
            content = f.read()

        # Extract metrics using regex patterns
        patterns = {
            "trades": r"Total Trades:\s*(\d+)",
            "pf": r"Profit Factor:\s*([\d.]+)",
            "wr": r"Win Rate:\s*([\d.]+)%?",
            "dd": r"Max Drawdown:\s*([\d.]+)%?",
            "sharpe": r"Sharpe Ratio:\s*([-\d.]+)",
            "return": r"Total Return:\s*([-\d.]+)%?",
            "wins": r"Winning Trades:\s*(\d+)",
            "losses": r"Losing Trades:\s*(\d+)",
            "avg_win": r"Average Win:\s*\$?([-\d.]+)",
            "avg_loss": r"Average Loss:\s*\$?([-\d.]+)",
            "max_win": r"Max Win:\s*\$?([-\d.]+)",
            "max_loss": r"Max Loss:\s*\$?([-\d.]+)"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1)
                if key in ["trades", "wins", "losses"]:
                    metrics[key] = int(value)
                else:
                    metrics[key] = float(value)

        # Determine status
        if metrics["trades"] > 0:
            metrics["status"] = "success"
        elif "error" in content.lower() or "failed" in content.lower():
            metrics["status"] = "error"
        elif "timeout" in content.lower():
            metrics["status"] = "timeout"
        else:
            metrics["status"] = "no_trades"

    except Exception as e:
        print(f"Warning: Failed to parse {log_path}: {e}")
        metrics["status"] = "parse_error"

    return metrics


def analyze_results(results: List[Dict], target_min: int, target_max: int) -> pd.DataFrame:
    """
    Analyze results and create summary DataFrame.

    Args:
        results: List of metric dictionaries
        target_min: Minimum trades for target range
        target_max: Maximum trades for target range

    Returns:
        DataFrame with analysis
    """
    df = pd.DataFrame(results)

    # Calculate additional metrics
    df["in_target_range"] = (
        (df["trades"] >= target_min) &
        (df["trades"] <= target_max) &
        (df["status"] == "success")
    )

    # Calculate risk-adjusted return (simplified)
    df["risk_adj_return"] = df["return"] / (df["dd"] + 1)  # Avoid division by zero

    # Calculate expectancy
    df["expectancy"] = (
        (df["avg_win"] * df["wr"] / 100) -
        (abs(df["avg_loss"]) * (100 - df["wr"]) / 100)
    )

    # Rank configs
    df["pf_rank"] = df["pf"].rank(ascending=False)
    df["sharpe_rank"] = df["sharpe"].rank(ascending=False)
    df["combined_rank"] = (df["pf_rank"] + df["sharpe_rank"]) / 2

    # Sort by combined rank
    df = df.sort_values("combined_rank")

    return df


def plot_trade_count_vs_pf(df: pd.DataFrame, output_path: Path, target_min: int, target_max: int):
    """
    Create scatter plot of trade count vs profit factor.

    Highlights target range and best performers.
    """
    plt.figure(figsize=(12, 8))

    # Plot all points
    plt.scatter(
        df["trades"],
        df["pf"],
        s=100,
        alpha=0.6,
        c=df["in_target_range"].map({True: "green", False: "gray"}),
        edgecolors="black"
    )

    # Highlight target range
    plt.axvspan(target_min, target_max, alpha=0.2, color="green", label="Target Range")

    # Add labels for each point
    for _, row in df.iterrows():
        plt.annotate(
            row["config"],
            (row["trades"], row["pf"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7
        )

    # Add reference lines
    plt.axhline(y=1.5, color='r', linestyle='--', alpha=0.3, label='PF=1.5 (Min Target)')
    plt.axhline(y=3.0, color='g', linestyle='--', alpha=0.3, label='PF=3.0 (Good)')

    plt.xlabel("Trade Count", fontsize=12)
    plt.ylabel("Profit Factor", fontsize=12)
    plt.title("Phase 1 Quick Test: Trade Count vs Profit Factor", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_performance_heatmap(df: pd.DataFrame, output_path: Path):
    """
    Create heatmap of key performance metrics.
    """
    # Select metrics for heatmap
    metrics_cols = ["trades", "pf", "wr", "sharpe", "risk_adj_return"]
    heatmap_data = df[["config"] + metrics_cols].set_index("config")

    # Normalize for comparison
    heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_norm.T,
        annot=heatmap_data.T,
        fmt=".2f",
        cmap="RdYlGn",
        cbar_kws={"label": "Normalized Score"},
        linewidths=0.5
    )

    plt.title("Phase 1 Quick Test: Performance Heatmap", fontsize=14, fontweight='bold')
    plt.xlabel("Config", fontsize=12)
    plt.ylabel("Metric", fontsize=12)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"Heatmap saved to: {output_path}")
    plt.close()


def generate_recommendations(df: pd.DataFrame, target_min: int, target_max: int) -> str:
    """
    Generate recommendations for Phase 2 optimization.

    Returns:
        Markdown-formatted recommendations
    """
    recommendations = []

    recommendations.append("# Phase 1 Quick Test Analysis")
    recommendations.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    recommendations.append(f"**Target Trade Range**: {target_min}-{target_max} trades\n")

    recommendations.append("---\n")
    recommendations.append("## Summary Statistics\n")

    # Overall stats
    total_configs = len(df)
    successful = len(df[df["status"] == "success"])
    in_target = len(df[df["in_target_range"]])

    recommendations.append(f"- Total configs tested: {total_configs}")
    recommendations.append(f"- Successful runs: {successful}")
    recommendations.append(f"- In target range: {in_target}\n")

    # Best performers
    recommendations.append("## Top Performers\n")

    # Best in target range
    target_df = df[df["in_target_range"]].head(3)
    if len(target_df) > 0:
        recommendations.append("### In Target Range (25-40 trades)\n")
        recommendations.append("| Rank | Config | Trades | PF | WR% | Sharpe | Return% |")
        recommendations.append("|------|--------|--------|----|----|--------|---------|")

        for i, (_, row) in enumerate(target_df.iterrows(), 1):
            recommendations.append(
                f"| {i} | {row['config']} | {row['trades']:.0f} | "
                f"{row['pf']:.2f} | {row['wr']:.1f} | "
                f"{row['sharpe']:.2f} | {row['return']:.1f} |"
            )

        recommendations.append("")
    else:
        recommendations.append("⚠️ **No configs in target range**\n")

    # Best overall
    recommendations.append("### Best Overall (by PF)\n")
    recommendations.append("| Rank | Config | Trades | PF | WR% | Sharpe | Return% |")
    recommendations.append("|------|--------|--------|----|----|--------|---------|")

    for i, (_, row) in enumerate(df.head(3).iterrows(), 1):
        recommendations.append(
            f"| {i} | {row['config']} | {row['trades']:.0f} | "
            f"{row['pf']:.2f} | {row['wr']:.1f} | "
            f"{row['sharpe']:.2f} | {row['return']:.1f} |"
        )

    recommendations.append("\n---\n")
    recommendations.append("## Recommendations for Phase 2\n")

    if len(target_df) > 0:
        best = target_df.iloc[0]
        recommendations.append(f"**Primary Recommendation**: `{best['config']}`\n")
        recommendations.append(f"- Trades: {best['trades']:.0f} (in target range)")
        recommendations.append(f"- Profit Factor: {best['pf']:.2f}")
        recommendations.append(f"- Sharpe Ratio: {best['sharpe']:.2f}")
        recommendations.append(f"- Win Rate: {best['wr']:.1f}%")
        recommendations.append(f"\n**Action**: Use this config as the base for Phase 2 optimization\n")
    else:
        # Find closest to target range
        df["distance_to_target"] = df["trades"].apply(
            lambda x: min(abs(x - target_min), abs(x - target_max))
        )
        closest = df.nsmallest(1, "distance_to_target").iloc[0]

        recommendations.append(f"**Alternative Recommendation**: `{closest['config']}`\n")
        recommendations.append(f"- Trades: {closest['trades']:.0f} (closest to target range)")
        recommendations.append(f"- Profit Factor: {closest['pf']:.2f}")
        recommendations.append(f"\n**Action**: Adjust thresholds to bring trade count into target range\n")

        # Suggest adjustments
        if closest["trades"] < target_min:
            recommendations.append(f"\n**Suggested Adjustment**: Lower entry thresholds to increase trade count")
        else:
            recommendations.append(f"\n**Suggested Adjustment**: Raise entry thresholds to decrease trade count")

    recommendations.append("\n---\n")
    recommendations.append("## Next Steps\n")
    recommendations.append("1. Review top performers in detail")
    recommendations.append("2. Select base config for Phase 2 Optuna optimization")
    recommendations.append("3. Define parameter search space")
    recommendations.append("4. Run multi-objective optimization (PF, Sharpe, Drawdown)")
    recommendations.append("5. Validate on out-of-sample data (2024)")

    return "\n".join(recommendations)


def main():
    args = parse_args()

    print("=" * 80)
    print("Phase 1 Quick Test Results Analysis")
    print("=" * 80)
    print()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all log files
    log_files = list(input_dir.glob("*_2022_bear_market.log"))
    print(f"Found {len(log_files)} log files in {input_dir}")

    if not log_files:
        print("ERROR: No log files found. Run Phase 1 validation first.")
        sys.exit(1)

    # Extract metrics from each log
    print("\nExtracting metrics...")
    results = []
    for log_file in log_files:
        print(f"  Processing: {log_file.name}")
        metrics = extract_metrics_from_log(log_file)
        results.append(metrics)

    # Analyze results
    print("\nAnalyzing results...")
    df = analyze_results(results, args.target_trades_min, args.target_trades_max)

    # Display summary table
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()

    summary_cols = ["config", "trades", "pf", "wr", "sharpe", "return", "dd", "in_target_range", "status"]
    print(df[summary_cols].to_string(index=False))
    print()

    # Save summary CSV
    csv_path = output_dir / "analysis_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Summary saved to: {csv_path}")

    # Create visualizations
    print("\nGenerating visualizations...")

    plot_path = output_dir / "trade_count_vs_pf.png"
    plot_trade_count_vs_pf(df, plot_path, args.target_trades_min, args.target_trades_max)

    heatmap_path = output_dir / "performance_heatmap.png"
    plot_performance_heatmap(df, heatmap_path)

    # Generate recommendations
    print("\nGenerating recommendations...")
    recommendations_md = generate_recommendations(df, args.target_trades_min, args.target_trades_max)

    # Save recommendations
    rec_path = output_dir / "recommendations.md"
    with open(rec_path, 'w') as f:
        f.write(recommendations_md)
    print(f"Recommendations saved to: {rec_path}")

    # Print recommendations to console
    print("\n" + "=" * 80)
    print(recommendations_md)
    print("=" * 80)

    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
