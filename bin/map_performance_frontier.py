#!/usr/bin/env python3
"""
Performance Frontier Mapper v1.0
Goal: Derive empirical PF/DD guardrails per asset by sweeping threshold space.

Instead of guessing at PF≥2.5 or PF≥1.6, we measure the achievable frontier
on historical data and set guardrails based on p50-p75 statistics.

Usage:
    python3 bin/map_performance_frontier.py --asset ETH --year 2024
    python3 bin/map_performance_frontier.py --asset BTC --year 2024
    python3 bin/map_performance_frontier.py --asset SPY --year 2024

Output:
    reports/frontier_{asset}_{year}.csv         - Raw sweep results
    reports/frontier_{asset}_{year}_plot.png    - PF vs DD scatter
    reports/guardrail_recommendations.md        - Derived targets
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import subprocess
import re
import tempfile

def parse_args():
    parser = argparse.ArgumentParser(description="Map PF/DD frontier for an asset")
    parser.add_argument('--asset', type=str, required=True, choices=['BTC', 'ETH', 'SPY'],
                        help='Asset to analyze')
    parser.add_argument('--year', type=int, default=2024,
                        help='Year to backtest (default: 2024)')
    parser.add_argument('--output', type=str, default='reports/frontier',
                        help='Output directory')
    parser.add_argument('--base-config', type=str, default=None,
                        help='Base config to use (default: profile_{asset.lower()}_seed.json)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Setup
    asset = args.asset
    year = args.year
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base config
    if args.base_config:
        config_path = args.base_config
    else:
        config_path = f'configs/profile_{asset.lower()}_seed.json'

    with open(config_path, 'r') as f:
        base_config = json.load(f)

    print("=" * 80)
    print(f"Performance Frontier Mapper - {asset} {year}")
    print("=" * 80)
    print(f"Base config: {config_path}")
    print()

    # Define sweep ranges
    if asset == 'ETH':
        fusion_floors = np.arange(0.24, 0.36, 0.02)  # ETH-specific range
        min_liquidity_vals = [0.02, 0.03, 0.04, 0.06, 0.08, 0.10]
    elif asset == 'BTC':
        fusion_floors = np.arange(0.30, 0.42, 0.02)  # BTC-specific range
        min_liquidity_vals = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    else:  # SPY
        fusion_floors = np.arange(0.24, 0.36, 0.02)  # Similar to ETH
        min_liquidity_vals = [0.02, 0.03, 0.04, 0.06, 0.08, 0.10]

    total_runs = len(fusion_floors) * len(min_liquidity_vals)
    print(f"Sweep parameters:")
    print(f"  fusion_floor: {fusion_floors.tolist()}")
    print(f"  min_liquidity: {min_liquidity_vals}")
    print(f"  Total runs: {total_runs}")
    print()

    # Run sweep
    results = []
    run_num = 0

    for fusion_floor in fusion_floors:
        for min_liq in min_liquidity_vals:
            run_num += 1
            print(f"[{run_num}/{total_runs}] Testing fusion={fusion_floor:.2f}, liq={min_liq:.2f}...", end=' ')

            # Create modified config (deep copy to avoid mutation)
            test_config = json.loads(json.dumps(base_config))

            # Update global fusion (for non-archetype logic)
            test_config['fusion']['entry_threshold_confidence'] = float(fusion_floor)

            # Update ALL archetype-specific fusion thresholds uniformly
            # This is what actually matters for entry decisions!
            archetypes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M']
            for arch in archetypes:
                if arch in test_config['archetypes']['thresholds']:
                    if 'fusion' in test_config['archetypes']['thresholds'][arch]:
                        # Apply relative scaling: if seed was 0.24, and we sweep to 0.28,
                        # that's a +16.7% increase, apply same to all archetypes
                        seed_global = base_config['fusion']['entry_threshold_confidence']
                        scale_factor = fusion_floor / seed_global

                        old_val = test_config['archetypes']['thresholds'][arch]['fusion']
                        new_val = old_val * scale_factor
                        test_config['archetypes']['thresholds'][arch]['fusion'] = float(min(new_val, 0.95))

            # Update min liquidity threshold
            test_config['archetypes']['thresholds']['min_liquidity'] = float(min_liq)

            # Write temp config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(test_config, f, indent=2)
                temp_config_path = f.name

            # Run backtest via subprocess
            try:
                cmd = [
                    "python3",
                    "bin/backtest_knowledge_v2.py",
                    "--asset", asset,
                    "--start", f"{year}-01-01",
                    "--end", f"{year}-12-31",
                    "--config", temp_config_path
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,  # 2 min timeout per run
                    check=False
                )

                output = result.stdout + result.stderr

                # Parse metrics from output (same regex as Optuna)
                trades = 0
                pf = 0.0
                dd = 0.0
                wr = 0.0
                pnl = 0.0

                pnl_match = re.search(r'Total PNL:\s+\$?([-\d,\.]+)', output)
                if pnl_match:
                    pnl = float(pnl_match.group(1).replace(',', ''))

                trades_match = re.search(r'Total Trades:\s+(\d+)', output)
                if trades_match:
                    trades = int(trades_match.group(1))

                wr_match = re.search(r'Win Rate:\s+([\d\.]+)%', output)
                if wr_match:
                    wr = float(wr_match.group(1))

                dd_match = re.search(r'Max Drawdown:\s+([\d\.]+)%', output)
                if dd_match:
                    dd = float(dd_match.group(1))

                pf_match = re.search(r'Profit Factor:\s+([\d\.]+)', output)
                if pf_match:
                    pf = float(pf_match.group(1))

                # Clean up temp config
                Path(temp_config_path).unlink()

                results.append({
                    'asset': asset,
                    'fusion_floor': fusion_floor,
                    'min_liquidity': min_liq,
                    'trades': trades,
                    'profit_factor': pf,
                    'max_dd_pct': dd,
                    'win_rate_pct': wr,
                    'total_pnl': pnl,
                    'avg_r_multiple': 0.0,  # Not parsed from output
                    'sharpe_ratio': 0.0,    # Not parsed from output
                    'pf_dd_ratio': pf / dd if dd > 0 else 0.0
                })

                print(f"✓ {trades} trades, PF={pf:.2f}, DD={dd:.1f}%, WR={wr:.1f}%")

            except Exception as e:
                print(f"✗ ERROR: {e}")
                # Clean up temp config if it exists
                if Path(temp_config_path).exists():
                    Path(temp_config_path).unlink()

                results.append({
                    'asset': asset,
                    'fusion_floor': fusion_floor,
                    'min_liquidity': min_liq,
                    'trades': 0,
                    'profit_factor': 0.0,
                    'max_dd_pct': 0.0,
                    'win_rate_pct': 0.0,
                    'total_pnl': 0.0,
                    'avg_r_multiple': 0.0,
                    'sharpe_ratio': 0.0,
                    'pf_dd_ratio': 0.0
                })

    # Save results to CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / f"{asset}_{year}_frontier.csv"
    df.to_csv(csv_path, index=False)
    print()
    print(f"✅ Results saved to {csv_path}")
    print()

    # Compute statistics (only on valid runs with trades ≥ 20)
    valid_df = df[(df['trades'] >= 20) & (df['profit_factor'] > 0)]

    if len(valid_df) == 0:
        print("⚠️  WARNING: No valid runs with ≥20 trades found!")
        print("    Thresholds may be too tight. Consider widening sweep ranges.")
        return

    print("=" * 80)
    print(f"FRONTIER STATISTICS ({len(valid_df)} valid runs with ≥20 trades)")
    print("=" * 80)

    # PF statistics
    pf_mean = valid_df['profit_factor'].mean()
    pf_p25 = valid_df['profit_factor'].quantile(0.25)
    pf_p50 = valid_df['profit_factor'].quantile(0.50)
    pf_p75 = valid_df['profit_factor'].quantile(0.75)
    pf_p90 = valid_df['profit_factor'].quantile(0.90)
    pf_max = valid_df['profit_factor'].max()

    print(f"\nProfit Factor Distribution:")
    print(f"  Mean:  {pf_mean:.2f}")
    print(f"  p25:   {pf_p25:.2f}")
    print(f"  p50:   {pf_p50:.2f}")
    print(f"  p75:   {pf_p75:.2f}")
    print(f"  p90:   {pf_p90:.2f}")
    print(f"  Max:   {pf_max:.2f}")

    # DD statistics
    dd_mean = valid_df['max_dd_pct'].mean()
    dd_p25 = valid_df['max_dd_pct'].quantile(0.25)
    dd_p50 = valid_df['max_dd_pct'].quantile(0.50)
    dd_p75 = valid_df['max_dd_pct'].quantile(0.75)
    dd_p90 = valid_df['max_dd_pct'].quantile(0.90)

    print(f"\nMax Drawdown Distribution:")
    print(f"  Mean:  {dd_mean:.1f}%")
    print(f"  p25:   {dd_p25:.1f}%")
    print(f"  p50:   {dd_p50:.1f}%")
    print(f"  p75:   {dd_p75:.1f}%")
    print(f"  p90:   {dd_p90:.1f}%")

    # Trade count statistics
    trades_mean = valid_df['trades'].mean()
    trades_p50 = valid_df['trades'].quantile(0.50)

    print(f"\nTrade Frequency:")
    print(f"  Mean:  {trades_mean:.0f} trades/year")
    print(f"  p50:   {trades_p50:.0f} trades/year")

    # WR statistics
    wr_mean = valid_df['win_rate_pct'].mean()
    wr_p50 = valid_df['win_rate_pct'].quantile(0.50)

    print(f"\nWin Rate:")
    print(f"  Mean:  {wr_mean:.1f}%")
    print(f"  p50:   {wr_p50:.1f}%")

    # Derived guardrails
    print()
    print("=" * 80)
    print("RECOMMENDED GUARDRAILS (Data-Driven)")
    print("=" * 80)

    # Conservative: p50 PF, p75 DD
    pf_conservative = round(pf_p50, 1)
    dd_conservative = round(dd_p75, 0)

    # Moderate: p60 PF, p65 DD
    pf_moderate = round(valid_df['profit_factor'].quantile(0.60), 1)
    dd_moderate = round(valid_df['max_dd_pct'].quantile(0.65), 0)

    # Aggressive: p75 PF, p50 DD
    pf_aggressive = round(pf_p75, 1)
    dd_aggressive = round(dd_p50, 0)

    print(f"\n{asset} Guardrail Recommendations:")
    print(f"\n  Conservative (50% of runs pass):")
    print(f"    PF ≥ {pf_conservative}")
    print(f"    DD ≤ {dd_conservative}%")
    print(f"    WR ≥ {round(wr_p50 - 2, 0)}%")
    print(f"    Trades: {round(trades_p50 * 0.7, 0)}-{round(trades_p50 * 1.5, 0)}")

    print(f"\n  Moderate (40% of runs pass):")
    print(f"    PF ≥ {pf_moderate}")
    print(f"    DD ≤ {dd_moderate}%")
    print(f"    WR ≥ {round(wr_mean - 1, 0)}%")
    print(f"    Trades: {round(trades_mean * 0.7, 0)}-{round(trades_mean * 1.3, 0)}")

    print(f"\n  Aggressive (25% of runs pass):")
    print(f"    PF ≥ {pf_aggressive}")
    print(f"    DD ≤ {dd_aggressive}%")
    print(f"    WR ≥ {round(wr_mean, 0)}%")
    print(f"    Trades: {round(trades_mean * 0.8, 0)}-{round(trades_mean * 1.2, 0)}")

    # Find best configurations
    print()
    print("=" * 80)
    print("TOP 5 CONFIGURATIONS (by PF/DD ratio)")
    print("=" * 80)
    top5 = valid_df.nlargest(5, 'pf_dd_ratio')
    print(top5[['fusion_floor', 'min_liquidity', 'trades', 'profit_factor', 'max_dd_pct', 'win_rate_pct', 'pf_dd_ratio']].to_string(index=False))

    # Save recommendations
    rec_path = output_dir / f"{asset}_{year}_recommendations.txt"
    with open(rec_path, 'w') as f:
        f.write(f"# {asset} {year} Performance Frontier Analysis\n\n")
        f.write(f"## Statistics (from {len(valid_df)} valid runs)\n\n")
        f.write(f"Profit Factor: mean={pf_mean:.2f}, p50={pf_p50:.2f}, p75={pf_p75:.2f}\n")
        f.write(f"Max Drawdown: mean={dd_mean:.1f}%, p50={dd_p50:.1f}%, p75={dd_p75:.1f}%\n")
        f.write(f"Trade Count: mean={trades_mean:.0f}, p50={trades_p50:.0f}\n")
        f.write(f"Win Rate: mean={wr_mean:.1f}%, p50={wr_p50:.1f}%\n\n")
        f.write(f"## Recommended Guardrails\n\n")
        f.write(f"Conservative: PF≥{pf_conservative}, DD≤{dd_conservative}%, WR≥{round(wr_p50-2,0)}%\n")
        f.write(f"Moderate: PF≥{pf_moderate}, DD≤{dd_moderate}%, WR≥{round(wr_mean-1,0)}%\n")
        f.write(f"Aggressive: PF≥{pf_aggressive}, DD≤{dd_aggressive}%, WR≥{round(wr_mean,0)}%\n")

    print()
    print(f"✅ Recommendations saved to {rec_path}")
    print()

    # Generate plot (if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8))

        # Scatter plot: PF vs DD, colored by trade count
        scatter = ax.scatter(
            valid_df['max_dd_pct'],
            valid_df['profit_factor'],
            c=valid_df['trades'],
            s=100,
            alpha=0.6,
            cmap='viridis',
            edgecolors='black',
            linewidth=0.5
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Trade Count', rotation=270, labelpad=20)

        # Add guardrail lines
        ax.axhline(y=pf_moderate, color='orange', linestyle='--', alpha=0.5, label=f'PF target (moderate): {pf_moderate}')
        ax.axvline(x=dd_moderate, color='red', linestyle='--', alpha=0.5, label=f'DD limit (moderate): {dd_moderate}%')

        # Labels and formatting
        ax.set_xlabel('Max Drawdown (%)', fontsize=12)
        ax.set_ylabel('Profit Factor', fontsize=12)
        ax.set_title(f'{asset} {year} Performance Frontier\n(Each point = fusion/liquidity threshold combo)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add annotation for efficient frontier
        best_row = valid_df.loc[valid_df['pf_dd_ratio'].idxmax()]
        ax.annotate(
            f"Best PF/DD ratio\nfusion={best_row['fusion_floor']:.2f}\nliq={best_row['min_liquidity']:.2f}",
            xy=(best_row['max_dd_pct'], best_row['profit_factor']),
            xytext=(best_row['max_dd_pct'] + 1, best_row['profit_factor'] + 0.2),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7)
        )

        plot_path = output_dir / f"{asset}_{year}_frontier_plot.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ Frontier plot saved to {plot_path}")
        print()

    except ImportError:
        print("⚠️  matplotlib not available - skipping plot generation")

    print("=" * 80)
    print("✅ Frontier mapping complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
