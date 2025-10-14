#!/usr/bin/env python3
"""
Quick PNL validation script - Shows P&L with $10k starting balance

Uses optimization results to show actual dollar P&L
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from datetime import datetime


def load_best_config(asset: str):
    """Load best config from Q3 2024 baseline results"""
    baseline_path = f"reports/v19/{asset}_q3_baseline.json"

    try:
        with open(baseline_path) as f:
            results = json.load(f)

        # Sort by Sharpe ratio
        results.sort(key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
        best = results[0]

        return {
            'fusion': {
                'entry_threshold_confidence': best['fusion_threshold'],
                'weights': {
                    'wyckoff': best['wyckoff_weight'],
                    'smc': best['smc_weight'],
                    'liquidity': best['hob_weight'],
                    'momentum': best['momentum_weight']
                }
            },
            'exits': {
                'atr_k': 1.0,
                'trail_atr_k': 1.0,
                'tp1_r': 1.0
            },
            'risk': {
                'base_risk_pct': 0.0075
            }
        }

    except Exception as e:
        print(f"⚠️  Failed to load best config for {asset}: {e}")
        print("Using default config...")
        return None


def format_currency(value: float) -> str:
    """Format value as USD"""
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    """Format value as percentage"""
    return f"{value*100:+.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Validate PNL with $10k starting balance")
    parser.add_argument('--asset', type=str, required=True, choices=['BTC', 'ETH'])
    parser.add_argument('--start', type=str, default='2024-07-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-09-30', help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=10000, help='Starting balance in USD')
    parser.add_argument('--config', type=str, default=None, help='Path to custom config (optional)')

    args = parser.parse_args()

    print("=" * 70)
    print(f"PNL VALIDATION - {args.asset}")
    print("=" * 70)
    print(f"Starting Balance: {format_currency(args.balance)}")
    print(f"Period: {args.start} to {args.end}")
    print("=" * 70)

    # Load config
    if args.config:
        print(f"\nLoading config from: {args.config}")
        with open(args.config) as f:
            config = json.load(f)
    else:
        print(f"\nLoading best config from Q3 2024 baseline results...")
        config = load_best_config(args.asset)

        if config is None:
            print("❌ Failed to load config")
            return 1

    # Show config
    print("\n📋 Config:")
    print(f"   Fusion Threshold: {config['fusion']['entry_threshold_confidence']:.2f}")
    print(f"   Wyckoff Weight: {config['fusion']['weights']['wyckoff']:.2f}")
    print(f"   SMC Weight: {config['fusion']['weights']['smc']:.2f}")
    print(f"   Liquidity Weight: {config['fusion']['weights']['liquidity']:.2f}")
    print(f"   Momentum Weight: {config['fusion']['weights']['momentum']:.2f}")
    print(f"   Risk per Trade: {config['risk']['base_risk_pct']*100:.2f}%")

    # Load results from baseline
    print(f"\n📊 Loading results from baseline test...")

    try:
        baseline_path = f"reports/v19/{args.asset}_q3_baseline.json"
        with open(baseline_path) as f:
            all_results = json.load(f)

        # Find the best result (already sorted by Sharpe)
        all_results.sort(key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
        results = all_results[0]

        # Show results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        if results['trades'] == 0:
            print("❌ No trades generated (check fusion threshold or date range)")
            return 1

        final_balance = args.balance * (1 + results['total_return'])
        pnl = final_balance - args.balance

        print(f"\n💰 P&L:")
        print(f"   Starting Balance: {format_currency(args.balance)}")
        print(f"   Final Balance: {format_currency(final_balance)}")
        print(f"   Net P&L: {format_currency(pnl)} ({format_percent(results['total_return'])})")

        print(f"\n📊 Performance:")
        print(f"   Total Trades: {results['trades']}")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"   Avg R-Multiple: {results['avg_r']:.2f}")

        if 'winners_avg' in results:
            print(f"\n📈 Trade Stats:")
            print(f"   Avg Winner: {format_percent(results['winners_avg'])}")
            print(f"   Avg Loser: {format_percent(results['losers_avg'])}")
            print(f"   Avg Hold Time: {results.get('avg_bars_held', 0):.1f} hours")

        print("\n" + "=" * 70)

        # Save detailed results
        output_path = f"reports/v19/{args.asset}_pnl_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump({
                'config': config,
                'parameters': {
                    'asset': args.asset,
                    'start_date': args.start,
                    'end_date': args.end,
                    'starting_balance': args.balance
                },
                'results': results
            }, f, indent=2)

        print(f"\n💾 Detailed results saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
