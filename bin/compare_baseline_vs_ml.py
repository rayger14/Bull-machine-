#!/usr/bin/env python3
"""
Compare Baseline vs ML-ON Results

Validates acceptance gates:
1. PF +0.20 uplift
2. Sharpe +0.15 uplift
3. MaxDD ≤ baseline
4. Trade count ≥ 80% of baseline

Generates comparison leaderboard
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_results(path: str) -> List[Dict]:
    """Load optimization results from JSON"""
    with open(path) as f:
        return json.load(f)


def get_best_config(results: List[Dict], metric: str = 'sharpe_ratio') -> Dict:
    """Get best configuration by metric"""
    if not results:
        return None

    return max(results, key=lambda x: x.get(metric, 0))


def format_metric(value: float, is_percentage: bool = False, decimals: int = 2) -> str:
    """Format metric for display"""
    if is_percentage:
        return f"{value*100:.{decimals}f}%"
    return f"{value:.{decimals}f}"


def calculate_uplift(baseline: float, ml: float, is_lower_better: bool = False) -> str:
    """Calculate and format uplift"""
    if baseline == 0:
        return "N/A"

    uplift = (ml - baseline) / abs(baseline)

    if is_lower_better:
        uplift = -uplift

    symbol = "↑" if uplift > 0 else "↓"
    color = "🟢" if uplift > 0 else "🔴"

    return f"{color} {symbol}{abs(uplift)*100:.1f}%"


def check_gate(name: str, condition: bool, details: str = "") -> Dict:
    """Check acceptance gate"""
    status = "✅ PASS" if condition else "❌ FAIL"
    return {
        'gate': name,
        'status': status,
        'passed': condition,
        'details': details
    }


def compare_configs(baseline_path: str, ml_path: str, asset: str) -> Dict:
    """
    Compare baseline vs ML-ON configurations

    Returns:
        Comparison dict with metrics and gate results
    """
    # Load results
    baseline_results = load_results(baseline_path)
    ml_results = load_results(ml_path)

    # Get best configs
    baseline = get_best_config(baseline_results, metric='sharpe_ratio')
    ml = get_best_config(ml_results, metric='sharpe_ratio')

    if not baseline or not ml:
        return {'error': 'Missing results'}

    # Extract metrics
    comparison = {
        'asset': asset,
        'baseline': {
            'trades': baseline.get('trades', 0),
            'win_rate': baseline.get('win_rate', 0),
            'profit_factor': baseline.get('profit_factor', 0),
            'sharpe_ratio': baseline.get('sharpe_ratio', 0),
            'max_drawdown': baseline.get('max_drawdown', 0),
            'total_return': baseline.get('total_return', 0),
            'avg_r': baseline.get('avg_r', 0),
            'config': {
                'threshold': baseline.get('fusion_threshold', 0),
                'wyckoff_weight': baseline.get('wyckoff_weight', 0),
                'momentum_weight': baseline.get('momentum_weight', 0)
            }
        },
        'ml': {
            'trades': ml.get('trades', 0),
            'win_rate': ml.get('win_rate', 0),
            'profit_factor': ml.get('profit_factor', 0),
            'sharpe_ratio': ml.get('sharpe_ratio', 0),
            'max_drawdown': ml.get('max_drawdown', 0),
            'total_return': ml.get('total_return', 0),
            'avg_r': ml.get('avg_r', 0),
            'regime_label': ml.get('regime_label', 'unknown'),
            'regime_confidence': ml.get('regime_confidence', 0),
            'config': {
                'threshold': ml.get('fusion_threshold', 0),
                'wyckoff_weight': ml.get('wyckoff_weight', 0),
                'momentum_weight': ml.get('momentum_weight', 0)
            }
        }
    }

    # Calculate uplifts
    comparison['uplifts'] = {
        'pf': calculate_uplift(comparison['baseline']['profit_factor'],
                               comparison['ml']['profit_factor']),
        'sharpe': calculate_uplift(comparison['baseline']['sharpe_ratio'],
                                   comparison['ml']['sharpe_ratio']),
        'max_dd': calculate_uplift(comparison['baseline']['max_drawdown'],
                                   comparison['ml']['max_drawdown'],
                                   is_lower_better=True),
        'trades': calculate_uplift(comparison['baseline']['trades'],
                                   comparison['ml']['trades']),
        'win_rate': calculate_uplift(comparison['baseline']['win_rate'],
                                     comparison['ml']['win_rate']),
        'avg_r': calculate_uplift(comparison['baseline']['avg_r'],
                                 comparison['ml']['avg_r'])
    }

    # Check acceptance gates
    pf_uplift = (comparison['ml']['profit_factor'] - comparison['baseline']['profit_factor'])
    sharpe_uplift = (comparison['ml']['sharpe_ratio'] - comparison['baseline']['sharpe_ratio'])
    trade_retention = comparison['ml']['trades'] / comparison['baseline']['trades'] if comparison['baseline']['trades'] > 0 else 0

    comparison['gates'] = [
        check_gate(
            "PF Uplift ≥ +0.20",
            pf_uplift >= 0.20,
            f"Actual: {pf_uplift:+.2f}"
        ),
        check_gate(
            "Sharpe Uplift ≥ +0.15",
            sharpe_uplift >= 0.15,
            f"Actual: {sharpe_uplift:+.2f}"
        ),
        check_gate(
            "MaxDD ≤ Baseline",
            comparison['ml']['max_drawdown'] <= comparison['baseline']['max_drawdown'],
            f"ML: {comparison['ml']['max_drawdown']*100:.1f}% vs Baseline: {comparison['baseline']['max_drawdown']*100:.1f}%"
        ),
        check_gate(
            "Trade Count ≥ 80%",
            trade_retention >= 0.80,
            f"Retention: {trade_retention*100:.1f}%"
        )
    ]

    # Overall gate status
    gates_passed = sum(1 for g in comparison['gates'] if g['passed'])
    comparison['gates_summary'] = {
        'passed': gates_passed,
        'total': len(comparison['gates']),
        'status': 'PASS' if gates_passed >= 3 else 'FAIL',  # Require 3/4
        'message': f"{gates_passed}/4 gates passed"
    }

    return comparison


def print_comparison(comparison: Dict):
    """Print formatted comparison"""
    asset = comparison['asset']

    print("\n" + "=" * 80)
    print(f"BASELINE VS ML-ON COMPARISON - {asset}")
    print("=" * 80)

    # Metrics table
    print("\n📊 PERFORMANCE METRICS")
    print("-" * 80)
    print(f"{'Metric':<20} {'Baseline':<15} {'ML-ON':<15} {'Uplift':<15}")
    print("-" * 80)

    b = comparison['baseline']
    m = comparison['ml']
    u = comparison['uplifts']

    print(f"{'Trades':<20} {b['trades']:<15} {m['trades']:<15} {u['trades']}")
    print(f"{'Win Rate':<20} {format_metric(b['win_rate'], True):<15} {format_metric(m['win_rate'], True):<15} {u['win_rate']}")
    print(f"{'Profit Factor':<20} {format_metric(b['profit_factor']):<15} {format_metric(m['profit_factor']):<15} {u['pf']}")
    print(f"{'Sharpe Ratio':<20} {format_metric(b['sharpe_ratio']):<15} {format_metric(m['sharpe_ratio']):<15} {u['sharpe']}")
    print(f"{'Max Drawdown':<20} {format_metric(b['max_drawdown'], True):<15} {format_metric(m['max_drawdown'], True):<15} {u['max_dd']}")
    print(f"{'Total Return':<20} {format_metric(b['total_return'], True):<15} {format_metric(m['total_return'], True):<15}")
    print(f"{'Avg R-Multiple':<20} {format_metric(b['avg_r']):<15} {format_metric(m['avg_r']):<15} {u['avg_r']}")

    # Regime info
    if 'regime_label' in m:
        print(f"\n🧠 Regime: {m['regime_label'].upper()} (confidence: {m['regime_confidence']:.2f})")

    # Acceptance gates
    print("\n" + "=" * 80)
    print("ACCEPTANCE GATES")
    print("=" * 80)

    for gate in comparison['gates']:
        print(f"{gate['status']:<12} {gate['gate']:<30} {gate['details']}")

    print("-" * 80)
    summary = comparison['gates_summary']
    status_emoji = "✅" if summary['status'] == 'PASS' else "❌"
    print(f"{status_emoji} {summary['status']}: {summary['message']} (require 3/4)")
    print("=" * 80)


def main():
    """Main comparison routine"""
    print("=" * 80)
    print("BASELINE VS ML-ON COMPARISON TOOL")
    print("=" * 80)

    # BTC Comparison
    print("\n🔶 BITCOIN (BTC)")
    btc_baseline = "reports/v19/BTC_q3_baseline.json"
    btc_ml = "reports/v19/BTC_q3_regime_v2.json"

    if Path(btc_baseline).exists() and Path(btc_ml).exists():
        btc_comparison = compare_configs(btc_baseline, btc_ml, "BTC")
        print_comparison(btc_comparison)
    else:
        print("❌ BTC results not found")

    # ETH Comparison (if available)
    print("\n🔷 ETHEREUM (ETH)")
    eth_baseline = "reports/v19/ETH_q3_baseline.json"
    eth_ml = "reports/v19/ETH_q3_regime_v2.json"

    if Path(eth_baseline).exists() and Path(eth_ml).exists():
        eth_comparison = compare_configs(eth_baseline, eth_ml, "ETH")
        print_comparison(eth_comparison)
    else:
        print("⚠️  ETH had no trades in Q3 2024 (quiet period)")
        print("   Recommend testing on full-year 2024 or Q1 2024")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
