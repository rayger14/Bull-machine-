#!/usr/bin/env python3
"""
Evaluation Script: Compare Baseline vs Regime-Enabled Backtests

Runs side-by-side comparison to measure impact of Phase 2 regime classification.

Usage:
    python3 scripts/eval_regime_backtest.py --asset BTC --start 2024-07-01 --end 2024-09-30 --config configs/v18/BTC_live.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
import logging

# Import components
from engine.context.regime_classifier import RegimeClassifier
from engine.context.regime_policy import RegimePolicy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load trading config"""
    with open(config_path) as f:
        return json.load(f)


def run_baseline_backtest(
    asset: str,
    start: str,
    end: str,
    config: Dict
) -> Dict:
    """
    Run baseline backtest without regime adaptation

    Args:
        asset: BTC or ETH
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        config: Trading config

    Returns:
        Dict with performance metrics
    """
    logger.info(f"Running baseline backtest (no regime adaptation)...")

    # TODO: Integrate with actual backtest engine
    # For now, return placeholder metrics
    # In production, this would call optimize_v19.py or hybrid_runner.py

    return {
        'trades': 0,
        'wins': 0,
        'losses': 0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'total_return': 0.0,
        'sharpe': 0.0,
        'max_dd': 0.0,
        'avg_r': 0.0
    }


def run_regime_backtest(
    asset: str,
    start: str,
    end: str,
    config: Dict,
    regime_classifier: RegimeClassifier,
    regime_policy: RegimePolicy
) -> Tuple[Dict, pd.DataFrame]:
    """
    Run backtest with regime-adaptive adjustments

    Args:
        asset: BTC or ETH
        start: Start date
        end: End date
        config: Trading config
        regime_classifier: Trained classifier
        regime_policy: Policy for adjustments

    Returns:
        (performance_metrics, regime_log)
    """
    logger.info(f"Running regime-enabled backtest...")

    # TODO: Integrate with actual backtest engine
    # This would:
    # 1. Load bar-by-bar data
    # 2. For each bar:
    #    a. Classify regime
    #    b. Apply policy adjustments
    #    c. Run trading logic with adjusted params
    #    d. Log regime + adjustments
    # 3. Return metrics + regime log

    regime_log = pd.DataFrame({
        'timestamp': [],
        'regime': [],
        'confidence': [],
        'threshold_delta': [],
        'risk_multiplier': [],
        'applied': []
    })

    metrics = {
        'trades': 0,
        'wins': 0,
        'losses': 0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'total_return': 0.0,
        'sharpe': 0.0,
        'max_dd': 0.0,
        'avg_r': 0.0
    }

    return metrics, regime_log


def compare_results(baseline: Dict, regime: Dict) -> Dict:
    """
    Compare baseline vs regime-enabled results

    Args:
        baseline: Baseline metrics
        regime: Regime-enabled metrics

    Returns:
        Dict with deltas and improvements
    """
    deltas = {}

    for key in ['trades', 'wins', 'win_rate', 'profit_factor', 'total_return', 'sharpe', 'avg_r']:
        baseline_val = baseline.get(key, 0.0)
        regime_val = regime.get(key, 0.0)

        if baseline_val > 0:
            pct_change = ((regime_val - baseline_val) / baseline_val) * 100
        else:
            pct_change = 0.0 if regime_val == 0 else 999.0

        deltas[key] = {
            'baseline': baseline_val,
            'regime': regime_val,
            'delta': regime_val - baseline_val,
            'pct_change': pct_change
        }

    return deltas


def print_comparison_table(deltas: Dict):
    """
    Print formatted comparison table

    Args:
        deltas: Output from compare_results()
    """
    print("\n" + "="*80)
    print("📊 BASELINE vs REGIME-ENABLED COMPARISON")
    print("="*80)

    print(f"{'Metric':<20} {'Baseline':>15} {'Regime':>15} {'Delta':>15} {'Change':>12}")
    print("-"*80)

    for metric, data in deltas.items():
        baseline = data['baseline']
        regime = data['regime']
        delta = data['delta']
        pct_change = data['pct_change']

        # Format based on metric type
        if metric in ['win_rate', 'total_return', 'pct_change']:
            baseline_str = f"{baseline:.1f}%"
            regime_str = f"{regime:.1f}%"
            delta_str = f"{delta:+.1f}%"
        elif metric in ['profit_factor', 'sharpe', 'avg_r']:
            baseline_str = f"{baseline:.2f}"
            regime_str = f"{regime:.2f}"
            delta_str = f"{delta:+.2f}"
        else:
            baseline_str = f"{int(baseline)}"
            regime_str = f"{int(regime)}"
            delta_str = f"{int(delta):+d}"

        # Color code the change
        if pct_change > 5.0:
            change_str = f"+{pct_change:.1f}% ✅"
        elif pct_change < -5.0:
            change_str = f"{pct_change:.1f}% ❌"
        else:
            change_str = f"{pct_change:+.1f}%"

        print(f"{metric:<20} {baseline_str:>15} {regime_str:>15} {delta_str:>15} {change_str:>12}")

    print("="*80)


def analyze_regime_distribution(regime_log: pd.DataFrame):
    """
    Analyze regime distribution during backtest

    Args:
        regime_log: DataFrame with regime classifications
    """
    if regime_log.empty:
        logger.warning("No regime log data available")
        return

    print("\n" + "="*80)
    print("📈 REGIME DISTRIBUTION ANALYSIS")
    print("="*80)

    # Regime counts
    regime_counts = regime_log['regime'].value_counts()
    total = len(regime_log)

    print("\nRegime Distribution:")
    for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
        count = regime_counts.get(regime, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        print(f"  {regime:12s}: {count:6d} bars ({pct:5.1f}%)")

    # Adjustments applied
    if 'applied' in regime_log.columns:
        applied_count = regime_log['applied'].sum()
        applied_pct = (applied_count / total * 100) if total > 0 else 0.0
        print(f"\nAdjustments Applied: {applied_count}/{total} bars ({applied_pct:.1f}%)")

    # Average confidence
    if 'confidence' in regime_log.columns:
        avg_confidence = regime_log['confidence'].mean()
        print(f"Average Confidence: {avg_confidence:.2f}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate regime-enabled backtest")
    parser.add_argument('--asset', required=True, choices=['BTC', 'ETH'], help="Asset to test")
    parser.add_argument('--start', required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end', required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument('--config', required=True, help="Path to trading config")
    parser.add_argument('--regime-model', default="models/regime_classifier_gmm.pkl", help="Path to regime model")
    parser.add_argument('--regime-policy', default="configs/v19/regime_policy.json", help="Path to regime policy")
    parser.add_argument('--output', help="Output path for results JSON")

    args = parser.parse_args()

    print("="*80)
    print("🔬 Bull Machine v1.9 - Regime-Enabled Backtest Evaluation")
    print("="*80)
    print(f"Asset: {args.asset}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Config: {args.config}")
    print("="*80)

    # Load config
    config = load_config(args.config)

    # Check if regime model exists
    if not Path(args.regime_model).exists():
        logger.error(f"Regime model not found: {args.regime_model}")
        logger.error("Please train the model first:")
        logger.error(f"  python3 bin/train/train_regime_classifier.py --data <macro_data> --output {args.regime_model}")
        return 1

    # Load regime components
    logger.info(f"Loading regime classifier from {args.regime_model}")
    feature_order = [
        "VIX", "DXY", "MOVE", "YIELD_2Y", "YIELD_10Y",
        "USDT.D", "BTC.D", "TOTAL", "TOTAL2",
        "funding", "oi", "rv_20d", "rv_60d"
    ]
    regime_classifier = RegimeClassifier.load(args.regime_model, feature_order)

    logger.info(f"Loading regime policy from {args.regime_policy}")
    regime_policy = RegimePolicy.load(args.regime_policy)

    # Run baseline backtest
    baseline_metrics = run_baseline_backtest(args.asset, args.start, args.end, config)

    # Run regime-enabled backtest
    regime_metrics, regime_log = run_regime_backtest(
        args.asset, args.start, args.end, config,
        regime_classifier, regime_policy
    )

    # Compare results
    deltas = compare_results(baseline_metrics, regime_metrics)
    print_comparison_table(deltas)

    # Analyze regime distribution
    analyze_regime_distribution(regime_log)

    # Save results
    if args.output:
        results = {
            'config': {
                'asset': args.asset,
                'start': args.start,
                'end': args.end,
                'config_path': args.config
            },
            'baseline': baseline_metrics,
            'regime': regime_metrics,
            'deltas': deltas,
            'regime_log': regime_log.to_dict('records') if not regime_log.empty else []
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    # Summary
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)

    # Highlight key improvements
    wr_change = deltas['win_rate']['pct_change']
    pf_change = deltas['profit_factor']['pct_change']
    return_change = deltas['total_return']['pct_change']

    print(f"\nKey Metrics:")
    print(f"  Win Rate: {wr_change:+.1f}%")
    print(f"  Profit Factor: {pf_change:+.1f}%")
    print(f"  Total Return: {return_change:+.1f}%")

    if wr_change > 3.0 or pf_change > 5.0:
        print("\n🎉 Phase 2 shows significant improvement! Ready for production validation.")
    elif wr_change > 0 or pf_change > 0:
        print("\n✅ Phase 2 shows positive impact. Consider additional tuning.")
    else:
        print("\n⚠️  Phase 2 needs refinement. Review regime policy bounds.")

    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
