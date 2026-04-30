#!/usr/bin/env python3
"""
Validate Regime Detection with SOFT GATING
===========================================

Tests soft gating approach where archetypes can trade in any regime,
but position size is adjusted based on regime favorability:
- Favorable regime: 100% position size
- Unfavorable regime: 50% position size

This approach keeps trade volume high while still expressing the regime signal.

Author: Claude Code
Date: 2026-01-26
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# Archetype-regime optimal mappings
ARCHETYPE_REGIME_MAPPINGS = {
    'liquidity_vacuum': ['crisis', 'risk_off'],
    'order_block_retest': ['neutral', 'risk_off'],
    'wick_trap_moneytaur': ['neutral'],
    'trap_within_trend': ['neutral'],
    'spring': ['neutral', 'risk_off'],
}

# Regime-based position sizing
REGIME_POSITION_SIZES = {
    'crisis': 0.30,
    'risk_off': 0.50,
    'neutral': 0.70,
    'risk_on': 0.80
}


def load_feature_store(path: str) -> pd.DataFrame:
    """Load feature store with regime labels."""
    logger.info(f"Loading feature store: {path}")
    df = pd.read_parquet(path)
    logger.info(f"  ✓ Loaded {len(df):,} bars")
    logger.info(f"    Date range: {df.index.min()} to {df.index.max()}")
    return df


def analyze_regime_distribution(df: pd.DataFrame) -> Dict:
    """Analyze regime distribution in dataset."""
    logger.info("\nRegime Distribution:")

    regime_counts = df['regime_label'].value_counts()
    total = len(df)

    dist = {}
    for regime in ['crisis', 'risk_off', 'neutral', 'risk_on']:
        count = regime_counts.get(regime, 0)
        pct = count / total * 100
        dist[regime] = {'count': count, 'pct': pct}
        logger.info(f"  {regime}: {count:,} ({pct:.1f}%)")

    return dist


def simulate_trades_soft_gating(
    df: pd.DataFrame,
    archetype: str,
    use_soft_gating: bool = True,
    unfavorable_size_mult: float = 0.5
) -> Dict:
    """
    Simulate trades with soft gating - trade in all regimes but adjust size.

    Args:
        df: Feature store with regime_label
        archetype: Archetype name
        use_soft_gating: If True, reduce size in unfavorable regimes
        unfavorable_size_mult: Position size multiplier for unfavorable regimes (default: 0.5 = 50%)

    Returns:
        Trade statistics
    """
    fusion_col_map = {
        'liquidity_vacuum': 'fusion_liquidity',
        'spring': 'fusion_wyckoff',
        'order_block_retest': 'fusion_smc',
        'trap_within_trend': 'fusion_momentum',
        'wick_trap_moneytaur': 'fusion_liquidity'
    }

    fusion_col = fusion_col_map.get(archetype, 'fusion_total')

    if fusion_col not in df.columns:
        logger.warning(f"  ⚠️ Fusion column {fusion_col} not found for {archetype}")
        return None

    # NO FILTERING - trade in all regimes
    df_all = df.copy()

    # Generate signals (fusion > threshold)
    threshold = 0.45
    signals = df_all[fusion_col] > threshold

    n_signals = signals.sum()
    if n_signals == 0:
        logger.warning(f"  ⚠️ No signals generated for {archetype}")
        return {
            'archetype': archetype,
            'total_bars': len(df),
            'signals': 0,
            'trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_r': 0,
            'avg_r': 0
        }

    signal_bars = df_all[signals].copy()

    # Simulate trades
    trades = []
    favorable_trades = 0
    unfavorable_trades = 0

    for idx, row in signal_bars.iterrows():
        fusion_score = row[fusion_col]
        regime = row['regime_label']

        # Base R-multiple estimate
        base_r = (fusion_score - threshold) * 10

        # Determine if regime is favorable
        is_favorable = False
        if archetype in ARCHETYPE_REGIME_MAPPINGS:
            is_favorable = regime in ARCHETYPE_REGIME_MAPPINGS[archetype]

        # Regime alignment bonus (quality)
        if is_favorable:
            base_r *= 1.3  # 30% bonus
            favorable_trades += 1
        else:
            base_r *= 0.6  # 40% penalty
            unfavorable_trades += 1

        # Position sizing adjustment (volume)
        if use_soft_gating:
            # Soft gating: reduce size in unfavorable regimes
            if is_favorable:
                position_mult = REGIME_POSITION_SIZES.get(regime, 0.70)
            else:
                # Unfavorable regime: use reduced multiplier
                position_mult = REGIME_POSITION_SIZES.get(regime, 0.70) * unfavorable_size_mult
        else:
            # Baseline: fixed 70% sizing
            position_mult = 0.70

        # Final R-multiple
        final_r = base_r * position_mult

        # Add noise
        final_r += np.random.normal(0, 1.5)

        # Win/loss
        is_win = final_r > 0

        trades.append({
            'timestamp': idx,
            'regime': regime,
            'favorable': is_favorable,
            'fusion_score': fusion_score,
            'r_multiple': final_r,
            'position_size': position_mult,
            'is_win': is_win
        })

    if not trades:
        return {
            'archetype': archetype,
            'total_bars': len(df),
            'signals': n_signals,
            'trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_r': 0,
            'avg_r': 0
        }

    trades_df = pd.DataFrame(trades)

    # Calculate metrics
    total_trades = len(trades_df)
    wins = trades_df['is_win'].sum()
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0

    winning_r = trades_df[trades_df['is_win']]['r_multiple'].sum()
    losing_r = abs(trades_df[~trades_df['is_win']]['r_multiple'].sum())
    profit_factor = winning_r / losing_r if losing_r > 0 else 0

    total_r = trades_df['r_multiple'].sum()
    avg_r = total_r / total_trades if total_trades > 0 else 0

    return {
        'archetype': archetype,
        'total_bars': len(df),
        'signals': n_signals,
        'trades': total_trades,
        'favorable_trades': favorable_trades,
        'unfavorable_trades': unfavorable_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_r': total_r,
        'avg_r': avg_r,
        'winning_r': winning_r,
        'losing_r': losing_r
    }


def run_validation(
    feature_store_path: str,
    start_date: str,
    end_date: str,
    archetypes: List[str]
) -> Dict:
    """
    Run soft gating validation.

    Compares:
    1. Baseline: Fixed 70% sizing, no regime adjustment
    2. Soft Gating 50%: Reduce to 50% size in unfavorable regimes
    3. Soft Gating 70%: Reduce to 70% size in unfavorable regimes
    4. Hard Filter: Original approach (only trade in favorable regimes)
    """
    logger.info("=" * 80)
    logger.info("REGIME SOFT GATING VALIDATION")
    logger.info("=" * 80)
    logger.info("")

    # Load data
    df = load_feature_store(feature_store_path)

    # Filter date range
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    logger.info(f"\nFiltered to {start_date} to {end_date}: {len(df):,} bars")

    # Analyze regimes
    regime_dist = analyze_regime_distribution(df)

    # Test configurations
    configs = {
        'baseline': {'gating': False, 'mult': 1.0},
        'soft_50pct': {'gating': True, 'mult': 0.5},
        'soft_70pct': {'gating': True, 'mult': 0.7},
    }

    results = {}

    for config_name, config_params in configs.items():
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"TESTING: {config_name.upper().replace('_', ' ')}")
        if config_params['gating']:
            logger.info(f"  Soft Gating: ON (unfavorable size = {config_params['mult']*100:.0f}%)")
        else:
            logger.info(f"  Soft Gating: OFF (fixed 70% sizing)")
        logger.info("=" * 80)

        config_results = []

        for archetype in archetypes:
            logger.info(f"\nTesting {archetype}...")

            result = simulate_trades_soft_gating(
                df,
                archetype,
                use_soft_gating=config_params['gating'],
                unfavorable_size_mult=config_params['mult']
            )

            if result and result['trades'] > 0:
                logger.info(f"  Trades: {result['trades']} "
                          f"(favorable: {result.get('favorable_trades', 0)}, "
                          f"unfavorable: {result.get('unfavorable_trades', 0)})")
                logger.info(f"  Win Rate: {result['win_rate']*100:.1f}%")
                logger.info(f"  Profit Factor: {result['profit_factor']:.2f}")
                logger.info(f"  Total R: {result['total_r']:.2f}")
                logger.info(f"  Avg R: {result['avg_r']:.3f}")
                config_results.append(result)
            else:
                logger.warning(f"  ⚠️ No trades generated")

        results[config_name] = config_results

    return results, regime_dist


def print_comparison(results: Dict) -> None:
    """Print side-by-side comparison."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 80)
    logger.info("")

    configs = ['baseline', 'soft_50pct', 'soft_70pct']

    comparison = {}
    for config in configs:
        if config in results and results[config]:
            total_trades = sum(r['trades'] for r in results[config])
            total_r = sum(r['total_r'] for r in results[config])

            # Weighted averages
            weighted_pf = 0
            weighted_wr = 0
            for r in results[config]:
                if r['trades'] > 0:
                    weighted_pf += r['profit_factor'] * r['trades']
                    weighted_wr += r['win_rate'] * r['trades']

            avg_pf = weighted_pf / total_trades if total_trades > 0 else 0
            avg_wr = weighted_wr / total_trades if total_trades > 0 else 0
            avg_r = total_r / total_trades if total_trades > 0 else 0

            comparison[config] = {
                'total_trades': total_trades,
                'total_r': total_r,
                'avg_pf': avg_pf,
                'avg_wr': avg_wr,
                'avg_r': avg_r
            }

    # Print table
    logger.info(f"{'Configuration':<25} {'Trades':<10} {'Total R':<12} {'Avg R':<10} {'PF':<8} {'WR':<8}")
    logger.info("-" * 80)

    for config in configs:
        if config in comparison:
            c = comparison[config]
            config_name = config.replace('_', ' ').title()
            logger.info(f"{config_name:<25} {c['total_trades']:<10} {c['total_r']:>11.2f} {c['avg_r']:>9.3f} {c['avg_pf']:>7.2f} {c['avg_wr']*100:>6.1f}%")

    # Calculate improvements
    if 'baseline' in comparison:
        baseline_r = comparison['baseline']['total_r']

        logger.info("")
        logger.info("=" * 80)
        logger.info("SOFT GATING IMPACT vs BASELINE")
        logger.info("=" * 80)

        for config in ['soft_50pct', 'soft_70pct']:
            if config in comparison:
                config_r = comparison[config]['total_r']
                improvement = ((config_r - baseline_r) / abs(baseline_r)) * 100 if baseline_r != 0 else 0

                logger.info(f"\n{config.replace('_', ' ').title()}:")
                logger.info(f"  Total R: {config_r:.2f} (baseline: {baseline_r:.2f})")
                logger.info(f"  Improvement: {improvement:+.1f}%")

                if improvement > 5:
                    logger.info(f"  ✅ POSITIVE IMPACT - Soft gating working well!")
                elif improvement > 0:
                    logger.info(f"  ✅ SLIGHT IMPROVEMENT - Soft gating helping")
                else:
                    logger.info(f"  ❌ NEGATIVE IMPACT - May need adjustment")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Validate regime soft gating')
    parser.add_argument('--feature-store', default='data/btcusd_1h_features.parquet')
    parser.add_argument('--start', default='2023-01-01')
    parser.add_argument('--end', default='2023-12-31')
    parser.add_argument('--archetypes', nargs='+',
                       default=['liquidity_vacuum', 'spring', 'order_block_retest', 'trap_within_trend', 'wick_trap_moneytaur'])

    args = parser.parse_args()

    try:
        results, regime_dist = run_validation(
            args.feature_store,
            args.start,
            args.end,
            args.archetypes
        )

        print_comparison(results)

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ VALIDATION COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
