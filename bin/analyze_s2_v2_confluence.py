#!/usr/bin/env python3
"""
S2 v2 Confluence Analysis - Research Experiment
================================================
Lightweight research experiment testing confluence hypothesis using ONLY existing features
that have actual variance in the baseline data.

CRITICAL CONSTRAINTS:
- Research only - no pipeline changes
- Uses only features that exist in baseline S2 data
- Creates analysis-only detector (log signals, don't trade)
- Compares baseline vs confluence-filtered performance

Author: Performance Engineer (Data-Driven Optimization)
Date: 2025-11-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def calculate_confluence_score(row):
    """
    Calculate confluence score using ONLY features with actual variance.

    Based on feature audit, we have:
    - volume_zscore: 335 unique values ✓
    - tf4h_fusion: 69 unique values ✓
    - macro_regime_risk_off: binary ✓

    Features that are constant (NOT USABLE):
    - rsi_14 = 50.0 (constant)
    - adx_14 = 0.0 (constant)
    - wyckoff_phase_score = 0.0 (constant)
    - entry_liquidity_score = 0.0 (constant)
    - boms_strength = 0.0 (constant)

    We'll create a simplified 4-condition confluence model:
    C1: Volume Fade (volume_zscore < 0)
    C2: Weak MTF Context (tf4h_fusion < 0.05)
    C3: Risk-Off Macro (macro_regime_risk_off = 1)
    C4: Low Entry Fusion (entry_fusion_score < 0.3)

    Returns:
        tuple: (confluence_count, condition_dict)
    """

    conditions = {}

    # C1: Volume Fade (below average volume = weakening momentum)
    volume_z = row.get('volume_zscore', 0)
    c1 = volume_z < 0.0
    conditions['c1_volume_fade'] = c1

    # C2: Weak MTF context (4H not supporting rally)
    tf4h_fus = row.get('tf4h_fusion', 0)
    c2 = tf4h_fus < 0.05  # Weak or no 4H alignment
    conditions['c2_weak_mtf'] = c2

    # C3: Risk-off macro environment
    risk_off = row.get('macro_regime_risk_off', 0)
    c3 = risk_off == 1
    conditions['c3_risk_off'] = c3

    # C4: Low entry fusion (weak setup)
    entry_fus = row.get('entry_fusion_score', 0)
    c4 = entry_fus < 0.3
    conditions['c4_low_fusion'] = c4

    # Count how many conditions are met
    confluence_count = sum([c1, c2, c3, c4])

    return confluence_count, conditions


def analyze_confluence_performance(df, min_confluence):
    """Analyze performance at a given confluence threshold"""

    filtered = df[df['confluence'] >= min_confluence]

    if len(filtered) == 0:
        return {
            'trades': 0,
            'pct_of_baseline': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_r': 0.0,
            'total_r': 0.0,
            'winners_r': 0.0,
            'losers_r': 0.0
        }

    winners = filtered[filtered['r_multiple'] > 0]
    losers = filtered[filtered['r_multiple'] < 0]

    winners_r = winners['r_multiple'].sum()
    losers_r = abs(losers['r_multiple'].sum())

    pf = winners_r / losers_r if losers_r > 0 else 0.0
    wr = len(winners) / len(filtered) * 100
    avg_r = filtered['r_multiple'].mean()
    total_r = filtered['r_multiple'].sum()

    return {
        'trades': len(filtered),
        'pct_of_baseline': len(filtered) / len(df) * 100,
        'win_rate': wr,
        'profit_factor': pf,
        'avg_r': avg_r,
        'total_r': total_r,
        'winners_r': winners_r,
        'losers_r': losers_r
    }


def main():
    """Main analysis function"""

    # Load S2 baseline trades
    baseline_path = Path('results/optimization/s2_baseline_trades.csv')
    if not baseline_path.exists():
        print(f"ERROR: Missing {baseline_path}")
        sys.exit(1)

    df = pd.read_csv(baseline_path)
    print(f"\nLoaded {len(df)} S2 baseline trades from 2022")
    print(f"Date range: {df['entry_time'].min()} to {df['entry_time'].max()}")

    # Calculate confluence for each trade
    print("\nCalculating confluence scores...")

    confluence_data = []
    for idx, row in df.iterrows():
        conf_score, conditions = calculate_confluence_score(row)

        confluence_data.append({
            'trade_id': idx,
            'entry_time': row['entry_time'],
            'r_multiple': row['r_multiple'],
            'trade_won': row['trade_won'],
            'confluence': conf_score,
            **conditions,
            'volume_zscore': row.get('volume_zscore', 0),
            'tf4h_fusion': row.get('tf4h_fusion', 0),
            'entry_fusion_score': row.get('entry_fusion_score', 0),
            'macro_regime_risk_off': row.get('macro_regime_risk_off', 0)
        })

    analysis_df = pd.DataFrame(confluence_data)

    # Analyze baseline (all trades)
    print("\n" + "="*80)
    print("S2 V2 CONFLUENCE ANALYSIS - RESEARCH EXPERIMENT")
    print("="*80)

    print("\nBASELINE (All Trades):")
    baseline_stats = analyze_confluence_performance(analysis_df, 0)
    print(f"  Trades: {baseline_stats['trades']:3d}")
    print(f"  Win Rate: {baseline_stats['win_rate']:5.1f}%")
    print(f"  Profit Factor: {baseline_stats['profit_factor']:5.2f}")
    print(f"  Avg R: {baseline_stats['avg_r']:+7.3f}R")
    print(f"  Total R: {baseline_stats['total_r']:+7.2f}R")

    # Analyze each confluence level
    print("\n" + "-"*80)
    print("CONFLUENCE FILTERING RESULTS")
    print("-"*80)

    results_table = []

    for conf_level in range(1, 5):
        stats = analyze_confluence_performance(analysis_df, conf_level)

        if stats['trades'] > 0:
            print(f"\nConfluence >= {conf_level}/4:")
            print(f"  Trades: {stats['trades']:3d} ({stats['pct_of_baseline']:5.1f}% of baseline)")
            print(f"  Win Rate: {stats['win_rate']:5.1f}%")
            print(f"  Profit Factor: {stats['profit_factor']:5.2f}")
            print(f"  Avg R: {stats['avg_r']:+7.3f}R")
            print(f"  Total R: {stats['total_r']:+7.2f}R")

            # Calculate improvement vs baseline
            pf_delta = stats['profit_factor'] - baseline_stats['profit_factor']
            wr_delta = stats['win_rate'] - baseline_stats['win_rate']

            print(f"  PF Δ: {pf_delta:+5.2f} ({pf_delta/baseline_stats['profit_factor']*100:+6.1f}%)")
            print(f"  WR Δ: {wr_delta:+5.1f}pp")

            results_table.append({
                'confluence': f"{conf_level}/4",
                'trades': stats['trades'],
                'pct_baseline': stats['pct_of_baseline'],
                'win_rate': stats['win_rate'],
                'profit_factor': stats['profit_factor'],
                'avg_r': stats['avg_r'],
                'total_r': stats['total_r'],
                'pf_delta': pf_delta,
                'wr_delta': wr_delta
            })

    # Condition frequency analysis
    print("\n" + "-"*80)
    print("CONDITION FREQUENCY ANALYSIS")
    print("-"*80)

    for condition in ['c1_volume_fade', 'c2_weak_mtf', 'c3_risk_off', 'c4_low_fusion']:
        count = analysis_df[condition].sum()
        pct = count / len(analysis_df) * 100

        # Performance when condition is TRUE
        cond_true = analysis_df[analysis_df[condition]]
        if len(cond_true) > 0:
            cond_wr = (cond_true['r_multiple'] > 0).mean() * 100
            cond_pf = (cond_true[cond_true['r_multiple'] > 0]['r_multiple'].sum() /
                      abs(cond_true[cond_true['r_multiple'] < 0]['r_multiple'].sum())
                      if (cond_true['r_multiple'] < 0).sum() > 0 else 0)
        else:
            cond_wr = 0
            cond_pf = 0

        print(f"\n{condition:20s}: {count:3d} trades ({pct:5.1f}%)")
        print(f"  When TRUE: WR={cond_wr:5.1f}%, PF={cond_pf:5.2f}")

    # Save detailed analysis
    output_path = Path('results/optimization/s2_v2_confluence_analysis.csv')
    analysis_df.to_csv(output_path, index=False)
    print(f"\n\nDetailed analysis saved to: {output_path}")

    # Generate comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)

    print(f"\n{'Version':<20s} {'Min Conf':<12s} {'Trades':<8s} {'WR':<8s} {'PF':<8s} {'Avg R':<10s} {'Total R':<10s}")
    print("-"*80)
    print(f"{'S2 Baseline':<20s} {'0/4 (all)':<12s} {baseline_stats['trades']:<8d} {baseline_stats['win_rate']:<7.1f}% {baseline_stats['profit_factor']:<8.2f} {baseline_stats['avg_r']:<+10.3f} {baseline_stats['total_r']:<+10.2f}")

    for result in results_table:
        print(f"{'S2 v2 (' + result['confluence'] + ')':<20s} {result['confluence']:<12s} {result['trades']:<8d} {result['win_rate']:<7.1f}% {result['profit_factor']:<8.2f} {result['avg_r']:<+10.3f} {result['total_r']:<+10.2f}")

    # Decision matrix
    print("\n" + "="*80)
    print("DECISION MATRIX")
    print("="*80)

    # Find best confluence level
    best_pf = 0
    best_conf = None
    for result in results_table:
        if result['profit_factor'] > best_pf and result['trades'] >= 50:
            best_pf = result['profit_factor']
            best_conf = result['confluence']

    if best_conf:
        best_result = [r for r in results_table if r['confluence'] == best_conf][0]

        print(f"\nBest Configuration: Confluence >= {best_conf}")
        print(f"  Trades: {best_result['trades']} ({best_result['pct_baseline']:.1f}% of baseline)")
        print(f"  Profit Factor: {best_result['profit_factor']:.2f}")
        print(f"  Win Rate: {best_result['win_rate']:.1f}%")
        print(f"  Improvement: PF {best_result['pf_delta']:+.2f} ({best_result['pf_delta']/baseline_stats['profit_factor']*100:+.1f}%), WR {best_result['wr_delta']:+.1f}pp")

        print("\nVERDICT:")
        if best_pf >= 1.3:
            print("  ✓ S2 v2 is SALVAGEABLE")
            print("  → Implement as 'failed_rally_v2' research archetype")
            print("  → Promote to production if 2023 validates")
        elif best_pf >= 1.0:
            print("  ⚠ MARGINAL - needs more work")
            print("  → Consider adding OI or ML-based confluence weighting")
            print("  → Keep in research branch")
        else:
            print("  ✗ S2 is DEAD even with confluence")
            print("  → Disable permanently")
            print("  → Focus on S5 and other patterns")
    else:
        print("\n✗ NO VIABLE CONFIGURATION FOUND")
        print("  → Not enough trades (min 50) at any confluence level with PF > baseline")
        print("  → S2 pattern is fundamentally broken")
        print("  → RECOMMENDATION: Disable permanently")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
