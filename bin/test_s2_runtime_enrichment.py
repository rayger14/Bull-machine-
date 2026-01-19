#!/usr/bin/env python3
"""
S2 Runtime Enrichment Test Script

Tests if S2 (Failed Rally) archetype is salvageable with runtime-calculated features.

**Mission:** Determine if poor performance (PF 0.56) is due to missing features or bad pattern logic.

**Approach:**
1. Load feature data for 2022 (bear market year)
2. Apply runtime feature enrichment (wick, volume, RSI, OB)
3. Run backtest with S2-only config
4. Compare results to baseline (no enrichment)

**Success Criteria:**
- PF > 1.0 (break even or better)
- WR > 45%
- Trade count < 200

**Author:** Claude Code (Performance Engineer)
**Date:** 2025-11-16
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.strategies.archetypes.bear.failed_rally_runtime import apply_runtime_enrichment

# ============================================================================
# Configuration
# ============================================================================

FEATURE_FILE = "data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet"
CONFIG_FILE = "configs/optimization/s2_runtime_enriched.json"
OUTPUT_DIR = "results/optimization"
TEST_START = "2022-01-01"
TEST_END = "2022-12-31"

BASELINE_RESULTS = {
    "pf": 0.38,
    "win_rate": 0.385,
    "trades": 335,
    "sharpe": -0.27,
    "max_dd": 0.352
}

OPTIMIZED_RESULTS = {
    "pf": 0.56,
    "win_rate": 0.426,
    "trades": 444,
    "sharpe": -0.14,
    "max_dd": 0.283
}

SUCCESS_CRITERIA = {
    "pf": 1.0,
    "win_rate": 0.45,
    "max_trades": 200
}


# ============================================================================
# Helper Functions
# ============================================================================

def load_and_enrich_features():
    """
    Load 2022 feature data and apply runtime enrichment.

    Returns:
        DataFrame with enriched features
    """
    print(f"Loading feature data from {FEATURE_FILE}...")

    try:
        df = pd.read_parquet(FEATURE_FILE)
        df_2022 = df[(df.index >= TEST_START) & (df.index < '2023-01-01')].copy()

        print(f"  - Loaded {len(df_2022)} bars from 2022")
        print(f"  - Date range: {df_2022.index.min()} to {df_2022.index.max()}")
        print(f"  - Columns: {len(df_2022.columns)}")

        print("\nApplying runtime feature enrichment...")
        df_enriched = apply_runtime_enrichment(df_2022, lookback=14)

        print(f"  - Enrichment complete!")
        print(f"  - New columns: wick_upper_ratio, wick_lower_ratio, volume_fade_flag, rsi_bearish_div, ob_retest_flag")

        # Validate enrichment
        new_cols = ['wick_upper_ratio', 'volume_fade_flag', 'rsi_bearish_div', 'ob_retest_flag']
        missing = [c for c in new_cols if c not in df_enriched.columns]

        if missing:
            raise ValueError(f"Enrichment failed! Missing columns: {missing}")

        # Statistics
        print("\nEnrichment Statistics:")
        print(f"  - Strong upper wicks (>0.4): {(df_enriched['wick_upper_ratio'] > 0.4).sum()} ({(df_enriched['wick_upper_ratio'] > 0.4).sum()/len(df_enriched)*100:.1f}%)")
        print(f"  - Volume fades: {df_enriched['volume_fade_flag'].sum()} ({df_enriched['volume_fade_flag'].sum()/len(df_enriched)*100:.1f}%)")
        print(f"  - RSI bearish divs: {df_enriched['rsi_bearish_div'].sum()} ({df_enriched['rsi_bearish_div'].sum()/len(df_enriched)*100:.1f}%)")
        print(f"  - OB retests: {df_enriched['ob_retest_flag'].sum()} ({df_enriched['ob_retest_flag'].sum()/len(df_enriched)*100:.1f}%)")

        # Combo signals (all 4 features aligned)
        combo_signals = (
            (df_enriched['wick_upper_ratio'] > 0.4) &
            (df_enriched['volume_fade_flag'] == True) &
            (df_enriched['rsi_bearish_div'] == True) &
            (df_enriched['ob_retest_flag'] == True)
        ).sum()

        print(f"  - PERFECT S2 signals (all 4 features): {combo_signals} ({combo_signals/len(df_enriched)*100:.2f}%)")

        return df_enriched

    except FileNotFoundError:
        print(f"ERROR: Feature file not found: {FEATURE_FILE}")
        print("Please run feature generation first.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR during enrichment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def save_enriched_features(df: pd.DataFrame):
    """
    Save enriched features to temporary file for backtest.

    Args:
        df: Enriched dataframe

    Returns:
        Path to saved file
    """
    output_file = f"{OUTPUT_DIR}/BTC_1H_2022_enriched.parquet"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nSaving enriched features to {output_file}...")
    df.to_parquet(output_file)
    print(f"  - Saved {len(df)} bars")

    return output_file


def run_backtest_with_enriched_features(enriched_file: str):
    """
    Run backtest using enriched features.

    Args:
        enriched_file: Path to enriched feature file

    Returns:
        Backtest results dict
    """
    print("\n" + "="*80)
    print("RUNNING BACKTEST WITH ENRICHED FEATURES")
    print("="*80)

    # Import backtest engine
    from bin.backtest_knowledge_v2 import main as backtest_main

    # Override feature file in config (hacky but works)
    # Alternative: pass enriched df directly to backtest
    # For now, we'll call backtest as subprocess with custom args

    import subprocess

    cmd = [
        "python3", "bin/backtest_knowledge_v2.py",
        "--config", CONFIG_FILE,
        "--asset", "BTC",
        "--start", TEST_START,
        "--end", TEST_END,
        "--output", f"{OUTPUT_DIR}/s2_enriched_results.json",
        # TODO: Add flag to override feature file path
    ]

    print(f"Command: {' '.join(cmd)}")
    print("\nRunning backtest...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("BACKTEST FAILED!")
        print(result.stderr)
        return None

    print(result.stdout)

    # Parse results
    try:
        with open(f"{OUTPUT_DIR}/s2_enriched_results.json", 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print("ERROR: Results file not found")
        return None


def compare_results(enriched_results: dict):
    """
    Compare enriched results to baseline and optimized versions.

    Args:
        enriched_results: Results from enriched backtest

    Returns:
        Comparison summary dict
    """
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)

    # Extract metrics (adjust keys based on actual backtest output structure)
    enriched_pf = enriched_results.get('profit_factor', 0.0)
    enriched_wr = enriched_results.get('win_rate', 0.0)
    enriched_trades = enriched_results.get('total_trades', 0)

    # Print comparison table
    print("\n| Metric         | Baseline | Optimized | Enriched | Target | Status |")
    print("|----------------|----------|-----------|----------|--------|--------|")

    # Profit Factor
    pf_status = "✅ PASS" if enriched_pf >= SUCCESS_CRITERIA['pf'] else "❌ FAIL"
    print(f"| Profit Factor  | {BASELINE_RESULTS['pf']:.2f}     | {OPTIMIZED_RESULTS['pf']:.2f}      | {enriched_pf:.2f}     | {SUCCESS_CRITERIA['pf']:.1f}    | {pf_status} |")

    # Win Rate
    wr_status = "✅ PASS" if enriched_wr >= SUCCESS_CRITERIA['win_rate'] else "❌ FAIL"
    print(f"| Win Rate       | {BASELINE_RESULTS['win_rate']*100:.1f}%   | {OPTIMIZED_RESULTS['win_rate']*100:.1f}%    | {enriched_wr*100:.1f}%   | {SUCCESS_CRITERIA['win_rate']*100:.0f}%  | {wr_status} |")

    # Trade Count
    trades_status = "✅ PASS" if enriched_trades <= SUCCESS_CRITERIA['max_trades'] else "❌ FAIL"
    print(f"| Trades         | {BASELINE_RESULTS['trades']}      | {OPTIMIZED_RESULTS['trades']}       | {enriched_trades}      | <{SUCCESS_CRITERIA['max_trades']}  | {trades_status} |")

    # Improvement calculations
    pf_improvement = (enriched_pf - BASELINE_RESULTS['pf']) / BASELINE_RESULTS['pf'] * 100
    wr_improvement = (enriched_wr - BASELINE_RESULTS['win_rate']) / BASELINE_RESULTS['win_rate'] * 100

    print(f"\nImprovement vs Baseline:")
    print(f"  - PF: {pf_improvement:+.1f}%")
    print(f"  - WR: {wr_improvement:+.1f}%")

    # Final verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    all_pass = (
        enriched_pf >= SUCCESS_CRITERIA['pf'] and
        enriched_wr >= SUCCESS_CRITERIA['win_rate'] and
        enriched_trades <= SUCCESS_CRITERIA['max_trades']
    )

    if all_pass:
        print("✅ S2 IS SALVAGEABLE with runtime feature enrichment!")
        print("\nRecommendation:")
        print("  1. Promote runtime features to feature store pipeline")
        print("  2. Run extended optimization on enriched features")
        print("  3. Include S2 in production bear market config")
    elif enriched_pf >= 0.7:
        print("⚠️ MARGINAL IMPROVEMENT detected")
        print("\nRecommendation:")
        print("  1. Runtime features helped but not enough")
        print("  2. Consider full OB pipeline implementation")
        print("  3. Test pattern inversion (use as LONG signal)")
        print("  4. OR disable for now, revisit after more feature engineering")
    else:
        print("❌ S2 IS NOT SALVAGEABLE even with runtime features")
        print("\nRecommendation:")
        print("  1. Disable S2 permanently")
        print("  2. Pattern has fundamental design flaws")
        print("  3. Document as case study for pattern validation")
        print("  4. Focus resources on S5 (Long Squeeze) instead")

    return {
        "enriched_pf": enriched_pf,
        "enriched_wr": enriched_wr,
        "enriched_trades": enriched_trades,
        "pf_improvement": pf_improvement,
        "wr_improvement": wr_improvement,
        "all_criteria_met": all_pass
    }


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main test execution flow.
    """
    print("="*80)
    print("S2 RUNTIME ENRICHMENT TEST")
    print("="*80)
    print(f"Date: {datetime.now()}")
    print(f"Test Period: {TEST_START} to {TEST_END}")
    print(f"Config: {CONFIG_FILE}")
    print("\n")

    # Step 1: Load and enrich features
    df_enriched = load_and_enrich_features()

    # Step 2: Save enriched features
    enriched_file = save_enriched_features(df_enriched)

    # Step 3: Run backtest
    print("\n⚠️  MANUAL STEP REQUIRED ⚠️")
    print("\nThe enriched features have been saved, but backtest integration requires:")
    print("1. Modify backtest_knowledge_v2.py to accept --feature-file argument")
    print("2. OR manually replace feature store data with enriched version")
    print("3. OR integrate enrichment directly into backtest engine")
    print("\nFor now, please run manually:")
    print(f"\n  python3 bin/backtest_knowledge_v2.py \\")
    print(f"    --config {CONFIG_FILE} \\")
    print(f"    --asset BTC \\")
    print(f"    --start {TEST_START} \\")
    print(f"    --end {TEST_END} \\")
    print(f"    --output {OUTPUT_DIR}/s2_enriched_results.json")
    print(f"\nBUT FIRST: Replace feature data with {enriched_file}")
    print("\nAlternatively, integrate enrichment into backtest engine startup.")

    # For now, skip backtest and just show feature stats
    print("\n" + "="*80)
    print("FEATURE CONTRIBUTION ANALYSIS")
    print("="*80)

    # Analyze which features are most common in potential S2 signals
    potential_signals = df_enriched[
        (df_enriched['wick_upper_ratio'] > 0.4) |
        (df_enriched['volume_fade_flag'] == True) |
        (df_enriched['rsi_bearish_div'] == True) |
        (df_enriched['ob_retest_flag'] == True)
    ]

    print(f"\nPotential S2 signals (any feature present): {len(potential_signals)} ({len(potential_signals)/len(df_enriched)*100:.1f}%)")
    print("\nFeature contribution:")
    print(f"  - Wick rejection:   {(potential_signals['wick_upper_ratio'] > 0.4).sum():4d} ({(potential_signals['wick_upper_ratio'] > 0.4).sum()/len(potential_signals)*100:.1f}%)")
    print(f"  - Volume fade:      {potential_signals['volume_fade_flag'].sum():4d} ({potential_signals['volume_fade_flag'].sum()/len(potential_signals)*100:.1f}%)")
    print(f"  - RSI divergence:   {potential_signals['rsi_bearish_div'].sum():4d} ({potential_signals['rsi_bearish_div'].sum()/len(potential_signals)*100:.1f}%)")
    print(f"  - OB retest:        {potential_signals['ob_retest_flag'].sum():4d} ({potential_signals['ob_retest_flag'].sum()/len(potential_signals)*100:.1f}%)")

    # Save enriched data for manual testing
    print(f"\n✅ Enriched features saved to: {enriched_file}")
    print("\nNext steps:")
    print("1. Integrate enrichment into backtest engine")
    print("2. Run backtest with enriched features")
    print("3. Compare results to baseline")
    print("4. Make final decision on S2 viability")


if __name__ == '__main__':
    main()
