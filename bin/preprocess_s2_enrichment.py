#!/usr/bin/env python3
"""
S2 Feature Enrichment Pre-processor

Enriches 2022 BTC feature data with S2 runtime features and saves to temp location.
Then you can run regular backtest with the enriched data.

**Usage:**
    python3 bin/preprocess_s2_enrichment.py
    python3 bin/backtest_knowledge_v2.py --config configs/optimization/s2_runtime_enriched.json ...

**Author:** Claude Code
**Date:** 2025-11-16
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import logging
from datetime import datetime

from engine.strategies.archetypes.bear.failed_rally_runtime import apply_runtime_enrichment

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("S2 FEATURE ENRICHMENT PRE-PROCESSOR")
    logger.info("="*80)
    logger.info(f"Date: {datetime.now()}")
    logger.info("")

    # Paths
    input_file = "data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet"
    output_file = "data/features_mtf/BTC_1H_2022_ENRICHED.parquet"

    # Load data
    logger.info(f"Loading: {input_file}")
    df = pd.read_parquet(input_file)

    # Filter to 2022
    df_2022 = df[df.index < '2023-01-01'].copy()
    logger.info(f"  - Loaded {len(df_2022)} bars from 2022")
    logger.info(f"  - Date range: {df_2022.index.min()} to {df_2022.index.max()}")

    # Apply enrichment
    logger.info("\nApplying S2 runtime enrichment...")
    df_enriched = apply_runtime_enrichment(df_2022, lookback=14)

    # Save
    logger.info(f"\nSaving enriched data to: {output_file}")
    df_enriched.to_parquet(output_file)

    logger.info("\n✅ Enrichment complete!")
    logger.info(f"\nEnriched file: {output_file}")
    logger.info(f"Bars: {len(df_enriched)}")
    logger.info(f"Columns: {len(df_enriched.columns)} (added 4 new features)")

    # Feature stats
    logger.info("\n" + "="*80)
    logger.info("FEATURE STATISTICS")
    logger.info("="*80)

    strong_wicks = (df_enriched['wick_upper_ratio'] > 0.4).sum()
    vol_fades = df_enriched['volume_fade_flag'].sum()
    rsi_divs = df_enriched['rsi_bearish_div'].sum()
    ob_retests = df_enriched['ob_retest_flag'].sum()

    logger.info(f"Strong upper wicks (>0.4):  {strong_wicks:4d} ({strong_wicks/len(df_enriched)*100:5.1f}%)")
    logger.info(f"Volume fades:                {vol_fades:4d} ({vol_fades/len(df_enriched)*100:5.1f}%)")
    logger.info(f"RSI bearish divergences:     {rsi_divs:4d} ({rsi_divs/len(df_enriched)*100:5.1f}%)")
    logger.info(f"OB retests:                  {ob_retests:4d} ({ob_retests/len(df_enriched)*100:5.1f}%)")

    # Perfect signals (all 4 features)
    perfect_signals = (
        (df_enriched['wick_upper_ratio'] > 0.4) &
        (df_enriched['volume_fade_flag'] == True) &
        (df_enriched['rsi_bearish_div'] == True) &
        (df_enriched['ob_retest_flag'] == True)
    ).sum()

    logger.info(f"\nPERFECT S2 signals (all 4):  {perfect_signals:4d} ({perfect_signals/len(df_enriched)*100:5.2f}%)")

    # Next steps
    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)
    logger.info("\n1. Temporarily replace feature file:")
    logger.info(f"   mv {input_file} {input_file}.backup")
    logger.info(f"   cp {output_file} {input_file}")
    logger.info("\n2. Run backtest with enriched features:")
    logger.info("   python3 bin/backtest_knowledge_v2.py \\")
    logger.info("     --config configs/optimization/s2_runtime_enriched.json \\")
    logger.info("     --asset BTC \\")
    logger.info("     --start 2022-01-01 \\")
    logger.info("     --end 2022-12-31 \\")
    logger.info("     --output results/optimization/s2_enriched_2022.json")
    logger.info("\n3. Restore original file:")
    logger.info(f"   mv {input_file}.backup {input_file}")
    logger.info("\n4. Compare results to baseline (PF 0.38, WR 38.5%, 335 trades)")
    logger.info("")


if __name__ == '__main__':
    main()
