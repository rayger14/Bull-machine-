#!/usr/bin/env python3
"""
S2 Enriched Backtest Wrapper

Integrates runtime feature enrichment directly into the backtest flow.
This is a wrapper around backtest_knowledge_v2.py that:

1. Loads feature data
2. Applies S2 runtime enrichment
3. Temporarily replaces feature store data
4. Runs backtest
5. Restores original data

**Author:** Claude Code (Performance Engineer)
**Date:** 2025-11-16
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import shutil
import logging
from datetime import datetime

# Import runtime enrichment
from engine.strategies.archetypes.bear.failed_rally_runtime import apply_runtime_enrichment

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Run S2 backtest with runtime feature enrichment.
    """
    import argparse
    parser = argparse.ArgumentParser(description='S2 Enriched Backtest')
    parser.add_argument('--asset', default='BTC', help='Asset to backtest')
    parser.add_argument('--start', default='2022-01-01', help='Start date')
    parser.add_argument('--end', default='2022-12-31', help='End date')
    parser.add_argument('--config', default='configs/optimization/s2_runtime_enriched.json', help='Config file')
    parser.add_argument('--output', default='results/optimization/s2_enriched_results.json', help='Output file')
    parser.add_argument('--lookback', type=int, default=14, help='Lookback for divergence detection')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("S2 RUNTIME ENRICHMENT BACKTEST")
    logger.info("="*80)
    logger.info(f"Asset: {args.asset}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Lookback: {args.lookback} bars")
    logger.info("")

    # ========================================================================
    # Step 1: Load feature data
    # ========================================================================
    feature_path = f"data/features_mtf/{args.asset}_1H_2022-01-01_to_2023-12-31.parquet"

    logger.info(f"Loading features from {feature_path}...")

    try:
        df = pd.read_parquet(feature_path)
        # Filter to test period
        df_test = df[(df.index >= args.start) & (df.index < '2023-01-01')].copy()

        logger.info(f"  - Loaded {len(df_test)} bars")
        logger.info(f"  - Date range: {df_test.index.min()} to {df_test.index.max()}")
        logger.info(f"  - Columns: {len(df_test.columns)}")

    except FileNotFoundError:
        logger.error(f"Feature file not found: {feature_path}")
        logger.error("Please run feature generation first.")
        return 1

    # ========================================================================
    # Step 2: Apply runtime enrichment
    # ========================================================================
    logger.info("\nApplying S2 runtime feature enrichment...")

    df_enriched = apply_runtime_enrichment(df_test, lookback=args.lookback)

    # Verify enrichment
    required_cols = ['wick_upper_ratio', 'volume_fade_flag', 'rsi_bearish_div', 'ob_retest_flag']
    missing = [c for c in required_cols if c not in df_enriched.columns]

    if missing:
        logger.error(f"Enrichment failed! Missing columns: {missing}")
        return 1

    logger.info("  ✓ Enrichment successful")

    # ========================================================================
    # Step 3: Save enriched features to temporary location
    # ========================================================================
    # Create backup of original features
    backup_path = feature_path + ".backup"
    temp_enriched_path = f"data/features_mtf/{args.asset}_1H_enriched_temp.parquet"

    logger.info(f"\nSaving enriched features to {temp_enriched_path}...")

    # Save enriched data
    df_enriched.to_parquet(temp_enriched_path)

    logger.info("  ✓ Saved enriched features")

    # ========================================================================
    # Step 4: Run backtest with enriched data
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("RUNNING BACKTEST WITH ENRICHED FEATURES")
    logger.info("="*80)

    # Import backtest main function
    try:
        from bin.backtest_knowledge_v2 import run_backtest

        # Run backtest (passing enriched dataframe directly)
        results = run_backtest(
            asset=args.asset,
            start_date=args.start,
            end_date=args.end,
            config_path=args.config,
            output_path=args.output,
            df_override=df_enriched  # Pass enriched data directly
        )

        logger.info("\n✓ Backtest complete!")

        # Print summary
        if results:
            logger.info("\n" + "="*80)
            logger.info("RESULTS SUMMARY")
            logger.info("="*80)
            logger.info(f"Profit Factor: {results.get('profit_factor', 0.0):.2f}")
            logger.info(f"Win Rate: {results.get('win_rate', 0.0)*100:.1f}%")
            logger.info(f"Total Trades: {results.get('total_trades', 0)}")
            logger.info(f"Sharpe Ratio: {results.get('sharpe_ratio', 0.0):.2f}")
            logger.info(f"Max Drawdown: {results.get('max_drawdown', 0.0)*100:.1f}%")

            # Compare to baseline
            logger.info("\n" + "="*80)
            logger.info("COMPARISON TO BASELINE")
            logger.info("="*80)

            baseline_pf = 0.38
            baseline_wr = 0.385
            baseline_trades = 335

            pf = results.get('profit_factor', 0.0)
            wr = results.get('win_rate', 0.0)
            trades = results.get('total_trades', 0)

            pf_improvement = (pf - baseline_pf) / baseline_pf * 100 if baseline_pf > 0 else 0
            wr_improvement = (wr - baseline_wr) / baseline_wr * 100 if baseline_wr > 0 else 0

            logger.info(f"PF improvement: {pf_improvement:+.1f}% ({baseline_pf:.2f} → {pf:.2f})")
            logger.info(f"WR improvement: {wr_improvement:+.1f}% ({baseline_wr*100:.1f}% → {wr*100:.1f}%)")
            logger.info(f"Trade count: {trades} (baseline: {baseline_trades})")

            # Verdict
            logger.info("\n" + "="*80)
            logger.info("VERDICT")
            logger.info("="*80)

            if pf >= 1.0:
                logger.info("✅ S2 IS SALVAGEABLE with runtime enrichment!")
                logger.info("\nNext steps:")
                logger.info("  1. Promote runtime features to feature store")
                logger.info("  2. Run extended optimization")
                logger.info("  3. Include S2 in production config")
            elif pf >= 0.7:
                logger.info("⚠️ MARGINAL IMPROVEMENT detected")
                logger.info("\nNext steps:")
                logger.info("  1. Consider full OB pipeline implementation")
                logger.info("  2. Test pattern inversion (LONG signals)")
                logger.info("  3. OR disable for now")
            else:
                logger.info("❌ S2 NOT SALVAGEABLE even with enrichment")
                logger.info("\nNext steps:")
                logger.info("  1. Disable S2 permanently")
                logger.info("  2. Focus on S5 (Long Squeeze) instead")

        return 0

    except ImportError as e:
        logger.error(f"Failed to import backtest engine: {e}")
        logger.error("\nFallback: Using subprocess to run backtest")

        # Fallback: Use subprocess (less elegant but works)
        import subprocess

        # First, replace feature file temporarily
        logger.info("\nTemporarily replacing feature file...")
        shutil.copy(feature_path, backup_path)
        shutil.copy(temp_enriched_path, feature_path)

        cmd = [
            "python3", "bin/backtest_knowledge_v2.py",
            "--config", args.config,
            "--asset", args.asset,
            "--start", args.start,
            "--end", args.end,
            "--output", args.output
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=False)

        # Restore original file
        logger.info("\nRestoring original feature file...")
        shutil.move(backup_path, feature_path)

        return result.returncode

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Cleanup temp file
        if Path(temp_enriched_path).exists():
            Path(temp_enriched_path).unlink()
            logger.info(f"\nCleaned up temp file: {temp_enriched_path}")


if __name__ == '__main__':
    sys.exit(main())
