#!/usr/bin/env python3
"""
Generate regime-stratified data splits for advanced ML training.

This script creates additional splits based on macro regime (risk_on vs risk_off)
extracted from the domain scores and features.

Outputs:
- results/ml_dataset/train_risk_on.csv
- results/ml_dataset/train_risk_off.csv
- results/ml_dataset/test_risk_on.csv
- results/ml_dataset/test_risk_off.csv
- results/ml_dataset/stratified_splits_summary.txt
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent


class RegimeStratifier:
    """Stratify trades by detected macro regime."""

    @staticmethod
    def classify_regime_from_macros(row: pd.Series) -> str:
        """Classify regime based on macro score components."""
        # Extract macro components from normalized row
        vix_z = row.get('vix_z_score', 0.5)
        btc_vol = row.get('btc_volatility_percentile', 0.5)
        atr_pct = row.get('atr_percentile', 0.5)

        # If any macro indicator is very high (>0.7), it's risk_off
        # If all are low (<0.3), it's risk_on
        macro_avg = np.mean([vix_z, btc_vol, atr_pct])

        if macro_avg > 0.65:
            return 'risk_off'
        elif macro_avg < 0.35:
            return 'risk_on'
        else:
            return 'neutral'

    @staticmethod
    def classify_regime_from_regime_col(row: pd.Series) -> str:
        """Classify from explicit macro_regime columns if available."""
        for regime_col in ['macro_regime_risk_on', 'macro_regime_risk_off',
                          'macro_regime_neutral', 'macro_regime_crisis']:
            if regime_col in row.index and row[regime_col] == 1:
                return regime_col.replace('macro_regime_', '')
        return None


def main():
    """Generate regime-stratified splits."""
    ml_dir = Path(project_root) / 'results' / 'ml_dataset'

    print("=" * 80)
    print("REGIME-STRATIFIED SPLIT GENERATION")
    print("=" * 80)

    # Load main datasets
    print("\nLoading main datasets...")
    train_df = pd.read_csv(ml_dir / 'train.csv')
    test_df = pd.read_csv(ml_dir / 'test.csv')

    print(f"  Train set: {len(train_df)} trades")
    print(f"  Test set: {len(test_df)} trades")

    # Stratify by regime
    print("\nStratifying by macro regime...")
    stratifier = RegimeStratifier()

    # Try to load original source data to get regime classification
    source_train = '/Users/raymondghandchi/Bull-machine-/Bull-machine-/results/validation/bear_2022_TRULY_FIXED.csv'
    source_test = '/Users/raymondghandchi/Bull-machine-/Bull-machine-/results/validation/bull_2024_TRULY_FIXED.csv'

    regime_info = {}

    # Load source data to get regime columns
    if Path(source_train).exists():
        source_df = pd.read_csv(source_train)
        print(f"  Loaded source train data: {source_train}")

        # Merge regime information
        if 'entry_time' in source_df.columns and 'entry_time' in train_df.columns:
            source_df['entry_time'] = pd.to_datetime(source_df['entry_time'], utc=True)
            train_df['entry_time'] = pd.to_datetime(train_df['entry_time'], utc=True)

            # Find regime columns
            regime_cols = [col for col in source_df.columns if 'macro_regime' in col]
            if regime_cols:
                # Get regime classification for each trade
                for idx, row in train_df.iterrows():
                    matching = source_df[source_df['entry_time'] == row['entry_time']]
                    if not matching.empty:
                        # Use the regime columns to determine the regime
                        source_row = matching.iloc[0]
                        regime = stratifier.classify_regime_from_regime_col(source_row)
                        if regime:
                            regime_info[idx] = regime
                        else:
                            regime_info[idx] = 'neutral'
                    else:
                        regime_info[idx] = 'neutral'

    if not regime_info:
        # Fall back to using macro score
        for idx, row in train_df.iterrows():
            regime_info[idx] = stratifier.classify_regime_from_macros(row)

    train_df['detected_regime'] = train_df.index.map(regime_info)
    print(f"  Train regime distribution:\n{train_df['detected_regime'].value_counts()}")

    # Same for test
    regime_info_test = {}
    if Path(source_test).exists():
        source_df = pd.read_csv(source_test)
        print(f"  Loaded source test data: {source_test}")

        if 'entry_time' in source_df.columns and 'entry_time' in test_df.columns:
            source_df['entry_time'] = pd.to_datetime(source_df['entry_time'], utc=True)
            test_df['entry_time'] = pd.to_datetime(test_df['entry_time'], utc=True)

            regime_cols = [col for col in source_df.columns if 'macro_regime' in col]
            if regime_cols:
                for idx, row in test_df.iterrows():
                    matching = source_df[source_df['entry_time'] == row['entry_time']]
                    if not matching.empty:
                        source_row = matching.iloc[0]
                        regime = stratifier.classify_regime_from_regime_col(source_row)
                        if regime:
                            regime_info_test[idx] = regime
                        else:
                            regime_info_test[idx] = 'neutral'
                    else:
                        regime_info_test[idx] = 'neutral'

    if not regime_info_test:
        for idx, row in test_df.iterrows():
            regime_info_test[idx] = stratifier.classify_regime_from_macros(row)

    test_df['detected_regime'] = test_df.index.map(regime_info_test)
    print(f"  Test regime distribution:\n{test_df['detected_regime'].value_counts()}")

    # Create splits
    print("\nCreating stratified splits...")

    splits_info = {}

    # Train splits
    train_risk_on = train_df[train_df['detected_regime'] == 'risk_on'].drop(columns=['detected_regime'])
    train_risk_off = train_df[train_df['detected_regime'].isin(['risk_off', 'crisis'])].drop(columns=['detected_regime'])

    # Test splits
    test_risk_on = test_df[test_df['detected_regime'] == 'risk_on'].drop(columns=['detected_regime'])
    test_risk_off = test_df[test_df['detected_regime'].isin(['risk_off', 'crisis'])].drop(columns=['detected_regime'])

    print(f"  Train risk_on: {len(train_risk_on)} trades")
    print(f"  Train risk_off: {len(train_risk_off)} trades")
    print(f"  Test risk_on: {len(test_risk_on)} trades")
    print(f"  Test risk_off: {len(test_risk_off)} trades")

    splits_info['train_risk_on'] = len(train_risk_on)
    splits_info['train_risk_off'] = len(train_risk_off)
    splits_info['test_risk_on'] = len(test_risk_on)
    splits_info['test_risk_off'] = len(test_risk_off)

    # Save splits
    print("\nSaving stratified splits...")

    if len(train_risk_on) > 0:
        train_risk_on.to_csv(ml_dir / 'train_risk_on.csv', index=False)
        print(f"  Saved: train_risk_on.csv")

    if len(train_risk_off) > 0:
        train_risk_off.to_csv(ml_dir / 'train_risk_off.csv', index=False)
        print(f"  Saved: train_risk_off.csv")

    if len(test_risk_on) > 0:
        test_risk_on.to_csv(ml_dir / 'test_risk_on.csv', index=False)
        print(f"  Saved: test_risk_on.csv")

    if len(test_risk_off) > 0:
        test_risk_off.to_csv(ml_dir / 'test_risk_off.csv', index=False)
        print(f"  Saved: test_risk_off.csv")

    # Generate summary
    print("\nGenerating summary...")
    summary_path = ml_dir / 'stratified_splits_summary.txt'

    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("REGIME-STRATIFIED SPLITS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        f.write("SPLIT COUNTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Train risk_on:  {splits_info['train_risk_on']:3d} trades\n")
        f.write(f"  Train risk_off: {splits_info['train_risk_off']:3d} trades\n")
        f.write(f"  Test risk_on:   {splits_info['test_risk_on']:3d} trades\n")
        f.write(f"  Test risk_off:  {splits_info['test_risk_off']:3d} trades\n\n")

        f.write("USE CASES\n")
        f.write("-" * 80 + "\n")
        f.write("  train_risk_on.csv:  Train engine weights for bull/rally markets\n")
        f.write("  train_risk_off.csv: Train engine weights for bear/correction markets\n")
        f.write("  test_risk_on.csv:   Test in bullish market conditions\n")
        f.write("  test_risk_off.csv:  Test in bearish market conditions\n\n")

        f.write("PERFORMANCE COMPARISON\n")
        f.write("-" * 80 + "\n")

        if len(train_risk_on) > 0:
            wr_on = train_risk_on['trade_won'].mean()
            rmult_on = train_risk_on['r_multiple'].mean()
            f.write(f"  Train risk_on:  WR={wr_on:5.1%}  Avg R={rmult_on:+.3f}\n")

        if len(train_risk_off) > 0:
            wr_off = train_risk_off['trade_won'].mean()
            rmult_off = train_risk_off['r_multiple'].mean()
            f.write(f"  Train risk_off: WR={wr_off:5.1%}  Avg R={rmult_off:+.3f}\n")

        if len(test_risk_on) > 0:
            wr_on = test_risk_on['trade_won'].mean()
            rmult_on = test_risk_on['r_multiple'].mean()
            f.write(f"  Test risk_on:   WR={wr_on:5.1%}  Avg R={rmult_on:+.3f}\n")

        if len(test_risk_off) > 0:
            wr_off = test_risk_off['trade_won'].mean()
            rmult_off = test_risk_off['r_multiple'].mean()
            f.write(f"  Test risk_off:  WR={wr_off:5.1%}  Avg R={rmult_off:+.3f}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"  Saved summary: {summary_path}")

    print("\n" + "=" * 80)
    print("REGIME-STRATIFIED SPLITS COMPLETE")
    print("=" * 80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
