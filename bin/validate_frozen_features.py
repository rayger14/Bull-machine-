#!/usr/bin/env python3
"""
Comprehensive Validation Script for All Frozen Feature Fixes
=============================================================

Validates that all three critical fixes are working:
1. MTF Fusion Scores (tf1h/tf4h/tf1d_fusion_score)
2. PTI Scores (tf1h/tf1d_pti_score, tf1h_pti_confidence)
3. SMC Fusion (fusion_smc)

Exit codes:
  0 - All validations passed
  1 - One or more validations failed
  2 - Critical error (file not found, etc.)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import argparse
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FrozenFeatureValidator:
    """Validates that frozen features are now varying properly"""

    def __init__(self, feature_store_path: str):
        self.feature_store_path = feature_store_path
        self.df = None
        self.validation_results = {}

    def load_feature_store(self) -> bool:
        """Load feature store for validation"""
        try:
            logger.info(f"Loading feature store: {self.feature_store_path}")
            self.df = pd.read_parquet(self.feature_store_path)
            logger.info(f"  Loaded {len(self.df):,} bars, {len(self.df.columns)} columns")
            logger.info(f"  Date range: {self.df.index.min()} to {self.df.index.max()}")
            return True
        except Exception as e:
            logger.error(f"Failed to load feature store: {e}")
            return False

    def validate_feature(self, col: str, min_unique: int, expected_range: Tuple[float, float],
                        frozen_value: float = None) -> Dict:
        """
        Validate a single feature for variance.

        Args:
            col: Column name
            min_unique: Minimum expected unique values
            expected_range: (min, max) expected value range
            frozen_value: If set, fail if feature is frozen at this value

        Returns:
            Dict with validation results
        """
        if col not in self.df.columns:
            return {
                'status': 'MISSING',
                'message': f"Column {col} not found in feature store",
                'passed': False
            }

        vals = self.df[col].dropna()
        if len(vals) == 0:
            return {
                'status': 'EMPTY',
                'message': f"Column {col} has no non-NaN values",
                'passed': False
            }

        unique_count = vals.nunique()
        mean_val = vals.mean()
        std_val = vals.std()
        min_val = vals.min()
        max_val = vals.max()

        # Check if frozen at specific value
        if frozen_value is not None and unique_count == 1 and abs(vals.iloc[0] - frozen_value) < 1e-6:
            return {
                'status': 'FROZEN',
                'message': f"Column {col} is frozen at {frozen_value}",
                'unique_count': unique_count,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'passed': False
            }

        # Check if has sufficient variance
        if unique_count < min_unique:
            return {
                'status': 'LOW_VARIANCE',
                'message': f"Column {col} has only {unique_count} unique values (expected >= {min_unique})",
                'unique_count': unique_count,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'passed': False
            }

        # Check if values are in expected range
        if min_val < expected_range[0] or max_val > expected_range[1]:
            return {
                'status': 'OUT_OF_RANGE',
                'message': f"Column {col} values outside expected range {expected_range}: [{min_val:.4f}, {max_val:.4f}]",
                'unique_count': unique_count,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'passed': False
            }

        # Check if standard deviation is too low (near-frozen)
        if std_val < 0.001:
            return {
                'status': 'NEAR_FROZEN',
                'message': f"Column {col} has very low variance (std={std_val:.6f})",
                'unique_count': unique_count,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'passed': False
            }

        # All checks passed
        return {
            'status': 'OK',
            'message': f"Column {col} is varying properly",
            'unique_count': unique_count,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'passed': True
        }

    def validate_mtf_fusion_scores(self) -> bool:
        """Validate MTF Fusion Scores are no longer frozen at 0.5"""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATING MTF FUSION SCORES")
        logger.info("=" * 80)

        mtf_features = {
            'tf1h_fusion_score': {
                'min_unique': 100,
                'expected_range': (0.0, 1.0),
                'frozen_value': 0.5
            },
            'tf4h_fusion_score': {
                'min_unique': 100,
                'expected_range': (0.0, 1.0),
                'frozen_value': 0.5
            },
            'tf1d_fusion_score': {
                'min_unique': 50,
                'expected_range': (0.0, 1.0),
                'frozen_value': 0.5
            },
            'k2_fusion_score': {
                'min_unique': 100,
                'expected_range': (0.0, 1.0),
                'frozen_value': 0.5
            }
        }

        all_passed = True
        for col, params in mtf_features.items():
            result = self.validate_feature(col, **params)
            self.validation_results[col] = result

            status_emoji = "✅" if result['passed'] else "❌"
            logger.info(f"\n{status_emoji} {col}:")
            logger.info(f"   Status: {result['status']}")
            logger.info(f"   Message: {result['message']}")
            if 'unique_count' in result:
                logger.info(f"   Unique values: {result['unique_count']:,}")
                logger.info(f"   Mean: {result['mean']:.6f}")
                logger.info(f"   Std:  {result['std']:.6f}")
                logger.info(f"   Range: [{result['min']:.6f}, {result['max']:.6f}]")

            if not result['passed']:
                all_passed = False

        return all_passed

    def validate_pti_scores(self) -> bool:
        """Validate PTI Scores are no longer frozen at 0.5"""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATING PTI SCORES")
        logger.info("=" * 80)

        pti_features = {
            'tf1h_pti_score': {
                'min_unique': 100,
                'expected_range': (0.0, 1.0),
                'frozen_value': 0.5
            },
            'tf1h_pti_confidence': {
                'min_unique': 100,
                'expected_range': (0.0, 1.0),
                'frozen_value': 0.5
            },
            'tf1d_pti_score': {
                'min_unique': 50,
                'expected_range': (0.0, 1.0),
                'frozen_value': 0.5
            }
        }

        all_passed = True
        for col, params in pti_features.items():
            result = self.validate_feature(col, **params)
            self.validation_results[col] = result

            status_emoji = "✅" if result['passed'] else "❌"
            logger.info(f"\n{status_emoji} {col}:")
            logger.info(f"   Status: {result['status']}")
            logger.info(f"   Message: {result['message']}")
            if 'unique_count' in result:
                logger.info(f"   Unique values: {result['unique_count']:,}")
                logger.info(f"   Mean: {result['mean']:.6f}")
                logger.info(f"   Std:  {result['std']:.6f}")
                logger.info(f"   Range: [{result['min']:.6f}, {result['max']:.6f}]")

            if not result['passed']:
                all_passed = False

        return all_passed

    def validate_smc_fusion(self) -> bool:
        """Validate SMC Fusion is no longer frozen (should have 1000+ unique values)"""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATING SMC FUSION")
        logger.info("=" * 80)

        result = self.validate_feature(
            'fusion_smc',
            min_unique=1000,  # Should have 1000+ unique values from continuous calculation
            expected_range=(0.0, 1.0),
            frozen_value=None  # Not frozen at specific value, just low variance
        )
        self.validation_results['fusion_smc'] = result

        status_emoji = "✅" if result['passed'] else "❌"
        logger.info(f"\n{status_emoji} fusion_smc:")
        logger.info(f"   Status: {result['status']}")
        logger.info(f"   Message: {result['message']}")
        if 'unique_count' in result:
            logger.info(f"   Unique values: {result['unique_count']:,}")
            logger.info(f"   Mean: {result['mean']:.6f}")
            logger.info(f"   Std:  {result['std']:.6f}")
            logger.info(f"   Range: [{result['min']:.6f}, {result['max']:.6f}]")

        return result['passed']

    def validate_dependencies(self) -> bool:
        """Validate that underlying features are also working"""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATING FEATURE DEPENDENCIES")
        logger.info("=" * 80)

        # Check SMC component scores (should be continuous)
        smc_components = {
            'smc_strength': {
                'min_unique': 100,
                'expected_range': (0.0, 1.0),
                'frozen_value': None
            },
            'smc_confidence': {
                'min_unique': 100,
                'expected_range': (0.0, 1.0),
                'frozen_value': None
            },
            'smc_confluence': {
                'min_unique': 50,
                'expected_range': (0.0, 1.0),
                'frozen_value': None
            }
        }

        all_passed = True
        for col, params in smc_components.items():
            if col not in self.df.columns:
                logger.warning(f"⚠️  {col}: MISSING (optional dependency)")
                continue

            result = self.validate_feature(col, **params)
            status_emoji = "✅" if result['passed'] else "❌"
            logger.info(f"\n{status_emoji} {col}:")
            logger.info(f"   Unique: {result['unique_count']:,}, Mean: {result['mean']:.4f}, Std: {result['std']:.4f}")

            if not result['passed']:
                all_passed = False

        return all_passed

    def print_summary(self) -> bool:
        """Print final summary of all validations"""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)

        passed_count = sum(1 for r in self.validation_results.values() if r['passed'])
        total_count = len(self.validation_results)

        logger.info(f"\nTotal features validated: {total_count}")
        logger.info(f"Passed: {passed_count}")
        logger.info(f"Failed: {total_count - passed_count}")

        if passed_count == total_count:
            logger.info("\n" + "=" * 80)
            logger.info("✅ ALL VALIDATIONS PASSED!")
            logger.info("=" * 80)
            logger.info("\nAll frozen features are now varying properly.")
            logger.info("Feature store is ready for backtesting.")
            return True
        else:
            logger.error("\n" + "=" * 80)
            logger.error("❌ VALIDATION FAILED")
            logger.error("=" * 80)
            logger.error(f"\n{total_count - passed_count} feature(s) still frozen or invalid:")

            for col, result in self.validation_results.items():
                if not result['passed']:
                    logger.error(f"  - {col}: {result['status']} - {result['message']}")

            return False

    def run_all_validations(self) -> bool:
        """Run all validation checks"""
        if not self.load_feature_store():
            return False

        mtf_passed = self.validate_mtf_fusion_scores()
        pti_passed = self.validate_pti_scores()
        smc_passed = self.validate_smc_fusion()
        dep_passed = self.validate_dependencies()

        return self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description='Validate all frozen feature fixes'
    )
    parser.add_argument(
        '--feature-store',
        type=str,
        required=True,
        help='Path to feature store parquet file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run validation
    validator = FrozenFeatureValidator(args.feature_store)
    success = validator.run_all_validations()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
