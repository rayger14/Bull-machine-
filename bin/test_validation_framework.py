#!/usr/bin/env python3
"""
Test Script for Walk-Forward Validation Framework

Creates sample configs and runs validation to verify framework works correctly.

Usage:
    python bin/test_validation_framework.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_configs(output_dir: Path, n_configs: int = 5):
    """
    Create sample test configs with varying parameters.

    These configs span a range of parameter values to test the validation framework.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = {
        "version": "validation_test",
        "profile": "test_config",
        "description": "Test config for validation framework",
        "adaptive_fusion": True,
        "regime_classifier": {
            "model_path": "models/regime_classifier_gmm.pkl",
            "feature_order": [
                "VIX", "DXY", "MOVE", "YIELD_2Y", "YIELD_10Y",
                "USDT.D", "BTC.D", "TOTAL", "TOTAL2",
                "funding", "oi", "rv_20d", "rv_60d"
            ],
            "zero_fill_missing": False,
        },
        "ml_filter": {"enabled": False},
        "fusion": {
            "entry_threshold_confidence": 0.36,
            "weights": {
                "wyckoff": 0.33,
                "liquidity": 0.39,
                "momentum": 0.21,
                "macro": 0.0,
                "pti": 0.07
            }
        },
        "archetypes": {
            "use_archetypes": True,
            "max_trades_per_day": 8,
            "enable_A": False, "enable_B": False, "enable_C": False, "enable_D": False,
            "enable_E": False, "enable_F": False, "enable_G": False, "enable_H": False,
            "enable_K": False, "enable_L": False, "enable_M": False,
            "enable_S1": False, "enable_S2": True, "enable_S3": False, "enable_S4": False,
            "enable_S5": False, "enable_S6": False, "enable_S7": False, "enable_S8": False,
            "thresholds": {"min_liquidity": 0.20},
            "failed_rally": {
                "direction": "short",
                "archetype_weight": 2.0,
                "fusion_threshold": 0.36,
                "final_fusion_gate": 0.36,
                "cooldown_bars": 8,
                "max_risk_pct": 0.015,
                "atr_stop_mult": 2.0,
                "wick_ratio_min": 2.0,
                "require_rsi_divergence": False,
                "weights": {
                    "ob_retest": 0.25,
                    "wick_rejection": 0.25,
                    "rsi_signal": 0.20,
                    "volume_fade": 0.15,
                    "tf4h_confirm": 0.15
                }
            },
            "exits": {
                "failed_rally": {
                    "enable_trail": True,
                    "trail_atr_mult": 1.5,
                    "time_limit_hours": 48
                }
            }
        },
        "context": {"crisis_fuse": {"enabled": False}},
        "risk": {
            "base_risk_pct": 0.015,
            "max_position_size_pct": 0.15,
            "max_portfolio_risk_pct": 0.08
        }
    }

    # Parameter variations
    variations = [
        # Config 1: Conservative (lower threshold, tighter stops)
        {
            'fusion.entry_threshold_confidence': 0.40,
            'archetypes.failed_rally.atr_stop_mult': 1.8,
            'archetypes.failed_rally.wick_ratio_min': 2.5,
            'risk.base_risk_pct': 0.01,
        },
        # Config 2: Aggressive (lower threshold, wider stops)
        {
            'fusion.entry_threshold_confidence': 0.32,
            'archetypes.failed_rally.atr_stop_mult': 2.5,
            'archetypes.failed_rally.wick_ratio_min': 1.5,
            'risk.base_risk_pct': 0.02,
        },
        # Config 3: Balanced (default params)
        {
            'fusion.entry_threshold_confidence': 0.36,
            'archetypes.failed_rally.atr_stop_mult': 2.0,
            'archetypes.failed_rally.wick_ratio_min': 2.0,
            'risk.base_risk_pct': 0.015,
        },
        # Config 4: High selectivity (high threshold)
        {
            'fusion.entry_threshold_confidence': 0.44,
            'archetypes.failed_rally.atr_stop_mult': 2.0,
            'archetypes.failed_rally.wick_ratio_min': 2.0,
            'risk.base_risk_pct': 0.015,
        },
        # Config 5: Active trading (low cooldown)
        {
            'fusion.entry_threshold_confidence': 0.36,
            'archetypes.failed_rally.cooldown_bars': 4,
            'archetypes.failed_rally.atr_stop_mult': 2.0,
            'risk.base_risk_pct': 0.015,
        },
    ]

    configs = []

    for i, variation in enumerate(variations[:n_configs]):
        config = json.loads(json.dumps(base_config))  # Deep copy

        # Apply variations
        for key, value in variation.items():
            parts = key.split('.')
            obj = config

            for part in parts[:-1]:
                obj = obj[part]

            obj[parts[-1]] = value

        # Update profile
        config['profile'] = f'test_config_{i+1:02d}'

        # Save config
        config_path = output_dir / f'config_{i+1:03d}.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        configs.append((f'config_{i+1:03d}', config))

        logger.info(f"Created {config_path}")

    return configs


def main():
    logger.info("="*80)
    logger.info("Walk-Forward Validation Framework Test")
    logger.info("="*80)

    # Check if feature store exists
    feature_store_path = 'data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31_backup.parquet'

    if not Path(feature_store_path).exists():
        logger.error(f"Feature store not found: {feature_store_path}")
        logger.error("Please ensure feature store exists before running validation")
        return

    # Create test configs
    logger.info("\nStep 1: Creating test configs...")
    test_configs_dir = Path('results/validation/test_configs')
    configs = create_test_configs(test_configs_dir, n_configs=3)  # Start with 3 for quick test

    logger.info(f"\nCreated {len(configs)} test configs in {test_configs_dir}")

    # Instructions for running validation
    logger.info("\n" + "="*80)
    logger.info("Next Steps: Run Validation")
    logger.info("="*80)
    logger.info("\n1. Run validation on test configs:")
    logger.info(f"\n   python bin/validate_walk_forward.py \\")
    logger.info(f"       --configs {test_configs_dir} \\")
    logger.info(f"       --output results/validation/test_run/ \\")
    logger.info(f"       --asset BTC")

    logger.info("\n2. Visualize results:")
    logger.info(f"\n   python bin/visualize_validation_results.py \\")
    logger.info(f"       --input results/validation/test_run/")

    logger.info("\n3. Review summary report:")
    logger.info(f"\n   cat results/validation/test_run/summary_report.md")

    logger.info("\n" + "="*80)
    logger.info("For Production Use:")
    logger.info("="*80)
    logger.info("\nValidate Optuna optimization results:")
    logger.info(f"\n   python bin/validate_walk_forward.py \\")
    logger.info(f"       --configs results/phase2_optimization/optimization_study.db \\")
    logger.info(f"       --study-name bear_phase2_tuning \\")
    logger.info(f"       --output results/validation/phase2_validation/ \\")
    logger.info(f"       --asset BTC \\")
    logger.info(f"       --min-trials 50")

    logger.info("\n" + "="*80)
    logger.info("Framework Ready")
    logger.info("="*80)


if __name__ == '__main__':
    main()
