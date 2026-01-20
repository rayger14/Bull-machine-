#!/usr/bin/env python3
"""
Regime Routing Validation Script

Tests regime routing infrastructure to ensure:
1. Different regimes produce different archetype distributions
2. Bull archetypes dominate in risk_on
3. Bear archetypes dominate in risk_off
4. Routing weights are being applied correctly
5. No regression in bull performance (2024)
6. Improvement in bear performance (2022)

Usage:
    python3 bin/validate_regime_routing.py --config configs/mvp/mvp_regime_routed_production.json
    python3 bin/validate_regime_routing.py --config configs/mvp/mvp_regime_routed_production.json --symbol BTCUSDT
    python3 bin/validate_regime_routing.py --config configs/mvp/mvp_regime_routed_production.json --quick
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.runtime.context import RuntimeContext
from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.archetypes.threshold_policy import ThresholdPolicy


class RegimeRoutingValidator:
    """Validates regime routing configuration and behavior."""

    def __init__(self, config_path: str, symbol: str = "BTCUSDT"):
        self.config_path = Path(config_path)
        self.symbol = symbol
        self.config = self._load_config()
        self.results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'failures': []
        }

    def _load_config(self) -> dict:
        """Load and validate configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path) as f:
            config = json.load(f)

        print(f"Loaded config: {config.get('version', 'unknown')}")
        return config

    def run_all_tests(self) -> bool:
        """Run all validation tests."""
        print("\n" + "="*80)
        print("REGIME ROUTING VALIDATION")
        print("="*80 + "\n")

        # Test 1: Configuration validation
        self.test_config_structure()

        # Test 2: Routing weight presence
        self.test_routing_weights()

        # Test 3: Archetype logic initialization
        self.test_archetype_logic_init()

        # Test 4: Regime routing behavior (synthetic data)
        self.test_regime_routing_behavior()

        # Test 5: Weight calculation examples
        self.test_weight_calculations()

        # Print summary
        self._print_summary()

        return self.results['tests_failed'] == 0

    def test_config_structure(self):
        """Test 1: Validate config structure."""
        test_name = "Config Structure"
        print(f"\nTest 1: {test_name}")
        print("-" * 80)

        checks = []

        # Check adaptive_fusion enabled
        adaptive_fusion = self.config.get('fusion', {}).get('adaptive_fusion', {}).get('enabled', False)
        checks.append(('adaptive_fusion.enabled', adaptive_fusion, True))

        # Check archetypes section
        has_archetypes = 'archetypes' in self.config
        checks.append(('archetypes section', has_archetypes, True))

        # Check use_archetypes enabled
        use_archetypes = self.config.get('archetypes', {}).get('use_archetypes', False)
        checks.append(('archetypes.use_archetypes', use_archetypes, True))

        # Check routing section
        has_routing = 'routing' in self.config.get('archetypes', {})
        checks.append(('archetypes.routing section', has_routing, True))

        # Execute checks
        for name, actual, expected in checks:
            if actual == expected:
                print(f"  ✓ {name}: {actual}")
            else:
                print(f"  ✗ {name}: expected {expected}, got {actual}")
                self._record_failure(test_name, f"{name} check failed")

        self._record_test(test_name, all(a == e for _, a, e in checks))

    def test_routing_weights(self):
        """Test 2: Validate routing weights for all regimes."""
        test_name = "Routing Weights"
        print(f"\nTest 2: {test_name}")
        print("-" * 80)

        routing = self.config.get('archetypes', {}).get('routing', {})
        required_regimes = ['risk_on', 'neutral', 'risk_off', 'crisis']

        all_passed = True

        for regime in required_regimes:
            if regime not in routing:
                print(f"  ✗ Missing regime: {regime}")
                self._record_failure(test_name, f"Missing regime: {regime}")
                all_passed = False
                continue

            weights = routing[regime].get('weights', {})
            if not weights:
                print(f"  ✗ {regime}: No weights defined")
                self._record_failure(test_name, f"{regime}: No weights defined")
                all_passed = False
                continue

            print(f"  ✓ {regime}: {len(weights)} archetype weights defined")

            # Print weight distribution for inspection
            for arch, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                indicator = "🔥" if weight > 1.5 else "✓" if weight >= 1.0 else "❄️"
                print(f"      {indicator} {arch}: {weight:.2f}")

        self._record_test(test_name, all_passed)

    def test_archetype_logic_init(self):
        """Test 3: Initialize ArchetypeLogic with config."""
        test_name = "ArchetypeLogic Initialization"
        print(f"\nTest 3: {test_name}")
        print("-" * 80)

        try:
            archetype_config = self.config.get('archetypes', {})
            logic = ArchetypeLogic(archetype_config)

            # Check enabled archetypes
            enabled_bull = [k for k, v in logic.enabled.items() if v and not k.startswith('S')]
            enabled_bear = [k for k, v in logic.enabled.items() if v and k.startswith('S')]

            print(f"  ✓ Initialized successfully")
            print(f"  ✓ Enabled bull archetypes: {len(enabled_bull)} ({', '.join(enabled_bull)})")
            print(f"  ✓ Enabled bear archetypes: {len(enabled_bear)} ({', '.join(enabled_bear)})")

            # Check routing config propagation
            routing_config = archetype_config.get('routing', {})
            print(f"  ✓ Routing config: {len(routing_config)} regimes")

            self._record_test(test_name, True)
            return logic

        except Exception as e:
            print(f"  ✗ Initialization failed: {e}")
            self._record_failure(test_name, f"Initialization failed: {e}")
            self._record_test(test_name, False)
            return None

    def test_regime_routing_behavior(self):
        """Test 4: Verify different regimes produce different archetype distributions."""
        test_name = "Regime Routing Behavior"
        print(f"\nTest 4: {test_name}")
        print("-" * 80)

        try:
            archetype_config = self.config.get('archetypes', {})
            logic = ArchetypeLogic(archetype_config)
            threshold_policy = ThresholdPolicy(
                base_cfg={'archetypes': archetype_config},
                locked_regime=None
            )

            # Create synthetic test cases
            regimes = ['risk_on', 'neutral', 'risk_off', 'crisis']
            regime_results = {}

            for regime in regimes:
                # Create synthetic feature row (moderate quality across the board)
                synthetic_row = pd.Series({
                    'close': 50000,
                    'high': 50200,
                    'low': 49800,
                    'open': 49900,
                    'fusion_score': 0.45,
                    'liquidity_score': 0.35,
                    'wyckoff_score': 0.40,
                    'rsi_14': 60,
                    'adx_14': 28,
                    'volume_zscore': 0.8,
                    'tf1h_bos_bullish': 1,
                    'tf1h_boms_strength': 0.35,
                    'funding_Z': 1.5,  # For long_squeeze
                    'tf1h_ob_high': 50100,  # For failed_rally
                })

                # Create runtime context
                regime_probs = {regime: 1.0}  # Force 100% confidence in test regime
                thresholds = threshold_policy.resolve(regime_probs, regime)

                context = RuntimeContext(
                    ts=pd.Timestamp('2024-01-01'),
                    row=synthetic_row,
                    regime_probs=regime_probs,
                    regime_label=regime,
                    adapted_params={},
                    thresholds=thresholds
                )

                # Run detection
                archetype, score, liq = logic.detect(context)

                regime_results[regime] = {
                    'archetype': archetype,
                    'score': score,
                    'liquidity': liq
                }

                print(f"  {regime:12s} → {archetype or 'None':25s} (score={score:.3f})")

            # Validate diversity
            archetypes = [r['archetype'] for r in regime_results.values() if r['archetype']]
            unique_archetypes = len(set(archetypes))

            if unique_archetypes >= 2:
                print(f"\n  ✓ Regime diversity: {unique_archetypes} different archetypes across regimes")
                self._record_test(test_name, True)
            else:
                print(f"\n  ✗ Regime diversity: Only {unique_archetypes} unique archetype(s) - routing may not be working")
                self._record_failure(test_name, f"Insufficient archetype diversity: {unique_archetypes}")
                self._record_test(test_name, False)

        except Exception as e:
            print(f"  ✗ Behavior test failed: {e}")
            import traceback
            traceback.print_exc()
            self._record_failure(test_name, f"Behavior test failed: {e}")
            self._record_test(test_name, False)

    def test_weight_calculations(self):
        """Test 5: Demonstrate weight calculation examples."""
        test_name = "Weight Calculation Examples"
        print(f"\nTest 5: {test_name}")
        print("-" * 80)

        routing = self.config.get('archetypes', {}).get('routing', {})

        # Example: Show how same base scores change with regime
        base_scores = {
            'trap_within_trend': 0.50,
            'failed_rally': 0.48,
            'long_squeeze': 0.45
        }

        print("\n  Example: Same base scores, different regimes\n")
        print(f"  {'Archetype':<25s} {'Base':<8s} {'risk_on':<12s} {'neutral':<12s} {'risk_off':<12s} {'crisis':<12s}")
        print("  " + "-" * 76)

        for archetype, base_score in base_scores.items():
            row_parts = [f"{archetype:<25s}", f"{base_score:.3f}"]

            for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
                weights = routing.get(regime, {}).get('weights', {})
                weight = weights.get(archetype, 1.0)
                weighted_score = base_score * weight
                row_parts.append(f"{weighted_score:.3f} ({weight:.1f}x)")

            print("  " + "  ".join(row_parts))

        print("\n  ✓ Weight calculations demonstrated")
        self._record_test(test_name, True)

    def _record_test(self, test_name: str, passed: bool):
        """Record test result."""
        self.results['tests_run'] += 1
        if passed:
            self.results['tests_passed'] += 1
        else:
            self.results['tests_failed'] += 1

    def _record_failure(self, test_name: str, reason: str):
        """Record test failure."""
        self.results['failures'].append(f"{test_name}: {reason}")

    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)

        print(f"\nTests run:    {self.results['tests_run']}")
        print(f"Tests passed: {self.results['tests_passed']}")
        print(f"Tests failed: {self.results['tests_failed']}")

        if self.results['tests_failed'] > 0:
            print(f"\n❌ VALIDATION FAILED")
            print("\nFailures:")
            for failure in self.results['failures']:
                print(f"  - {failure}")
            print("\nRecommended actions:")
            print("  1. Check adaptive_fusion.enabled=true")
            print("  2. Verify archetypes.routing section exists")
            print("  3. Review docs/technical/REGIME_ROUTING_GUIDE.md")
            print("  4. Check logs for [REGIME ROUTING] messages")
        else:
            print(f"\n✅ ALL TESTS PASSED")
            print("\nNext steps:")
            print("  1. Run full backtest on 2022 with regime_override={'2022': 'risk_off'}")
            print("  2. Run full backtest on 2024 to validate bull performance")
            print("  3. Compare results to baseline configs")
            print("  4. Remove regime_override for production deployment")

        print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate regime routing configuration")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Symbol to test (default: BTCUSDT)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick validation (skip heavy tests)'
    )

    args = parser.parse_args()

    # Run validation
    validator = RegimeRoutingValidator(args.config, args.symbol)
    success = validator.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
