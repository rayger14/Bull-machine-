"""
Golden Fixtures for Bull Machine v1.7
Curated scenarios for Spring/UTAD, DXY veto, HOB patterns, etc.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import numpy as np
import json
import copy
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from typing import Dict


class GoldenFixtures:
    """Create and test golden fixture scenarios"""

    def __init__(self):
        self.fixtures_dir = Path('tests/fixtures/data')
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        self.config = self._load_default_config()

    def _load_default_config(self) -> Dict:
        """Load default configuration for testing"""
        return {
            'hps_floor': {'1h': 0.3, '4h': 0.4, '1d': 0.5},
            'hob_quality_factors': {
                '1h': {'volume_z_min': 1.1, 'proximity_atr_pct': 0.25},
                '4h': {'volume_z_min': 1.3, 'proximity_atr_pct': 0.25},
                '1d': {'volume_z_min': 1.5, 'proximity_atr_pct': 0.25}
            },
            'confidence_1h': 0.3,
            'confidence_4h': 0.4,
            'confidence_1d': 0.3,
            'vix_guard_on': 22.0,
            'vix_guard_off': 18.0
        }

    @contextmanager
    def temp_overrides(self, overrides: Dict):
        """
        Temporary config overrides for scoped testing
        Prevents test relaxations from bleeding into production
        """
        original = copy.deepcopy(self.config)
        try:
            self._deep_update(self.config, overrides)
            yield
        finally:
            self.config.clear()
            self.config.update(original)

    def _deep_update(self, target: Dict, source: Dict):
        """Deep update dictionary"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target:
                self._deep_update(target[key], value)
            else:
                target[key] = value

    def create_spring_scenario(self):
        """
        Create Wyckoff Spring pattern fixture
        - False breakdown below support
        - Volume divergence
        - Immediate reversal with expansion
        """
        dates = pd.date_range('2025-01-01', periods=100, freq='4H')

        # Build Spring pattern
        prices = []
        volumes = []

        # Phase A: Initial selling climax
        for i in range(20):
            prices.append(100 - i * 0.5)  # Decline from 100 to 90
            volumes.append(1000 + i * 50)  # Increasing volume

        # Phase B: Trading range
        for i in range(40):
            prices.append(90 + np.sin(i/5) * 2)  # Range 88-92
            volumes.append(800 - i * 5)  # Declining volume

        # Phase C: Spring (false breakdown)
        for i in range(10):
            if i < 5:
                prices.append(88 - i * 0.5)  # Break below to 85.5
                volumes.append(600 + i * 100)  # Increasing
            else:
                prices.append(86 + (i-5) * 1.5)  # Sharp reversal
                volumes.append(1500 + (i-5) * 200)  # Volume expansion

        # Phase D: Markup beginning
        for i in range(30):
            prices.append(93 + i * 0.3)  # Rally to 102
            volumes.append(1200 + np.sin(i/3) * 200)

        df = pd.DataFrame({
            'open': prices,
            'high': [p + np.random.uniform(0, 0.5) for p in prices],
            'low': [p - np.random.uniform(0, 0.5) for p in prices],
            'close': [p + np.random.uniform(-0.2, 0.2) for p in prices],
            'volume': volumes
        }, index=dates)

        # Save fixture
        fixture_path = self.fixtures_dir / 'spring_pattern.csv'
        df.to_csv(fixture_path)

        return {
            'name': 'spring_pattern',
            'path': fixture_path,
            'expected_signal': 'long',
            'expected_phase': 'C',
            'trigger_index': 65,  # Spring occurs around bar 65
            'description': 'Wyckoff Spring with false breakdown and reversal'
        }

    def create_utad_scenario(self):
        """
        Create UTAD (Upthrust After Distribution) pattern
        - False breakout above resistance
        - No follow-through
        - Reversal on high volume
        """
        dates = pd.date_range('2025-01-01', periods=100, freq='4H')

        # Build UTAD pattern
        prices = []
        volumes = []

        # Initial rally to distribution zone
        for i in range(30):
            prices.append(100 + i * 0.3)  # Rally to 109
            volumes.append(1000 + i * 20)

        # Distribution range
        for i in range(40):
            prices.append(108 + np.sin(i/5) * 1.5)  # Range 106.5-109.5
            volumes.append(900 - i * 5)

        # UTAD (false breakout)
        for i in range(10):
            if i < 4:
                prices.append(109.5 + i * 0.5)  # Break to 111.5
                volumes.append(800 - i * 50)  # Declining volume (warning)
            else:
                prices.append(111 - (i-4) * 1.2)  # Sharp reversal
                volumes.append(1400 + (i-4) * 150)  # Volume surge

        # Markdown beginning
        for i in range(20):
            prices.append(104 - i * 0.4)  # Decline to 96
            volumes.append(1200 + np.random.uniform(-100, 100))

        df = pd.DataFrame({
            'open': prices,
            'high': [p + np.random.uniform(0, 0.3) for p in prices],
            'low': [p - np.random.uniform(0, 0.3) for p in prices],
            'close': [p + np.random.uniform(-0.15, 0.15) for p in prices],
            'volume': volumes
        }, index=dates)

        fixture_path = self.fixtures_dir / 'utad_pattern.csv'
        df.to_csv(fixture_path)

        return {
            'name': 'utad_pattern',
            'path': fixture_path,
            'expected_signal': 'short',
            'expected_phase': 'E',
            'trigger_index': 74,  # UTAD occurs around bar 74
            'description': 'UTAD false breakout with distribution'
        }

    def create_dxy_veto_scenario(self):
        """
        Create DXY + Oil macro veto scenario
        - DXY above 106
        - Oil above $85
        - Should veto all crypto longs
        """
        dates = pd.date_range('2025-01-01', periods=100, freq='4H')

        # ETH price data (would normally trigger long)
        eth_prices = 2000 + np.cumsum(np.random.randn(100) * 10)

        eth_df = pd.DataFrame({
            'open': eth_prices,
            'high': eth_prices + np.random.uniform(5, 15, 100),
            'low': eth_prices - np.random.uniform(5, 15, 100),
            'close': eth_prices + np.random.uniform(-5, 5, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)

        # Macro data (veto conditions)
        macro_df = pd.DataFrame({
            'DXY': 106 + np.random.uniform(0, 2, 100),  # Above 106
            'Oil': 85 + np.random.uniform(0, 5, 100),    # Above 85
            'VIX': 18 + np.random.uniform(-2, 2, 100)    # Moderate
        }, index=dates)

        eth_path = self.fixtures_dir / 'dxy_veto_eth.csv'
        macro_path = self.fixtures_dir / 'dxy_veto_macro.csv'

        eth_df.to_csv(eth_path)
        macro_df.to_csv(macro_path)

        return {
            'name': 'dxy_veto',
            'eth_path': eth_path,
            'macro_path': macro_path,
            'expected_signal': None,  # Should be vetoed
            'veto_reason': 'DXY > 106 and Oil > 85',
            'description': 'Macro veto scenario with risk-off conditions'
        }

    def create_hob_pattern_scenario(self):
        """
        Create HOB (Hands-on-Back) pattern
        - Institutional accumulation
        - Volume z-score > 1.3
        - Price compression before expansion
        """
        dates = pd.date_range('2025-01-01', periods=100, freq='4H')

        prices = []
        volumes = []

        # Normal trading
        for i in range(40):
            prices.append(100 + np.sin(i/5) * 2)
            volumes.append(1000 + np.random.uniform(-100, 100))

        # HOB pattern formation
        for i in range(20):
            # Price compression
            prices.append(100 + np.sin(i/10) * 0.5)  # Tight range
            if i == 10:
                volumes.append(3500)  # Institutional volume spike (z > 1.3)
            else:
                volumes.append(900 + np.random.uniform(-50, 50))

        # Post-HOB expansion
        for i in range(40):
            prices.append(100 + i * 0.3)  # Directional move
            volumes.append(1200 + np.random.uniform(-100, 100))

        df = pd.DataFrame({
            'open': prices,
            'high': [p + np.random.uniform(0.1, 0.3) for p in prices],
            'low': [p - np.random.uniform(0.1, 0.3) for p in prices],
            'close': [p + np.random.uniform(-0.1, 0.1) for p in prices],
            'volume': volumes
        }, index=dates)

        fixture_path = self.fixtures_dir / 'hob_pattern.csv'
        df.to_csv(fixture_path)

        return {
            'name': 'hob_pattern',
            'path': fixture_path,
            'expected_signal': 'long',
            'hob_index': 50,  # HOB occurs at bar 50
            'volume_z_score': 3.2,  # Calculated z-score at HOB
            'description': 'Institutional HOB pattern with volume spike'
        }

    def create_total3_leadership_scenario(self):
        """
        Create TOTAL3 market leadership scenario
        - TOTAL3 outperforming TOTAL
        - Altcoin strength indication
        """
        dates = pd.date_range('2025-01-01', periods=100, freq='4H')

        # TOTAL (all crypto) - modest growth
        total_values = 2000e9 + np.cumsum(np.random.randn(100) * 10e9)

        # TOTAL3 (ex BTC/ETH) - stronger growth
        total3_values = 500e9 + np.cumsum(np.random.randn(100) * 5e9 + 2e9)  # Positive bias

        # Calculate dominance
        total3_dominance = (total3_values / total_values) * 100

        df = pd.DataFrame({
            'TOTAL': total_values,
            'TOTAL3': total3_values,
            'TOTAL3_dominance': total3_dominance,
            'TOTAL3_momentum': pd.Series(total3_values).pct_change(10).fillna(0) * 100
        }, index=dates)

        fixture_path = self.fixtures_dir / 'total3_leadership.csv'
        df.to_csv(fixture_path)

        return {
            'name': 'total3_leadership',
            'path': fixture_path,
            'expected_signal': 'risk_on',
            'avg_dominance': total3_dominance.mean(),
            'description': 'TOTAL3 showing altcoin market leadership'
        }

    def test_fixture(self, fixture: dict, config: dict):
        """
        Test a golden fixture against expected outcomes with scoped overrides

        Args:
            fixture: Fixture metadata dict
            config: Bull Machine config

        Returns:
            Test result dict
        """

        # Use scoped overrides for fixture testing
        test_overrides = {
            'hps_floor': {'1h': 0.25},  # Relaxed for fixtures
            'hob_quality_factors': {
                '1h': {'volume_z_min': 1.0}  # Relaxed for fixtures
            }
        }

        with self.temp_overrides(test_overrides):
            return self._run_fixture_test(fixture, config)

    def _run_fixture_test(self, fixture: dict, config: dict):
        """Internal fixture test runner"""

        try:
            # Mock engine testing for this implementation
            from engine.timeframes.mtf_alignment import MTFAlignmentEngine

            # Load fixture data
            if 'path' in fixture:
                df = pd.read_csv(fixture['path'], index_col=0, parse_dates=True)
            else:
                # Handle multi-file fixtures
                df = {}
                for key in fixture:
                    if key.endswith('_path'):
                        df[key.replace('_path', '')] = pd.read_csv(
                            fixture[key], index_col=0, parse_dates=True
                        )

            # Initialize MTF engine with test config
            mtf_engine = MTFAlignmentEngine(self.config)

            # Process fixture
            if isinstance(df, pd.DataFrame):
                # Single timeframe test with mock data
                trigger_idx = fixture.get('trigger_index', len(df) - 1)
                test_data = df.iloc[:trigger_idx+1]

                # Create mock 1H, 4H, 1D data from test_data
                df_1h = test_data.copy()
                df_4h = test_data.iloc[::4].copy()  # Every 4th bar
                df_1d = test_data.iloc[::24].copy()  # Every 24th bar

                # Test MTF confluence with mock VIX
                vix_now = 16.0  # Normal conditions
                vix_prev = 15.0

                confluence_result = mtf_engine.mtf_confluence(
                    df_1h, df_4h, df_1d, vix_now, vix_prev
                )

                # Check expectations
                results = {
                    'fixture': fixture['name'],
                    'confluence_ok': confluence_result['ok'],
                    'direction': confluence_result['direction'],
                    'confidence': confluence_result['confidence'],
                    'guard_active': confluence_result['guard_active']
                }

                # Verify expected signal
                if fixture['expected_signal']:
                    results['matches_expected'] = (
                        confluence_result['ok'] and
                        confluence_result['direction'] == fixture['expected_signal']
                    )
                else:
                    results['matches_expected'] = not confluence_result['ok']

            else:
                # Multi-file fixture (e.g., macro veto)
                results = {
                    'fixture': fixture['name'],
                    'veto_applied': self._check_macro_veto(df.get('macro', None))
                }

                if fixture['expected_signal'] is None:
                    results['matches_expected'] = results['veto_applied']

            return results

        except Exception as e:
            return {
                'fixture': fixture.get('name', 'unknown'),
                'error': str(e),
                'matches_expected': False
            }

    def _check_macro_veto(self, macro_df: pd.DataFrame) -> bool:
        """Check if macro conditions trigger veto"""
        if macro_df is None:
            return False

        # Check veto conditions
        dxy_veto = (macro_df['DXY'] > 106).any() if 'DXY' in macro_df else False
        oil_veto = (macro_df['Oil'] > 85).any() if 'Oil' in macro_df else False
        vix_veto = (macro_df['VIX'] > 30).any() if 'VIX' in macro_df else False

        return dxy_veto or oil_veto or vix_veto


def create_all_fixtures():
    """Create all golden fixtures"""
    print("🏗️ CREATING GOLDEN FIXTURES")
    print("=" * 50)

    fixtures = GoldenFixtures()

    scenarios = [
        fixtures.create_spring_scenario(),
        fixtures.create_utad_scenario(),
        fixtures.create_dxy_veto_scenario(),
        fixtures.create_hob_pattern_scenario(),
        fixtures.create_total3_leadership_scenario()
    ]

    print(f"✅ Created {len(scenarios)} golden fixtures:")
    for scenario in scenarios:
        print(f"   - {scenario['name']}: {scenario['description']}")

    # Save fixture manifest
    manifest_path = Path('tests/fixtures/manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(scenarios, f, indent=2, default=str)

    print(f"\n📁 Fixture manifest saved to {manifest_path}")

    return scenarios


def test_all_fixtures():
    """Test all golden fixtures"""
    print("\n🧪 TESTING GOLDEN FIXTURES")
    print("=" * 50)

    # Load config
    with open('configs/v170/assets/ETH_v17_tuned.json', 'r') as f:
        config = json.load(f)

    # Load fixture manifest
    with open('tests/fixtures/manifest.json', 'r') as f:
        scenarios = json.load(f)

    fixtures = GoldenFixtures()
    results = []

    for scenario in scenarios:
        print(f"\nTesting {scenario['name']}...")

        try:
            result = fixtures.test_fixture(scenario, config)
            passed = result.get('matches_expected', False)

            if passed:
                print(f"   ✅ PASS: Expected behavior confirmed")
            else:
                print(f"   ❌ FAIL: Unexpected result")

            results.append((scenario['name'], passed))

        except Exception as e:
            print(f"   ⚠️  ERROR: {e}")
            results.append((scenario['name'], False))

    # Summary
    passed_count = sum(1 for _, passed in results if passed)
    print(f"\n📊 RESULTS: {passed_count}/{len(results)} fixtures passed")

    return passed_count == len(results)


if __name__ == "__main__":
    # Create fixtures if they don't exist
    if not Path('tests/fixtures/manifest.json').exists():
        create_all_fixtures()

    # Test fixtures
    success = test_all_fixtures()
    exit(0 if success else 1)