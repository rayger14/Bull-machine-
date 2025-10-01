"""
Perturbation and Stability Tests for Bull Machine v1.7
Tests robustness via entry randomization, block bootstrap, and Monte Carlo
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import numpy as np
import random
import json
from typing import Dict, List, Tuple
from pathlib import Path

from engine.io.tradingview_loader import load_tv
from engine.risk.transaction_costs import TransactionCostModel


class PerturbationTester:
    """
    Test system robustness under various perturbations
    - Entry timing randomization (±1 bar)
    - Price jitter (±0.5 spread)
    - Block bootstrap
    - Monte Carlo simulation
    """

    def __init__(self, config: Dict):
        """Initialize with Bull Machine config"""
        self.config = config
        self.cost_model = TransactionCostModel()

    def entry_randomization_test(self,
                                df: pd.DataFrame,
                                trades: List[Dict],
                                n_trials: int = 100) -> Dict:
        """
        Test robustness to ±1 bar entry timing variation

        Args:
            df: Price data
            trades: Original trade list
            n_trials: Number of randomization trials

        Returns:
            Stability metrics
        """
        print(f"🎲 Running entry randomization test ({n_trials} trials)")

        original_pnl = self._calculate_total_pnl(trades)
        perturbed_pnls = []

        for trial in range(n_trials):
            # Randomize entry/exit by ±1 bar
            perturbed_trades = self._randomize_entries(trades, df, max_shift=1)
            perturbed_pnl = self._calculate_total_pnl(perturbed_trades)
            perturbed_pnls.append(perturbed_pnl)

        # Calculate stability metrics
        mean_pnl = np.mean(perturbed_pnls)
        std_pnl = np.std(perturbed_pnls)
        cv_pnl = abs(std_pnl / mean_pnl) if mean_pnl != 0 else float('inf')

        # Check for sign changes
        sign_changes = sum(1 for pnl in perturbed_pnls if np.sign(pnl) != np.sign(original_pnl))
        sign_stability = (n_trials - sign_changes) / n_trials

        return {
            'original_pnl': original_pnl,
            'mean_perturbed_pnl': mean_pnl,
            'std_perturbed_pnl': std_pnl,
            'coefficient_of_variation': cv_pnl,
            'sign_stability': sign_stability,
            'trials': n_trials,
            'stable': cv_pnl < 0.1 and sign_stability > 0.9  # <10% CV, >90% sign stability
        }

    def price_jitter_test(self,
                         df: pd.DataFrame,
                         trades: List[Dict],
                         jitter_pct: float = 0.0005) -> Dict:
        """
        Test robustness to price jitter (±0.05% typical spread)

        Args:
            df: Price data
            trades: Trade list
            jitter_pct: Jitter as % of price

        Returns:
            Jitter impact analysis
        """
        print(f"📊 Running price jitter test (±{jitter_pct:.4f})")

        original_pnl = self._calculate_total_pnl(trades)
        jittered_results = []

        for trial in range(50):  # 50 trials for jitter
            jittered_trades = []

            for trade in trades:
                # Add random jitter to entry/exit prices
                entry_jitter = np.random.uniform(-jitter_pct, jitter_pct)
                exit_jitter = np.random.uniform(-jitter_pct, jitter_pct)

                jittered_trade = trade.copy()
                jittered_trade['entry_price'] *= (1 + entry_jitter)
                jittered_trade['exit_price'] *= (1 + exit_jitter)

                # Recalculate P&L
                if trade['direction'] == 'long':
                    pnl = (jittered_trade['exit_price'] - jittered_trade['entry_price']) * trade.get('quantity', 1)
                else:
                    pnl = (jittered_trade['entry_price'] - jittered_trade['exit_price']) * trade.get('quantity', 1)

                jittered_trade['pnl'] = pnl
                jittered_trades.append(jittered_trade)

            total_pnl = sum(t['pnl'] for t in jittered_trades)
            jittered_results.append(total_pnl)

        # Analysis
        mean_jittered = np.mean(jittered_results)
        impact_pct = abs((mean_jittered - original_pnl) / original_pnl * 100) if original_pnl != 0 else 0

        return {
            'original_pnl': original_pnl,
            'mean_jittered_pnl': mean_jittered,
            'impact_percent': impact_pct,
            'stable_to_jitter': impact_pct < 2.0  # <2% impact acceptable
        }

    def block_bootstrap_test(self,
                           df: pd.DataFrame,
                           block_size_days: int = 7) -> Dict:
        """
        Block bootstrap test with regime-aware blocks

        Args:
            df: Price data
            block_size_days: Size of blocks in days

        Returns:
            Bootstrap stability metrics
        """
        print(f"🔄 Running block bootstrap test (block size: {block_size_days} days)")

        # Calculate block size in bars (assuming 4H data)
        bars_per_day = 6  # 4H bars
        block_size = block_size_days * bars_per_day

        # Detect regimes
        regimes = self._detect_regimes(df)

        # Create regime-aware blocks
        blocks = self._create_regime_blocks(df, regimes, block_size)

        bootstrap_results = []

        for trial in range(20):  # 20 bootstrap trials
            # Shuffle blocks maintaining regime characteristics
            shuffled_df = self._shuffle_blocks(blocks)

            # Run simplified backtest on shuffled data
            bootstrap_pnl = self._simplified_backtest(shuffled_df)
            bootstrap_results.append(bootstrap_pnl)

        # Calculate stability
        mean_bootstrap = np.mean(bootstrap_results)
        std_bootstrap = np.std(bootstrap_results)
        cv_bootstrap = abs(std_bootstrap / mean_bootstrap) if mean_bootstrap != 0 else float('inf')

        return {
            'block_size_days': block_size_days,
            'bootstrap_trials': len(bootstrap_results),
            'mean_bootstrap_pnl': mean_bootstrap,
            'std_bootstrap_pnl': std_bootstrap,
            'coefficient_of_variation': cv_bootstrap,
            'stable': cv_bootstrap < 0.15  # <15% CV for bootstrap
        }

    def monte_carlo_stress_test(self,
                              df: pd.DataFrame,
                              n_simulations: int = 100) -> Dict:
        """
        Monte Carlo stress testing with various market conditions

        Args:
            df: Price data
            n_simulations: Number of MC simulations

        Returns:
            Stress test results
        """
        print(f"⚡ Running Monte Carlo stress test ({n_simulations} simulations)")

        stress_scenarios = [
            ('normal', {'vol_mult': 1.0, 'trend_mult': 1.0}),
            ('high_vol', {'vol_mult': 2.0, 'trend_mult': 1.0}),
            ('low_vol', {'vol_mult': 0.5, 'trend_mult': 1.0}),
            ('strong_trend', {'vol_mult': 1.0, 'trend_mult': 1.5}),
            ('choppy', {'vol_mult': 1.5, 'trend_mult': 0.3}),
        ]

        results = {}

        for scenario_name, params in stress_scenarios:
            scenario_pnls = []

            for sim in range(n_simulations // len(stress_scenarios)):
                # Generate stressed market data
                stressed_df = self._apply_stress_scenario(df, params)

                # Run backtest
                stressed_pnl = self._simplified_backtest(stressed_df)
                scenario_pnls.append(stressed_pnl)

            # Calculate scenario metrics
            results[scenario_name] = {
                'mean_pnl': np.mean(scenario_pnls),
                'std_pnl': np.std(scenario_pnls),
                'min_pnl': np.min(scenario_pnls),
                'max_pnl': np.max(scenario_pnls),
                'simulations': len(scenario_pnls)
            }

        return results

    def stability_report(self, df: pd.DataFrame, trades: List[Dict]) -> Dict:
        """
        Generate comprehensive stability report

        Args:
            df: Price data
            trades: Historical trades

        Returns:
            Complete stability assessment
        """
        print("\n📋 GENERATING STABILITY REPORT")
        print("=" * 50)

        # Run all tests
        entry_results = self.entry_randomization_test(df, trades)
        jitter_results = self.price_jitter_test(df, trades)
        bootstrap_results = self.block_bootstrap_test(df)
        mc_results = self.monte_carlo_stress_test(df)

        # Overall stability score
        stability_checks = [
            entry_results['stable'],
            jitter_results['stable_to_jitter'],
            bootstrap_results['stable']
        ]

        overall_stable = all(stability_checks)
        stability_score = sum(stability_checks) / len(stability_checks)

        # Risk assessment
        max_cv = max(
            entry_results['coefficient_of_variation'],
            bootstrap_results['coefficient_of_variation']
        )

        risk_level = 'LOW' if max_cv < 0.05 else 'MEDIUM' if max_cv < 0.15 else 'HIGH'

        report = {
            'timestamp': pd.Timestamp.now(),
            'overall_stable': overall_stable,
            'stability_score': stability_score,
            'risk_level': risk_level,
            'entry_randomization': entry_results,
            'price_jitter': jitter_results,
            'block_bootstrap': bootstrap_results,
            'monte_carlo': mc_results,
            'recommendations': self._generate_recommendations(entry_results, jitter_results, bootstrap_results)
        }

        return report

    def _randomize_entries(self, trades: List[Dict], df: pd.DataFrame, max_shift: int) -> List[Dict]:
        """Randomize trade entry/exit timing by ±max_shift bars"""
        randomized = []

        for trade in trades:
            new_trade = trade.copy()

            # Find trade indices in DataFrame
            try:
                entry_idx = df.index.get_loc(trade['entry_timestamp'])
                exit_idx = df.index.get_loc(trade['exit_timestamp'])

                # Apply random shifts
                entry_shift = random.randint(-max_shift, max_shift)
                exit_shift = random.randint(-max_shift, max_shift)

                new_entry_idx = max(0, min(len(df)-1, entry_idx + entry_shift))
                new_exit_idx = max(new_entry_idx+1, min(len(df)-1, exit_idx + exit_shift))

                # Update trade
                new_trade['entry_timestamp'] = df.index[new_entry_idx]
                new_trade['exit_timestamp'] = df.index[new_exit_idx]
                new_trade['entry_price'] = df.iloc[new_entry_idx]['close']
                new_trade['exit_price'] = df.iloc[new_exit_idx]['close']

                # Recalculate P&L
                if trade['direction'] == 'long':
                    pnl = (new_trade['exit_price'] - new_trade['entry_price']) * trade.get('quantity', 1)
                else:
                    pnl = (new_trade['entry_price'] - new_trade['exit_price']) * trade.get('quantity', 1)

                new_trade['pnl'] = pnl
                randomized.append(new_trade)

            except (KeyError, ValueError):
                # Keep original if timestamp not found
                randomized.append(trade)

        return randomized

    def _calculate_total_pnl(self, trades: List[Dict]) -> float:
        """Calculate total P&L from trade list"""
        return sum(trade.get('pnl', 0) for trade in trades)

    def _detect_regimes(self, df: pd.DataFrame) -> pd.Series:
        """Detect market regimes (bull/bear/chop)"""
        # Simple regime detection using trend strength
        returns = df['close'].pct_change(20)  # 20-bar returns

        regime = pd.Series(index=df.index, dtype=str)
        regime[returns > 0.05] = 'bull'      # >5% positive momentum
        regime[returns < -0.05] = 'bear'     # <-5% negative momentum
        regime[regime.isna()] = 'chop'       # Neutral/choppy

        return regime.fillna('chop')

    def _create_regime_blocks(self, df: pd.DataFrame, regimes: pd.Series, block_size: int) -> List[pd.DataFrame]:
        """Create blocks grouped by regime"""
        blocks = []

        for i in range(0, len(df) - block_size, block_size):
            block = df.iloc[i:i+block_size].copy()
            block_regime = regimes.iloc[i:i+block_size].mode().iloc[0] if len(regimes.iloc[i:i+block_size]) > 0 else 'chop'
            block['regime'] = block_regime
            blocks.append(block)

        return blocks

    def _shuffle_blocks(self, blocks: List[pd.DataFrame]) -> pd.DataFrame:
        """Shuffle blocks while maintaining some regime structure"""
        # Group by regime
        bull_blocks = [b for b in blocks if b['regime'].iloc[0] == 'bull']
        bear_blocks = [b for b in blocks if b['regime'].iloc[0] == 'bear']
        chop_blocks = [b for b in blocks if b['regime'].iloc[0] == 'chop']

        # Shuffle within regimes
        random.shuffle(bull_blocks)
        random.shuffle(bear_blocks)
        random.shuffle(chop_blocks)

        # Combine randomly but maintain rough proportions
        all_blocks = bull_blocks + bear_blocks + chop_blocks
        random.shuffle(all_blocks)

        # Concatenate
        shuffled = pd.concat([b.drop('regime', axis=1) for b in all_blocks], ignore_index=True)
        return shuffled

    def _apply_stress_scenario(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Apply stress scenario transformations"""
        stressed = df.copy()

        vol_mult = params.get('vol_mult', 1.0)
        trend_mult = params.get('trend_mult', 1.0)

        # Calculate returns
        returns = stressed['close'].pct_change().fillna(0)

        # Apply volatility scaling
        stressed_returns = returns * vol_mult

        # Apply trend scaling
        trend = returns.rolling(20).mean().fillna(0)
        adjusted_returns = stressed_returns + (trend * (trend_mult - 1.0))

        # Reconstruct prices
        stressed['close'] = (1 + adjusted_returns).cumprod() * stressed['close'].iloc[0]

        # Adjust OHLC proportionally
        price_ratio = stressed['close'] / df['close']
        for col in ['open', 'high', 'low']:
            stressed[col] = df[col] * price_ratio

        return stressed

    def _simplified_backtest(self, df: pd.DataFrame) -> float:
        """Simplified backtest for robustness testing"""
        # Mock backtest - replace with actual logic
        returns = df['close'].pct_change().dropna()

        # Simple momentum strategy
        signals = (returns.rolling(5).mean() > 0.001).astype(int)
        strategy_returns = signals.shift(1) * returns

        total_return = strategy_returns.sum()
        return total_return * 100000  # Scale to notional

    def _generate_recommendations(self, entry_results: Dict, jitter_results: Dict, bootstrap_results: Dict) -> List[str]:
        """Generate recommendations based on stability tests"""
        recommendations = []

        if not entry_results['stable']:
            recommendations.append(
                f"Entry timing sensitivity detected (CV: {entry_results['coefficient_of_variation']:.3f}). "
                "Consider longer holding periods or different entry criteria."
            )

        if not jitter_results['stable_to_jitter']:
            recommendations.append(
                f"High sensitivity to price jitter ({jitter_results['impact_percent']:.1f}% impact). "
                "Consider wider spreads or different position sizing."
            )

        if not bootstrap_results['stable']:
            recommendations.append(
                f"Bootstrap instability detected (CV: {bootstrap_results['coefficient_of_variation']:.3f}). "
                "Strategy may be overfitted to specific market sequences."
            )

        if not recommendations:
            recommendations.append("System shows good stability across all perturbation tests.")

        return recommendations


def run_perturbation_tests():
    """Run complete perturbation test suite"""
    print("🧪 BULL MACHINE v1.7 PERTURBATION TESTS")
    print("=" * 60)

    # Load test data
    try:
        # Use BTC data since it's available
        eth_data = load_tv('BTC_4H').tail(200)
        print(f"✅ Loaded {len(eth_data)} bars of BTC data for testing")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

    # Load config
    try:
        with open('configs/v170/assets/ETH_v17_tuned.json', 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded config v{config['version']}")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

    # Create mock trades for testing
    mock_trades = [
        {
            'entry_timestamp': eth_data.index[50],
            'exit_timestamp': eth_data.index[65],
            'entry_price': eth_data.iloc[50]['close'],
            'exit_price': eth_data.iloc[65]['close'],
            'direction': 'long',
            'quantity': 1.0,
            'pnl': (eth_data.iloc[65]['close'] - eth_data.iloc[50]['close'])
        },
        {
            'entry_timestamp': eth_data.index[100],
            'exit_timestamp': eth_data.index[120],
            'entry_price': eth_data.iloc[100]['close'],
            'exit_price': eth_data.iloc[120]['close'],
            'direction': 'short',
            'quantity': 1.0,
            'pnl': (eth_data.iloc[100]['close'] - eth_data.iloc[120]['close'])
        }
    ]

    # Run perturbation tests
    tester = PerturbationTester(config)
    stability_report = tester.stability_report(eth_data, mock_trades)

    # Display results
    print(f"\n📊 STABILITY REPORT")
    print("=" * 40)
    print(f"Overall Stable: {'✅ YES' if stability_report['overall_stable'] else '❌ NO'}")
    print(f"Stability Score: {stability_report['stability_score']:.1%}")
    print(f"Risk Level: {stability_report['risk_level']}")

    print(f"\n🎲 Entry Randomization:")
    entry = stability_report['entry_randomization']
    print(f"   CV: {entry['coefficient_of_variation']:.3f}")
    print(f"   Sign Stability: {entry['sign_stability']:.1%}")
    print(f"   Stable: {'✅' if entry['stable'] else '❌'}")

    print(f"\n📊 Price Jitter:")
    jitter = stability_report['price_jitter']
    print(f"   Impact: {jitter['impact_percent']:.1f}%")
    print(f"   Stable: {'✅' if jitter['stable_to_jitter'] else '❌'}")

    print(f"\n🔄 Block Bootstrap:")
    bootstrap = stability_report['block_bootstrap']
    print(f"   CV: {bootstrap['coefficient_of_variation']:.3f}")
    print(f"   Stable: {'✅' if bootstrap['stable'] else '❌'}")

    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(stability_report['recommendations'], 1):
        print(f"   {i}. {rec}")

    return stability_report['overall_stable']


if __name__ == "__main__":
    success = run_perturbation_tests()
    exit(0 if success else 1)