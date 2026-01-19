#!/usr/bin/env python3
"""
Statistical Significance Validation Framework

Validates that observed performance is statistically significant and not due to random chance.
Uses bootstrap resampling and permutation tests to establish confidence intervals.

Validation Strategy:
===================
**1. Bootstrap Resampling (1000 iterations):**
   - Resample trades with replacement
   - Compute PF distribution
   - Calculate 95% confidence interval
   - Requirement: Lower bound of 95% CI for PF > 1.2

**2. Permutation Test:**
   - Shuffle trade outcomes randomly
   - Test if observed PF could occur by chance
   - Requirement: p-value < 0.05 (95% confidence)

**3. Regime Comparison:**
   - Test if PF_risk_off significantly different from PF_risk_on
   - Use permutation test or t-test
   - Validates regime-specific edge

**4. Sample Size Requirements:**
   - Minimum 10 trades for bootstrap
   - Minimum 20 trades for robust CI
   - Flag if insufficient sample size

Statistical Tests:
==================
**Bootstrap Confidence Interval:**
- Resamples trades 1000x to build PF distribution
- Computes 95% CI using percentile method
- Tests: Is lower bound > 1.2?

**Permutation Test for Edge:**
- Null hypothesis: PF = 1.0 (no edge)
- Randomly shuffles trade outcomes
- Computes p-value: P(PF_random >= PF_observed)
- Rejects null if p < 0.05

**Regime Comparison Test:**
- Null hypothesis: PF_regime1 = PF_regime2
- Permutation test for difference in means
- Tests if regime-specific edge is real

Usage:
    # Test single config
    python bin/validate_statistical_significance.py \
        --config configs/mvp/mvp_bear_market_v1.json \
        --output results/validation/statistical/

    # Test with custom parameters
    python bin/validate_statistical_significance.py \
        --config configs/optimized/s5_balanced.json \
        --n-bootstrap 2000 \
        --n-permutations 1000 \
        --alpha 0.05 \
        --output results/validation/s5_stats/

Output:
    results/validation/{config_name}/
        statistical_summary.json    - All test results
        bootstrap_distribution.png  - PF distribution from bootstrap
        permutation_test.png        - Permutation test visualization
        confidence_intervals.csv    - CI for all metrics
        regime_comparison.csv       - Regime-vs-regime statistical tests
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import backtest engine
from bin.backtest_knowledge_v2 import KnowledgeAwareBacktest, KnowledgeParams

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Results from bootstrap resampling"""
    metric_name: str
    observed_value: float
    mean: float
    std: float
    ci_lower: float      # 95% CI lower bound
    ci_upper: float      # 95% CI upper bound
    is_significant: bool  # Lower bound > threshold

    def to_dict(self):
        return asdict(self)


@dataclass
class PermutationTestResult:
    """Results from permutation test"""
    test_name: str
    observed_value: float
    null_hypothesis: str
    p_value: float
    is_significant: bool  # p < alpha
    alpha: float

    def to_dict(self):
        return asdict(self)


@dataclass
class StatisticalValidationResult:
    """Complete statistical validation results"""
    config_name: str
    config_path: str
    timestamp: str

    # Sample size
    total_trades: int
    sufficient_sample: bool

    # Bootstrap results
    bootstrap_pf: BootstrapResult
    bootstrap_wr: BootstrapResult
    bootstrap_sharpe: BootstrapResult

    # Permutation tests
    perm_edge_test: PermutationTestResult
    perm_regime_tests: Dict[str, PermutationTestResult]

    # Overall validation
    statistically_significant: bool
    warnings: List[str]
    errors: List[str]

    def to_dict(self):
        return {
            'config_name': self.config_name,
            'config_path': self.config_path,
            'timestamp': self.timestamp,
            'sample_size': {
                'total_trades': self.total_trades,
                'sufficient_sample': self.sufficient_sample,
            },
            'bootstrap_results': {
                'profit_factor': self.bootstrap_pf.to_dict(),
                'win_rate': self.bootstrap_wr.to_dict(),
                'sharpe_ratio': self.bootstrap_sharpe.to_dict(),
            },
            'permutation_tests': {
                'edge_test': self.perm_edge_test.to_dict(),
                'regime_tests': {
                    regime: result.to_dict()
                    for regime, result in self.perm_regime_tests.items()
                },
            },
            'statistically_significant': self.statistically_significant,
            'warnings': self.warnings,
            'errors': self.errors,
        }


class StatisticalValidator:
    """
    Statistical significance validation framework.

    Uses bootstrap resampling and permutation tests to validate that
    observed performance is statistically significant.
    """

    def __init__(
        self,
        feature_store_path: str,
        output_dir: str,
        n_bootstrap: int = 1000,
        n_permutations: int = 1000,
        alpha: float = 0.05,
        min_pf_threshold: float = 1.2
    ):
        """
        Initialize statistical validator.

        Args:
            feature_store_path: Path to feature store parquet
            output_dir: Directory for validation outputs
            n_bootstrap: Number of bootstrap iterations
            n_permutations: Number of permutation test iterations
            alpha: Significance level (default 0.05 for 95% confidence)
            min_pf_threshold: Minimum PF for CI lower bound
        """
        self.feature_store_path = feature_store_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.min_pf_threshold = min_pf_threshold

        # Sample size requirements
        self.min_trades_bootstrap = 10
        self.min_trades_robust = 20

        # Load feature store
        logger.info(f"Loading feature store: {feature_store_path}")
        self.df = pd.read_parquet(feature_store_path)
        logger.info(f"Loaded {len(self.df)} bars")

        # Check for regime labels
        if 'macro_regime' not in self.df.columns:
            logger.warning("No macro_regime column")
            self.df['macro_regime'] = 'neutral'

    def validate_config(
        self,
        config_path: str,
        period: Tuple[str, str] = ('2022-01-01', '2023-06-30')
    ) -> StatisticalValidationResult:
        """
        Run statistical validation on a config.

        Args:
            config_path: Path to config JSON file
            period: (start_date, end_date) for validation

        Returns:
            StatisticalValidationResult with all statistical tests
        """
        config_name = Path(config_path).stem

        logger.info(f"\n{'='*80}")
        logger.info(f"STATISTICAL VALIDATION: {config_name}")
        logger.info(f"Config: {config_path}")
        logger.info(f"{'='*80}\n")

        # Load config
        with open(config_path) as f:
            config = json.load(f)

        # Run backtest
        trades_df = self._run_backtest(config, period)

        if len(trades_df) == 0:
            logger.error("No trades generated")
            return self._empty_result(config_name, config_path)

        # Check sample size
        total_trades = len(trades_df)
        sufficient_sample = total_trades >= self.min_trades_robust

        warnings = []
        errors = []

        if total_trades < self.min_trades_bootstrap:
            errors.append(f"Insufficient trades ({total_trades} < {self.min_trades_bootstrap})")
            return self._empty_result(config_name, config_path, errors=errors)

        if total_trades < self.min_trades_robust:
            warnings.append(f"Small sample size ({total_trades} < {self.min_trades_robust}) - results may be unstable")

        # Run bootstrap resampling
        logger.info("\nRunning bootstrap resampling...")
        bootstrap_pf = self._bootstrap_metric(trades_df, 'profit_factor', self.min_pf_threshold)
        bootstrap_wr = self._bootstrap_metric(trades_df, 'win_rate', 0.45)
        bootstrap_sharpe = self._bootstrap_metric(trades_df, 'sharpe_ratio', 0.5)

        # Run permutation test for edge
        logger.info("\nRunning permutation test for edge significance...")
        perm_edge = self._permutation_test_edge(trades_df)

        # Run regime comparison tests
        logger.info("\nRunning regime comparison tests...")
        perm_regime = self._permutation_test_regimes(trades_df)

        # Overall significance
        statistically_significant = all([
            bootstrap_pf.is_significant,
            perm_edge.is_significant,
            sufficient_sample,
        ])

        # Add warnings for marginal results
        if not bootstrap_pf.is_significant:
            warnings.append(f"PF CI lower bound ({bootstrap_pf.ci_lower:.2f}) below threshold ({self.min_pf_threshold})")

        if not perm_edge.is_significant:
            warnings.append(f"Edge not statistically significant (p={perm_edge.p_value:.3f})")

        # Create result
        result = StatisticalValidationResult(
            config_name=config_name,
            config_path=str(config_path),
            timestamp=datetime.now().isoformat(),
            total_trades=total_trades,
            sufficient_sample=sufficient_sample,
            bootstrap_pf=bootstrap_pf,
            bootstrap_wr=bootstrap_wr,
            bootstrap_sharpe=bootstrap_sharpe,
            perm_edge_test=perm_edge,
            perm_regime_tests=perm_regime,
            statistically_significant=statistically_significant,
            warnings=warnings,
            errors=errors,
        )

        # Create visualizations
        self._create_visualizations(config_name, result, trades_df)

        # Save results
        self._save_results(config_name, result)

        # Print summary
        self._print_summary(result)

        return result

    def _run_backtest(self, config: Dict, period: Tuple[str, str]) -> pd.DataFrame:
        """Run backtest and return trades"""
        start_date, end_date = period

        mask = (self.df.index >= start_date) & (self.df.index <= end_date)
        period_df = self.df[mask].copy()

        logger.info(f"Backtest period: {len(period_df)} bars from {start_date} to {end_date}")

        bt = KnowledgeAwareBacktest(
            df=period_df,
            params=self._config_to_params(config),
            runtime_config=config
        )

        bt.run()
        trades_df = bt.get_trades_dataframe()

        logger.info(f"Generated {len(trades_df)} trades")

        return trades_df

    def _bootstrap_metric(
        self,
        trades_df: pd.DataFrame,
        metric: str,
        threshold: float
    ) -> BootstrapResult:
        """
        Bootstrap resample to compute confidence interval for a metric.

        Args:
            trades_df: Trades dataframe
            metric: 'profit_factor', 'win_rate', or 'sharpe_ratio'
            threshold: Minimum value for significance test

        Returns:
            BootstrapResult with CI and significance
        """
        logger.info(f"  Bootstrap {metric}: {self.n_bootstrap} iterations")

        # Compute observed metric
        observed = self._compute_metric(trades_df, metric)

        # Bootstrap resampling
        bootstrap_values = []

        for i in range(self.n_bootstrap):
            # Resample with replacement
            sample_df = trades_df.sample(n=len(trades_df), replace=True)
            value = self._compute_metric(sample_df, metric)
            bootstrap_values.append(value)

        bootstrap_values = np.array(bootstrap_values)

        # Compute statistics
        mean = bootstrap_values.mean()
        std = bootstrap_values.std()

        # 95% confidence interval (percentile method)
        ci_lower = np.percentile(bootstrap_values, 2.5)
        ci_upper = np.percentile(bootstrap_values, 97.5)

        # Significance test: is lower bound > threshold?
        is_significant = ci_lower > threshold

        logger.info(f"    Observed: {observed:.3f}")
        logger.info(f"    95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        logger.info(f"    Significant: {is_significant} (threshold={threshold})")

        return BootstrapResult(
            metric_name=metric,
            observed_value=observed,
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            is_significant=is_significant,
        )

    def _permutation_test_edge(self, trades_df: pd.DataFrame) -> PermutationTestResult:
        """
        Permutation test to check if edge is statistically significant.

        Null hypothesis: PF = 1.0 (no edge, wins = losses by chance)
        """
        logger.info(f"  Permutation test for edge: {self.n_permutations} iterations")

        # Observed PF
        observed_pf = self._compute_metric(trades_df, 'profit_factor')

        # Permutation test: shuffle outcomes
        perm_pfs = []

        for i in range(self.n_permutations):
            # Shuffle PNL outcomes
            shuffled_df = trades_df.copy()
            shuffled_df['net_pnl'] = np.random.permutation(shuffled_df['net_pnl'].values)
            perm_pf = self._compute_metric(shuffled_df, 'profit_factor')
            perm_pfs.append(perm_pf)

        perm_pfs = np.array(perm_pfs)

        # p-value: proportion of permutations >= observed
        p_value = (perm_pfs >= observed_pf).mean()

        is_significant = p_value < self.alpha

        logger.info(f"    Observed PF: {observed_pf:.3f}")
        logger.info(f"    Permutation mean: {perm_pfs.mean():.3f}")
        logger.info(f"    p-value: {p_value:.4f}")
        logger.info(f"    Significant: {is_significant} (alpha={self.alpha})")

        return PermutationTestResult(
            test_name='edge_significance',
            observed_value=observed_pf,
            null_hypothesis='PF = 1.0 (no edge)',
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha,
        )

    def _permutation_test_regimes(self, trades_df: pd.DataFrame) -> Dict[str, PermutationTestResult]:
        """
        Test if PF differs significantly between regimes.

        Compares risk_on vs risk_off, and risk_off vs crisis.
        """
        if 'macro_regime' not in trades_df.columns:
            logger.warning("  No regime labels - skipping regime comparison")
            return {}

        results = {}

        # Test pairs
        test_pairs = [
            ('risk_on', 'risk_off'),
            ('risk_off', 'crisis'),
            ('neutral', 'risk_off'),
        ]

        for regime1, regime2 in test_pairs:
            trades1 = trades_df[trades_df['macro_regime'] == regime1]
            trades2 = trades_df[trades_df['macro_regime'] == regime2]

            if len(trades1) < 5 or len(trades2) < 5:
                logger.info(f"  Skipping {regime1} vs {regime2}: insufficient trades")
                continue

            logger.info(f"  Comparing {regime1} vs {regime2}")

            pf1 = self._compute_metric(trades1, 'profit_factor')
            pf2 = self._compute_metric(trades2, 'profit_factor')

            observed_diff = abs(pf1 - pf2)

            # Permutation test for difference
            perm_diffs = []

            combined = pd.concat([trades1, trades2])

            for i in range(self.n_permutations):
                # Shuffle regime labels
                shuffled_regimes = np.random.permutation(combined['macro_regime'].values)
                shuffled_df = combined.copy()
                shuffled_df['macro_regime'] = shuffled_regimes

                perm_trades1 = shuffled_df[shuffled_df['macro_regime'] == regime1]
                perm_trades2 = shuffled_df[shuffled_df['macro_regime'] == regime2]

                perm_pf1 = self._compute_metric(perm_trades1, 'profit_factor')
                perm_pf2 = self._compute_metric(perm_trades2, 'profit_factor')
                perm_diff = abs(perm_pf1 - perm_pf2)

                perm_diffs.append(perm_diff)

            perm_diffs = np.array(perm_diffs)

            # p-value
            p_value = (perm_diffs >= observed_diff).mean()

            is_significant = p_value < self.alpha

            logger.info(f"    {regime1} PF: {pf1:.3f}")
            logger.info(f"    {regime2} PF: {pf2:.3f}")
            logger.info(f"    Difference: {observed_diff:.3f}")
            logger.info(f"    p-value: {p_value:.4f}")
            logger.info(f"    Significant: {is_significant}")

            results[f"{regime1}_vs_{regime2}"] = PermutationTestResult(
                test_name=f"{regime1}_vs_{regime2}",
                observed_value=observed_diff,
                null_hypothesis=f"PF_{regime1} = PF_{regime2}",
                p_value=p_value,
                is_significant=is_significant,
                alpha=self.alpha,
            )

        return results

    def _compute_metric(self, trades_df: pd.DataFrame, metric: str) -> float:
        """Compute a specific metric from trades"""
        if len(trades_df) == 0:
            return 0.0

        if metric == 'profit_factor':
            wins = trades_df[trades_df['net_pnl'] > 0]
            losses = trades_df[trades_df['net_pnl'] < 0]

            gross_profit = wins['net_pnl'].sum() if len(wins) > 0 else 0.0
            gross_loss = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 0.0

            return gross_profit / gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)

        elif metric == 'win_rate':
            wins = trades_df[trades_df['net_pnl'] > 0]
            return len(wins) / len(trades_df)

        elif metric == 'sharpe_ratio':
            returns = trades_df['net_pnl'].values
            return returns.mean() / returns.std() if returns.std() > 0 else 0.0

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _create_visualizations(self, config_name: str, result: StatisticalValidationResult, trades_df: pd.DataFrame):
        """Create statistical visualization plots"""
        config_dir = self.output_dir / config_name
        config_dir.mkdir(exist_ok=True)

        # 1. Bootstrap distribution for PF
        fig, ax = plt.subplots(figsize=(10, 6))

        # Re-run bootstrap to get distribution (for plotting)
        bootstrap_pfs = []
        for i in range(self.n_bootstrap):
            sample_df = trades_df.sample(n=len(trades_df), replace=True)
            pf = self._compute_metric(sample_df, 'profit_factor')
            bootstrap_pfs.append(pf)

        ax.hist(bootstrap_pfs, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(result.bootstrap_pf.observed_value, color='red', linestyle='--',
                   linewidth=2, label=f'Observed: {result.bootstrap_pf.observed_value:.2f}')
        ax.axvline(result.bootstrap_pf.ci_lower, color='green', linestyle='--',
                   label=f'95% CI: [{result.bootstrap_pf.ci_lower:.2f}, {result.bootstrap_pf.ci_upper:.2f}]')
        ax.axvline(result.bootstrap_pf.ci_upper, color='green', linestyle='--')
        ax.axvline(self.min_pf_threshold, color='orange', linestyle=':',
                   label=f'Threshold: {self.min_pf_threshold}')

        ax.set_xlabel('Profit Factor')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Bootstrap Distribution of Profit Factor ({self.n_bootstrap} iterations)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(config_dir / 'bootstrap_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Permutation test visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        # Re-run permutation test
        perm_pfs = []
        for i in range(self.n_permutations):
            shuffled_df = trades_df.copy()
            shuffled_df['net_pnl'] = np.random.permutation(shuffled_df['net_pnl'].values)
            pf = self._compute_metric(shuffled_df, 'profit_factor')
            perm_pfs.append(pf)

        ax.hist(perm_pfs, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(result.perm_edge_test.observed_value, color='red', linestyle='--',
                   linewidth=2, label=f'Observed: {result.perm_edge_test.observed_value:.2f}')
        ax.axvline(1.0, color='orange', linestyle=':', label='Breakeven (PF=1.0)')

        ax.set_xlabel('Profit Factor')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Permutation Test (p={result.perm_edge_test.p_value:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(config_dir / 'permutation_test.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualizations saved to {config_dir}")

    def _config_to_params(self, config: Dict) -> KnowledgeParams:
        """Convert config dict to KnowledgeParams"""
        fusion = config.get('fusion', {})
        weights = fusion.get('weights', {})

        return KnowledgeParams(
            wyckoff_weight=weights.get('wyckoff', 0.33),
            liquidity_weight=weights.get('liquidity', 0.39),
            momentum_weight=weights.get('momentum', 0.21),
            macro_weight=weights.get('macro', 0.0),
            pti_weight=weights.get('pti', 0.07),
            tier3_threshold=fusion.get('entry_threshold_confidence', 0.37),
        )

    def _empty_result(self, config_name: str, config_path: str, errors: List[str] = None) -> StatisticalValidationResult:
        """Create empty result for failed validations"""
        if errors is None:
            errors = ['No trades generated']

        return StatisticalValidationResult(
            config_name=config_name,
            config_path=config_path,
            timestamp=datetime.now().isoformat(),
            total_trades=0,
            sufficient_sample=False,
            bootstrap_pf=BootstrapResult('profit_factor', 0, 0, 0, 0, 0, False),
            bootstrap_wr=BootstrapResult('win_rate', 0, 0, 0, 0, 0, False),
            bootstrap_sharpe=BootstrapResult('sharpe_ratio', 0, 0, 0, 0, 0, False),
            perm_edge_test=PermutationTestResult('edge', 0, 'N/A', 1.0, False, self.alpha),
            perm_regime_tests={},
            statistically_significant=False,
            warnings=[],
            errors=errors,
        )

    def _save_results(self, config_name: str, result: StatisticalValidationResult):
        """Save validation results"""
        config_dir = self.output_dir / config_name
        config_dir.mkdir(exist_ok=True)

        # Save statistical summary JSON
        with open(config_dir / 'statistical_summary.json', 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save confidence intervals CSV
        ci_data = []
        for bootstrap_result in [result.bootstrap_pf, result.bootstrap_wr, result.bootstrap_sharpe]:
            ci_data.append({
                'metric': bootstrap_result.metric_name,
                'observed': bootstrap_result.observed_value,
                'mean': bootstrap_result.mean,
                'std': bootstrap_result.std,
                'ci_lower_95': bootstrap_result.ci_lower,
                'ci_upper_95': bootstrap_result.ci_upper,
                'significant': bootstrap_result.is_significant,
            })

        pd.DataFrame(ci_data).to_csv(config_dir / 'confidence_intervals.csv', index=False)

        # Save regime comparison CSV
        if result.perm_regime_tests:
            regime_data = []
            for test_name, test_result in result.perm_regime_tests.items():
                regime_data.append({
                    'comparison': test_name,
                    'observed_difference': test_result.observed_value,
                    'p_value': test_result.p_value,
                    'significant': test_result.is_significant,
                })

            pd.DataFrame(regime_data).to_csv(config_dir / 'regime_comparison.csv', index=False)

        logger.info(f"Results saved to {config_dir}")

    def _print_summary(self, result: StatisticalValidationResult):
        """Print validation summary"""
        print(f"\n{'='*80}")
        print(f"STATISTICAL VALIDATION SUMMARY: {result.config_name}")
        print(f"{'='*80}\n")

        print(f"Sample Size: {result.total_trades} trades")
        print(f"Sufficient Sample: {'YES' if result.sufficient_sample else 'NO'}")
        print()

        print("BOOTSTRAP CONFIDENCE INTERVALS (95%):")
        print(f"  Profit Factor:  {result.bootstrap_pf.observed_value:.3f} "
              f"[{result.bootstrap_pf.ci_lower:.3f}, {result.bootstrap_pf.ci_upper:.3f}] "
              f"{'✓' if result.bootstrap_pf.is_significant else '✗'}")
        print(f"  Win Rate:       {result.bootstrap_wr.observed_value:.3f} "
              f"[{result.bootstrap_wr.ci_lower:.3f}, {result.bootstrap_wr.ci_upper:.3f}] "
              f"{'✓' if result.bootstrap_wr.is_significant else '✗'}")
        print(f"  Sharpe Ratio:   {result.bootstrap_sharpe.observed_value:.3f} "
              f"[{result.bootstrap_sharpe.ci_lower:.3f}, {result.bootstrap_sharpe.ci_upper:.3f}] "
              f"{'✓' if result.bootstrap_sharpe.is_significant else '✗'}")
        print()

        print("PERMUTATION TESTS:")
        print(f"  Edge Significance: p={result.perm_edge_test.p_value:.4f} "
              f"{'✓ SIGNIFICANT' if result.perm_edge_test.is_significant else '✗ NOT SIGNIFICANT'}")

        if result.perm_regime_tests:
            print("  Regime Comparisons:")
            for test_name, test_result in result.perm_regime_tests.items():
                print(f"    {test_name}: p={test_result.p_value:.4f} "
                      f"{'✓' if test_result.is_significant else '✗'}")

        print()

        if result.warnings:
            print("WARNINGS:")
            for warning in result.warnings:
                print(f"  - {warning}")
            print()

        if result.errors:
            print("ERRORS:")
            for error in result.errors:
                print(f"  - {error}")
            print()

        print(f"STATISTICALLY SIGNIFICANT: {'YES' if result.statistically_significant else 'NO'}")
        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Statistical Significance Validation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config JSON file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results/validation/statistical',
        help='Output directory'
    )

    parser.add_argument(
        '--feature-store',
        type=str,
        default='data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31_backup.parquet',
        help='Path to feature store'
    )

    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=1000,
        help='Number of bootstrap iterations'
    )

    parser.add_argument(
        '--n-permutations',
        type=int,
        default=1000,
        help='Number of permutation test iterations'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default 0.05)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default='2022-01-01',
        help='Validation period start'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default='2023-06-30',
        help='Validation period end'
    )

    args = parser.parse_args()

    # Verify inputs
    if not Path(args.config).exists():
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    if not Path(args.feature_store).exists():
        logger.error(f"Feature store not found: {args.feature_store}")
        sys.exit(1)

    # Initialize validator
    validator = StatisticalValidator(
        feature_store_path=args.feature_store,
        output_dir=args.output,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        alpha=args.alpha,
    )

    # Run validation
    result = validator.validate_config(
        config_path=args.config,
        period=(args.start_date, args.end_date)
    )

    # Exit with appropriate code
    sys.exit(0 if result.statistically_significant else 1)


if __name__ == '__main__':
    main()
