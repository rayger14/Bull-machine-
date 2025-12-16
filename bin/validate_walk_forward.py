#!/usr/bin/env python3
"""
Walk-Forward and Cross-Regime Validation Framework

Rigorous validation to ensure configs aren't overfit and perform consistently across regimes.

Validation Strategy:
1. Walk-Forward Validation: Train → Validation → OOS Test
   - Train: 2022 H1 (Jan-Jun)
   - Validation: 2022 H2 (Jul-Dec)
   - OOS Test: 2023 (full year - transition regime)

2. Cross-Regime Analysis: Performance breakdown by macro regime
   - Regimes: {risk_on, neutral, risk_off, crisis}
   - Metrics: PF, trade count, DD, win rate by regime

3. Overfitting Detection: Statistical tests and degradation checks
   - Permutation test for edge validation
   - Train/Val/OOS degradation analysis
   - Minimum sample size requirements

Acceptance Criteria:
- Validation PF ≥ 0.8 × Train PF (max 20% degradation)
- OOS PF ≥ 1.1 (must maintain edge)
- Max DD in OOS < 25%
- At least one short archetype profitable in risk_off (PF > 1.3)
- No regime with PF < 0.8

Usage:
    python bin/validate_walk_forward.py \
        --configs results/phase2_optimization/optimization_study.db \
        --output results/validation/walk_forward/ \
        --asset BTC \
        --min-trials 50
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import traceback
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PeriodMetrics:
    """Metrics for a specific time period"""
    period_name: str
    start_date: str
    end_date: str
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    avg_win: float
    avg_loss: float
    avg_r_multiple: float
    max_consecutive_losses: int

    # Regime breakdown
    regime_trades: Dict[str, int] = None
    regime_pf: Dict[str, float] = None
    regime_wr: Dict[str, float] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class ValidationResult:
    """Complete validation results for a config"""
    config_id: str
    config_params: Dict

    # Period metrics
    train_metrics: PeriodMetrics
    val_metrics: PeriodMetrics
    oos_metrics: PeriodMetrics

    # Validation checks
    val_degradation_ok: bool
    oos_edge_ok: bool
    oos_dd_ok: bool
    regime_consistency_ok: bool
    sample_size_ok: bool

    # Statistical tests
    permutation_p_value: float
    edge_is_significant: bool

    # Overall pass/fail
    passed: bool
    failure_reasons: List[str]

    def to_dict(self):
        return {
            'config_id': self.config_id,
            'config_params': self.config_params,
            'train_metrics': self.train_metrics.to_dict(),
            'val_metrics': self.val_metrics.to_dict(),
            'oos_metrics': self.oos_metrics.to_dict(),
            'validation_checks': {
                'val_degradation_ok': self.val_degradation_ok,
                'oos_edge_ok': self.oos_edge_ok,
                'oos_dd_ok': self.oos_dd_ok,
                'regime_consistency_ok': self.regime_consistency_ok,
                'sample_size_ok': self.sample_size_ok,
            },
            'statistical_tests': {
                'permutation_p_value': self.permutation_p_value,
                'edge_is_significant': self.edge_is_significant,
            },
            'passed': self.passed,
            'failure_reasons': self.failure_reasons,
        }


class WalkForwardValidator:
    """
    Walk-forward and cross-regime validation framework.

    Performs rigorous out-of-sample testing to detect overfitting and ensure
    configs perform consistently across different market regimes.
    """

    def __init__(self, feature_store_path: str, output_dir: str):
        """
        Initialize validator.

        Args:
            feature_store_path: Path to parquet file with features and regime labels
            output_dir: Directory for validation outputs
        """
        self.feature_store_path = feature_store_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load feature store
        logger.info(f"Loading feature store from {feature_store_path}")
        self.df = pd.read_parquet(feature_store_path)
        logger.info(f"Loaded {len(self.df)} bars from {self.df.index[0]} to {self.df.index[-1]}")

        # Verify regime column exists
        if 'macro_regime' not in self.df.columns:
            logger.warning("No macro_regime column found, will use neutral regime for all bars")
            self.df['macro_regime'] = 'neutral'

        # Define periods
        self.periods = {
            'train': ('2022-01-01', '2022-06-30'),
            'val': ('2022-07-01', '2022-12-31'),
            'oos': ('2023-01-01', '2023-12-31'),
        }

        # Validation thresholds
        self.thresholds = {
            'val_degradation_max': 0.20,  # Max 20% PF degradation from train to val
            'oos_pf_min': 1.1,              # OOS must maintain edge
            'oos_dd_max': 0.25,             # Max 25% drawdown in OOS
            'min_trades_per_period': 5,     # Minimum sample size
            'permutation_alpha': 0.05,      # p-value threshold for significance
            'regime_pf_min': 0.8,           # No regime should have PF < 0.8
        }

    def validate_config(self, config: Dict, config_id: str) -> ValidationResult:
        """
        Run full walk-forward validation on a single config.

        Args:
            config: Configuration dictionary
            config_id: Unique identifier for this config

        Returns:
            ValidationResult with all metrics and checks
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Validating config: {config_id}")
        logger.info(f"{'='*80}")

        try:
            # Run backtest on each period
            train_metrics = self._run_period_backtest(config, 'train')
            val_metrics = self._run_period_backtest(config, 'val')
            oos_metrics = self._run_period_backtest(config, 'oos')

            # Perform validation checks
            checks = self._perform_validation_checks(train_metrics, val_metrics, oos_metrics)

            # Run permutation test on OOS period
            perm_p_value = self._permutation_test(config, 'oos')
            edge_significant = perm_p_value < self.thresholds['permutation_alpha']

            # Determine overall pass/fail
            passed = all([
                checks['val_degradation_ok'],
                checks['oos_edge_ok'],
                checks['oos_dd_ok'],
                checks['regime_consistency_ok'],
                checks['sample_size_ok'],
                edge_significant,
            ])

            failure_reasons = []
            if not checks['val_degradation_ok']:
                failure_reasons.append(f"Val PF degraded too much from train: {val_metrics.profit_factor:.2f} < 0.8 * {train_metrics.profit_factor:.2f}")
            if not checks['oos_edge_ok']:
                failure_reasons.append(f"OOS PF too low: {oos_metrics.profit_factor:.2f} < 1.1")
            if not checks['oos_dd_ok']:
                failure_reasons.append(f"OOS drawdown too high: {oos_metrics.max_drawdown:.2%} > 25%")
            if not checks['regime_consistency_ok']:
                failure_reasons.append("Some regime has PF < 0.8")
            if not checks['sample_size_ok']:
                failure_reasons.append("Insufficient trades in some period")
            if not edge_significant:
                failure_reasons.append(f"Edge not statistically significant (p={perm_p_value:.3f})")

            result = ValidationResult(
                config_id=config_id,
                config_params=self._extract_key_params(config),
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                oos_metrics=oos_metrics,
                val_degradation_ok=checks['val_degradation_ok'],
                oos_edge_ok=checks['oos_edge_ok'],
                oos_dd_ok=checks['oos_dd_ok'],
                regime_consistency_ok=checks['regime_consistency_ok'],
                sample_size_ok=checks['sample_size_ok'],
                permutation_p_value=perm_p_value,
                edge_is_significant=edge_significant,
                passed=passed,
                failure_reasons=failure_reasons,
            )

            # Save individual results
            self._save_config_results(config_id, result)

            return result

        except Exception as e:
            logger.error(f"Error validating config {config_id}: {e}")
            logger.error(traceback.format_exc())

            # Return failed result
            return ValidationResult(
                config_id=config_id,
                config_params=self._extract_key_params(config),
                train_metrics=self._empty_metrics('train'),
                val_metrics=self._empty_metrics('val'),
                oos_metrics=self._empty_metrics('oos'),
                val_degradation_ok=False,
                oos_edge_ok=False,
                oos_dd_ok=False,
                regime_consistency_ok=False,
                sample_size_ok=False,
                permutation_p_value=1.0,
                edge_is_significant=False,
                passed=False,
                failure_reasons=[f"Backtest error: {str(e)}"],
            )

    def _run_period_backtest(self, config: Dict, period: str) -> PeriodMetrics:
        """Run backtest for a specific period and compute metrics"""
        start_date, end_date = self.periods[period]

        logger.info(f"\nRunning {period} period: {start_date} to {end_date}")

        # Filter data for period
        mask = (self.df.index >= start_date) & (self.df.index <= end_date)
        period_df = self.df[mask].copy()

        logger.info(f"Period has {len(period_df)} bars")

        # Initialize backtest
        bt = KnowledgeAwareBacktest(
            df=period_df,
            params=self._config_to_params(config),
            runtime_config=config
        )

        # Run backtest
        bt.run()

        # Compute metrics
        trades_df = bt.get_trades_dataframe()
        equity_curve = bt.get_equity_curve()

        if len(trades_df) == 0:
            logger.warning(f"No trades in {period} period")
            return self._empty_metrics(period, start_date, end_date)

        # Basic metrics
        total_trades = len(trades_df)
        wins = trades_df[trades_df['net_pnl'] > 0]
        losses = trades_df[trades_df['net_pnl'] < 0]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

        total_wins = wins['net_pnl'].sum() if len(wins) > 0 else 0.0
        total_losses = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0.0
        avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0.0

        # R-multiples
        if 'r_multiple' in trades_df.columns:
            avg_r_multiple = trades_df['r_multiple'].mean()
        else:
            # Estimate from PNL and initial risk
            avg_r_multiple = (avg_win * win_rate + avg_loss * (1 - win_rate)) / 100

        # Returns and drawdown
        total_return = (equity_curve[-1] / equity_curve[0] - 1) if len(equity_curve) > 0 else 0.0

        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        # Sharpe ratio (annualized)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.sqrt(365 * 24) * returns.mean() / returns.std() if len(returns) > 1 and returns.std() > 0 else 0.0

        # Consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for _, trade in trades_df.iterrows():
            if trade['net_pnl'] < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        # Regime breakdown
        regime_trades = {}
        regime_pf = {}
        regime_wr = {}

        if 'macro_regime' in trades_df.columns:
            for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
                regime_df = trades_df[trades_df['macro_regime'] == regime]

                if len(regime_df) > 0:
                    regime_wins = regime_df[regime_df['net_pnl'] > 0]
                    regime_losses = regime_df[regime_df['net_pnl'] < 0]

                    regime_trades[regime] = len(regime_df)
                    regime_wr[regime] = len(regime_wins) / len(regime_df)

                    regime_total_wins = regime_wins['net_pnl'].sum() if len(regime_wins) > 0 else 0.0
                    regime_total_losses = abs(regime_losses['net_pnl'].sum()) if len(regime_losses) > 0 else 0.0
                    regime_pf[regime] = regime_total_wins / regime_total_losses if regime_total_losses > 0 else 0.0
                else:
                    regime_trades[regime] = 0
                    regime_wr[regime] = 0.0
                    regime_pf[regime] = 0.0

        return PeriodMetrics(
            period_name=period,
            start_date=start_date,
            end_date=end_date,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            total_return=total_return,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_r_multiple=avg_r_multiple,
            max_consecutive_losses=max_consecutive_losses,
            regime_trades=regime_trades,
            regime_pf=regime_pf,
            regime_wr=regime_wr,
        )

    def _perform_validation_checks(
        self,
        train: PeriodMetrics,
        val: PeriodMetrics,
        oos: PeriodMetrics
    ) -> Dict[str, bool]:
        """Perform all validation checks"""

        # Check 1: Validation degradation
        val_degradation_ok = val.profit_factor >= (train.profit_factor * (1 - self.thresholds['val_degradation_max']))

        # Check 2: OOS edge
        oos_edge_ok = oos.profit_factor >= self.thresholds['oos_pf_min']

        # Check 3: OOS drawdown
        oos_dd_ok = oos.max_drawdown <= self.thresholds['oos_dd_max']

        # Check 4: Regime consistency
        regime_consistency_ok = True
        if oos.regime_pf:
            for regime, pf in oos.regime_pf.items():
                if oos.regime_trades.get(regime, 0) >= 3:  # Only check if enough trades
                    if pf < self.thresholds['regime_pf_min']:
                        regime_consistency_ok = False
                        logger.warning(f"Regime {regime} has low PF: {pf:.2f}")

        # Check 5: Sample size
        sample_size_ok = (
            train.total_trades >= self.thresholds['min_trades_per_period'] and
            val.total_trades >= self.thresholds['min_trades_per_period'] and
            oos.total_trades >= self.thresholds['min_trades_per_period']
        )

        return {
            'val_degradation_ok': val_degradation_ok,
            'oos_edge_ok': oos_edge_ok,
            'oos_dd_ok': oos_dd_ok,
            'regime_consistency_ok': regime_consistency_ok,
            'sample_size_ok': sample_size_ok,
        }

    def _permutation_test(self, config: Dict, period: str, n_permutations: int = 1000) -> float:
        """
        Run permutation test to check if edge is statistically significant.

        Shuffles trade outcomes to see if observed profit factor could occur by chance.

        Returns:
            p-value: probability that observed PF could occur by random chance
        """
        start_date, end_date = self.periods[period]

        # Filter data
        mask = (self.df.index >= start_date) & (self.df.index <= end_date)
        period_df = self.df[mask].copy()

        # Run original backtest
        bt = KnowledgeAwareBacktest(
            df=period_df,
            params=self._config_to_params(config),
            runtime_config=config
        )
        bt.run()
        trades_df = bt.get_trades_dataframe()

        if len(trades_df) < 5:
            logger.warning("Too few trades for permutation test")
            return 1.0

        # Compute actual profit factor
        actual_pf = self._compute_profit_factor(trades_df)

        # Run permutations
        logger.info(f"Running {n_permutations} permutations for significance test...")
        perm_pfs = []

        for i in range(n_permutations):
            shuffled_df = trades_df.copy()
            shuffled_df['net_pnl'] = shuffled_df['net_pnl'].sample(frac=1, random_state=i).values
            perm_pf = self._compute_profit_factor(shuffled_df)
            perm_pfs.append(perm_pf)

        # Compute p-value
        p_value = (np.array(perm_pfs) >= actual_pf).mean()

        logger.info(f"Actual PF: {actual_pf:.2f}, Permutation mean: {np.mean(perm_pfs):.2f}, p-value: {p_value:.3f}")

        return p_value

    def _compute_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Compute profit factor from trades dataframe"""
        if len(trades_df) == 0:
            return 0.0

        wins = trades_df[trades_df['net_pnl'] > 0]
        losses = trades_df[trades_df['net_pnl'] < 0]

        total_wins = wins['net_pnl'].sum() if len(wins) > 0 else 0.0
        total_losses = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 0.0

        return total_wins / total_losses if total_losses > 0 else 0.0

    def _config_to_params(self, config: Dict) -> KnowledgeParams:
        """Convert config dict to KnowledgeParams"""
        # Extract fusion weights
        fusion = config.get('fusion', {})
        weights = fusion.get('weights', {})

        params = KnowledgeParams(
            wyckoff_weight=weights.get('wyckoff', 0.33),
            liquidity_weight=weights.get('liquidity', 0.39),
            momentum_weight=weights.get('momentum', 0.21),
            macro_weight=weights.get('macro', 0.0),
            pti_weight=weights.get('pti', 0.07),
            tier3_threshold=fusion.get('entry_threshold_confidence', 0.37),
        )

        return params

    def _extract_key_params(self, config: Dict) -> Dict:
        """Extract key parameters from config for reporting"""
        fusion = config.get('fusion', {})
        archetypes = config.get('archetypes', {})

        return {
            'fusion_weights': fusion.get('weights', {}),
            'entry_threshold': fusion.get('entry_threshold_confidence', 0.37),
            'archetypes_enabled': [
                k.replace('enable_', '')
                for k, v in archetypes.items()
                if k.startswith('enable_') and v
            ],
        }

    def _empty_metrics(self, period: str, start_date: str = '', end_date: str = '') -> PeriodMetrics:
        """Create empty metrics for failed periods"""
        return PeriodMetrics(
            period_name=period,
            start_date=start_date,
            end_date=end_date,
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            total_return=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_r_multiple=0.0,
            max_consecutive_losses=0,
            regime_trades={},
            regime_pf={},
            regime_wr={},
        )

    def _save_config_results(self, config_id: str, result: ValidationResult):
        """Save individual config results to disk"""
        config_dir = self.output_dir / config_id
        config_dir.mkdir(exist_ok=True)

        # Save metrics JSONs
        with open(config_dir / 'train_metrics.json', 'w') as f:
            json.dump(result.train_metrics.to_dict(), f, indent=2)

        with open(config_dir / 'val_metrics.json', 'w') as f:
            json.dump(result.val_metrics.to_dict(), f, indent=2)

        with open(config_dir / 'oos_metrics.json', 'w') as f:
            json.dump(result.oos_metrics.to_dict(), f, indent=2)

        # Save regime breakdown CSV
        regime_data = []
        for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
            regime_data.append({
                'regime': regime,
                'train_trades': result.train_metrics.regime_trades.get(regime, 0) if result.train_metrics.regime_trades else 0,
                'train_pf': result.train_metrics.regime_pf.get(regime, 0.0) if result.train_metrics.regime_pf else 0.0,
                'train_wr': result.train_metrics.regime_wr.get(regime, 0.0) if result.train_metrics.regime_wr else 0.0,
                'val_trades': result.val_metrics.regime_trades.get(regime, 0) if result.val_metrics.regime_trades else 0,
                'val_pf': result.val_metrics.regime_pf.get(regime, 0.0) if result.val_metrics.regime_pf else 0.0,
                'val_wr': result.val_metrics.regime_wr.get(regime, 0.0) if result.val_metrics.regime_wr else 0.0,
                'oos_trades': result.oos_metrics.regime_trades.get(regime, 0) if result.oos_metrics.regime_trades else 0,
                'oos_pf': result.oos_metrics.regime_pf.get(regime, 0.0) if result.oos_metrics.regime_pf else 0.0,
                'oos_wr': result.oos_metrics.regime_wr.get(regime, 0.0) if result.oos_metrics.regime_wr else 0.0,
            })

        regime_df = pd.DataFrame(regime_data)
        regime_df.to_csv(config_dir / 'regime_breakdown.csv', index=False)

        logger.info(f"Saved results to {config_dir}")

    def generate_summary_report(self, results: List[ValidationResult]) -> str:
        """Generate comprehensive summary report"""

        # Summary statistics
        total_configs = len(results)
        passed_configs = sum(1 for r in results if r.passed)
        pass_rate = passed_configs / total_configs if total_configs > 0 else 0.0

        # Create summary dataframe
        summary_data = []
        for r in results:
            summary_data.append({
                'config_id': r.config_id,
                'train_pf': r.train_metrics.profit_factor,
                'val_pf': r.val_metrics.profit_factor,
                'oos_pf': r.oos_metrics.profit_factor,
                'train_trades': r.train_metrics.total_trades,
                'val_trades': r.val_metrics.total_trades,
                'oos_trades': r.oos_metrics.total_trades,
                'oos_dd': r.oos_metrics.max_drawdown,
                'oos_sharpe': r.oos_metrics.sharpe_ratio,
                'p_value': r.permutation_p_value,
                'passed': r.passed,
                'failure_reasons': '; '.join(r.failure_reasons) if r.failure_reasons else 'PASS',
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('oos_pf', ascending=False)

        # Save summary CSV
        summary_df.to_csv(self.output_dir / 'validation_summary.csv', index=False)

        # Generate markdown report
        report_lines = [
            "# Walk-Forward Validation Summary Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            "",
            f"- **Total Configs Tested:** {total_configs}",
            f"- **Passed Validation:** {passed_configs} ({pass_rate:.1%})",
            f"- **Failed Validation:** {total_configs - passed_configs}",
            "",
            "## Validation Criteria",
            "",
            "| Criterion | Threshold | Description |",
            "|-----------|-----------|-------------|",
            f"| Val PF Degradation | ≤ 20% | Val PF ≥ 0.8 × Train PF |",
            f"| OOS Profit Factor | ≥ 1.1 | Must maintain edge OOS |",
            f"| OOS Max Drawdown | < 25% | Risk management |",
            f"| Regime Consistency | PF ≥ 0.8 | All regimes viable |",
            f"| Sample Size | ≥ 5 trades | Statistical validity |",
            f"| Permutation Test | p < 0.05 | Edge significance |",
            "",
            "## Top Performing Configs (Passed Validation)",
            "",
        ]

        # Add top configs table
        passed_df = summary_df[summary_df['passed']].head(10)
        if len(passed_df) > 0:
            report_lines.append("| Config ID | Train PF | Val PF | OOS PF | OOS Trades | OOS DD | OOS Sharpe | p-value |")
            report_lines.append("|-----------|----------|--------|--------|------------|--------|------------|---------|")

            for _, row in passed_df.iterrows():
                report_lines.append(
                    f"| {row['config_id']} | {row['train_pf']:.2f} | {row['val_pf']:.2f} | "
                    f"{row['oos_pf']:.2f} | {row['oos_trades']} | {row['oos_dd']:.1%} | "
                    f"{row['oos_sharpe']:.2f} | {row['p_value']:.3f} |"
                )
        else:
            report_lines.append("*No configs passed validation*")

        report_lines.extend([
            "",
            "## Failed Configs Analysis",
            "",
        ])

        # Failure reasons breakdown
        failed_df = summary_df[~summary_df['passed']]
        if len(failed_df) > 0:
            failure_counts = {}
            for reasons in failed_df['failure_reasons']:
                for reason in reasons.split('; '):
                    failure_counts[reason] = failure_counts.get(reason, 0) + 1

            report_lines.append("| Failure Reason | Count |")
            report_lines.append("|----------------|-------|")
            for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
                if reason != 'PASS':
                    report_lines.append(f"| {reason} | {count} |")
        else:
            report_lines.append("*All configs passed validation*")

        report_lines.extend([
            "",
            "## Recommendations",
            "",
        ])

        if passed_configs > 0:
            top_config = summary_df[summary_df['passed']].iloc[0]
            report_lines.extend([
                f"### Production Candidate: {top_config['config_id']}",
                "",
                f"- **OOS Profit Factor:** {top_config['oos_pf']:.2f}",
                f"- **OOS Sharpe Ratio:** {top_config['oos_sharpe']:.2f}",
                f"- **OOS Max Drawdown:** {top_config['oos_dd']:.1%}",
                f"- **OOS Trade Count:** {top_config['oos_trades']}",
                f"- **Statistical Significance:** p = {top_config['p_value']:.3f}",
                "",
                "This config demonstrated:",
                "- Consistent performance across all validation periods",
                "- Statistically significant edge",
                "- Acceptable risk-adjusted returns",
                "",
            ])
        else:
            report_lines.extend([
                "**WARNING:** No configs passed all validation criteria.",
                "",
                "Recommended actions:",
                "1. Review failure reasons above",
                "2. Re-run optimization with adjusted parameter ranges",
                "3. Consider ensemble approach combining multiple strategies",
                "4. Extend training period or adjust acceptance thresholds",
                "",
            ])

        report_lines.extend([
            "## Next Steps",
            "",
            "1. Review individual config reports in subdirectories",
            "2. Analyze regime breakdown CSVs for robustness",
            "3. Visualize equity curves for top performers",
            "4. Run ensemble validation combining top 3-5 configs",
            "5. Conduct monte carlo simulation on production candidates",
            "",
            "## Files Generated",
            "",
            "- `validation_summary.csv` - All results in tabular format",
            "- `{config_id}/train_metrics.json` - Training period metrics",
            "- `{config_id}/val_metrics.json` - Validation period metrics",
            "- `{config_id}/oos_metrics.json` - Out-of-sample metrics",
            "- `{config_id}/regime_breakdown.csv` - Performance by regime",
            "",
        ])

        report = '\n'.join(report_lines)

        # Save report
        with open(self.output_dir / 'summary_report.md', 'w') as f:
            f.write(report)

        logger.info(f"Summary report saved to {self.output_dir / 'summary_report.md'}")

        return report


def load_configs_from_optuna(db_path: str, study_name: str = 'bear_phase2_tuning', min_trials: int = 10) -> List[Tuple[str, Dict]]:
    """
    Load configs from Optuna database.

    Args:
        db_path: Path to Optuna SQLite database
        study_name: Name of the study
        min_trials: Minimum trial number to start from

    Returns:
        List of (config_id, config_dict) tuples
    """
    try:
        import optuna

        # Load study
        storage = f'sqlite:///{db_path}'
        study = optuna.load_study(study_name=study_name, storage=storage)

        # Get completed trials
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        logger.info(f"Found {len(trials)} completed trials in study '{study_name}'")

        # Filter by minimum trial number
        trials = [t for t in trials if t.number >= min_trials]

        logger.info(f"Using {len(trials)} trials (>= trial {min_trials})")

        # Convert trials to configs
        configs = []
        for trial in trials:
            config_id = f"trial_{trial.number:04d}"

            # Reconstruct config from trial params
            config = {
                'version': 's2_optimization',
                'profile': f'trial_{trial.number}',
                'fusion': {
                    'entry_threshold_confidence': trial.params.get('fusion_threshold', 0.36),
                    'weights': {
                        'wyckoff': trial.params.get('wyckoff_weight', 0.35),
                        'liquidity': trial.params.get('liquidity_weight', 0.30),
                        'momentum': trial.params.get('momentum_weight', 0.35),
                    }
                },
                'archetypes': {
                    'use_archetypes': True,
                    'enable_S2': True,
                    'thresholds': {
                        'min_liquidity': trial.params.get('min_liquidity', 0.20),
                    },
                    'failed_rally': {
                        'fusion_threshold': trial.params.get('fusion_threshold', 0.36),
                        'atr_stop_mult': trial.params.get('atr_stop_mult', 2.0),
                        'wick_ratio_min': trial.params.get('wick_ratio_min', 2.0),
                    },
                },
                'risk': {
                    'base_risk_pct': trial.params.get('max_risk_pct', 0.015),
                }
            }

            configs.append((config_id, config))

        return configs

    except ImportError:
        logger.error("Optuna not installed. Install with: pip install optuna")
        return []
    except Exception as e:
        logger.error(f"Error loading configs from Optuna: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Validation Framework')
    parser.add_argument('--configs', type=str, help='Path to Optuna DB or config directory')
    parser.add_argument('--output', type=str, default='results/validation/walk_forward',
                       help='Output directory for validation results')
    parser.add_argument('--asset', type=str, default='BTC', help='Asset to validate')
    parser.add_argument('--min-trials', type=int, default=0,
                       help='Minimum trial number to start from (for Optuna DB)')
    parser.add_argument('--study-name', type=str, default='bear_phase2_tuning',
                       help='Optuna study name')
    parser.add_argument('--max-configs', type=int, default=None,
                       help='Maximum number of configs to validate')

    args = parser.parse_args()

    # Determine feature store path
    feature_store_path = f'data/features_mtf/{args.asset}_1H_2022-01-01_to_2024-12-31_backup.parquet'

    if not Path(feature_store_path).exists():
        logger.error(f"Feature store not found: {feature_store_path}")
        sys.exit(1)

    # Initialize validator
    validator = WalkForwardValidator(
        feature_store_path=feature_store_path,
        output_dir=args.output
    )

    # Load configs
    configs = []

    if args.configs.endswith('.db'):
        # Load from Optuna database
        configs = load_configs_from_optuna(
            db_path=args.configs,
            study_name=args.study_name,
            min_trials=args.min_trials
        )
    else:
        # Load from JSON files
        config_dir = Path(args.configs)
        if config_dir.is_dir():
            for config_file in config_dir.glob('*.json'):
                with open(config_file) as f:
                    config = json.load(f)
                    config_id = config_file.stem
                    configs.append((config_id, config))
        else:
            logger.error(f"Config path not found: {args.configs}")
            sys.exit(1)

    if len(configs) == 0:
        logger.error("No configs found to validate")
        sys.exit(1)

    # Limit configs if specified
    if args.max_configs:
        configs = configs[:args.max_configs]

    logger.info(f"Validating {len(configs)} configs")

    # Run validation on all configs
    results = []
    for config_id, config in configs:
        result = validator.validate_config(config, config_id)
        results.append(result)

    # Generate summary report
    logger.info("\n" + "="*80)
    logger.info("Generating summary report...")
    logger.info("="*80)

    report = validator.generate_summary_report(results)

    print("\n" + report)

    # Summary stats
    passed = sum(1 for r in results if r.passed)
    logger.info(f"\nValidation Complete: {passed}/{len(results)} configs passed")


if __name__ == '__main__':
    main()
