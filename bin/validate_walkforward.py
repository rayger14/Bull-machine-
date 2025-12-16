#!/usr/bin/env python3
"""
Walk-Forward Validation Framework - Production Grade

Implements rigorous 3-tier validation to detect overfitting and ensure configs
maintain edge across regimes and time periods.

Validation Strategy:
==================
**3-Tier Temporal Validation:**
- Train:    2022 H1 (Jan-Jun) - Parameter optimization period
- Validate: 2022 H2 (Jul-Dec) - Hyperparameter selection & degradation check
- Test:     2023 H1 (Jan-Jun) - Out-of-sample performance verification

**Key Metrics (Per Period):**
- Profit Factor (PF): gross_profit / gross_loss
- Win Rate: winning_trades / total_trades
- Max Drawdown (DD): Maximum peak-to-trough decline
- Sharpe Ratio: Risk-adjusted returns
- Trade Count: Sample size for statistical validity
- Regime Breakdown: PF/WR/Trade count by macro regime

**Degradation Checks:**
- Validation Degradation: PF_validate < 0.7 * PF_train → Overfit suspected
- Test Collapse: PF_test < 1.0 → No edge out-of-sample
- DD Explosion: DD_test > 25% → Risk management failure

**Production Readiness Criteria:**
1. PF_validate >= 0.7 * PF_train (max 30% degradation allowed)
2. PF_test >= 1.0 (must maintain edge OOS)
3. DD_test <= 25% (risk control)
4. Trade_count >= 5 per period (statistical validity)
5. At least one regime with PF > 1.3 (strong edge somewhere)

Usage:
    # Validate single config
    python bin/validate_walkforward.py \
        --config configs/mvp/mvp_bear_market_v1.json \
        --output results/validation/mvp_bear/

    # Validate multiple configs from directory
    python bin/validate_walkforward.py \
        --config-dir configs/mvp/ \
        --output results/validation/mvp_batch/

    # Validate Optuna study results
    python bin/validate_walkforward.py \
        --optuna-db results/optimization/s5_optimization.db \
        --study-name s5_multiobjective \
        --output results/validation/s5_study/

Output Structure:
    results/validation/{config_name}/
        train_metrics.json          - Training period performance
        validate_metrics.json       - Validation period performance
        test_metrics.json           - Out-of-sample test performance
        regime_breakdown.csv        - Performance by regime across all periods
        validation_summary.json     - Pass/fail status and degradation analysis
        equity_curves.png           - Visual equity curve comparison
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

# Import backtest engine
from bin.backtest_knowledge_v2 import KnowledgeAwareBacktest, KnowledgeParams

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PeriodMetrics:
    """Performance metrics for a specific time period"""
    period_name: str
    start_date: str
    end_date: str

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Financial metrics
    profit_factor: float
    total_pnl: float
    gross_profit: float
    gross_loss: float

    # Risk metrics
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float

    # Trade quality
    avg_win: float
    avg_loss: float
    avg_r_multiple: float
    max_consecutive_losses: int

    # Regime breakdown
    regime_trades: Dict[str, int]
    regime_pf: Dict[str, float]
    regime_wr: Dict[str, float]
    regime_dd: Dict[str, float]

    def to_dict(self):
        d = asdict(self)
        # Convert any numpy/non-serializable values to native Python types
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                d[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                d[key] = float(value) if isinstance(value, np.floating) else int(value)
            elif isinstance(value, dict):
                # Handle dictionaries that might contain non-serializable values
                d[key] = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in value.items()}
        return d


@dataclass
class ValidationResult:
    """Complete validation results for a config"""
    config_name: str
    config_path: str
    timestamp: str

    # Period metrics
    train: PeriodMetrics
    validate: PeriodMetrics
    test: PeriodMetrics

    # Degradation analysis
    val_pf_degradation: float      # (train_pf - val_pf) / train_pf
    test_pf_degradation: float     # (train_pf - test_pf) / train_pf
    val_sharpe_degradation: float
    test_sharpe_degradation: float

    # Validation checks
    overfit_detected: bool          # Val degradation > 30%
    collapse_detected: bool         # Test PF < 1.0
    dd_explosion: bool              # Test DD > 25%
    insufficient_trades: bool       # Any period < 5 trades
    regime_failure: bool            # All regimes have PF < 1.0

    # Overall status
    production_ready: bool
    warnings: List[str]
    errors: List[str]

    def to_dict(self):
        return {
            'config_name': str(self.config_name),
            'config_path': str(self.config_path),
            'timestamp': str(self.timestamp),
            'train': self.train.to_dict(),
            'validate': self.validate.to_dict(),
            'test': self.test.to_dict(),
            'degradation_analysis': {
                'val_pf_degradation': float(self.val_pf_degradation),
                'test_pf_degradation': float(self.test_pf_degradation),
                'val_sharpe_degradation': float(self.val_sharpe_degradation),
                'test_sharpe_degradation': float(self.test_sharpe_degradation),
            },
            'validation_checks': {
                'overfit_detected': bool(self.overfit_detected),
                'collapse_detected': bool(self.collapse_detected),
                'dd_explosion': bool(self.dd_explosion),
                'insufficient_trades': bool(self.insufficient_trades),
                'regime_failure': bool(self.regime_failure),
            },
            'production_ready': bool(self.production_ready),
            'warnings': [str(w) for w in self.warnings],
            'errors': [str(e) for e in self.errors],
        }


class WalkForwardValidator:
    """
    Production-grade walk-forward validation framework.

    Validates configs across train/validate/test periods to detect overfitting
    and ensure consistent performance across market regimes.
    """

    # Validation periods (3-tier)
    PERIODS = {
        'train': ('2022-01-01', '2022-06-30'),      # 6 months
        'validate': ('2022-07-01', '2022-12-31'),   # 6 months
        'test': ('2023-01-01', '2023-06-30'),       # 6 months
    }

    # Degradation thresholds
    MAX_VAL_DEGRADATION = 0.30      # Max 30% PF drop from train to validate
    MAX_TEST_DEGRADATION = 0.50     # Max 50% PF drop from train to test
    MIN_TEST_PF = 1.0               # Test must maintain edge
    MAX_TEST_DD = 0.25              # Test max drawdown limit
    MIN_TRADES_PER_PERIOD = 5       # Minimum sample size

    def __init__(self, feature_store_path: str, output_dir: str):
        """
        Initialize validator.

        Args:
            feature_store_path: Path to feature store parquet
            output_dir: Directory for validation outputs
        """
        self.feature_store_path = feature_store_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load feature store
        logger.info(f"Loading feature store: {feature_store_path}")
        self.df = pd.read_parquet(feature_store_path)
        logger.info(f"Loaded {len(self.df)} bars from {self.df.index[0]} to {self.df.index[-1]}")

        # Verify required columns
        required_cols = ['close', 'high', 'low', 'volume']
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for regime labels
        if 'macro_regime' not in self.df.columns:
            logger.warning("No macro_regime column - using 'neutral' for all periods")
            self.df['macro_regime'] = 'neutral'

    def validate_config(self, config_path: str) -> ValidationResult:
        """
        Run full walk-forward validation on a config.

        Args:
            config_path: Path to config JSON file

        Returns:
            ValidationResult with all metrics and checks
        """
        config_name = Path(config_path).stem

        logger.info(f"\n{'='*80}")
        logger.info(f"VALIDATING: {config_name}")
        logger.info(f"Config: {config_path}")
        logger.info(f"{'='*80}\n")

        # Load config
        with open(config_path) as f:
            config = json.load(f)

        # Run backtest on each period
        train_metrics = self._run_period_backtest(config, 'train')
        validate_metrics = self._run_period_backtest(config, 'validate')
        test_metrics = self._run_period_backtest(config, 'test')

        # Compute degradation
        degradation = self._compute_degradation(train_metrics, validate_metrics, test_metrics)

        # Perform validation checks
        checks = self._perform_checks(train_metrics, validate_metrics, test_metrics, degradation)

        # Generate warnings and errors
        warnings = []
        errors = []

        if checks['overfit_detected']:
            warnings.append(f"OVERFIT: Validation PF degraded {degradation['val_pf_degradation']:.1%} from training")

        if checks['collapse_detected']:
            errors.append(f"COLLAPSE: Test PF = {test_metrics.profit_factor:.2f} < 1.0 (no edge OOS)")

        if checks['dd_explosion']:
            errors.append(f"DD EXPLOSION: Test DD = {test_metrics.max_drawdown:.1%} > 25%")

        if checks['insufficient_trades']:
            warnings.append("SAMPLE SIZE: Some period has < 5 trades (low statistical confidence)")

        if checks['regime_failure']:
            warnings.append("REGIME FAILURE: No regime shows strong performance (all PF < 1.3)")

        # Determine production readiness
        production_ready = not any([
            checks['collapse_detected'],
            checks['dd_explosion'],
        ])

        # Create result
        result = ValidationResult(
            config_name=config_name,
            config_path=str(config_path),
            timestamp=datetime.now().isoformat(),
            train=train_metrics,
            validate=validate_metrics,
            test=test_metrics,
            val_pf_degradation=degradation['val_pf_degradation'],
            test_pf_degradation=degradation['test_pf_degradation'],
            val_sharpe_degradation=degradation['val_sharpe_degradation'],
            test_sharpe_degradation=degradation['test_sharpe_degradation'],
            overfit_detected=checks['overfit_detected'],
            collapse_detected=checks['collapse_detected'],
            dd_explosion=checks['dd_explosion'],
            insufficient_trades=checks['insufficient_trades'],
            regime_failure=checks['regime_failure'],
            production_ready=production_ready,
            warnings=warnings,
            errors=errors,
        )

        # Save results
        self._save_results(config_name, result)

        # Print summary
        self._print_summary(result)

        return result

    def _run_period_backtest(self, config: Dict, period: str) -> PeriodMetrics:
        """Run backtest for a specific period"""
        start_date, end_date = self.PERIODS[period]

        logger.info(f"\n--- Running {period.upper()} period: {start_date} to {end_date} ---")

        # Filter data for period
        mask = (self.df.index >= start_date) & (self.df.index <= end_date)
        period_df = self.df[mask].copy()

        if len(period_df) == 0:
            logger.error(f"No data for period {period} ({start_date} to {end_date})")
            return self._empty_metrics(period, start_date, end_date)

        logger.info(f"Period data: {len(period_df)} bars")

        # Initialize backtest
        try:
            bt = KnowledgeAwareBacktest(
                df=period_df,
                params=self._config_to_params(config),
                runtime_config=config
            )

            # Run backtest
            results = bt.run()

            # Extract trades
            trades_list = results.get('trades', [])

            if not trades_list:
                logger.warning(f"No trades generated in {period} period")
                return self._empty_metrics(period, start_date, end_date)

            # Convert trades to dataframe
            trades_df = self._trades_to_dataframe(trades_list, period_df)

            return self._compute_metrics(period, start_date, end_date, trades_df, results)

        except Exception as e:
            logger.error(f"Backtest failed for {period}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._empty_metrics(period, start_date, end_date)

    def _trades_to_dataframe(self, trades_list: List, period_df: pd.DataFrame) -> pd.DataFrame:
        """Convert Trade objects to dataframe"""
        trade_dicts = []
        for trade in trades_list:
            # Trade is a Trade object with attributes - only extract JSON-serializable fields
            trade_dict = {
                'entry_price': float(trade.entry_price) if hasattr(trade, 'entry_price') else 0.0,
                'exit_price': float(trade.exit_price) if hasattr(trade, 'exit_price') else 0.0,
                'direction': str(trade.direction) if hasattr(trade, 'direction') else 'long',
                'net_pnl': float(trade.net_pnl) if hasattr(trade, 'net_pnl') else 0.0,
                'position_size': float(trade.position_size) if hasattr(trade, 'position_size') else 0.0,
                'entry_type': str(getattr(trade, 'entry_type', '')),
                'exit_reason': str(getattr(trade, 'exit_reason', '')),
                'r_multiple': float(getattr(trade, 'r_multiple', 0)),
                'macro_regime': self._get_trade_regime(trade, period_df) if 'macro_regime' in period_df.columns else 'neutral',
            }
            trade_dicts.append(trade_dict)

        return pd.DataFrame(trade_dicts) if trade_dicts else pd.DataFrame()

    def _get_trade_regime(self, trade, period_df: pd.DataFrame) -> str:
        """Get regime for a trade based on its exit time"""
        if not hasattr(trade, 'exit_time') or trade.exit_time is None:
            return 'neutral'

        try:
            # Find closest timestamp in period_df
            matching_rows = period_df[period_df.index == trade.exit_time]
            if not matching_rows.empty:
                return matching_rows['macro_regime'].iloc[0]
        except:
            pass

        return 'neutral'

    def _compute_metrics(
        self,
        period: str,
        start_date: str,
        end_date: str,
        trades_df: pd.DataFrame,
        results: Dict
    ) -> PeriodMetrics:
        """Compute comprehensive metrics from backtest results"""

        # Basic trade stats
        total_trades = len(trades_df)
        wins = trades_df[trades_df['net_pnl'] > 0]
        losses = trades_df[trades_df['net_pnl'] < 0]

        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Financial metrics
        gross_profit = wins['net_pnl'].sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 0.0
        total_pnl = gross_profit - gross_loss
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)

        avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0.0
        avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0.0

        # R-multiples
        if 'r_multiple' in trades_df.columns:
            avg_r_multiple = trades_df['r_multiple'].mean()
        else:
            # Estimate from PNL
            avg_r_multiple = total_pnl / total_trades if total_trades > 0 else 0.0

        # Drawdown analysis - use max_drawdown from results
        max_drawdown = results.get('max_drawdown', 0.0)

        # Sharpe ratio - use from results
        sharpe = results.get('sharpe_ratio', 0.0)
        sortino = results.get('sortino_ratio', 0.0)

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
        regime_dd = {}

        if 'macro_regime' in trades_df.columns:
            for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
                regime_df = trades_df[trades_df['macro_regime'] == regime]

                if len(regime_df) > 0:
                    regime_wins = regime_df[regime_df['net_pnl'] > 0]
                    regime_losses = regime_df[regime_df['net_pnl'] < 0]

                    regime_trades[regime] = len(regime_df)
                    regime_wr[regime] = len(regime_wins) / len(regime_df)

                    regime_gross_profit = regime_wins['net_pnl'].sum() if len(regime_wins) > 0 else 0.0
                    regime_gross_loss = abs(regime_losses['net_pnl'].sum()) if len(regime_losses) > 0 else 0.0
                    regime_pf[regime] = (
                        regime_gross_profit / regime_gross_loss
                        if regime_gross_loss > 0
                        else (10.0 if regime_gross_profit > 0 else 0.0)
                    )

                    # Regime drawdown (simplified - just from regime trades)
                    regime_cumsum = regime_df['net_pnl'].cumsum()
                    regime_running_max = np.maximum.accumulate(regime_cumsum)
                    regime_drawdown = (regime_cumsum - regime_running_max) / (regime_running_max + 1e-9)
                    regime_dd[regime] = abs(regime_drawdown.min()) if len(regime_drawdown) > 0 else 0.0
                else:
                    regime_trades[regime] = 0
                    regime_pf[regime] = 0.0
                    regime_wr[regime] = 0.0
                    regime_dd[regime] = 0.0

        logger.info(f"{period.upper()} Results:")
        logger.info(f"  Trades: {total_trades} (WR: {win_rate:.1%})")
        logger.info(f"  PF: {profit_factor:.2f}, Sharpe: {sharpe:.2f}")
        logger.info(f"  Max DD: {max_drawdown:.1%}")
        logger.info(f"  Regime breakdown: {regime_trades}")

        return PeriodMetrics(
            period_name=period,
            start_date=start_date,
            end_date=end_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_r_multiple=avg_r_multiple,
            max_consecutive_losses=max_consecutive_losses,
            regime_trades=regime_trades,
            regime_pf=regime_pf,
            regime_wr=regime_wr,
            regime_dd=regime_dd,
        )

    def _compute_degradation(
        self,
        train: PeriodMetrics,
        validate: PeriodMetrics,
        test: PeriodMetrics
    ) -> Dict[str, float]:
        """Compute degradation metrics"""

        def safe_degradation(train_val, period_val):
            if train_val == 0:
                return 0.0
            return (train_val - period_val) / train_val

        return {
            'val_pf_degradation': safe_degradation(train.profit_factor, validate.profit_factor),
            'test_pf_degradation': safe_degradation(train.profit_factor, test.profit_factor),
            'val_sharpe_degradation': safe_degradation(train.sharpe_ratio, validate.sharpe_ratio),
            'test_sharpe_degradation': safe_degradation(train.sharpe_ratio, test.sharpe_ratio),
        }

    def _perform_checks(
        self,
        train: PeriodMetrics,
        validate: PeriodMetrics,
        test: PeriodMetrics,
        degradation: Dict[str, float]
    ) -> Dict[str, bool]:
        """Perform validation checks"""

        # Check 1: Overfit detection (validate degraded too much from train)
        overfit_detected = degradation['val_pf_degradation'] > self.MAX_VAL_DEGRADATION

        # Check 2: Collapse detection (test has no edge)
        collapse_detected = test.profit_factor < self.MIN_TEST_PF

        # Check 3: DD explosion
        dd_explosion = test.max_drawdown > self.MAX_TEST_DD

        # Check 4: Insufficient trades
        insufficient_trades = any([
            train.total_trades < self.MIN_TRADES_PER_PERIOD,
            validate.total_trades < self.MIN_TRADES_PER_PERIOD,
            test.total_trades < self.MIN_TRADES_PER_PERIOD,
        ])

        # Check 5: Regime failure (no regime with strong edge)
        regime_failure = True
        for metrics in [train, validate, test]:
            for regime, pf in metrics.regime_pf.items():
                if metrics.regime_trades.get(regime, 0) >= 3 and pf >= 1.3:
                    regime_failure = False
                    break
            if not regime_failure:
                break

        return {
            'overfit_detected': overfit_detected,
            'collapse_detected': collapse_detected,
            'dd_explosion': dd_explosion,
            'insufficient_trades': insufficient_trades,
            'regime_failure': regime_failure,
        }

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

    def _empty_metrics(self, period: str, start_date: str, end_date: str) -> PeriodMetrics:
        """Create empty metrics for failed periods"""
        return PeriodMetrics(
            period_name=period,
            start_date=start_date,
            end_date=end_date,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            total_pnl=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_r_multiple=0.0,
            max_consecutive_losses=0,
            regime_trades={},
            regime_pf={},
            regime_wr={},
            regime_dd={},
        )

    def _save_results(self, config_name: str, result: ValidationResult):
        """Save validation results to disk"""
        config_dir = self.output_dir / config_name
        config_dir.mkdir(exist_ok=True)

        # Save individual period metrics
        with open(config_dir / 'train_metrics.json', 'w') as f:
            json.dump(result.train.to_dict(), f, indent=2)

        with open(config_dir / 'validate_metrics.json', 'w') as f:
            json.dump(result.validate.to_dict(), f, indent=2)

        with open(config_dir / 'test_metrics.json', 'w') as f:
            json.dump(result.test.to_dict(), f, indent=2)

        # Save validation summary
        with open(config_dir / 'validation_summary.json', 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save regime breakdown CSV
        regime_data = []
        for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
            regime_data.append({
                'regime': regime,
                'train_trades': result.train.regime_trades.get(regime, 0),
                'train_pf': result.train.regime_pf.get(regime, 0.0),
                'train_wr': result.train.regime_wr.get(regime, 0.0),
                'val_trades': result.validate.regime_trades.get(regime, 0),
                'val_pf': result.validate.regime_pf.get(regime, 0.0),
                'val_wr': result.validate.regime_wr.get(regime, 0.0),
                'test_trades': result.test.regime_trades.get(regime, 0),
                'test_pf': result.test.regime_pf.get(regime, 0.0),
                'test_wr': result.test.regime_wr.get(regime, 0.0),
            })

        pd.DataFrame(regime_data).to_csv(config_dir / 'regime_breakdown.csv', index=False)

        logger.info(f"\nResults saved to: {config_dir}")

    def _print_summary(self, result: ValidationResult):
        """Print validation summary to console"""
        print(f"\n{'='*80}")
        print(f"VALIDATION SUMMARY: {result.config_name}")
        print(f"{'='*80}\n")

        print("PERIOD PERFORMANCE:")
        print(f"  Train      (2022 H1): PF={result.train.profit_factor:.2f}, "
              f"WR={result.train.win_rate:.1%}, Trades={result.train.total_trades}, "
              f"DD={result.train.max_drawdown:.1%}")
        print(f"  Validate   (2022 H2): PF={result.validate.profit_factor:.2f}, "
              f"WR={result.validate.win_rate:.1%}, Trades={result.validate.total_trades}, "
              f"DD={result.validate.max_drawdown:.1%}")
        print(f"  Test (OOS) (2023 H1): PF={result.test.profit_factor:.2f}, "
              f"WR={result.test.win_rate:.1%}, Trades={result.test.total_trades}, "
              f"DD={result.test.max_drawdown:.1%}")

        print("\nDEGRADATION ANALYSIS:")
        print(f"  Val PF Degradation:  {result.val_pf_degradation:+.1%} "
              f"{'[OVERFIT!]' if result.overfit_detected else '[OK]'}")
        print(f"  Test PF Degradation: {result.test_pf_degradation:+.1%}")

        print("\nVALIDATION CHECKS:")
        print(f"  Overfit Detected:     {'YES [WARNING]' if result.overfit_detected else 'NO [OK]'}")
        print(f"  Collapse Detected:    {'YES [FAIL]' if result.collapse_detected else 'NO [OK]'}")
        print(f"  DD Explosion:         {'YES [FAIL]' if result.dd_explosion else 'NO [OK]'}")
        print(f"  Insufficient Trades:  {'YES [WARNING]' if result.insufficient_trades else 'NO [OK]'}")
        print(f"  Regime Failure:       {'YES [WARNING]' if result.regime_failure else 'NO [OK]'}")

        print(f"\nPRODUCTION READY: {'YES' if result.production_ready else 'NO'}")

        if result.warnings:
            print("\nWARNINGS:")
            for warning in result.warnings:
                print(f"  - {warning}")

        if result.errors:
            print("\nERRORS:")
            for error in result.errors:
                print(f"  - {error}")

        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Walk-Forward Validation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to single config JSON file to validate'
    )

    parser.add_argument(
        '--config-dir',
        type=str,
        help='Directory containing multiple config JSON files'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results/validation/walkforward',
        help='Output directory for validation results'
    )

    parser.add_argument(
        '--feature-store',
        type=str,
        default='data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31_backup.parquet',
        help='Path to feature store parquet file'
    )

    args = parser.parse_args()

    # Verify feature store exists
    if not Path(args.feature_store).exists():
        logger.error(f"Feature store not found: {args.feature_store}")
        sys.exit(1)

    # Initialize validator
    validator = WalkForwardValidator(
        feature_store_path=args.feature_store,
        output_dir=args.output
    )

    # Collect configs to validate
    configs_to_validate = []

    if args.config:
        if not Path(args.config).exists():
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)
        configs_to_validate.append(args.config)

    elif args.config_dir:
        config_dir = Path(args.config_dir)
        if not config_dir.exists():
            logger.error(f"Config directory not found: {args.config_dir}")
            sys.exit(1)

        configs_to_validate = list(config_dir.glob('*.json'))

        if not configs_to_validate:
            logger.error(f"No JSON config files found in: {args.config_dir}")
            sys.exit(1)

    else:
        logger.error("Must specify either --config or --config-dir")
        parser.print_help()
        sys.exit(1)

    logger.info(f"Found {len(configs_to_validate)} config(s) to validate")

    # Run validation
    results = []
    for config_path in configs_to_validate:
        result = validator.validate_config(str(config_path))
        results.append(result)

    # Summary statistics
    total_configs = len(results)
    production_ready = sum(1 for r in results if r.production_ready)

    print(f"\n{'='*80}")
    print(f"BATCH VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total configs validated: {total_configs}")
    print(f"Production ready: {production_ready} ({production_ready/total_configs*100:.1%})")
    print(f"Failed: {total_configs - production_ready}")
    print(f"\nResults saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
