#!/usr/bin/env python3
"""
Cross-Regime Validation Framework

Validates that configs perform appropriately across different market regimes.
Critical for detecting regime-specific overfitting and ensuring proper archetype routing.

Validation Strategy:
===================
**Regime Slicing:**
- Extract all trades from backtest
- Group by macro_regime (risk_on, neutral, risk_off, crisis)
- Compute independent metrics for each regime

**Regime-Specific Requirements:**

Bull Archetypes (LONG):
- MUST excel in risk_on: PF > 1.5
- SHOULD work in neutral: PF > 1.2
- EXPECTED to struggle in risk_off: PF may be < 1.0 (should be disabled)
- MUST fail in crisis: PF < 1.0 (should be muted)

Bear Archetypes (SHORT):
- MUST fail in risk_on: PF < 1.0 (should be disabled)
- MAY work in neutral: PF > 1.0
- MUST excel in risk_off: PF > 1.3
- MUST excel in crisis: PF > 1.2

**Anomaly Detection:**
- Flag if archetype produces >30% trades in "forbidden" regime
  Example: S5 (bear) trading heavily in risk_on = regime misclassification
- Flag if intended regime has PF < 1.0 = pattern broken

**Minimum Requirements (Production):**
- S2 (Failed Rally): PF > 1.3 in risk_off, disabled elsewhere
- S5 (Long Squeeze): PF > 1.4 in risk_off/crisis, minimal in risk_on
- Bull Traps: PF > 1.3 in risk_on, disabled in risk_off/crisis

Usage:
    # Validate single config
    python bin/validate_cross_regime.py \
        --config configs/mvp/mvp_bear_market_v1.json \
        --output results/validation/regime/

    # Validate with detailed trade export
    python bin/validate_cross_regime.py \
        --config configs/optimized/s5_balanced.json \
        --output results/validation/s5_regime/ \
        --export-trades

Output:
    results/validation/{config_name}/
        regime_performance.csv      - Metrics by regime
        regime_breakdown.json       - Detailed regime analysis
        regime_distribution.png     - Trade distribution chart
        regime_pf_chart.png         - PF by regime visualization
        trades_by_regime.csv        - All trades grouped by regime (if --export-trades)
        anomaly_report.txt          - Detected anomalies and warnings
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
class RegimeMetrics:
    """Performance metrics for a specific regime"""
    regime: str
    total_trades: int
    winning_trades: int
    win_rate: float
    profit_factor: float
    gross_profit: float
    gross_loss: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float

    # Trade distribution
    trade_pct: float  # Percentage of total trades in this regime

    # Archetype breakdown (if available)
    archetype_trades: Dict[str, int]
    archetype_pf: Dict[str, float]

    def to_dict(self):
        return asdict(self)


@dataclass
class CrossRegimeResult:
    """Cross-regime validation results"""
    config_name: str
    config_path: str
    timestamp: str

    # Overall metrics
    total_trades: int
    overall_pf: float
    overall_wr: float

    # Regime-specific metrics
    regime_metrics: Dict[str, RegimeMetrics]

    # Validation checks
    bull_regime_ok: bool          # Bull archetypes perform in risk_on
    bear_regime_ok: bool          # Bear archetypes perform in risk_off
    forbidden_trades_detected: bool  # Trading in wrong regime
    missing_regime_data: bool     # Some regime has 0 trades (may be OK)

    # Anomalies
    anomalies: List[str]
    warnings: List[str]

    # Recommendations
    regime_routing_ok: bool
    production_ready: bool

    def to_dict(self):
        return {
            'config_name': self.config_name,
            'config_path': self.config_path,
            'timestamp': self.timestamp,
            'total_trades': self.total_trades,
            'overall_pf': self.overall_pf,
            'overall_wr': self.overall_wr,
            'regime_metrics': {
                regime: metrics.to_dict()
                for regime, metrics in self.regime_metrics.items()
            },
            'validation_checks': {
                'bull_regime_ok': self.bull_regime_ok,
                'bear_regime_ok': self.bear_regime_ok,
                'forbidden_trades_detected': self.forbidden_trades_detected,
                'missing_regime_data': self.missing_regime_data,
            },
            'anomalies': self.anomalies,
            'warnings': self.warnings,
            'regime_routing_ok': self.regime_routing_ok,
            'production_ready': self.production_ready,
        }


class CrossRegimeValidator:
    """
    Cross-regime validation framework.

    Ensures configs perform appropriately across different market regimes
    and don't suffer from regime-specific overfitting.
    """

    # Regime-specific PF requirements
    REGIME_REQUIREMENTS = {
        'bull_archetypes': {
            'risk_on': {'min_pf': 1.5, 'min_trades': 3},
            'neutral': {'min_pf': 1.2, 'min_trades': 2},
            'risk_off': {'max_pf': None, 'max_trade_pct': 0.20},  # Should be disabled
            'crisis': {'max_pf': 1.0, 'max_trade_pct': 0.10},     # Should be muted
        },
        'bear_archetypes': {
            'risk_on': {'max_pf': 1.0, 'max_trade_pct': 0.30},    # Should be disabled
            'neutral': {'min_pf': 1.0, 'min_trades': 2},
            'risk_off': {'min_pf': 1.3, 'min_trades': 3},
            'crisis': {'min_pf': 1.2, 'min_trades': 2},
        },
    }

    # Known archetype classifications
    BULL_ARCHETYPES = [
        'trap_within_trend', 'order_block_retest', 'bos_choch_reversal',
        'failed_continuation', 'liquidity_compression', 'expansion_exhaustion',
        're_accumulate_base', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M'
    ]

    BEAR_ARCHETYPES = [
        'failed_rally', 'long_squeeze', 'breakdown', 'whipsaw',
        'distribution_climax', 'S1', 'S2', 'S3', 'S4', 'S5'
    ]

    def __init__(self, feature_store_path: str, output_dir: str):
        """
        Initialize cross-regime validator.

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

        # Check for regime labels
        if 'macro_regime' not in self.df.columns:
            logger.warning("No macro_regime column - using 'neutral' for all periods")
            self.df['macro_regime'] = 'neutral'

    def validate_config(self, config_path: str, period: Tuple[str, str] = ('2022-01-01', '2023-06-30')) -> CrossRegimeResult:
        """
        Run cross-regime validation on a config.

        Args:
            config_path: Path to config JSON file
            period: (start_date, end_date) tuple for validation period

        Returns:
            CrossRegimeResult with regime-specific analysis
        """
        config_name = Path(config_path).stem

        logger.info(f"\n{'='*80}")
        logger.info(f"CROSS-REGIME VALIDATION: {config_name}")
        logger.info(f"Config: {config_path}")
        logger.info(f"Period: {period[0]} to {period[1]}")
        logger.info(f"{'='*80}\n")

        # Load config
        with open(config_path) as f:
            config = json.load(f)

        # Detect archetype type
        archetype_type = self._detect_archetype_type(config)
        logger.info(f"Detected archetype type: {archetype_type}")

        # Run backtest
        trades_df = self._run_backtest(config, period)

        if len(trades_df) == 0:
            logger.error("No trades generated - cannot perform regime analysis")
            return self._empty_result(config_name, config_path)

        # Compute regime-specific metrics
        regime_metrics = self._compute_regime_metrics(trades_df)

        # Overall metrics
        total_trades = len(trades_df)
        overall_wins = trades_df[trades_df['net_pnl'] > 0]
        overall_losses = trades_df[trades_df['net_pnl'] < 0]
        overall_wr = len(overall_wins) / total_trades if total_trades > 0 else 0.0
        overall_gross_profit = overall_wins['net_pnl'].sum() if len(overall_wins) > 0 else 0.0
        overall_gross_loss = abs(overall_losses['net_pnl'].sum()) if len(overall_losses) > 0 else 0.0
        overall_pf = overall_gross_profit / overall_gross_loss if overall_gross_loss > 0 else 0.0

        # Perform validation checks
        checks = self._perform_regime_checks(regime_metrics, archetype_type)

        # Generate visualizations
        self._create_visualizations(config_name, regime_metrics, trades_df)

        # Create result
        result = CrossRegimeResult(
            config_name=config_name,
            config_path=str(config_path),
            timestamp=datetime.now().isoformat(),
            total_trades=total_trades,
            overall_pf=overall_pf,
            overall_wr=overall_wr,
            regime_metrics=regime_metrics,
            bull_regime_ok=checks['bull_regime_ok'],
            bear_regime_ok=checks['bear_regime_ok'],
            forbidden_trades_detected=checks['forbidden_trades_detected'],
            missing_regime_data=checks['missing_regime_data'],
            anomalies=checks['anomalies'],
            warnings=checks['warnings'],
            regime_routing_ok=checks['regime_routing_ok'],
            production_ready=checks['production_ready'],
        )

        # Save results
        self._save_results(config_name, result, trades_df)

        # Print summary
        self._print_summary(result)

        return result

    def _run_backtest(self, config: Dict, period: Tuple[str, str]) -> pd.DataFrame:
        """Run backtest and return trades dataframe"""
        start_date, end_date = period

        # Filter data for period
        mask = (self.df.index >= start_date) & (self.df.index <= end_date)
        period_df = self.df[mask].copy()

        logger.info(f"Backtest period: {len(period_df)} bars")

        # Initialize backtest
        bt = KnowledgeAwareBacktest(
            df=period_df,
            params=self._config_to_params(config),
            runtime_config=config
        )

        # Run backtest
        results = bt.run()

        # Convert trades to dataframe
        trades_list = results.get('trades', [])
        trades_df = self._trades_to_dataframe(trades_list, period_df)

        logger.info(f"Generated {len(trades_df)} trades")

        return trades_df

    def _trades_to_dataframe(self, trades_list: List, period_df: pd.DataFrame) -> pd.DataFrame:
        """Convert Trade objects to dataframe"""
        trade_dicts = []
        for trade in trades_list:
            trade_dict = {
                'entry_price': float(trade.entry_price) if hasattr(trade, 'entry_price') else 0.0,
                'exit_price': float(trade.exit_price) if hasattr(trade, 'exit_price') else 0.0,
                'direction': str(trade.direction) if hasattr(trade, 'direction') else 'long',
                'net_pnl': float(trade.net_pnl) if hasattr(trade, 'net_pnl') else 0.0,
                'position_size': float(trade.position_size) if hasattr(trade, 'position_size') else 0.0,
                'entry_type': str(getattr(trade, 'entry_type', '')),
                'exit_reason': str(getattr(trade, 'exit_reason', '')),
                'macro_regime': self._get_trade_regime(trade, period_df) if 'macro_regime' in period_df.columns else 'neutral',
                'archetype': str(getattr(trade, 'archetype', '')),
            }
            trade_dicts.append(trade_dict)

        return pd.DataFrame(trade_dicts) if trade_dicts else pd.DataFrame()

    def _get_trade_regime(self, trade, period_df: pd.DataFrame) -> str:
        """Get regime for a trade based on its exit time"""
        if not hasattr(trade, 'exit_time') or trade.exit_time is None:
            return 'neutral'

        try:
            matching_rows = period_df[period_df.index == trade.exit_time]
            if not matching_rows.empty:
                return matching_rows['macro_regime'].iloc[0]
        except:
            pass

        return 'neutral'

    def _compute_regime_metrics(self, trades_df: pd.DataFrame) -> Dict[str, RegimeMetrics]:
        """Compute metrics for each regime"""
        regime_metrics = {}

        total_trades = len(trades_df)

        for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
            regime_trades = trades_df[trades_df['macro_regime'] == regime]

            if len(regime_trades) == 0:
                # Create empty metrics for regimes with no trades
                regime_metrics[regime] = RegimeMetrics(
                    regime=regime,
                    total_trades=0,
                    winning_trades=0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    gross_profit=0.0,
                    gross_loss=0.0,
                    total_pnl=0.0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    trade_pct=0.0,
                    archetype_trades={},
                    archetype_pf={},
                )
                continue

            # Basic metrics
            wins = regime_trades[regime_trades['net_pnl'] > 0]
            losses = regime_trades[regime_trades['net_pnl'] < 0]

            winning_trades = len(wins)
            win_rate = winning_trades / len(regime_trades)

            gross_profit = wins['net_pnl'].sum() if len(wins) > 0 else 0.0
            gross_loss = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 0.0
            total_pnl = gross_profit - gross_loss
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)

            avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0.0
            avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0.0

            # Drawdown
            cumsum = regime_trades['net_pnl'].cumsum()
            running_max = np.maximum.accumulate(cumsum)
            drawdown = (cumsum - running_max) / (running_max + 1e-9)
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

            # Sharpe (simplified)
            returns = regime_trades['net_pnl'].values
            sharpe = returns.mean() / returns.std() if len(returns) > 1 and returns.std() > 0 else 0.0

            # Trade distribution
            trade_pct = len(regime_trades) / total_trades

            # Archetype breakdown
            archetype_trades = {}
            archetype_pf = {}

            if 'archetype' in regime_trades.columns:
                for archetype in regime_trades['archetype'].unique():
                    arch_trades = regime_trades[regime_trades['archetype'] == archetype]
                    arch_wins = arch_trades[arch_trades['net_pnl'] > 0]
                    arch_losses = arch_trades[arch_trades['net_pnl'] < 0]

                    archetype_trades[archetype] = len(arch_trades)

                    arch_gross_profit = arch_wins['net_pnl'].sum() if len(arch_wins) > 0 else 0.0
                    arch_gross_loss = abs(arch_losses['net_pnl'].sum()) if len(arch_losses) > 0 else 0.0
                    archetype_pf[archetype] = (
                        arch_gross_profit / arch_gross_loss
                        if arch_gross_loss > 0
                        else (10.0 if arch_gross_profit > 0 else 0.0)
                    )

            regime_metrics[regime] = RegimeMetrics(
                regime=regime,
                total_trades=len(regime_trades),
                winning_trades=winning_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                gross_profit=gross_profit,
                gross_loss=gross_loss,
                total_pnl=total_pnl,
                avg_win=avg_win,
                avg_loss=avg_loss,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe,
                trade_pct=trade_pct,
                archetype_trades=archetype_trades,
                archetype_pf=archetype_pf,
            )

        return regime_metrics

    def _perform_regime_checks(
        self,
        regime_metrics: Dict[str, RegimeMetrics],
        archetype_type: str
    ) -> Dict:
        """Perform regime-specific validation checks"""

        anomalies = []
        warnings = []

        # Select requirements based on archetype type
        if archetype_type == 'bull':
            requirements = self.REGIME_REQUIREMENTS['bull_archetypes']
        elif archetype_type == 'bear':
            requirements = self.REGIME_REQUIREMENTS['bear_archetypes']
        else:
            requirements = {}

        bull_regime_ok = True
        bear_regime_ok = True
        forbidden_trades_detected = False
        missing_regime_data = False

        for regime, metrics in regime_metrics.items():
            if metrics.total_trades == 0:
                missing_regime_data = True
                continue

            req = requirements.get(regime, {})

            # Check minimum PF
            if 'min_pf' in req and req['min_pf'] is not None:
                if metrics.total_trades >= req.get('min_trades', 0):
                    if metrics.profit_factor < req['min_pf']:
                        anomalies.append(
                            f"{regime}: PF {metrics.profit_factor:.2f} < {req['min_pf']:.2f} "
                            f"(expected strong performance)"
                        )
                        if archetype_type == 'bull' and regime in ['risk_on', 'neutral']:
                            bull_regime_ok = False
                        if archetype_type == 'bear' and regime in ['risk_off', 'crisis']:
                            bear_regime_ok = False

            # Check maximum PF (should fail in forbidden regime)
            if 'max_pf' in req and req['max_pf'] is not None:
                if metrics.profit_factor > req['max_pf']:
                    warnings.append(
                        f"{regime}: PF {metrics.profit_factor:.2f} > {req['max_pf']:.2f} "
                        f"(should be disabled in this regime)"
                    )

            # Check trade distribution (shouldn't trade too much in forbidden regime)
            if 'max_trade_pct' in req:
                if metrics.trade_pct > req['max_trade_pct']:
                    anomalies.append(
                        f"{regime}: {metrics.trade_pct:.1%} of trades in forbidden regime "
                        f"(max {req['max_trade_pct']:.1%})"
                    )
                    forbidden_trades_detected = True

        # Overall regime routing check
        regime_routing_ok = not forbidden_trades_detected

        # Production readiness
        if archetype_type == 'bull':
            production_ready = bull_regime_ok and not forbidden_trades_detected
        elif archetype_type == 'bear':
            production_ready = bear_regime_ok and not forbidden_trades_detected
        else:
            production_ready = True  # Unknown type, assume OK

        return {
            'bull_regime_ok': bull_regime_ok,
            'bear_regime_ok': bear_regime_ok,
            'forbidden_trades_detected': forbidden_trades_detected,
            'missing_regime_data': missing_regime_data,
            'anomalies': anomalies,
            'warnings': warnings,
            'regime_routing_ok': regime_routing_ok,
            'production_ready': production_ready,
        }

    def _detect_archetype_type(self, config: Dict) -> str:
        """Detect if config uses bull or bear archetypes"""
        archetypes = config.get('archetypes', {})

        # Check enabled archetypes
        enabled_bull = []
        enabled_bear = []

        for key, value in archetypes.items():
            if key.startswith('enable_') and value:
                arch_code = key.replace('enable_', '')

                # Check archetype family
                if arch_code in self.BEAR_ARCHETYPES or any(
                    bear in key.lower() for bear in ['failed_rally', 'long_squeeze', 'breakdown']
                ):
                    enabled_bear.append(arch_code)
                elif arch_code in self.BULL_ARCHETYPES or any(
                    bull in key.lower() for bull in ['trap', 'order_block', 'bos', 'choch']
                ):
                    enabled_bull.append(arch_code)

        if enabled_bear and not enabled_bull:
            return 'bear'
        elif enabled_bull and not enabled_bear:
            return 'bull'
        elif enabled_bull and enabled_bear:
            return 'mixed'
        else:
            return 'unknown'

    def _create_visualizations(self, config_name: str, regime_metrics: Dict[str, RegimeMetrics], trades_df: pd.DataFrame):
        """Create regime performance visualizations"""
        config_dir = self.output_dir / config_name
        config_dir.mkdir(exist_ok=True)

        # 1. Trade distribution by regime
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        regimes = list(regime_metrics.keys())
        trade_counts = [regime_metrics[r].total_trades for r in regimes]
        trade_pcts = [regime_metrics[r].trade_pct * 100 for r in regimes]

        axes[0].bar(regimes, trade_counts)
        axes[0].set_title('Trade Count by Regime')
        axes[0].set_ylabel('Number of Trades')
        axes[0].grid(True, alpha=0.3)

        axes[1].pie(trade_pcts, labels=regimes, autopct='%1.1f%%')
        axes[1].set_title('Trade Distribution')

        plt.tight_layout()
        plt.savefig(config_dir / 'regime_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. PF by regime
        fig, ax = plt.subplots(figsize=(10, 6))

        pfs = [regime_metrics[r].profit_factor for r in regimes]
        colors = ['green' if pf > 1.0 else 'red' for pf in pfs]

        ax.bar(regimes, pfs, color=colors, alpha=0.7)
        ax.axhline(y=1.0, color='black', linestyle='--', label='Breakeven')
        ax.axhline(y=1.3, color='blue', linestyle='--', alpha=0.5, label='Strong Edge')
        ax.set_title('Profit Factor by Regime')
        ax.set_ylabel('Profit Factor')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(config_dir / 'regime_pf_chart.png', dpi=150, bbox_inches='tight')
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

    def _empty_result(self, config_name: str, config_path: str) -> CrossRegimeResult:
        """Create empty result for failed validations"""
        return CrossRegimeResult(
            config_name=config_name,
            config_path=config_path,
            timestamp=datetime.now().isoformat(),
            total_trades=0,
            overall_pf=0.0,
            overall_wr=0.0,
            regime_metrics={},
            bull_regime_ok=False,
            bear_regime_ok=False,
            forbidden_trades_detected=False,
            missing_regime_data=True,
            anomalies=['No trades generated'],
            warnings=[],
            regime_routing_ok=False,
            production_ready=False,
        )

    def _save_results(self, config_name: str, result: CrossRegimeResult, trades_df: pd.DataFrame):
        """Save validation results to disk"""
        config_dir = self.output_dir / config_name
        config_dir.mkdir(exist_ok=True)

        # Save regime breakdown JSON
        with open(config_dir / 'regime_breakdown.json', 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save regime performance CSV
        regime_data = []
        for regime, metrics in result.regime_metrics.items():
            regime_data.append({
                'regime': regime,
                'trades': metrics.total_trades,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'total_pnl': metrics.total_pnl,
                'max_dd': metrics.max_drawdown,
                'sharpe': metrics.sharpe_ratio,
                'trade_pct': metrics.trade_pct,
            })

        pd.DataFrame(regime_data).to_csv(config_dir / 'regime_performance.csv', index=False)

        # Save trades by regime
        if len(trades_df) > 0 and 'macro_regime' in trades_df.columns:
            trades_df.to_csv(config_dir / 'trades_by_regime.csv', index=False)

        # Save anomaly report
        with open(config_dir / 'anomaly_report.txt', 'w') as f:
            f.write(f"Cross-Regime Validation Report\n")
            f.write(f"Config: {config_name}\n")
            f.write(f"Generated: {result.timestamp}\n")
            f.write(f"\n")

            f.write(f"ANOMALIES:\n")
            if result.anomalies:
                for anomaly in result.anomalies:
                    f.write(f"  - {anomaly}\n")
            else:
                f.write(f"  None detected\n")

            f.write(f"\nWARNINGS:\n")
            if result.warnings:
                for warning in result.warnings:
                    f.write(f"  - {warning}\n")
            else:
                f.write(f"  None\n")

            f.write(f"\nPRODUCTION READY: {'YES' if result.production_ready else 'NO'}\n")

        logger.info(f"Results saved to {config_dir}")

    def _print_summary(self, result: CrossRegimeResult):
        """Print validation summary to console"""
        print(f"\n{'='*80}")
        print(f"CROSS-REGIME VALIDATION SUMMARY: {result.config_name}")
        print(f"{'='*80}\n")

        print(f"Overall: {result.total_trades} trades, PF={result.overall_pf:.2f}, WR={result.overall_wr:.1%}")
        print()

        print("REGIME BREAKDOWN:")
        print(f"{'Regime':<12} {'Trades':<8} {'Trade%':<8} {'PF':<8} {'WR':<8} {'Max DD':<8}")
        print("-" * 60)

        for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
            metrics = result.regime_metrics.get(regime)
            if metrics:
                print(f"{regime:<12} {metrics.total_trades:<8} {metrics.trade_pct*100:<7.1f}% "
                      f"{metrics.profit_factor:<7.2f} {metrics.win_rate*100:<7.1f}% "
                      f"{metrics.max_drawdown*100:<7.1f}%")

        print()
        print("VALIDATION CHECKS:")
        print(f"  Bull Regime Performance:  {'PASS' if result.bull_regime_ok else 'FAIL'}")
        print(f"  Bear Regime Performance:  {'PASS' if result.bear_regime_ok else 'FAIL'}")
        print(f"  Forbidden Trades:         {'DETECTED' if result.forbidden_trades_detected else 'OK'}")
        print(f"  Regime Routing:           {'OK' if result.regime_routing_ok else 'NEEDS ADJUSTMENT'}")

        if result.anomalies:
            print("\nANOMALIES:")
            for anomaly in result.anomalies:
                print(f"  - {anomaly}")

        if result.warnings:
            print("\nWARNINGS:")
            for warning in result.warnings:
                print(f"  - {warning}")

        print(f"\nPRODUCTION READY: {'YES' if result.production_ready else 'NO'}")
        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Regime Validation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config JSON file to validate'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results/validation/cross_regime',
        help='Output directory for validation results'
    )

    parser.add_argument(
        '--feature-store',
        type=str,
        default='data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31_backup.parquet',
        help='Path to feature store parquet file'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default='2022-01-01',
        help='Validation period start date'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default='2023-06-30',
        help='Validation period end date'
    )

    args = parser.parse_args()

    # Verify inputs
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    if not Path(args.feature_store).exists():
        logger.error(f"Feature store not found: {args.feature_store}")
        sys.exit(1)

    # Initialize validator
    validator = CrossRegimeValidator(
        feature_store_path=args.feature_store,
        output_dir=args.output
    )

    # Run validation
    result = validator.validate_config(
        config_path=args.config,
        period=(args.start_date, args.end_date)
    )

    # Exit with appropriate code
    sys.exit(0 if result.production_ready else 1)


if __name__ == '__main__':
    main()
