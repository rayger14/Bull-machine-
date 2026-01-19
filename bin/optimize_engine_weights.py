#!/usr/bin/env python3
"""
Engine-Level Weight Optimization System

Optimizes how different domain engines are weighted in the final fusion score
after individual archetypes have been calibrated.

Domain Engines:
    - Structure: Wyckoff + SMC (BOS, OB, FVG quality)
    - Liquidity: liquidity_score, sweep strength
    - Momentum: RSI, ADX, MACD
    - Wyckoff Events: LPS, Spring, BC, UTAD detection
    - Macro: VIX/DXY/regime alignment
    - PTI: Psychological trap index (future)

Optimization Strategy:
    Uses Optuna to find optimal weight combination that maximizes:
    - Profit factor across all regimes
    - Consistent trade count (25-40 per regime-year)
    - Minimal regime-specific degradation

Output:
    - results/engine_weights/optuna_study.db
    - results/engine_weights/optimal_weights.json
    - results/engine_weights/weight_sensitivity.png
    - results/engine_weights/regime_breakdown.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Import backtest engine
from bin.backtest_knowledge_v2 import KnowledgeParams, KnowledgeAwareBacktest

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EngineWeightOptimizer:
    """
    Optimizes domain engine weights for meta-fusion scoring.

    Given calibrated archetype configurations, finds optimal weights
    that maximize performance across all market regimes.
    """

    def __init__(self, feature_store_path: str, output_dir: str = "results/engine_weights"):
        """
        Initialize optimizer.

        Args:
            feature_store_path: Path to feature store parquet
            output_dir: Directory for output files
        """
        self.feature_store_path = Path(feature_store_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load feature store
        logger.info(f"Loading feature store: {self.feature_store_path}")
        self.df = pd.read_parquet(self.feature_store_path)
        logger.info(f"Loaded {len(self.df)} bars from {self.df.index[0]} to {self.df.index[-1]}")

        # Validate required features
        self._validate_features()

        # Store baseline weights (current system)
        self.baseline_weights = {
            'structure': 0.331,      # Wyckoff + SMC combined
            'liquidity': 0.392,      # Liquidity score
            'momentum': 0.205,       # RSI + ADX + MACD
            'macro': 0.072,          # VIX/DXY/regime (residual)
        }

    def _validate_features(self):
        """Validate that required domain score features exist."""
        required = [
            'close', 'open', 'high', 'low', 'volume',
            'tf1h_fusion_score', 'tf4h_fusion_score',
            'liquidity_score', 'rsi_14', 'adx_14',
            'regime_label', 'k2_fusion_score'
        ]

        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        logger.info(f"✓ Feature validation passed ({len(required)} required features)")

    def _compute_fusion_with_weights(
        self,
        row: pd.Series,
        weights: Dict[str, float]
    ) -> float:
        """
        Compute meta-fusion score using custom weights.

        Combines domain scores:
            - Structure: tf4h_fusion_score (Wyckoff + SMC)
            - Liquidity: liquidity_score
            - Momentum: derived from rsi_14, adx_14
            - Macro: VIX_Z, regime alignment

        Args:
            row: Feature row
            weights: Dict of domain weights

        Returns:
            Weighted fusion score [0, 1]
        """
        # Extract domain scores
        structure_score = row.get('tf4h_fusion_score', 0.5)
        liquidity_score = row.get('liquidity_score', 0.5)

        # Derive momentum score from RSI and ADX
        rsi = row.get('rsi_14', 50.0)
        adx = row.get('adx_14', 20.0)

        # Momentum: RSI deviation from 50, scaled by ADX strength
        rsi_deviation = abs(rsi - 50.0) / 50.0  # 0-1
        adx_strength = min(adx / 40.0, 1.0)     # 0-1
        momentum_score = 0.5 + (rsi_deviation * adx_strength * 0.5)

        # Macro score: VIX alignment + regime quality
        vix_z = row.get('VIX_Z', 0.0)
        regime_conf = row.get('regime_confidence', 0.5)

        # Macro: favorable when VIX is low (negative Z) and regime confidence high
        macro_score = 0.5 - (vix_z * 0.1) + (regime_conf * 0.3)
        macro_score = np.clip(macro_score, 0.0, 1.0)

        # Weighted combination
        fusion = (
            weights['structure'] * structure_score +
            weights['liquidity'] * liquidity_score +
            weights['momentum'] * momentum_score +
            weights['macro'] * macro_score
        )

        return np.clip(fusion, 0.0, 1.0)

    def _apply_weights_to_dataframe(
        self,
        df: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Apply custom weights to compute new fusion scores.

        Args:
            df: Feature store dataframe
            weights: Domain weights

        Returns:
            Modified dataframe with 'fusion_score_weighted' column
        """
        df = df.copy()

        # Compute weighted fusion for each row
        df['fusion_score_weighted'] = df.apply(
            lambda row: self._compute_fusion_with_weights(row, weights),
            axis=1
        )

        return df

    def _run_backtest_with_weights(
        self,
        weights: Dict[str, float],
        config_params: Dict = None
    ) -> Dict:
        """
        Run backtest with custom domain weights.

        Args:
            weights: Domain engine weights
            config_params: Optional backtest parameters

        Returns:
            Backtest results dict
        """
        # Apply weights to dataframe
        df_weighted = self._apply_weights_to_dataframe(self.df, weights)

        # Override k2_fusion_score with weighted version
        df_weighted['k2_fusion_score'] = df_weighted['fusion_score_weighted']

        # Default backtest params (from calibrated configs)
        if config_params is None:
            config_params = {
                'tier1_threshold': 0.45,
                'tier2_threshold': 0.35,
                'tier3_threshold': 0.25,
                'require_m1m2_confirmation': False,
                'require_macro_alignment': True,
                'atr_stop_mult': 2.0,
                'trailing_atr_mult': 2.0,
                'max_hold_bars': 168,  # 7 days
                'max_risk_pct': 0.02,
                'volatility_scaling': True,
                'use_smart_exits': True,
                'breakeven_after_tp1': True,
            }

        # Create params object
        params = KnowledgeParams(**config_params)

        # Run backtest
        try:
            backtest = KnowledgeAwareBacktest(
                df_weighted,
                params,
                starting_capital=10000.0
            )
            results = backtest.run()
            return results
        except Exception as e:
            logger.warning(f"Backtest failed with weights {weights}: {e}")
            return {
                'total_pnl': -1000.0,
                'total_trades': 0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'win_rate': 0.0
            }

    def _split_by_regime(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split dataframe by regime for regime-specific analysis."""
        regimes = {}

        if 'regime_label' in df.columns:
            for regime in df['regime_label'].unique():
                if pd.notna(regime):
                    regimes[regime] = df[df['regime_label'] == regime].copy()
        else:
            # No regime data - return full dataset
            regimes['ALL'] = df.copy()

        return regimes

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.

        Samples domain weights and evaluates performance.

        Args:
            trial: Optuna trial object

        Returns:
            Composite score (higher is better)
        """
        # Sample weights (constrained to sum = 1.0)
        w_structure = trial.suggest_float('w_structure', 0.20, 0.50)
        w_liquidity = trial.suggest_float('w_liquidity', 0.20, 0.50)
        w_momentum = trial.suggest_float('w_momentum', 0.10, 0.30)
        w_macro = trial.suggest_float('w_macro', 0.05, 0.20)

        # Normalize to sum = 1.0
        total = w_structure + w_liquidity + w_momentum + w_macro
        weights = {
            'structure': w_structure / total,
            'liquidity': w_liquidity / total,
            'momentum': w_momentum / total,
            'macro': w_macro / total,
        }

        # Run backtest with these weights
        results = self._run_backtest_with_weights(weights)

        # Log metrics
        trial.set_user_attr('total_pnl', results['total_pnl'])
        trial.set_user_attr('total_trades', results['total_trades'])
        trial.set_user_attr('profit_factor', results['profit_factor'])
        trial.set_user_attr('sharpe_ratio', results['sharpe_ratio'])
        trial.set_user_attr('max_drawdown', results['max_drawdown'])
        trial.set_user_attr('win_rate', results['win_rate'])

        # Penalize low trade counts
        if results['total_trades'] < 10:
            return -1e6

        # Penalize low profit factor
        if results['profit_factor'] < 0.5:
            return -1e6

        # Composite score
        pf = results['profit_factor']
        trades = results['total_trades']
        dd = results['max_drawdown']
        sharpe = results['sharpe_ratio']

        # Trade frequency penalty (want 25-40 trades per regime-year)
        # For full 2022-2024 (3 years), expect 75-120 trades
        trade_penalty = min(np.sqrt(trades / 30.0), 1.0)

        # Drawdown penalty
        dd_penalty = 1.0 / (1.0 + dd)

        # Composite
        score = pf * trade_penalty * dd_penalty

        # Sharpe bonus
        if sharpe > 1.0:
            score *= (1.0 + sharpe / 10.0)

        return score

    def optimize(self, n_trials: int = 100) -> Dict:
        """
        Run Optuna optimization to find optimal weights.

        Args:
            n_trials: Number of trials

        Returns:
            Dict with optimization results
        """
        logger.info("=" * 80)
        logger.info("ENGINE WEIGHT OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"Feature store: {self.feature_store_path}")
        logger.info(f"Trials: {n_trials}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("")

        # Create study
        study_path = self.output_dir / "optuna_study.db"
        sampler = TPESampler(seed=42)

        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='engine_weights',
            storage=f'sqlite:///{study_path}',
            load_if_exists=True
        )

        # Run optimization
        logger.info("Running optimization...")
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=1
        )

        # Extract best trial
        best_trial = study.best_trial
        logger.info("")
        logger.info("=" * 80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best score: {best_trial.value:.4f}")
        logger.info("")
        logger.info("Optimal Weights:")

        # Normalize weights
        total = sum([
            best_trial.params['w_structure'],
            best_trial.params['w_liquidity'],
            best_trial.params['w_momentum'],
            best_trial.params['w_macro']
        ])

        optimal_weights = {
            'structure': best_trial.params['w_structure'] / total,
            'liquidity': best_trial.params['w_liquidity'] / total,
            'momentum': best_trial.params['w_momentum'] / total,
            'macro': best_trial.params['w_macro'] / total,
        }

        for domain, weight in optimal_weights.items():
            baseline = self.baseline_weights[domain]
            change = ((weight - baseline) / baseline) * 100
            logger.info(f"  {domain:12s}: {weight:.3f} (baseline: {baseline:.3f}, Δ{change:+.1f}%)")

        logger.info("")
        logger.info("Best Trial Metrics:")
        logger.info(f"  PNL: ${best_trial.user_attrs['total_pnl']:.2f}")
        logger.info(f"  Trades: {best_trial.user_attrs['total_trades']}")
        logger.info(f"  Win Rate: {best_trial.user_attrs['win_rate']:.1%}")
        logger.info(f"  Profit Factor: {best_trial.user_attrs['profit_factor']:.2f}")
        logger.info(f"  Sharpe: {best_trial.user_attrs['sharpe_ratio']:.2f}")
        logger.info(f"  Max DD: {best_trial.user_attrs['max_drawdown']:.1%}")

        # Save optimal weights
        weights_path = self.output_dir / "optimal_weights.json"
        with open(weights_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'n_trials': n_trials,
                'best_score': best_trial.value,
                'optimal_weights': optimal_weights,
                'baseline_weights': self.baseline_weights,
                'best_metrics': {
                    'total_pnl': best_trial.user_attrs['total_pnl'],
                    'total_trades': best_trial.user_attrs['total_trades'],
                    'win_rate': best_trial.user_attrs['win_rate'],
                    'profit_factor': best_trial.user_attrs['profit_factor'],
                    'sharpe_ratio': best_trial.user_attrs['sharpe_ratio'],
                    'max_drawdown': best_trial.user_attrs['max_drawdown'],
                }
            }, f, indent=2)

        logger.info(f"\n✓ Saved optimal weights to: {weights_path}")

        return {
            'study': study,
            'optimal_weights': optimal_weights,
            'baseline_weights': self.baseline_weights,
            'best_trial': best_trial
        }

    def analyze_sensitivity(self, study: optuna.Study):
        """
        Analyze weight sensitivity and generate visualization.

        Args:
            study: Completed Optuna study
        """
        logger.info("\nAnalyzing weight sensitivity...")

        # Extract trial data
        trials_df = study.trials_dataframe()

        # Filter successful trials
        trials_df = trials_df[trials_df['value'] > 0].copy()

        if len(trials_df) < 10:
            logger.warning("Too few successful trials for sensitivity analysis")
            return

        # Normalize weights
        weight_cols = ['params_w_structure', 'params_w_liquidity',
                       'params_w_momentum', 'params_w_macro']

        trials_df['weight_sum'] = trials_df[weight_cols].sum(axis=1)
        for col in weight_cols:
            trials_df[col + '_norm'] = trials_df[col] / trials_df['weight_sum']

        # Create sensitivity plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Domain Weight Sensitivity Analysis', fontsize=16, fontweight='bold')

        weight_names = {
            'params_w_structure_norm': 'Structure (Wyckoff + SMC)',
            'params_w_liquidity_norm': 'Liquidity',
            'params_w_momentum_norm': 'Momentum (RSI + ADX)',
            'params_w_macro_norm': 'Macro (VIX + Regime)'
        }

        for ax, (col, name) in zip(axes.flat, weight_names.items()):
            ax.scatter(trials_df[col], trials_df['value'], alpha=0.5, s=20)
            ax.set_xlabel(f'{name} Weight', fontsize=11)
            ax.set_ylabel('Optimization Score', fontsize=11)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add trend line
            z = np.polyfit(trials_df[col], trials_df['value'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(trials_df[col].min(), trials_df[col].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
            ax.legend()

        plt.tight_layout()

        sensitivity_path = self.output_dir / "weight_sensitivity.png"
        plt.savefig(sensitivity_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved sensitivity plot to: {sensitivity_path}")

        plt.close()

    def validate_regime_breakdown(self, optimal_weights: Dict):
        """
        Validate optimal weights across different market regimes.

        Args:
            optimal_weights: Optimal weight configuration
        """
        logger.info("\nValidating regime-specific performance...")

        # Split by regime
        regimes = self._split_by_regime(self.df)

        results = []

        for regime_name, regime_df in regimes.items():
            if len(regime_df) < 100:
                logger.warning(f"Skipping {regime_name} (too few bars: {len(regime_df)})")
                continue

            # Apply weights
            df_weighted = self._apply_weights_to_dataframe(regime_df, optimal_weights)
            df_weighted['k2_fusion_score'] = df_weighted['fusion_score_weighted']

            # Run backtest
            params = KnowledgeParams(
                tier1_threshold=0.45,
                tier2_threshold=0.35,
                tier3_threshold=0.25,
                require_macro_alignment=True,
                atr_stop_mult=2.0,
                trailing_atr_mult=2.0,
                max_hold_bars=168,
                max_risk_pct=0.02,
                volatility_scaling=True,
                use_smart_exits=True,
                breakeven_after_tp1=True,
            )

            backtest = KnowledgeAwareBacktest(df_weighted, params, starting_capital=10000.0)
            regime_results = backtest.run()

            # Calculate annualized metrics
            years = (regime_df.index[-1] - regime_df.index[0]).days / 365.25
            trades_per_year = regime_results['total_trades'] / max(years, 0.1)

            results.append({
                'regime': regime_name,
                'bars': len(regime_df),
                'years': years,
                'trades': regime_results['total_trades'],
                'trades_per_year': trades_per_year,
                'pnl': regime_results['total_pnl'],
                'win_rate': regime_results['win_rate'],
                'profit_factor': regime_results['profit_factor'],
                'sharpe': regime_results['sharpe_ratio'],
                'max_dd': regime_results['max_drawdown']
            })

            logger.info(f"  {regime_name:15s}: PF={regime_results['profit_factor']:.2f}, "
                       f"Trades={regime_results['total_trades']:3d} "
                       f"({trades_per_year:.1f}/yr), WR={regime_results['win_rate']:.1%}")

        # Save breakdown
        breakdown_df = pd.DataFrame(results)
        breakdown_path = self.output_dir / "regime_breakdown.csv"
        breakdown_df.to_csv(breakdown_path, index=False)
        logger.info(f"\n✓ Saved regime breakdown to: {breakdown_path}")

        # Check success criteria
        logger.info("\nSuccess Criteria Check:")
        all_pf_positive = all(r['profit_factor'] >= 1.0 for r in results)
        logger.info(f"  All regimes PF ≥ 1.0: {'✓' if all_pf_positive else '✗'}")

        avg_trades_per_year = np.mean([r['trades_per_year'] for r in results])
        trade_count_ok = 25 <= avg_trades_per_year <= 40
        logger.info(f"  Trade frequency OK (25-40/yr): {'✓' if trade_count_ok else '✗'} "
                   f"(actual: {avg_trades_per_year:.1f}/yr)")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Optimize domain engine weights for meta-fusion'
    )
    parser.add_argument(
        '--feature-store',
        default='data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet',
        help='Path to feature store parquet'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=100,
        help='Number of optimization trials'
    )
    parser.add_argument(
        '--output-dir',
        default='results/engine_weights',
        help='Output directory'
    )

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = EngineWeightOptimizer(
        feature_store_path=args.feature_store,
        output_dir=args.output_dir
    )

    # Run optimization
    opt_results = optimizer.optimize(n_trials=args.trials)

    # Analyze sensitivity
    optimizer.analyze_sensitivity(opt_results['study'])

    # Validate regime breakdown
    optimizer.validate_regime_breakdown(opt_results['optimal_weights'])

    logger.info("\n" + "=" * 80)
    logger.info("✅ ENGINE WEIGHT OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {args.output_dir}/")
    logger.info("  - optimal_weights.json")
    logger.info("  - weight_sensitivity.png")
    logger.info("  - regime_breakdown.csv")
    logger.info("  - optuna_study.db")


if __name__ == '__main__':
    main()
