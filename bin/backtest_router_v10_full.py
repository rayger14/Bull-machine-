#!/usr/bin/env python3
"""
Router v10 Full Backtest - Integrated with Knowledge-Aware Engine

Integrates Router v10 regime-aware config switching with the full
backtest_knowledge_v2.py engine to get accurate trade execution and PNL.

This preserves the correct fusion score calculation that configs were optimized against.

Usage:
    python3 bin/backtest_router_v10_full.py \\
        --asset BTC \\
        --start 2022-01-01 \\
        --end 2023-12-31 \\
        --bull-config configs/baseline_btc_bull_pf20.json \\
        --bear-config configs/baseline_btc_bear_defensive.json \\
        --output results/router_v10_full_2022_2023
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import json
from typing import Dict, Optional
import logging

# Router v10 components
from engine.regime_detector import RegimeDetector
from engine.event_calendar import EventCalendar
from engine.router_v10 import RouterV10

# Import the full backtest engine
from bin.backtest_knowledge_v2 import KnowledgeAwareBacktest, KnowledgeParams

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class RouterAwareBacktest(KnowledgeAwareBacktest):
    """
    Extended version of KnowledgeAwareBacktest with Router v10 integration.

    Dynamically switches configs based on regime classification at each bar.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        params: KnowledgeParams,
        bull_config: Dict,
        bear_config: Dict,
        router: RouterV10,
        regime_detector: RegimeDetector,
        event_calendar: EventCalendar,
        starting_capital: float = 10000.0,
        asset: str = 'BTC'
    ):
        """Initialize router-aware backtest."""

        # Initialize with bull config as default
        super().__init__(df, params, starting_capital, asset, runtime_config=bull_config)

        # Store configs
        self.bull_config = bull_config
        self.bear_config = bear_config

        # Router components
        self.router = router
        self.regime_detector = regime_detector
        self.event_calendar = event_calendar

        # Tracking
        self.config_switches = 0
        self.active_config_name = 'BULL'
        self.regime_history = []
        self.config_history = []

        logger.info("RouterAwareBacktest initialized")
        logger.info(f"  Bull config: {router.bull_config_path.name}")
        logger.info(f"  Bear config: {router.bear_config_path.name}")

    def run(self) -> Dict:
        """
        Run backtest with dynamic regime-aware config switching.

        Override of KnowledgeAwareBacktest.run() with router integration.
        """
        logger.info(f"Starting router-aware backtest on {len(self.df):,} bars...")

        # Step 1: Classify all regimes upfront
        logger.info("Step 1: Classifying regimes...")
        df_classified = self.regime_detector.classify_batch(self.df)

        regime_dist = df_classified['regime_label'].value_counts()
        logger.info(f"  Regime distribution:")
        for regime, count in regime_dist.items():
            pct = 100 * count / len(df_classified)
            logger.info(f"    {regime:12s}: {count:5d} bars ({pct:5.1f}%)")

        # Step 2: Run bar-by-bar with dynamic config switching
        logger.info("Step 2: Running backtest with dynamic config switching...")

        for bar_idx, (idx, row) in enumerate(df_classified.iterrows()):
            # Skip early bars without indicators
            if pd.isna(row.get('atr_14')):
                continue

            # Ensure row has proper name/timestamp for parent class methods
            if 'timestamp' in row:
                # If timestamp is a column, make it the row name too
                row = row.copy()
                row.name = row['timestamp']

            # Get regime classification
            regime_label = row['regime_label']
            regime_confidence = row['regime_confidence']
            timestamp = row.get('timestamp', idx)

            # Check event suppression
            event_flag = self.event_calendar.is_suppression_window(timestamp)

            # Router decision: which config to use?
            decision = self.router.select_config(
                timestamp=timestamp,
                regime_label=regime_label,
                regime_confidence=regime_confidence,
                event_flag=event_flag
            )

            action = decision['action']  # 'BULL', 'BEAR', or 'CASH'
            reason = decision['reason']

            # Track regime and config history
            self.regime_history.append({
                'timestamp': timestamp,
                'regime_label': regime_label,
                'regime_confidence': regime_confidence,
                'event_flag': event_flag
            })

            self.config_history.append({
                'timestamp': timestamp,
                'action': action,
                'reason': reason
            })

            # Switch config if needed
            target_config_name = action
            if action == 'BULL':
                target_config = self.bull_config
            elif action == 'BEAR':
                target_config = self.bear_config
            else:  # CASH
                target_config = None

            if target_config_name != self.active_config_name:
                self.config_switches += 1
                if bar_idx % 100 == 0:  # Log every 100 bars to avoid spam
                    logger.debug(f"Bar {bar_idx}: Config switch {self.active_config_name} → {target_config_name}")
                self.active_config_name = target_config_name

                # Update runtime config (this affects fusion calculation)
                if target_config is not None:
                    self.runtime_config = target_config
                    # Re-initialize components that depend on config
                    self._reinitialize_config_dependent_components()

            # If CASH mode, close any position and skip
            if target_config is None:
                if self.current_position is not None:
                    self._close_trade(row, row['close'], f"cash_mode_{reason}")
                continue

            # PR#6B: Adaptive Fusion - Classify regime and get adapted parameters
            adapted_params = None
            if self.adaptive_fusion and self.regime_classifier:
                try:
                    macro_row = {feat: row.get(feat, np.nan)
                                for feat in self.runtime_config.get('regime_classifier', {}).get('feature_order', [])}
                    regime_info = self.regime_classifier.classify(macro_row)
                    adapted_params = self.adaptive_fusion.update(regime_info)

                    if adapted_params.get('ml_threshold') is not None:
                        self.ml_threshold = adapted_params['ml_threshold']
                except Exception as e:
                    if bar_idx % 1000 == 0:
                        logger.warning(f"Adaptive fusion failed at bar {bar_idx}: {e}")
                    adapted_params = None

            # Compute fusion score (with adapted parameters if available)
            fusion_score, context = self.compute_advanced_fusion_score(row, adapted_params)

            # Store adapted_params in context
            context['adapted_params'] = adapted_params
            context['router_regime'] = regime_label
            context['router_action'] = action

            # PR#4: Compute runtime liquidity score (if enabled)
            if self.liquidity_enabled:
                side = "long"
                from engine.liquidity.score import compute_liquidity_score
                liquidity_score = compute_liquidity_score(context, side)
                context['liquidity_score'] = liquidity_score
                self.liquidity_scores.append(liquidity_score)
            else:
                context['liquidity_score'] = 0.0

            # Check for open position
            if self.current_position is not None:
                # Update peak profit and MAE
                current_price = row['close']
                pnl_pct = (current_price - self.current_position.entry_price) / self.current_position.entry_price * self.current_position.direction

                if pnl_pct > self.current_position.peak_profit:
                    self.current_position.peak_profit = pnl_pct

                if pnl_pct < -self.current_position.max_adverse_excursion:
                    self.current_position.max_adverse_excursion = -pnl_pct

                # Check exit conditions
                exit_result = self.check_exit_conditions(row, self.current_position, bar_idx)

                if exit_result:
                    exit_reason, exit_price = exit_result
                    self._close_trade(row, exit_price, exit_reason, bar_idx)

            # Check for new entry (only if no position)
            if self.current_position is None:
                # Check re-entry conditions first
                reentry_result = self._check_reentry_conditions(row, fusion_score, context, bar_idx)

                if reentry_result:
                    entry_type, entry_price, reentry_size_mult = reentry_result
                    self._open_trade(row, entry_price, entry_type, fusion_score, context, bar_idx, reentry_size_mult=reentry_size_mult)
                    self._reentry_count += 1
                else:
                    # Check regular entry conditions
                    entry_result = self.check_entry_conditions(row, fusion_score, context)

                    if entry_result:
                        entry_type, entry_price = entry_result
                        self._open_trade(row, entry_price, entry_type, fusion_score, context, bar_idx)

        # Close any remaining position at end
        if self.current_position is not None:
            last_row = df_classified.iloc[-1]
            self._close_trade(last_row, last_row['close'], "end_of_period")

        # Calculate metrics (use parent class method)
        results = self._calculate_metrics()

        # Add router-specific stats
        results['config_switches'] = self.config_switches
        results['regime_history'] = self.regime_history
        results['config_history'] = self.config_history
        results['router_stats'] = self.router.get_stats()

        # Diagnostic output
        logger.info("=" * 60)
        logger.info("ROUTER STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Config switches: {self.config_switches}")
        logger.info(f"Regime switches: {results['router_stats']['regime_switches']}")

        config_dist = {}
        for entry in self.config_history:
            action = entry['action']
            config_dist[action] = config_dist.get(action, 0) + 1

        logger.info("Config usage:")
        for action, count in sorted(config_dist.items()):
            pct = 100 * count / len(self.config_history)
            logger.info(f"  {action:5s}: {pct:5.1f}%")

        return results

    def _reinitialize_config_dependent_components(self):
        """Reinitialize components that depend on runtime_config."""
        # Update ML filter threshold if config changed
        ml_config = self.runtime_config.get('ml_filter', {})
        if ml_config.get('enabled') and self.ml_model is not None:
            self.ml_threshold = float(ml_config.get('threshold', 0.707))

        # Note: We don't reinitialize the entire ML model or archetype logic
        # to avoid overhead - just update thresholds


def load_feature_store(asset: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load MTF feature store."""
    feature_dir = Path('data/features_mtf')

    filename = f"{asset}_1H_{start_date}_to_{end_date}.parquet"
    filepath = feature_dir / filename

    if filepath.exists():
        logger.info(f"Loading feature store: {filepath}")
        df = pd.read_parquet(filepath)

        if 'timestamp' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = df.index
                df = df.reset_index(drop=True)

        return df

    # Try to find overlapping file
    pattern = f"{asset}_1H_*.parquet"
    matching_files = list(feature_dir.glob(pattern))

    if not matching_files:
        raise FileNotFoundError(f"No feature stores found for {asset}")

    start = pd.to_datetime(start_date, utc=True)
    end = pd.to_datetime(end_date, utc=True)

    for file in sorted(matching_files):
        df = pd.read_parquet(file)

        if 'timestamp' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = df.index
                df = df.reset_index(drop=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df_start = df['timestamp'].min()
        df_end = df['timestamp'].max()

        if df_start <= start and df_end >= end:
            logger.info(f"Found covering file: {file.name}")
            mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
            return df[mask].copy()

    raise FileNotFoundError(f"No feature store covers {start_date} to {end_date}")


def main():
    parser = argparse.ArgumentParser(description='Router v10 Full Backtest with Trade Execution')
    parser.add_argument('--asset', type=str, required=True, help='Asset symbol')
    parser.add_argument('--start', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--bull-config', type=str, required=True, help='Bull market config')
    parser.add_argument('--bear-config', type=str, required=True, help='Bear market config')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--capital', type=float, default=10000.0, help='Starting capital')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("ROUTER V10 - FULL BACKTEST WITH TRADE EXECUTION")
    print("="*80)
    print(f"Asset: {args.asset}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Bull config: {args.bull_config}")
    print(f"Bear config: {args.bear_config}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load feature store
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print('='*80)
    df = load_feature_store(args.asset, args.start, args.end)
    print(f"✅ Loaded {len(df):,} bars")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Load configs
    print(f"\n{'='*80}")
    print("LOADING CONFIGS")
    print('='*80)

    with open(args.bull_config) as f:
        bull_config = json.load(f)
    print(f"✅ Bull config: {Path(args.bull_config).name}")
    print(f"   Entry threshold: {bull_config.get('fusion', {}).get('entry_threshold_confidence', 'N/A')}")

    with open(args.bear_config) as f:
        bear_config = json.load(f)
    print(f"✅ Bear config: {Path(args.bear_config).name}")
    print(f"   Entry threshold: {bear_config.get('fusion', {}).get('entry_threshold_confidence', 'N/A')}")

    # Initialize router components
    print(f"\n{'='*80}")
    print("INITIALIZING ROUTER COMPONENTS")
    print('='*80)

    regime_detector = RegimeDetector()
    print(f"✅ RegimeDetector (GMM v3.1)")

    event_calendar = EventCalendar()
    print(f"✅ EventCalendar ({len(event_calendar.events)} events)")

    router = RouterV10(
        bull_config_path=args.bull_config,
        bear_config_path=args.bear_config,
        confidence_threshold=0.60,
        event_suppression=True,
        hysteresis_bars=0
    )
    print(f"✅ RouterV10 (confidence threshold: {router.confidence_threshold})")

    # Run backtest
    print(f"\n{'='*80}")
    print("RUNNING BACKTEST")
    print('='*80)

    params = KnowledgeParams()

    backtest = RouterAwareBacktest(
        df=df,
        params=params,
        bull_config=bull_config,
        bear_config=bear_config,
        router=router,
        regime_detector=regime_detector,
        event_calendar=event_calendar,
        starting_capital=args.capital,
        asset=args.asset
    )

    results = backtest.run()

    # Print results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"\n📊 Performance:")
    print(f"   Total PNL:      ${results['total_pnl']:>10,.2f}")
    print(f"   Final Equity:   ${results['final_equity']:>10,.2f}")
    print(f"   Return:         {(results['final_equity'] / args.capital - 1):>10.2%}")
    print(f"   Total Trades:   {results['total_trades']:>10}")
    print(f"   Win Rate:       {results['win_rate']:>10.1%}")
    print(f"   Profit Factor:  {results['profit_factor']:>10.2f}")
    print(f"   Sharpe Ratio:   {results['sharpe_ratio']:>10.2f}")
    print(f"   Max Drawdown:   {results['max_drawdown']:>10.1%}")

    # Config usage
    print(f"\n📊 Config Usage:")
    router_stats = results['router_stats']
    for action, pct in sorted(router_stats['action_distribution'].items()):
        bar = '█' * int(pct / 2)
        print(f"   {action:5s}: {pct:5.1f}% {bar}")

    print(f"\n📈 Regime Switches: {router_stats['regime_switches']}")
    print(f"   Config Switches: {results['config_switches']}")

    # Trade log (first 10)
    print(f"\n{'='*80}")
    print("TRADE LOG (first 10)")
    print('='*80)
    for i, trade in enumerate(results['trades'][:10], 1):
        print(f"\nTrade {i}:")
        print(f"  Entry:  {trade.entry_time} @ ${trade.entry_price:.2f}")
        print(f"  Exit:   {trade.exit_time} @ ${trade.exit_price:.2f}")
        print(f"  PNL:    ${trade.net_pnl:.2f} ({trade.net_pnl/trade.position_size:.2%})")
        print(f"  Reason: {trade.entry_reason} → {trade.exit_reason}")

    if len(results['trades']) > 10:
        print(f"\n... ({len(results['trades']) - 10} more trades)")

    # Export results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats_file = output_dir / 'results.json'
        with open(stats_file, 'w') as f:
            export_results = {
                'asset': args.asset,
                'period': f"{args.start} to {args.end}",
                'performance': {
                    'total_pnl': results['total_pnl'],
                    'final_equity': results['final_equity'],
                    'return_pct': (results['final_equity'] / args.capital - 1) * 100,
                    'total_trades': results['total_trades'],
                    'win_rate': results['win_rate'],
                    'profit_factor': results['profit_factor'],
                    'sharpe_ratio': results['sharpe_ratio'],
                    'max_drawdown': results['max_drawdown']
                },
                'router': {
                    'config_switches': results['config_switches'],
                    'regime_switches': router_stats['regime_switches'],
                    'action_distribution': router_stats['action_distribution']
                }
            }
            json.dump(export_results, f, indent=2)
        print(f"\n💾 Exported: {stats_file}")

        # Export trades
        if results['trades']:
            trades_data = []
            for trade in results['trades']:
                trades_data.append({
                    'entry_time': str(trade.entry_time),
                    'entry_price': trade.entry_price,
                    'exit_time': str(trade.exit_time),
                    'exit_price': trade.exit_price,
                    'net_pnl': trade.net_pnl,
                    'entry_reason': trade.entry_reason,
                    'exit_reason': trade.exit_reason
                })

            trades_df = pd.DataFrame(trades_data)
            trades_file = output_dir / 'trades.csv'
            trades_df.to_csv(trades_file, index=False)
            print(f"💾 Exported: {trades_file}")

    print("\n" + "="*80)
    print("✅ BACKTEST COMPLETE")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == '__main__':
    exit(main())
