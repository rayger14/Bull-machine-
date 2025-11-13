#!/usr/bin/env python3
"""
Router v10 Integrated Backtest - Full Trade Execution with Regime-Aware Config Switching

Combines:
- Regime detection (GMM v3.1)
- Event calendar suppression
- Router v10 config selection logic
- Full trade execution from backtest_knowledge_v2.py

This produces actual trades and PNL based on dynamic regime-aware strategy switching.

Usage:
    python3 bin/backtest_router_v10_integrated.py --asset BTC --start 2022-01-01 --end 2023-12-31 \\
        --bull-config configs/v10_bases/btc_bull_v10_best.json \\
        --bear-config configs/v10_bases/btc_bear_v10_best.json \\
        --output results/router_v10_integrated_2022_2023
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import json
from typing import Dict, Optional, Tuple
import logging

# Router v10 components
from engine.regime_detector import RegimeDetector
from engine.event_calendar import EventCalendar
from engine.router_v10 import RouterV10

# Backtest engine (we'll import the class directly)
from bin.backtest_knowledge_v2 import KnowledgeAwareBacktest, KnowledgeParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedRouterBacktest:
    """
    Integrated backtest engine that combines Router v10 regime-aware config switching
    with full trade execution.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        bull_config: Dict,
        bear_config: Dict,
        router: RouterV10,
        regime_detector: RegimeDetector,
        event_calendar: EventCalendar,
        starting_capital: float = 10000.0,
        asset: str = 'BTC'
    ):
        """
        Initialize integrated router backtest.

        Args:
            df: Feature store DataFrame
            bull_config: Bull market config (loaded from JSON)
            bear_config: Bear market config (loaded from JSON)
            router: RouterV10 instance
            regime_detector: RegimeDetector instance
            event_calendar: EventCalendar instance
            starting_capital: Starting capital
            asset: Asset symbol
        """
        self.df = df.copy()
        self.bull_config = bull_config
        self.bear_config = bear_config
        self.router = router
        self.regime_detector = regime_detector
        self.event_calendar = event_calendar
        self.starting_capital = starting_capital
        self.asset = asset

        # Track which config is active at each bar
        self.config_history = []

        # Current runtime config (switches between bull/bear)
        self.active_config = None
        self.active_config_name = None

        # Backtest state
        self.equity = starting_capital
        self.peak_equity = starting_capital
        self.trades = []
        self.current_position = None

        # Regime classification telemetry
        self.regime_stats = {
            'risk_on': 0,
            'risk_off': 0,
            'neutral': 0,
            'crisis': 0,
            'low_confidence': 0,
            'event_suppression': 0
        }

        self.config_stats = {
            'BULL': 0,
            'BEAR': 0,
            'CASH': 0
        }

    def run(self) -> Dict:
        """
        Run integrated backtest with dynamic regime-aware config switching.

        Returns:
            Results dict with trades, metrics, and router telemetry
        """
        logger.info(f"Starting integrated router backtest on {len(self.df):,} bars...")
        logger.info(f"  Bull config: {self.router.bull_config_path.name}")
        logger.info(f"  Bear config: {self.router.bear_config_path.name}")

        # Step 1: Classify all regimes upfront
        logger.info("Step 1: Classifying regimes for all bars...")
        df_classified = self.regime_detector.classify_batch(self.df)

        regime_dist = df_classified['regime_label'].value_counts()
        logger.info(f"  Regime distribution:")
        for regime, count in regime_dist.items():
            pct = 100 * count / len(df_classified)
            logger.info(f"    {regime:12s}: {count:5d} bars ({pct:5.1f}%)")

        # Step 2: Run bar-by-bar backtest with dynamic config switching
        logger.info("Step 2: Running backtest with dynamic config switching...")

        for bar_idx, (idx, row) in enumerate(df_classified.iterrows()):
            # Skip early bars without indicators
            if pd.isna(row.get('atr_14')):
                continue

            # Get regime classification
            regime_label = row['regime_label']
            regime_confidence = row['regime_confidence']
            timestamp = row['timestamp'] if 'timestamp' in row else idx

            # Check event suppression
            event_flag = self.event_calendar.is_suppression_window(timestamp)

            # Update regime stats
            if regime_label in self.regime_stats:
                self.regime_stats[regime_label] += 1
            if event_flag:
                self.regime_stats['event_suppression'] += 1
            if regime_confidence < 0.6:
                self.regime_stats['low_confidence'] += 1

            # Router decision: which config to use?
            decision = self.router.select_config(
                timestamp=timestamp,
                regime_label=regime_label,
                regime_confidence=regime_confidence,
                event_flag=event_flag
            )

            action = decision['action']  # 'BULL', 'BEAR', or 'CASH'
            reason = decision['reason']

            self.config_stats[action] += 1
            self.config_history.append({
                'timestamp': timestamp,
                'regime_label': regime_label,
                'regime_confidence': regime_confidence,
                'event_flag': event_flag,
                'action': action,
                'reason': reason
            })

            # Determine active config
            if action == 'BULL':
                target_config = self.bull_config
                target_config_name = 'BULL'
            elif action == 'BEAR':
                target_config = self.bear_config
                target_config_name = 'BEAR'
            else:  # CASH
                target_config = None
                target_config_name = 'CASH'

            # If config changed, log the transition
            if target_config_name != self.active_config_name:
                if self.active_config_name is not None:
                    logger.debug(f"Bar {bar_idx}: Config switch {self.active_config_name} → {target_config_name} (reason: {reason})")
                self.active_config = target_config
                self.active_config_name = target_config_name

            # Process trades with active config
            if self.active_config is not None:
                # Check exits first (if we have a position)
                if self.current_position is not None:
                    exit_result = self._check_exit_with_config(row, self.current_position, self.active_config)
                    if exit_result:
                        exit_price, exit_reason = exit_result
                        self._close_position(row, exit_price, exit_reason)

                # Check entries (if no position and not CASH)
                if self.current_position is None and self.active_config_name != 'CASH':
                    entry_result = self._check_entry_with_config(row, self.active_config, regime_label)
                    if entry_result:
                        entry_price, entry_reason = entry_result
                        self._open_position(row, entry_price, entry_reason, regime_label)

            else:
                # CASH mode: close any existing position
                if self.current_position is not None:
                    self._close_position(row, row['close'], f"cash_mode_{reason}")

        # Close any remaining position at end
        if self.current_position is not None:
            last_row = df_classified.iloc[-1]
            self._close_position(last_row, last_row['close'], "end_of_period")

        # Step 3: Calculate performance metrics
        results = self._calculate_metrics()
        results['regime_stats'] = self.regime_stats
        results['config_stats'] = self.config_stats
        results['router_stats'] = self.router.get_stats()
        results['config_history'] = self.config_history

        return results

    def _check_entry_with_config(self, row: pd.Series, config: Dict, regime_label: str) -> Optional[Tuple[float, str]]:
        """
        Check if we should enter a position using the given config's rules.

        Simplified entry logic based on fusion score and archetype thresholds.
        """
        # Get fusion score (try multiple column names)
        fusion_score = row.get('fusion_score', row.get('k2_fusion_score', 0.0))

        if fusion_score == 0.0:
            # Fallback: compute basic fusion score from available features
            fusion_weights = config.get('fusion', {}).get('weights', {})

            # Try to extract component scores
            wyckoff = row.get('wyckoff_score', row.get('wyckoff_m2_1h', 0.0)) * fusion_weights.get('wyckoff', 0.3)
            liquidity = row.get('liquidity_score', row.get('liquidity_1h', 0.0)) * fusion_weights.get('liquidity', 0.3)
            momentum = row.get('momentum_score', row.get('rsi_14', 50.0) / 100.0) * fusion_weights.get('momentum', 0.2)
            smc = row.get('smc_score', row.get('smc_bos_strength_1h', 0.0)) * fusion_weights.get('smc', 0.2)

            fusion_score = wyckoff + liquidity + momentum + smc

        # Check fusion threshold
        fusion_threshold = config.get('fusion', {}).get('entry_threshold_confidence', 0.4)

        if fusion_score < fusion_threshold:
            return None

        # Additional filters from config
        # 1. Macro veto (crisis regime)
        if regime_label == 'crisis':
            crisis_fuse = config.get('context', {}).get('crisis_fuse', {})
            if not crisis_fuse.get('enabled', False):
                return None
            if fusion_score < crisis_fuse.get('allow_one_trade_if_fusion_confidence_ge', 0.8):
                return None

        # 2. Check archetype thresholds (simplified - just check min_liquidity)
        archetype_config = config.get('archetypes', {})
        if archetype_config.get('use_archetypes', False):
            min_liquidity = archetype_config.get('thresholds', {}).get('min_liquidity', 0.2)
            liquidity_score = row.get('liquidity_score', 0.0)
            if liquidity_score < min_liquidity:
                return None

        # Entry price
        entry_price = row['close']

        return (entry_price, f"fusion_{fusion_score:.3f}_regime_{regime_label}")

    def _check_exit_with_config(self, row: pd.Series, position: Dict, config: Dict) -> Optional[Tuple[float, str]]:
        """
        Check if we should exit position using the given config's rules.
        """
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        current_price = row['close']
        timestamp = row.get('timestamp', row.name)

        # Calculate current PNL%
        pnl_pct = (current_price - entry_price) / entry_price

        # Exit rule 1: Stop loss (use config's ATR-based stop)
        exits_config = config.get('pnl_tracker', {}).get('exits', {})
        atr_k = exits_config.get('trail_atr_mult', 1.2)
        atr_14 = row.get('atr_14', 0.0)

        # Initial stop: entry_price - (atr_k * atr_14)
        stop_loss = entry_price - (atr_k * atr_14)
        if current_price <= stop_loss:
            return (current_price, f"stop_loss_atr{atr_k}")

        # Exit rule 2: Take profit (first TP level)
        tp1_r = exits_config.get('scale_out_rr', 1.0)
        tp1_price = entry_price + (tp1_r * atr_k * atr_14)

        if current_price >= tp1_price and exits_config.get('enable_partial', False):
            # For simplicity, full exit at TP1 (not implementing partial exits in this simplified version)
            return (current_price, f"tp1_{tp1_r}R")

        # Exit rule 3: Max bars in trade
        max_bars = exits_config.get('max_bars_in_trade', 72)
        bars_held = (timestamp - entry_time).total_seconds() / 3600  # hours

        if bars_held >= max_bars:
            return (current_price, f"max_hold_{max_bars}h")

        # Exit rule 4: Macro exit (VIX spike)
        if exits_config.get('macro_exit_enabled', False):
            vix_exit_level = exits_config.get('vix_exit_level', 30.0)
            vix = row.get('VIX', 0.0)
            if vix >= vix_exit_level:
                return (current_price, f"macro_vix_{vix:.1f}")

        return None

    def _open_position(self, row: pd.Series, entry_price: float, entry_reason: str, regime_label: str):
        """Open a new position."""
        # Position sizing: simple fixed percentage
        position_size_usd = self.equity * 0.02  # 2% risk per trade

        self.current_position = {
            'entry_time': row.get('timestamp', row.name),
            'entry_price': entry_price,
            'position_size': position_size_usd,
            'entry_reason': entry_reason,
            'regime_label': regime_label,
            'config_name': self.active_config_name,
            'entry_equity': self.equity
        }

        logger.debug(f"ENTRY: {self.current_position['entry_time']} @ ${entry_price:.2f} | {entry_reason} | {self.active_config_name} config")

    def _close_position(self, row: pd.Series, exit_price: float, exit_reason: str):
        """Close the current position."""
        if self.current_position is None:
            return

        entry_price = self.current_position['entry_price']
        position_size = self.current_position['position_size']

        # Calculate PNL
        pnl_pct = (exit_price - entry_price) / entry_price
        gross_pnl = position_size * pnl_pct

        # Apply costs (simplified)
        fees = position_size * 0.0002  # 2 bps
        slippage = position_size * 0.0002  # 2 bps
        net_pnl = gross_pnl - fees - slippage

        # Update equity
        self.equity += net_pnl
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Record trade
        trade_record = {
            **self.current_position,
            'exit_time': row.get('timestamp', row.name),
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'gross_pnl': gross_pnl,
            'fees': fees,
            'slippage': slippage,
            'net_pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'exit_equity': self.equity
        }

        self.trades.append(trade_record)

        logger.debug(f"EXIT: {trade_record['exit_time']} @ ${exit_price:.2f} | {exit_reason} | PNL: ${net_pnl:.2f} ({pnl_pct:+.2%})")

        self.current_position = None

    def _calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'final_equity': self.equity,
                'gross_profit': 0.0,
                'gross_loss': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'winners': 0,
                'losers': 0,
                'trades': [],
                'equity_curve': [self.starting_capital]
            }

        # Basic metrics
        total_trades = len(self.trades)
        total_pnl = sum(t['net_pnl'] for t in self.trades)

        winners = [t for t in self.trades if t['net_pnl'] > 0]
        losers = [t for t in self.trades if t['net_pnl'] <= 0]

        win_rate = len(winners) / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(t['net_pnl'] for t in winners) if winners else 0.0
        gross_loss = abs(sum(t['net_pnl'] for t in losers)) if losers else 0.0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = gross_profit / len(winners) if winners else 0.0
        avg_loss = gross_loss / len(losers) if losers else 0.0

        # Drawdown calculation (simplified)
        equity_curve = [self.starting_capital]
        for trade in self.trades:
            equity_curve.append(trade['exit_equity'])

        peak = self.starting_capital
        max_dd = 0.0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (simplified - assumes trades are independent)
        if len(self.trades) > 1:
            returns = [t['pnl_pct'] for t in self.trades]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'final_equity': self.equity,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'winners': len(winners),
            'losers': len(losers),
            'trades': self.trades,
            'equity_curve': equity_curve
        }


def load_feature_store(asset: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load MTF feature store for given asset and date range."""
    feature_dir = Path('data/features_mtf')

    # Try exact match first
    filename = f"{asset}_1H_{start_date}_to_{end_date}.parquet"
    filepath = feature_dir / filename

    if filepath.exists():
        print(f"✅ Loading feature store: {filepath}")
        df = pd.read_parquet(filepath)

        # Ensure timestamp column
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

    # Load and filter by date range
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
            print(f"✅ Found covering file: {file.name}")
            mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
            return df[mask].copy()

    raise FileNotFoundError(f"No feature store covers full range {start_date} to {end_date}")


def main():
    parser = argparse.ArgumentParser(description='Router v10 Integrated Backtest - Full Trade Execution')
    parser.add_argument('--asset', type=str, required=True, help='Asset symbol (BTC, ETH, etc.)')
    parser.add_argument('--start', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--bull-config', type=str, required=True, help='Bull market config JSON')
    parser.add_argument('--bear-config', type=str, required=True, help='Bear market config JSON')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--capital', type=float, default=10000.0, help='Starting capital')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("ROUTER V10 - INTEGRATED BACKTEST WITH FULL TRADE EXECUTION")
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
    print(f"   Columns: {len(df.columns)}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Load configs
    print(f"\n{'='*80}")
    print("LOADING CONFIGS")
    print('='*80)

    with open(args.bull_config) as f:
        bull_config = json.load(f)
    print(f"✅ Bull config loaded: {Path(args.bull_config).name}")

    with open(args.bear_config) as f:
        bear_config = json.load(f)
    print(f"✅ Bear config loaded: {Path(args.bear_config).name}")

    # Initialize components
    print(f"\n{'='*80}")
    print("INITIALIZING ROUTER COMPONENTS")
    print('='*80)

    regime_detector = RegimeDetector()
    print(f"✅ RegimeDetector initialized (GMM v3.1)")
    print(f"   Features: {len(regime_detector.features)}")
    print(f"   Clusters: {len(regime_detector.label_map)}")

    event_calendar = EventCalendar()
    print(f"✅ EventCalendar initialized")
    print(f"   Total events: {len(event_calendar.events)}")
    print(f"   Suppression window: T-{event_calendar.pre_event_hours}h to T+{event_calendar.post_event_hours}h")

    router = RouterV10(
        bull_config_path=args.bull_config,
        bear_config_path=args.bear_config,
        confidence_threshold=0.60,
        event_suppression=True,
        hysteresis_bars=0
    )
    print(f"✅ RouterV10 initialized")
    print(f"   Confidence threshold: {router.confidence_threshold}")

    # Run integrated backtest
    print(f"\n{'='*80}")
    print("RUNNING INTEGRATED BACKTEST")
    print('='*80)

    backtest = IntegratedRouterBacktest(
        df=df,
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
    print(f"\n📊 Performance Metrics:")
    print(f"   Total PNL:      ${results['total_pnl']:>10,.2f}")
    print(f"   Final Equity:   ${results['final_equity']:>10,.2f}")
    print(f"   Return:         {(results['final_equity'] / args.capital - 1):>10.2%}")
    print(f"   Total Trades:   {results['total_trades']:>10}")
    print(f"   Win Rate:       {results['win_rate']:>10.1%}")
    print(f"   Profit Factor:  {results['profit_factor']:>10.2f}")
    print(f"   Sharpe Ratio:   {results['sharpe_ratio']:>10.2f}")
    print(f"   Max Drawdown:   {results['max_drawdown']:>10.1%}")
    print(f"   Gross Profit:   ${results['gross_profit']:>10,.2f}")
    print(f"   Gross Loss:     ${results['gross_loss']:>10,.2f}")
    print(f"   Avg Win:        ${results['avg_win']:>10,.2f}")
    print(f"   Avg Loss:       ${results['avg_loss']:>10,.2f}")
    print(f"   Winners:        {results['winners']:>10}")
    print(f"   Losers:         {results['losers']:>10}")

    print(f"\n📊 Regime Distribution:")
    for regime, count in results['regime_stats'].items():
        pct = 100 * count / len(df)
        bar = '█' * int(pct / 2)
        print(f"   {regime:20s}: {count:5d} bars ({pct:5.1f}%) {bar}")

    print(f"\n📊 Config Usage:")
    router_stats = results['router_stats']
    for action, pct in sorted(router_stats['action_distribution'].items()):
        bar = '█' * int(pct / 2)
        print(f"   {action:5s}: {pct:5.1f}% {bar}")

    print(f"\n📈 Router Statistics:")
    print(f"   Regime switches: {router_stats['regime_switches']}")
    print(f"   Switches per day: {router_stats['regime_switches'] / (len(df) / 24):.2f}")

    # Trade log
    print(f"\n{'='*80}")
    print("TRADE LOG")
    print('='*80)
    for i, trade in enumerate(results['trades'][:20], 1):  # Show first 20 trades
        print(f"\nTrade {i}: {trade['entry_reason']}")
        print(f"  Entry:  {trade['entry_time']} @ ${trade['entry_price']:.2f} | {trade['config_name']} config | {trade['regime_label']}")
        print(f"  Exit:   {trade['exit_time']} @ ${trade['exit_price']:.2f} | {trade['exit_reason']}")
        print(f"  PNL:    ${trade['net_pnl']:.2f} ({trade['pnl_pct']:+.2%})")

    if len(results['trades']) > 20:
        print(f"\n... ({len(results['trades']) - 20} more trades)")

    # Export results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export summary stats
        stats_file = output_dir / 'backtest_results.json'
        with open(stats_file, 'w') as f:
            export_results = {
                'asset': args.asset,
                'start_date': args.start,
                'end_date': args.end,
                'bull_config': args.bull_config,
                'bear_config': args.bear_config,
                'starting_capital': args.capital,
                'performance': {
                    'total_pnl': results['total_pnl'],
                    'final_equity': results['final_equity'],
                    'return_pct': (results['final_equity'] / args.capital - 1) * 100,
                    'total_trades': results['total_trades'],
                    'win_rate': results['win_rate'],
                    'profit_factor': results['profit_factor'],
                    'sharpe_ratio': results['sharpe_ratio'],
                    'max_drawdown': results['max_drawdown'],
                    'winners': results['winners'],
                    'losers': results['losers']
                },
                'regime_stats': results['regime_stats'],
                'config_stats': results['config_stats'],
                'router_stats': {
                    'total_decisions': router_stats['total_decisions'],
                    'action_distribution': router_stats['action_distribution'],
                    'regime_switches': router_stats['regime_switches']
                }
            }
            json.dump(export_results, f, indent=2)
        print(f"\n💾 Exported summary: {stats_file}")

        # Export trades to CSV
        trades_file = output_dir / 'trades.csv'
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(trades_file, index=False)
        print(f"💾 Exported trades: {trades_file}")

        # Export equity curve
        equity_file = output_dir / 'equity_curve.csv'
        equity_df = pd.DataFrame({
            'trade_num': range(len(results['equity_curve'])),
            'equity': results['equity_curve']
        })
        equity_df.to_csv(equity_file, index=False)
        print(f"💾 Exported equity curve: {equity_file}")

        # Export config history
        config_hist_file = output_dir / 'config_history.parquet'
        config_hist_df = pd.DataFrame(results['config_history'])
        config_hist_df.to_parquet(config_hist_file, index=False)
        print(f"💾 Exported config history: {config_hist_file}")

    print("\n" + "="*80)
    print("✅ INTEGRATED ROUTER BACKTEST COMPLETE")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == '__main__':
    exit(main())
