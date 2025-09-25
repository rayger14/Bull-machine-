#!/usr/bin/env python3
"""
Bull Machine v1.5.0 - Enhanced ETH Backtest
Comprehensive testing with v1.5.0 alphas integrated into v1.4.2 baseline
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Add project to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bull_machine.version import __version__, get_version_banner
from bull_machine.core.config_loader import load_config, is_feature_enabled
from bull_machine.core.fusion_enhanced import FusionEngineV150
from bull_machine.core.telemetry import log_telemetry, clear_telemetry_logs

# Import analyzers
from bull_machine.modules.wyckoff.analyzer import WyckoffAnalyzer
from bull_machine.modules.liquidity.analyzer import LiquidityAnalyzer
from bull_machine.modules.structure.analyzer import StructureAnalyzer
from bull_machine.modules.momentum.analyzer import MomentumAnalyzer
from bull_machine.modules.volume.analyzer import VolumeAnalyzer
from bull_machine.modules.context.analyzer import ContextAnalyzer

@dataclass
class Trade:
    """Trade data structure."""
    timestamp: str
    direction: str
    entry: float
    exit: float
    stop: float
    pnl_pct: float
    pnl_dollar: float
    bars_held: int
    exit_reason: str
    fusion_score: float
    profile_used: str
    alphas_active: List[str]
    layer_scores: Dict[str, float]

@dataclass
class BacktestResult:
    """Comprehensive backtest results."""
    timeframe: str
    profile: str
    total_bars: int
    trades: List[Trade]
    summary: Dict[str, Any]

# ETH data sources (use available data)
ETH_DATA_SOURCES = [
    "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv",
    "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 1D_fa116.csv",
    "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv",
    "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 240_ab8a9.csv",
    "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv"
]

def load_and_prepare_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load and prepare ETH data."""
    if not Path(file_path).exists():
        return None

    try:
        df = pd.read_csv(file_path)

        # Standardize columns
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            return None

        # Handle timestamp
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        elif 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')

        # Handle volume
        if 'volume' not in df.columns:
            volume_candidates = ['BUY+SELL V', 'Total Buy Volume', 'Volume']
            for candidate in volume_candidates:
                if candidate in df.columns:
                    df['volume'] = df[candidate]
                    break
            else:
                df['volume'] = 100000  # Default

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def determine_timeframe(file_path: str) -> str:
    """Determine timeframe from filename."""
    filename = Path(file_path).name.lower()
    if '1d_' in filename or ', 1d' in filename:
        return '1D'
    elif '240_' in filename or ', 240' in filename:
        return '4H'
    elif '60_' in filename or ', 60' in filename:
        return '1H'
    elif '360_' in filename:
        return '6H'
    elif '720_' in filename:
        return '12H'
    else:
        return 'UNKNOWN'

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators required for analysis."""
    df = df.copy()

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.inf)
    df['rsi'] = 100 - (100 / (1 + rs))

    # SMAs
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    return df

def run_v150_backtest(df: pd.DataFrame, config: Dict[str, Any], timeframe: str) -> BacktestResult:
    """Run enhanced v1.5.0 backtest."""

    # Initialize analyzers
    wyckoff = WyckoffAnalyzer()
    liquidity = LiquidityAnalyzer(config)
    structure = StructureAnalyzer()
    momentum = MomentumAnalyzer()
    volume = VolumeAnalyzer()
    context = ContextAnalyzer()
    fusion = FusionEngineV150(config)

    # Results tracking
    trades = []
    balance = config.get('initial_balance', 10000)
    peak_balance = balance
    active_position = None

    # Processing parameters
    min_history = 200
    risk_per_trade = config.get('risk_per_trade', 0.01)
    profile_name = config.get('profile_name', 'base')

    print(f"\n🔄 Processing {len(df)} bars for {timeframe} timeframe")
    print(f"📊 Profile: {profile_name}")
    print(f"🎚️ Entry Threshold: {config.get('entry_threshold', 0.45)}")

    # Active alphas
    active_alphas = [f for f in ['mtf_dl2', 'six_candle_leg', 'orderflow_lca', 'negative_vip']
                    if is_feature_enabled(config, f)]
    print(f"🚀 Active Alphas: {active_alphas if active_alphas else 'None (baseline only)'}")

    signals_generated = 0
    signals_above_threshold = 0

    # Process bars
    for i in range(min_history, len(df) - 1):  # Leave one bar for exit
        current_bar = df.iloc[i]
        history = df.iloc[max(0, i-200):i+1]

        # Exit check for active position
        if active_position:
            bars_held = i - active_position['entry_bar']
            current_price = current_bar['close']

            # Calculate current PnL
            if active_position['direction'] == 'long':
                current_pnl_pct = ((current_price - active_position['entry']) / active_position['entry']) * 100
            else:
                current_pnl_pct = ((active_position['entry'] - current_price) / active_position['entry']) * 100

            # Exit conditions
            should_exit = False
            exit_reason = ""

            # Stop loss
            if active_position['direction'] == 'long':
                if current_price <= active_position['stop']:
                    should_exit = True
                    exit_reason = "Stop Loss"
            else:
                if current_price >= active_position['stop']:
                    should_exit = True
                    exit_reason = "Stop Loss"

            # Time-based exit
            if bars_held >= 25:  # Longer holding period
                should_exit = True
                exit_reason = "Time Limit"

            # Profit target
            if current_pnl_pct >= 10:  # 10% profit target
                should_exit = True
                exit_reason = "Profit Target"

            # Dynamic exit based on negative VIP
            if is_feature_enabled(config, 'negative_vip'):
                from bull_machine.modules.sentiment.negative_vip import negative_vip_score
                vip_score = negative_vip_score(history, config)
                if vip_score >= 0.7 and bars_held >= 3:  # High reversal risk
                    should_exit = True
                    exit_reason = "Negative VIP Signal"

            if should_exit:
                # Execute exit
                exit_price = current_price
                pnl_dollar = balance * risk_per_trade * (current_pnl_pct / 100)
                balance += pnl_dollar

                # Create trade record
                trade = Trade(
                    timestamp=current_bar['timestamp'].isoformat(),
                    direction=active_position['direction'],
                    entry=active_position['entry'],
                    exit=exit_price,
                    stop=active_position['stop'],
                    pnl_pct=current_pnl_pct,
                    pnl_dollar=pnl_dollar,
                    bars_held=bars_held,
                    exit_reason=exit_reason,
                    fusion_score=active_position['fusion_score'],
                    profile_used=profile_name,
                    alphas_active=active_position['alphas_active'],
                    layer_scores=active_position['layer_scores']
                )

                trades.append(trade)
                print(f"  EXIT {trade.direction.upper()} @ ${trade.exit:.2f} | PnL: {trade.pnl_pct:+.2f}% | {trade.exit_reason}")

                active_position = None

        # Signal generation (only if no active position)
        if not active_position and i % 6 == 0:  # Check every 6 bars

            # Run layer analyses
            try:
                wyckoff_result = wyckoff.analyze(history)
                liquidity_result = liquidity.analyze(history)
                structure_result = structure.analyze(history)
                momentum_result = momentum.analyze(history)
                volume_result = volume.analyze(history)
                context_result = context.analyze(history)

                # Prepare layer scores
                layer_scores = {
                    'wyckoff': wyckoff_result.get('score', 0),
                    'liquidity': liquidity_result.get('score', 0),
                    'structure': structure_result.get('score', 0),
                    'momentum': momentum_result.get('score', 0),
                    'volume': volume_result.get('score', 0),
                    'context': context_result.get('score', 0),
                    'mtf': 0.5  # Default MTF score
                }

                signals_generated += 1

                # Run enhanced fusion
                fusion_result = fusion.fuse(
                    layer_scores=layer_scores,
                    df=history,
                    wyckoff_context=wyckoff_result
                )

                if fusion_result['score'] >= config.get('entry_threshold', 0.45):
                    signals_above_threshold += 1

                    # Determine direction
                    direction = fusion_result.get('signal', 'neutral')

                    if direction != 'neutral':
                        # Risk management
                        entry_price = current_bar['close']
                        atr = current_bar.get('atr', entry_price * 0.02)

                        # Asset-specific stop calculation
                        if profile_name == 'ETH':
                            stop_multiplier = config.get('atr', {}).get('mult', {}).get('trend', 1.8)
                        else:
                            stop_multiplier = 2.0

                        if direction == 'long':
                            stop_price = entry_price - (atr * stop_multiplier)
                        else:
                            stop_price = entry_price + (atr * stop_multiplier)

                        # Create position
                        active_position = {
                            'entry': entry_price,
                            'entry_bar': i,
                            'direction': direction,
                            'stop': stop_price,
                            'fusion_score': fusion_result['score'],
                            'alphas_active': fusion_result.get('alphas_enabled', []),
                            'layer_scores': fusion_result['layer_scores']
                        }

                        print(f"\n[{timeframe}] Bar {i} | {current_bar['timestamp']}")
                        print(f"  📊 Fusion Score: {fusion_result['score']:.3f} (threshold: {config.get('entry_threshold', 0.45)})")
                        print(f"  🎯 Profile: {profile_name} | Alphas: {active_position['alphas_active']}")
                        print(f"  ENTER {direction.upper()} @ ${entry_price:.2f} | Stop: ${stop_price:.2f}")

            except Exception as e:
                print(f"  Error in analysis at bar {i}: {e}")
                continue

        # Update peak balance
        if balance > peak_balance:
            peak_balance = balance

    # Close any remaining position
    if active_position:
        final_price = df.iloc[-1]['close']
        if active_position['direction'] == 'long':
            final_pnl_pct = ((final_price - active_position['entry']) / active_position['entry']) * 100
        else:
            final_pnl_pct = ((active_position['entry'] - final_price) / active_position['entry']) * 100

        final_pnl_dollar = balance * risk_per_trade * (final_pnl_pct / 100)
        balance += final_pnl_dollar

        trade = Trade(
            timestamp=df.iloc[-1]['timestamp'].isoformat(),
            direction=active_position['direction'],
            entry=active_position['entry'],
            exit=final_price,
            stop=active_position['stop'],
            pnl_pct=final_pnl_pct,
            pnl_dollar=final_pnl_dollar,
            bars_held=len(df) - 1 - active_position['entry_bar'],
            exit_reason="End of Data",
            fusion_score=active_position['fusion_score'],
            profile_used=profile_name,
            alphas_active=active_position['alphas_active'],
            layer_scores=active_position['layer_scores']
        )

        trades.append(trade)

    # Calculate summary statistics
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.pnl_pct > 0])
    losing_trades = len([t for t in trades if t.pnl_pct <= 0])

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum(t.pnl_pct for t in trades)
    avg_win = np.mean([t.pnl_pct for t in trades if t.pnl_pct > 0]) if winning_trades > 0 else 0
    avg_loss = np.mean([t.pnl_pct for t in trades if t.pnl_pct <= 0]) if losing_trades > 0 else 0
    max_drawdown = ((peak_balance - min([balance] + [t.pnl_dollar for t in trades])) / peak_balance * 100) if peak_balance > 0 else 0

    # Calculate monthly trade frequency (approximate)
    if len(df) > 0:
        days_covered = (df['timestamp'].max() - df['timestamp'].min()).days
        trades_per_month = (total_trades / max(days_covered, 1)) * 30.44 if days_covered > 0 else 0
    else:
        trades_per_month = 0

    summary = {
        'signals_generated': signals_generated,
        'signals_above_threshold': signals_above_threshold,
        'signal_efficiency': (signals_above_threshold / signals_generated * 100) if signals_generated > 0 else 0,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_drawdown,
        'final_balance': balance,
        'total_return': ((balance - config.get('initial_balance', 10000)) / config.get('initial_balance', 10000)) * 100,
        'trades_per_month': trades_per_month,
        'profile_used': profile_name,
        'active_alphas': active_alphas
    }

    return BacktestResult(
        timeframe=timeframe,
        profile=profile_name,
        total_bars=len(df),
        trades=trades,
        summary=summary
    )

def main():
    """Run comprehensive v1.5.0 ETH backtest."""
    print("=" * 80)
    print(f"🚀 {get_version_banner()} - Enhanced ETH Comprehensive Backtest")
    print("=" * 80)
    print(f"📅 Analysis Date: {datetime.now()}")
    print("=" * 80)

    # Clear previous telemetry
    clear_telemetry_logs()

    # Test configurations
    test_configs = [
        {
            'name': 'v1.5.0_ETH_Base',
            'config': load_config('ETH', 'v150'),
            'description': 'ETH profile with all features disabled (baseline)'
        },
        {
            'name': 'v1.5.0_ETH_Enhanced',
            'description': 'ETH profile with v1.5.0 alphas enabled'
        }
    ]

    # Enable features for enhanced config
    enhanced_config = load_config('ETH', 'v150')
    enhanced_config['features']['mtf_dl2'] = True
    enhanced_config['features']['six_candle_leg'] = True
    enhanced_config['features']['orderflow_lca'] = True
    enhanced_config['features']['negative_vip'] = True
    test_configs[1]['config'] = enhanced_config

    all_results = {}

    # Run backtests on available data
    for file_path in ETH_DATA_SOURCES:
        if not Path(file_path).exists():
            continue

        print(f"\n{'='*70}")
        print(f"📂 Processing: {Path(file_path).name}")
        print(f"{'='*70}")

        # Load data
        df = load_and_prepare_data(file_path)
        if df is None or len(df) < 250:
            print(f"❌ Insufficient data: {len(df) if df is not None else 0} bars")
            continue

        # Add technical indicators
        df = calculate_technical_indicators(df)

        timeframe = determine_timeframe(file_path)
        print(f"✅ Loaded {len(df)} bars for {timeframe} timeframe")
        print(f"📅 Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

        # Test each configuration
        for test_config in test_configs:
            config_name = test_config['name']
            config = test_config['config']
            description = test_config['description']

            print(f"\n🧪 Testing: {config_name}")
            print(f"   {description}")

            try:
                result = run_v150_backtest(df, config, timeframe)

                # Store result
                key = f"{timeframe}_{config_name}_{Path(file_path).stem[:15]}"
                all_results[key] = result

                # Display summary
                s = result.summary
                print(f"\n📊 Results Summary:")
                print(f"  Signals Generated: {s['signals_generated']}")
                print(f"  Above Threshold: {s['signals_above_threshold']} ({s['signal_efficiency']:.1f}%)")
                print(f"  Trades Executed: {s['total_trades']}")
                print(f"  Trades/Month: {s['trades_per_month']:.1f}")

                if s['total_trades'] > 0:
                    print(f"  Win Rate: {s['win_rate']:.1f}%")
                    print(f"  Total PnL: {s['total_pnl']:+.2f}%")
                    print(f"  Avg Win: {s['avg_win']:+.2f}%")
                    print(f"  Avg Loss: {s['avg_loss']:.2f}%")
                    print(f"  Max Drawdown: {s['max_drawdown']:.1f}%")
                    print(f"  Final Balance: ${s['final_balance']:,.2f}")
                    print(f"  Total Return: {s['total_return']:+.1f}%")

                    # Show recent trades
                    if len(result.trades) > 0:
                        print(f"\n  Recent Trades:")
                        for trade in result.trades[-3:]:
                            print(f"    {trade.direction.upper()} | "
                                  f"Entry: ${trade.entry:.2f} | Exit: ${trade.exit:.2f} | "
                                  f"PnL: {trade.pnl_pct:+.2f}% | {trade.exit_reason}")

                else:
                    print(f"  No trades executed")

            except Exception as e:
                print(f"❌ Error running {config_name}: {e}")

    # Generate comparative analysis
    print(f"\n{'='*80}")
    print(f"📊 COMPARATIVE ANALYSIS - v1.5.0 vs BASELINE")
    print(f"{'='*80}")

    # Group results by timeframe
    baseline_results = {k: v for k, v in all_results.items() if 'Base' in k}
    enhanced_results = {k: v for k, v in all_results.items() if 'Enhanced' in k}

    print(f"\n🎯 Performance Comparison:")
    print(f"{'Timeframe':<12} {'Config':<20} {'Trades':<8} {'Win Rate':<10} {'PnL':<12} {'Return':<12}")
    print("-" * 80)

    for key, result in all_results.items():
        s = result.summary
        timeframe = result.timeframe
        config_type = 'Enhanced' if 'Enhanced' in key else 'Baseline'

        print(f"{timeframe:<12} {config_type:<20} {s['total_trades']:<8} "
              f"{s['win_rate']:<10.1f}% {s['total_pnl']:<12.1f}% {s['total_return']:<12.1f}%")

    # Calculate aggregated improvements
    if baseline_results and enhanced_results:
        baseline_trades = sum(r.summary['total_trades'] for r in baseline_results.values())
        enhanced_trades = sum(r.summary['total_trades'] for r in enhanced_results.values())

        baseline_wins = sum(r.summary['winning_trades'] for r in baseline_results.values())
        enhanced_wins = sum(r.summary['winning_trades'] for r in enhanced_results.values())

        if baseline_trades > 0 and enhanced_trades > 0:
            baseline_wr = (baseline_wins / baseline_trades) * 100
            enhanced_wr = (enhanced_wins / enhanced_trades) * 100

            print(f"\n🏆 Overall Improvements:")
            print(f"  Win Rate: {baseline_wr:.1f}% → {enhanced_wr:.1f}% ({enhanced_wr - baseline_wr:+.1f}pp)")

            # Find best performing setup
            best_result = max(all_results.values(), key=lambda x: x.summary['total_return'])
            if best_result.summary['total_trades'] > 0:
                print(f"\n🥇 Best Performer:")
                print(f"  Setup: {best_result.profile} on {best_result.timeframe}")
                print(f"  Return: {best_result.summary['total_return']:+.1f}%")
                print(f"  Win Rate: {best_result.summary['win_rate']:.1f}%")
                print(f"  Trades/Month: {best_result.summary['trades_per_month']:.1f}")

    # Save comprehensive results
    output_file = f"eth_v150_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert results to JSON-serializable format
    serializable_results = {}
    for key, result in all_results.items():
        serializable_results[key] = {
            'timeframe': result.timeframe,
            'profile': result.profile,
            'total_bars': result.total_bars,
            'trades': [asdict(trade) for trade in result.trades],
            'summary': result.summary
        }

    with open(output_file, 'w') as f:
        json.dump({
            'version': __version__,
            'timestamp': datetime.now().isoformat(),
            'test_configs': [{k: v for k, v in tc.items() if k != 'config'} for tc in test_configs],
            'results': serializable_results
        }, f, indent=2, default=str)

    print(f"\n💾 Comprehensive results saved to: {output_file}")

    print(f"\n{'='*80}")
    print(f"✅ Enhanced ETH Backtest Complete!")
    print(f"   {get_version_banner()}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()