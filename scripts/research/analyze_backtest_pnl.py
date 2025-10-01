#!/usr/bin/env python3
"""
Analyze Bull Machine v1.7 Backtest P&L
Extract and calculate exact trade P&L from the extended validation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from engine.io.tradingview_loader import load_tv
from engine.smc.smc_engine import SMCEngine
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.liquidity.hob import HOBDetector
from engine.momentum.momentum_engine import MomentumEngine

def calculate_exact_backtest_pnl():
    """Calculate exact P&L from the backtest trades"""

    print("💰 BULL MACHINE v1.7 EXACT P&L ANALYSIS")
    print("="*60)

    # Load config and data
    with open('configs/v170/assets/ETH_v17_tuned.json', 'r') as f:
        config = json.load(f)

    # Load the exact same dataset as the backtest
    eth_4h = load_tv('ETH_4H')
    eth_1d = load_tv('ETH_1D')

    # Get the validation slice (last 300 bars)
    validation_4h = eth_4h.tail(300)

    print(f"📊 VALIDATION DATASET")
    print(f"   Period: {validation_4h.index[0]} to {validation_4h.index[-1]}")
    print(f"   Total bars: {len(validation_4h)}")

    # Initialize engines
    smc_engine = SMCEngine(config['domains']['smc'])
    momentum_engine = MomentumEngine(config['domains']['momentum'])
    wyckoff_engine = WyckoffEngine(config['domains']['wyckoff'])
    hob_engine = HOBDetector(config['domains']['liquidity']['hob_detection'])

    # Thresholds
    conf_threshold = config['fusion']['calibration_thresholds']['confidence']
    strength_threshold = config['fusion']['calibration_thresholds']['strength']

    # Track portfolio and trades
    portfolio = {
        'capital': 100000.0,
        'position': 0.0,
        'entry_price': 0.0,
        'trade_count': 0
    }

    trades_log = []

    print(f"\n⚡ RECREATING EXACT BACKTEST TRADES")
    print("-" * 50)

    # Process the same way as the backtest
    for i in range(50, len(validation_4h)):
        current_bar = validation_4h.iloc[i]
        historical_4h = validation_4h.iloc[:i+1]
        recent_4h = historical_4h.tail(60)

        try:
            # Generate domain signals (same as backtest)
            domain_signals = {}

            try:
                domain_signals['smc'] = smc_engine.analyze(recent_4h)
            except:
                domain_signals['smc'] = None

            try:
                domain_signals['momentum'] = momentum_engine.analyze(recent_4h)
            except:
                domain_signals['momentum'] = None

            try:
                domain_signals['wyckoff'] = wyckoff_engine.analyze(recent_4h, usdt_stagnation=0.5)
            except:
                domain_signals['wyckoff'] = None

            try:
                domain_signals['hob'] = hob_engine.detect_hob(recent_4h)
            except:
                domain_signals['hob'] = None

            # MTF confluence check (same logic as backtest)
            mtf_aligned = check_mtf_confluence_simple(eth_1d, current_bar, historical_4h)

            # Signal fusion (same as backtest)
            active_signals = [s for s in domain_signals.values() if s is not None]

            if len(active_signals) >= 1:
                directions = []
                confidences = []
                engine_names = []

                for engine, signal in domain_signals.items():
                    if signal and hasattr(signal, 'direction') and hasattr(signal, 'confidence'):
                        directions.append(signal.direction)
                        confidences.append(signal.confidence)
                        engine_names.append(engine)

                if directions and confidences:
                    # Direction consensus
                    long_votes = directions.count('long')
                    short_votes = directions.count('short')

                    if long_votes > short_votes:
                        fusion_direction = 'long'
                        fusion_strength = long_votes / len(directions)
                    elif short_votes > long_votes:
                        fusion_direction = 'short'
                        fusion_strength = short_votes / len(directions)
                    else:
                        continue

                    avg_confidence = np.mean(confidences)

                    # Check entry criteria and MTF filter
                    if (avg_confidence >= conf_threshold and
                        fusion_strength >= strength_threshold and
                        mtf_aligned):

                        # Execute trade logic (same as backtest)
                        trade_executed = execute_trade_with_pnl(
                            portfolio, current_bar, fusion_direction,
                            avg_confidence, len(active_signals), trades_log, engine_names
                        )

        except Exception as e:
            continue

    # Close final position if any
    if portfolio['position'] != 0:
        close_final_position(portfolio, validation_4h.iloc[-1], trades_log)

    # Calculate and display exact P&L
    display_exact_pnl_analysis(trades_log, portfolio)

def check_mtf_confluence_simple(eth_1d, current_bar, historical_4h):
    """Simple MTF confluence check (same as backtest)"""
    try:
        current_time = current_bar.name

        # 4H trend
        if len(historical_4h) >= 12:
            sma_4h_fast = historical_4h['close'].tail(6).mean()
            sma_4h_slow = historical_4h['close'].tail(12).mean()

            if sma_4h_fast > sma_4h_slow * 1.005:
                trend_4h = 'bullish'
            elif sma_4h_fast < sma_4h_slow * 0.995:
                trend_4h = 'bearish'
            else:
                trend_4h = 'neutral'
        else:
            return False

        # 1D trend
        aligned_1d = eth_1d[eth_1d.index <= current_time]
        if len(aligned_1d) >= 10:
            sma_1d_fast = aligned_1d['close'].tail(5).mean()
            sma_1d_slow = aligned_1d['close'].tail(10).mean()

            if sma_1d_fast > sma_1d_slow * 1.01:
                trend_1d = 'bullish'
            elif sma_1d_fast < sma_1d_slow * 0.99:
                trend_1d = 'bearish'
            else:
                trend_1d = 'neutral'
        else:
            return False

        # Check alignment
        return trend_4h == trend_1d and trend_4h != 'neutral'

    except:
        return False

def execute_trade_with_pnl(portfolio, current_bar, direction, confidence, engines, trades_log, engine_names):
    """Execute trade with exact P&L calculation"""

    current_price = current_bar['close']

    # Close opposite position
    if portfolio['position'] != 0:
        if ((portfolio['position'] > 0 and direction == 'short') or
            (portfolio['position'] < 0 and direction == 'long')):
            close_position_with_pnl(portfolio, current_bar, trades_log)

    # Open new position
    if portfolio['position'] == 0:
        # Calculate position sizing (same as backtest)
        base_risk = 0.075  # 7.5%
        confidence_multiplier = min(1.3, confidence / 0.25)
        mtf_multiplier = 1.5  # Full alignment bonus

        final_sizing = base_risk * confidence_multiplier * mtf_multiplier
        final_sizing = min(final_sizing, 0.15)  # Cap at 15%

        position_value = portfolio['capital'] * final_sizing

        if direction == 'long':
            portfolio['position'] = position_value / current_price
        else:
            portfolio['position'] = -position_value / current_price

        portfolio['entry_price'] = current_price
        portfolio['trade_count'] += 1

        # Log trade opening
        trade = {
            'trade_id': portfolio['trade_count'],
            'entry_timestamp': current_bar.name,
            'entry_price': current_price,
            'direction': direction,
            'position_size': abs(portfolio['position']),
            'position_value': position_value,
            'confidence': confidence,
            'engines': engines,
            'engine_names': '+'.join(engine_names),
            'sizing_multiplier': final_sizing / base_risk,
            'capital_at_entry': portfolio['capital']
        }

        trades_log.append(trade)
        return True

    return False

def close_position_with_pnl(portfolio, current_bar, trades_log):
    """Close position with exact P&L calculation"""

    if portfolio['position'] == 0 or not trades_log:
        return

    current_price = current_bar['close']

    # Calculate exact P&L
    if portfolio['position'] > 0:  # Long position
        pnl = portfolio['position'] * (current_price - portfolio['entry_price'])
    else:  # Short position
        pnl = abs(portfolio['position']) * (portfolio['entry_price'] - current_price)

    # Update portfolio
    portfolio['capital'] += pnl

    # Update trade record with exit details
    trade = trades_log[-1]
    trade.update({
        'exit_timestamp': current_bar.name,
        'exit_price': current_price,
        'pnl_dollars': pnl,
        'return_pct': (pnl / trade['position_value']) * 100,
        'capital_after_trade': portfolio['capital'],
        'hold_duration': str(current_bar.name - trade['entry_timestamp'])
    })

    # Reset position
    portfolio['position'] = 0
    portfolio['entry_price'] = 0

def close_final_position(portfolio, final_bar, trades_log):
    """Close final position at end of backtest"""
    if portfolio['position'] != 0:
        close_position_with_pnl(portfolio, final_bar, trades_log)

def display_exact_pnl_analysis(trades_log, portfolio):
    """Display comprehensive P&L analysis"""

    print(f"\n" + "="*80)
    print("💰 EXACT TRADE-BY-TRADE P&L ANALYSIS")
    print("="*80)

    completed_trades = [t for t in trades_log if 'exit_price' in t]

    if not completed_trades:
        print("❌ No completed trades found")
        return

    print(f"\n📊 PORTFOLIO SUMMARY")
    print("-" * 40)
    print(f"Starting Capital: $100,000.00")
    print(f"Final Capital: ${portfolio['capital']:,.2f}")
    print(f"Total P&L: ${portfolio['capital'] - 100000:+,.2f}")
    print(f"Total Return: {((portfolio['capital'] - 100000) / 100000) * 100:+.2f}%")

    print(f"\n📈 TRADE STATISTICS")
    print("-" * 40)
    print(f"Total Trades: {len(completed_trades)}")

    winning_trades = [t for t in completed_trades if t['pnl_dollars'] > 0]
    losing_trades = [t for t in completed_trades if t['pnl_dollars'] <= 0]

    print(f"Winning Trades: {len(winning_trades)}")
    print(f"Losing Trades: {len(losing_trades)}")
    print(f"Win Rate: {len(winning_trades) / len(completed_trades) * 100:.1f}%")

    if winning_trades:
        avg_win = np.mean([t['pnl_dollars'] for t in winning_trades])
        print(f"Average Win: ${avg_win:+,.2f}")

    if losing_trades:
        avg_loss = np.mean([t['pnl_dollars'] for t in losing_trades])
        print(f"Average Loss: ${avg_loss:+,.2f}")

    # Profit factor
    gross_profit = sum(t['pnl_dollars'] for t in winning_trades)
    gross_loss = abs(sum(t['pnl_dollars'] for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    print(f"Gross Profit: ${gross_profit:+,.2f}")
    print(f"Gross Loss: ${gross_loss:+,.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")

    print(f"\n💵 DETAILED TRADE-BY-TRADE P&L")
    print("="*80)
    print(f"{'#':<3} {'Date':<12} {'Dir':<5} {'Entry':<10} {'Exit':<10} {'P&L $':<12} {'P&L %':<8} {'Size':<6} {'Engines'}")
    print("-"*80)

    for i, trade in enumerate(completed_trades, 1):
        entry_date = trade['entry_timestamp'].strftime('%m/%d')
        direction = trade['direction'].upper()
        entry_price = f"${trade['entry_price']:,.2f}"
        exit_price = f"${trade['exit_price']:,.2f}"
        pnl_dollars = f"${trade['pnl_dollars']:+,.2f}"
        pnl_percent = f"{trade['return_pct']:+.1f}%"
        size_mult = f"{trade['sizing_multiplier']:.1f}x"
        engines = trade['engine_names']

        print(f"{i:<3} {entry_date:<12} {direction:<5} {entry_price:<10} {exit_price:<10} "
              f"{pnl_dollars:<12} {pnl_percent:<8} {size_mult:<6} {engines}")

    # Best and worst trades
    best_trade = max(completed_trades, key=lambda x: x['pnl_dollars'])
    worst_trade = min(completed_trades, key=lambda x: x['pnl_dollars'])

    print(f"\n🏆 BEST TRADE")
    print("-" * 20)
    print(f"Date: {best_trade['entry_timestamp'].strftime('%Y-%m-%d')} to {best_trade['exit_timestamp'].strftime('%Y-%m-%d')}")
    print(f"Direction: {best_trade['direction'].upper()}")
    print(f"Entry: ${best_trade['entry_price']:,.2f} → Exit: ${best_trade['exit_price']:,.2f}")
    print(f"P&L: ${best_trade['pnl_dollars']:+,.2f} ({best_trade['return_pct']:+.1f}%)")
    print(f"Engines: {best_trade['engine_names']}")

    print(f"\n💸 WORST TRADE")
    print("-" * 20)
    print(f"Date: {worst_trade['entry_timestamp'].strftime('%Y-%m-%d')} to {worst_trade['exit_timestamp'].strftime('%Y-%m-%d')}")
    print(f"Direction: {worst_trade['direction'].upper()}")
    print(f"Entry: ${worst_trade['entry_price']:,.2f} → Exit: ${worst_trade['exit_price']:,.2f}")
    print(f"P&L: ${worst_trade['pnl_dollars']:+,.2f} ({worst_trade['return_pct']:+.1f}%)")
    print(f"Engines: {worst_trade['engine_names']}")

    # Running capital progression
    print(f"\n📈 CAPITAL PROGRESSION")
    print("-" * 30)
    running_capital = 100000
    for i, trade in enumerate(completed_trades, 1):
        running_capital += trade['pnl_dollars']
        print(f"After Trade {i}: ${running_capital:,.2f} ({trade['pnl_dollars']:+,.2f})")

if __name__ == "__main__":
    calculate_exact_backtest_pnl()