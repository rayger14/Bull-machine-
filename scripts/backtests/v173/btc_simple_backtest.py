#!/usr/bin/env python3
"""
Simplified BTC 1-Year Backtest using Price Action + MTF + Macro
Simpler signal generation without mocking fusion modules
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from engine.io.tradingview_loader import load_tv
from engine.timeframes.mtf_alignment import MTFAlignmentEngine
from engine.context.loader import load_macro_data, fetch_macro_snapshot
from engine.context.macro_engine import analyze_macro, create_default_macro_config

class SimpleBTCBacktest:
    """Simplified BTC backtest using price action signals"""

    def __init__(self, starting_balance=10000):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.position = None
        self.trades = []

        # Config - TUNED for better risk/reward
        self.risk_pct = 0.05  # 5% per trade (reduced for wider stops)
        self.entry_threshold = 0.45  # Higher threshold (was 0.30)
        self.stop_loss_pct = 0.08  # Wider stop (was 0.06)
        self.take_profit_pct = 0.15  # Higher target (was 0.12)

        # Initialize engines
        self.mtf_engine = MTFAlignmentEngine()
        self.macro_config = create_default_macro_config()
        self.macro_config['macro_veto_threshold'] = 0.90  # Less strict

        # Load macro data
        print("📊 Loading macro context data...")
        self.macro_data = load_macro_data()

        # Tracking
        self.vix_prev = None
        self.signals_checked = 0
        self.macro_vetoed = 0
        self.signals_generated = 0

    def load_data(self):
        """Load BTC data"""
        print("\n📈 Loading BTC data...")

        df_1h = load_tv('BTC_1H')
        df_4h = load_tv('BTC_4H')
        df_1d = load_tv('BTC_1D')

        # Filter to last year
        start_date = '2024-10-01'
        end_date = '2025-10-01'

        df_1h = df_1h[(df_1h.index >= start_date) & (df_1h.index <= end_date)]
        df_4h = df_4h[(df_4h.index >= start_date) & (df_4h.index <= end_date)]
        df_1d = df_1d[(df_1d.index >= start_date) & (df_1d.index <= end_date)]

        print(f"✅ BTC data loaded (Oct 2024 - Oct 2025):")
        print(f"   1H: {len(df_1h)}, 4H: {len(df_4h)}, 1D: {len(df_1d)}")

        return df_1h, df_4h, df_1d

    def generate_simple_signal(self, df_1h, df_4h, df_1d, idx_4h, timestamp):
        """Generate trading signal using simple price action + MTF"""

        if idx_4h < 50:
            return None

        self.signals_checked += 1

        # Get windows
        window_1h = df_1h[df_1h.index <= timestamp].tail(100)
        window_4h = df_4h.iloc[:idx_4h+1].tail(50)
        window_1d = df_1d[df_1d.index <= timestamp].tail(20)

        if len(window_1h) < 20 or len(window_4h) < 20 or len(window_1d) < 5:
            return None

        # Get macro context
        ts = pd.Timestamp(timestamp)
        if ts.tz is not None:
            ts = ts.tz_localize(None)
        macro_snapshot = fetch_macro_snapshot(self.macro_data, ts)
        macro_result = analyze_macro(macro_snapshot, self.macro_config, asset_type='crypto')

        # Get VIX
        vix_now = macro_snapshot.get('VIX', {}).get('value', 20.0)
        if vix_now is None:
            vix_now = 20.0
        if self.vix_prev is None:
            self.vix_prev = vix_now

        # Check macro veto FIRST
        if macro_result['veto_strength'] >= self.macro_config['macro_veto_threshold']:
            self.macro_vetoed += 1
            self.vix_prev = vix_now
            return None

        # MTF confluence check
        try:
            sync_report = self.mtf_engine.mtf_confluence(
                window_1h, window_4h, window_1d, vix_now, self.vix_prev
            )
        except Exception as e:
            self.vix_prev = vix_now
            return None

        self.vix_prev = vix_now

        # SIMPLE PRICE ACTION SIGNAL
        # Calculate momentum across timeframes
        price_1h = window_1h['close'].iloc[-1]
        sma_20_1h = window_1h['close'].tail(20).mean()
        sma_50_1h = window_1h['close'].tail(50).mean() if len(window_1h) >= 50 else sma_20_1h

        price_4h = window_4h['close'].iloc[-1]
        sma_20_4h = window_4h['close'].tail(20).mean()

        price_1d = window_1d['close'].iloc[-1]
        sma_10_1d = window_1d['close'].tail(10).mean()

        # RSI calculation (simple)
        def calc_rsi(prices, period=14):
            deltas = prices.diff()
            gain = (deltas.where(deltas > 0, 0)).rolling(window=period).mean()
            loss = (-deltas.where(deltas < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if len(rsi) > 0 else 50

        rsi_1h = calc_rsi(window_1h['close'])
        rsi_4h = calc_rsi(window_4h['close'])

        # ADX - Average Directional Index (trend strength filter)
        def calc_adx(df, period=14):
            high = df['high']
            low = df['low']
            close = df['close']

            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()

            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            adx = dx.rolling(period).mean()

            return adx.iloc[-1] if len(adx) > 0 else 0

        adx_4h = calc_adx(window_4h)

        # Only trade in trending markets (ADX > 20)
        if adx_4h < 20:
            self.vix_prev = vix_now
            return None

        # Determine signal direction and strength
        signal = {'direction': 'neutral', 'confidence': 0, 'reasons': []}

        # Bullish conditions - TUNED: Require stronger alignment
        bullish_score = 0
        if price_1h > sma_20_1h > sma_50_1h:
            bullish_score += 0.20  # Increased weight
            signal['reasons'].append('1H strong trend up')
        if price_4h > sma_20_4h:
            bullish_score += 0.20  # Increased weight
            signal['reasons'].append('4H trend up')
        if price_1d > sma_10_1d:
            bullish_score += 0.20  # Increased weight
            signal['reasons'].append('1D trend up')
        if 35 < rsi_1h < 65:  # Stricter RSI range
            bullish_score += 0.05
        if 35 < rsi_4h < 65:
            bullish_score += 0.05
        if sync_report.get('htf_aligned', False):
            bullish_score += 0.20  # Increased weight
            signal['reasons'].append('MTF aligned')
        if macro_result.get('macro_delta', 0) > 0:
            bullish_score += 0.10
            signal['reasons'].append('macro greenlight')

        # Bearish conditions - TUNED: Require stronger alignment
        bearish_score = 0
        if price_1h < sma_20_1h < sma_50_1h:
            bearish_score += 0.20  # Increased weight
            signal['reasons'].append('1H strong trend down')
        if price_4h < sma_20_4h:
            bearish_score += 0.20  # Increased weight
            signal['reasons'].append('4H trend down')
        if price_1d < sma_10_1d:
            bearish_score += 0.20  # Increased weight
            signal['reasons'].append('1D trend down')
        if 35 < rsi_1h < 65:  # Stricter RSI range
            bearish_score += 0.05
        if 35 < rsi_4h < 65:
            bearish_score += 0.05
        if sync_report.get('htf_aligned', False):
            bearish_score += 0.20  # Increased weight
        if macro_result.get('macro_delta', 0) < 0:
            bearish_score += 0.10
            signal['reasons'].append('macro risk-off')

        # Determine direction
        if bullish_score > bearish_score and bullish_score >= self.entry_threshold:
            signal['direction'] = 'long'
            signal['confidence'] = bullish_score
        elif bearish_score > bullish_score and bearish_score >= self.entry_threshold:
            signal['direction'] = 'short'
            signal['confidence'] = bearish_score
        else:
            return None

        self.signals_generated += 1

        return {
            'direction': signal['direction'],
            'confidence': signal['confidence'],
            'reasons': signal['reasons'],
            'vix': vix_now,
            'macro_delta': macro_result.get('macro_delta', 0),
            'rsi_1h': rsi_1h,
            'rsi_4h': rsi_4h
        }

    def execute_trade(self, signal, price, timestamp):
        """Execute trade"""
        position_size = self.balance * self.risk_pct

        if signal['direction'] == 'long':
            stop_price = price * (1 - self.stop_loss_pct)
            target_price = price * (1 + self.take_profit_pct)
        else:
            stop_price = price * (1 + self.stop_loss_pct)
            target_price = price * (1 - self.take_profit_pct)

        self.position = {
            'entry_time': timestamp,
            'entry_price': price,
            'direction': signal['direction'],
            'size': position_size / price,
            'position_value': position_size,
            'confidence': signal['confidence'],
            'reasons': signal['reasons'],
            'stop_price': stop_price,
            'target_price': target_price,
            'vix': signal['vix']
        }

        print(f"\n🎯 Entry {timestamp.strftime('%Y-%m-%d %H:%M')} | {signal['direction'].upper():>6} | ${price:,.2f}")
        print(f"   Conf: {signal['confidence']:.2f} | Reasons: {', '.join(signal['reasons'][:3])}")
        print(f"   Stop: ${stop_price:,.2f} | Target: ${target_price:,.2f}")

    def update_position(self, df_4h, idx_4h):
        """Check stops/targets"""
        if not self.position:
            return

        bar = df_4h.iloc[idx_4h]
        price = bar['close']
        high = bar['high']
        low = bar['low']
        timestamp = bar.name

        exit_reason = None
        exit_price = None

        if self.position['direction'] == 'long':
            if low <= self.position['stop_price']:
                exit_reason = 'stop_loss'
                exit_price = self.position['stop_price']
            elif high >= self.position['target_price']:
                exit_reason = 'take_profit'
                exit_price = self.position['target_price']
        else:
            if high >= self.position['stop_price']:
                exit_reason = 'stop_loss'
                exit_price = self.position['stop_price']
            elif low <= self.position['target_price']:
                exit_reason = 'take_profit'
                exit_price = self.position['target_price']

        if exit_reason:
            self._close_position(exit_price, timestamp, exit_reason)

    def _close_position(self, exit_price, timestamp, reason):
        """Close position"""
        entry_price = self.position['entry_price']
        direction = self.position['direction']

        if direction == 'long':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        pnl_dollars = self.position['position_value'] * pnl_pct
        fees = self.position['position_value'] * 0.002
        pnl_dollars -= fees

        self.balance += pnl_dollars

        trade = {
            **self.position,
            'exit_time': timestamp,
            'exit_price': exit_price,
            'exit_reason': reason,
            'pnl_pct': pnl_pct * 100,
            'pnl_dollars': pnl_dollars,
            'fees': fees,
            'balance_after': self.balance
        }

        self.trades.append(trade)

        print(f"🔄 Exit  {timestamp.strftime('%Y-%m-%d %H:%M')} | {direction.upper():>6} | ${exit_price:,.2f}")
        print(f"   P&L: {pnl_pct*100:+.2f}% (${pnl_dollars:+,.2f}) | {reason} | Balance: ${self.balance:,.0f}")

        self.position = None

    def run(self):
        """Run backtest"""
        print("\n" + "="*80)
        print("🏛️  SIMPLIFIED BTC 1-YEAR BACKTEST")
        print("="*80)

        df_1h, df_4h, df_1d = self.load_data()

        print(f"\n💰 Starting Balance: ${self.starting_balance:,.2f}")
        print(f"⚖️  Risk per Trade: {self.risk_pct*100:.1f}%")
        print(f"🎯 Entry Threshold: {self.entry_threshold:.2f}")

        print(f"\n🚀 EXECUTING BACKTEST...")
        print("="*80)

        for idx_4h in range(len(df_4h)):
            bar = df_4h.iloc[idx_4h]
            timestamp = bar.name
            price = bar['close']

            if self.position:
                self.update_position(df_4h, idx_4h)
                continue

            signal = self.generate_simple_signal(df_1h, df_4h, df_1d, idx_4h, timestamp)

            if signal and signal['direction'] != 'neutral':
                self.execute_trade(signal, price, timestamp)

        if self.position:
            final_price = df_4h.iloc[-1]['close']
            final_time = df_4h.iloc[-1].name
            self._close_position(final_price, final_time, 'backtest_end')

        self._print_results()
        return self.trades

    def _print_results(self):
        """Print results"""
        print("\n" + "="*80)
        print("📊 BACKTEST COMPLETE")
        print("="*80)

        total_return = ((self.balance - self.starting_balance) / self.starting_balance) * 100

        print(f"\n💰 PERFORMANCE:")
        print(f"   Starting Balance:     ${self.starting_balance:>10,.2f}")
        print(f"   Final Balance:        ${self.balance:>10,.2f}")
        print(f"   Total Return:         {total_return:>10.2f}%")
        print(f"   P&L:                  ${self.balance - self.starting_balance:>10,.2f}")

        if self.trades:
            wins = [t for t in self.trades if t['pnl_dollars'] > 0]
            losses = [t for t in self.trades if t['pnl_dollars'] <= 0]
            win_rate = len(wins) / len(self.trades) * 100
            avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0

            print(f"\n📈 TRADE STATISTICS:")
            print(f"   Total Trades:         {len(self.trades):>10}")
            print(f"   Wins / Losses:        {len(wins):>5} / {len(losses):<5}")
            print(f"   Win Rate:             {win_rate:>10.1f}%")
            print(f"   Avg Win:              {avg_win:>10.2f}%")
            print(f"   Avg Loss:             {avg_loss:>10.2f}%")

            if wins and losses:
                pf = abs(sum([t['pnl_dollars'] for t in wins]) / sum([t['pnl_dollars'] for t in losses]))
                print(f"   Profit Factor:        {pf:>10.2f}")

            best = max(self.trades, key=lambda t: t['pnl_pct'])
            worst = min(self.trades, key=lambda t: t['pnl_pct'])

            print(f"\n🏆 BEST: {best['entry_time'].strftime('%Y-%m-%d')} | {best['direction'].upper()} | {best['pnl_pct']:+.2f}%")
            print(f"💔 WORST: {worst['entry_time'].strftime('%Y-%m-%d')} | {worst['direction'].upper()} | {worst['pnl_pct']:+.2f}%")

        print(f"\n🎯 SIGNAL STATISTICS:")
        print(f"   Bars Checked:         {self.signals_checked:>10}")
        print(f"   Macro Vetoed:         {self.macro_vetoed:>10}")
        print(f"   Signals Generated:    {self.signals_generated:>10}")
        print(f"   Trades Executed:      {len(self.trades):>10}")

        print("\n" + "="*80)


if __name__ == '__main__':
    backtest = SimpleBTCBacktest(starting_balance=10000)
    trades = backtest.run()
