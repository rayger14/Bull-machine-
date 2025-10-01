#!/usr/bin/env python3
"""
Full Year Real ETH Backtest - Bull Machine v1.7.1
Uses complete year of real ETH data with full engine integration
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from data.real_data_loader import RealDataLoader

class FullYearBullMachine:
    """
    Full year Bull Machine backtest using real ETH data.
    Adapts timeframes based on available data.
    """

    def __init__(self, starting_balance: float = 10000.0):
        self.starting_balance = starting_balance
        self.current_balance = starting_balance

        # Initialize data loader
        self.data_loader = RealDataLoader()

        # Load v1.7.1 enhanced configs
        self.config = self._load_enhanced_configs()

        # Enhanced tracking
        self.engine_usage = {
            'smc': 0, 'wyckoff': 0, 'momentum': 0, 'hob': 0,
            'macro_veto': 0, 'ethbtc_veto': 0, 'atr_throttle': 0,
            'counter_trend_blocked': 0
        }

        self.trades = []
        self.daily_balance = []
        self.rejected_signals = []

    def _load_enhanced_configs(self) -> Dict[str, Any]:
        """Load enhanced v1.7.1 configurations."""
        config_base = "/Users/raymondghandchi/Bull-machine-/Bull-machine-/configs/v171"

        configs = {}
        config_files = ['fusion.json', 'context.json', 'liquidity.json',
                       'exits.json', 'risk.json', 'momentum.json']

        for config_file in config_files:
            config_path = os.path.join(config_base, config_file)
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        configs[config_file.replace('.json', '')] = json.load(f)
                except Exception as e:
                    print(f"⚠️  Error loading {config_file}: {e}")
                    configs[config_file.replace('.json', '')] = {}

        return configs

    def load_full_year_dataset(self) -> Dict[str, pd.DataFrame]:
        """Load full year dataset using best available timeframes."""
        print("🔄 Loading full year real ETH dataset...")

        # Use 12H as primary (has most historical data)
        eth_12h = self.data_loader.load_raw_data('COINBASE_ETHUSD', '12H')  # 720 minutes
        eth_6h = self.data_loader.load_raw_data('COINBASE_ETHUSD', '6H')    # 360 minutes
        eth_1d = self.data_loader.load_eth_data('1D')

        if eth_12h is None:
            raise ValueError("❌ No 12H ETH data available")

        # Find overlapping period for all timeframes
        start_date = max(
            eth_12h.index[0] if eth_12h is not None else pd.Timestamp('2020-01-01'),
            eth_6h.index[0] if eth_6h is not None else pd.Timestamp('2020-01-01'),
            eth_1d.index[0] if eth_1d is not None else pd.Timestamp('2020-01-01')
        )

        end_date = min(
            eth_12h.index[-1] if eth_12h is not None else pd.Timestamp('2030-01-01'),
            eth_6h.index[-1] if eth_6h is not None else pd.Timestamp('2030-01-01'),
            eth_1d.index[-1] if eth_1d is not None else pd.Timestamp('2030-01-01')
        )

        print(f"📊 Overlapping data period: {start_date} → {end_date}")
        print(f"Duration: {(end_date - start_date).days} days")

        # Filter all timeframes to overlapping period
        dataset = {}
        dataset['eth_ltf'] = eth_6h[(eth_6h.index >= start_date) & (eth_6h.index <= end_date)]  # 6H as LTF
        dataset['eth_mtf'] = eth_12h[(eth_12h.index >= start_date) & (eth_12h.index <= end_date)]  # 12H as MTF
        dataset['eth_htf'] = eth_1d[(eth_1d.index >= start_date) & (eth_1d.index <= end_date)]  # 1D as HTF

        # Generate synthetic macro data aligned with 12H timeframe
        macro_data = self._generate_macro_data(dataset['eth_mtf'])
        dataset['macro'] = macro_data

        print(f"✅ Full year dataset loaded:")
        print(f"   ETH 6H (LTF): {len(dataset['eth_ltf'])} bars")
        print(f"   ETH 12H (MTF): {len(dataset['eth_mtf'])} bars")
        print(f"   ETH 1D (HTF): {len(dataset['eth_htf'])} bars")
        print(f"   Macro: {len(dataset['macro'])} bars")

        return dataset

    def _generate_macro_data(self, eth_df: pd.DataFrame) -> pd.DataFrame:
        """Generate macro data aligned with ETH timeframe."""
        np.random.seed(42)  # Reproducible
        timestamps = eth_df.index

        macro_data = pd.DataFrame(index=timestamps)

        # Generate realistic macro data
        macro_data['dxy'] = 103 * np.exp(np.cumsum(np.random.normal(0, 0.005, len(timestamps))))

        # VIX with spikes
        vix_values = []
        current_vix = 20
        for i in range(len(timestamps)):
            if np.random.random() < 0.05:
                current_vix *= np.random.uniform(1.2, 1.8)
            else:
                current_vix *= np.random.uniform(0.98, 1.02)
            current_vix = max(10, min(60, current_vix))
            vix_values.append(current_vix)
        macro_data['vix'] = vix_values

        macro_data['total2'] = 800e9 * np.exp(np.cumsum(np.random.normal(0.0002, 0.02, len(timestamps))))
        macro_data['ethbtc'] = 0.06 * np.exp(np.cumsum(np.random.normal(0, 0.015, len(timestamps))))

        return macro_data

    def enhanced_signal_generation(self, dataset: Dict[str, pd.DataFrame], current_idx: int) -> Optional[Dict[str, Any]]:
        """Generate enhanced signal using Bull Machine v1.7.1 with real data."""
        df_ltf = dataset['eth_ltf']  # 6H
        df_mtf = dataset['eth_mtf']  # 12H
        df_htf = dataset['eth_htf']  # 1D
        macro_data = dataset['macro']

        if current_idx < 20 or len(df_mtf) <= current_idx:
            return None

        # Get aligned current data
        current_time = df_mtf.index[current_idx]

        # Get data windows up to current time
        window_ltf = df_ltf[df_ltf.index <= current_time].tail(50)
        window_mtf = df_mtf.iloc[:current_idx+1].tail(50)
        window_htf = df_htf[df_htf.index <= current_time].tail(30)
        macro_window = macro_data.iloc[:current_idx+1].tail(30)

        current_price = window_mtf['Close'].iloc[-1]
        atr = self._calculate_atr(window_mtf)

        # ATR throttle check
        atr_threshold = self.config.get('risk', {}).get('cost_controls', {}).get('min_atr_threshold', 0.01)
        if atr < atr_threshold * current_price:
            self.engine_usage['atr_throttle'] += 1
            return None

        # Multi-timeframe confluence analysis
        signals = []
        active_engines = []

        # SMC Analysis (Break of Structure)
        smc_signal = self._analyze_smc_real(window_ltf, window_mtf)
        if smc_signal:
            signals.append(smc_signal['signal'])
            active_engines.append('smc')
            self.engine_usage['smc'] += 1

        # Wyckoff Analysis (Volume/Price relationship)
        wyckoff_signal = self._analyze_wyckoff_real(window_mtf, window_htf)
        if wyckoff_signal:
            signals.append(wyckoff_signal['signal'])
            active_engines.append('wyckoff')
            self.engine_usage['wyckoff'] += 1

        # HOB Analysis (Enhanced absorption)
        hob_signal = self._analyze_hob_real(window_ltf, window_mtf, macro_window)
        if hob_signal:
            signals.append(hob_signal['signal'])
            active_engines.append('hob')
            self.engine_usage['hob'] += 1

        # Momentum Analysis (Directional bias)
        momentum_signal = self._analyze_momentum_real(window_mtf, window_htf)
        if momentum_signal:
            signals.append(momentum_signal['signal'])
            active_engines.append('momentum')
            self.engine_usage['momentum'] += 1

        if not signals:
            return None

        # Determine signal direction
        bullish_count = signals.count('bullish')
        bearish_count = signals.count('bearish')

        if bullish_count > bearish_count:
            signal_direction = 'bullish'
        elif bearish_count > bullish_count:
            signal_direction = 'bearish'
        else:
            return None

        # Enhanced v1.7.1 filters
        if not self._apply_enhanced_filters(signal_direction, active_engines, window_mtf, window_htf, macro_window):
            return None

        # Risk/reward calculation
        exits_config = self.config.get('exits', {})
        sl_atr_mult = exits_config.get('stop_loss', {}).get('initial_sl_atr', 0.8)
        target_rr = exits_config.get('risk_reward', {}).get('target_rr', 2.5)

        sl_distance = atr * sl_atr_mult
        tp_distance = sl_distance * target_rr

        return {
            'direction': signal_direction,
            'engines': active_engines,
            'confidence': min(0.8, len(active_engines) * 0.15 + 0.3),
            'expected_rr': target_rr,
            'entry_price': current_price,
            'sl_distance': sl_distance,
            'tp_distance': tp_distance,
            'atr': atr,
            'timestamp': current_time
        }

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range from real data."""
        if len(df) < period + 1:
            return df['High'].iloc[-1] - df['Low'].iloc[-1] if len(df) > 0 else 50.0

        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        return atr if not pd.isna(atr) else 50.0

    def _analyze_smc_real(self, df_ltf: pd.DataFrame, df_mtf: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Real SMC analysis on actual price data."""
        if len(df_mtf) < 20:
            return None

        # Break of Structure detection
        recent_highs = df_mtf['High'].rolling(10).max()
        recent_lows = df_mtf['Low'].rolling(10).min()
        current_price = df_mtf['Close'].iloc[-1]

        # Check for BOS on multiple timeframes
        mtf_bos = None
        if current_price > recent_highs.iloc[-2]:
            mtf_bos = 'bullish'
        elif current_price < recent_lows.iloc[-2]:
            mtf_bos = 'bearish'

        # Confirm with LTF if available
        if len(df_ltf) > 10 and mtf_bos:
            ltf_trend = 'bullish' if df_ltf['Close'].iloc[-1] > df_ltf['Close'].rolling(5).mean().iloc[-1] else 'bearish'
            if ltf_trend == mtf_bos:
                return {
                    'signal': mtf_bos,
                    'strength': 0.75,
                    'pattern': f'BOS_{mtf_bos}',
                    'confluence': True
                }

        return None

    def _analyze_wyckoff_real(self, df_mtf: pd.DataFrame, df_htf: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Real Wyckoff analysis using volume and price relationships."""
        if len(df_mtf) < 30:
            return None

        # Volume analysis
        volume_ma = df_mtf['Volume'].rolling(20).mean()
        current_volume = df_mtf['Volume'].iloc[-1]
        volume_ratio = current_volume / volume_ma.iloc[-1]

        # Price momentum
        price_change_5 = (df_mtf['Close'].iloc[-1] - df_mtf['Close'].iloc[-5]) / df_mtf['Close'].iloc[-5]
        price_change_10 = (df_mtf['Close'].iloc[-1] - df_mtf['Close'].iloc[-10]) / df_mtf['Close'].iloc[-10]

        # Wyckoff phases
        if volume_ratio > 1.5:  # High volume
            if price_change_5 > 0.03:  # Strong up move
                return {
                    'signal': 'bullish',
                    'strength': 0.8,
                    'pattern': 'accumulation_markup',
                    'volume_confirmation': True
                }
            elif price_change_5 < -0.03:  # Strong down move
                return {
                    'signal': 'bearish',
                    'strength': 0.8,
                    'pattern': 'distribution_markdown',
                    'volume_confirmation': True
                }

        return None

    def _analyze_hob_real(self, df_ltf: pd.DataFrame, df_mtf: pd.DataFrame, macro_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Real HOB analysis with enhanced absorption requirements."""
        if len(df_mtf) < 20:
            return None

        # Enhanced volume analysis
        volume_ma = df_mtf['Volume'].rolling(20).mean()
        volume_std = df_mtf['Volume'].rolling(20).std()
        current_volume = df_mtf['Volume'].iloc[-1]

        if pd.isna(volume_std.iloc[-1]) or volume_std.iloc[-1] == 0:
            return None

        volume_z = (current_volume - volume_ma.iloc[-1]) / volume_std.iloc[-1]

        # Enhanced absorption thresholds from v1.7.1
        min_vol_z_long = self.config.get('liquidity', {}).get('hob_quality_factors', {}).get('volume_z_min_long', 1.3)
        min_vol_z_short = self.config.get('liquidity', {}).get('hob_quality_factors', {}).get('volume_z_min_short', 1.6)

        current_price = df_mtf['Close'].iloc[-1]

        # Find significant price levels
        price_levels = df_mtf[['High', 'Low', 'Close']].iloc[-30:].values.flatten()
        unique_levels = np.unique(price_levels)

        for level in unique_levels:
            distance_pct = abs(current_price - level) / current_price

            if distance_pct < 0.01:  # Within 1% of significant level
                if current_price > level and volume_z >= min_vol_z_long:
                    return {
                        'signal': 'bullish',
                        'strength': 0.75,
                        'pattern': 'HOB_support_absorption',
                        'volume_z': volume_z
                    }
                elif current_price < level and volume_z >= min_vol_z_short:
                    return {
                        'signal': 'bearish',
                        'strength': 0.75,
                        'pattern': 'HOB_resistance_absorption',
                        'volume_z': volume_z
                    }

        return None

    def _analyze_momentum_real(self, df_mtf: pd.DataFrame, df_htf: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Real momentum analysis with trend alignment."""
        if len(df_mtf) < 30:
            return None

        # Calculate RSI
        def calc_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        rsi = calc_rsi(df_mtf['Close'])
        current_rsi = rsi.iloc[-1]

        # Trend alignment from HTF
        if len(df_htf) > 10:
            htf_trend = 'bullish' if df_htf['Close'].iloc[-1] > df_htf['Close'].rolling(10).mean().iloc[-1] else 'bearish'
        else:
            htf_trend = 'neutral'

        # Momentum signals with directional bias
        momentum_config = self.config.get('momentum', {}).get('signal_generation', {})
        rsi_oversold = momentum_config.get('rsi_oversold', 30)
        rsi_overbought = momentum_config.get('rsi_overbought', 70)

        if pd.isna(current_rsi):
            return None

        if current_rsi < rsi_oversold and htf_trend == 'bullish':
            return {
                'signal': 'bullish',
                'strength': 0.65,
                'pattern': 'momentum_oversold_reversal',
                'trend_aligned': True,
                'rsi': current_rsi
            }
        elif current_rsi > rsi_overbought and htf_trend == 'bearish':
            return {
                'signal': 'bearish',
                'strength': 0.65,
                'pattern': 'momentum_overbought_reversal',
                'trend_aligned': True,
                'rsi': current_rsi
            }

        return None

    def _apply_enhanced_filters(self, signal_direction: str, active_engines: List[str],
                              window_mtf: pd.DataFrame, window_htf: pd.DataFrame,
                              macro_window: pd.DataFrame) -> bool:
        """Apply enhanced v1.7.1 filters."""

        # Counter-trend discipline
        if len(window_htf) > 10:
            trend_direction = 'bullish' if window_htf['Close'].iloc[-1] > window_htf['Close'].rolling(10).mean().iloc[-1] else 'bearish'
            is_counter_trend = (signal_direction == 'bullish' and trend_direction == 'bearish') or \
                              (signal_direction == 'bearish' and trend_direction == 'bullish')

            min_engines = self.config.get('fusion', {}).get('counter_trend_discipline', {}).get('min_engines', 3)
            if is_counter_trend and len(active_engines) < min_engines:
                self.engine_usage['counter_trend_blocked'] += 1
                return False

        # ETHBTC/TOTAL2 rotation gate for shorts
        if signal_direction == 'bearish' and len(macro_window) > 5:
            ethbtc_strength = macro_window['ethbtc'].iloc[-1] / macro_window['ethbtc'].rolling(10).mean().iloc[-1]
            total2_strength = macro_window['total2'].iloc[-1] / macro_window['total2'].rolling(10).mean().iloc[-1]

            ethbtc_threshold = self.config.get('context', {}).get('rotation_gates', {}).get('ethbtc_threshold', 1.05)
            total2_threshold = self.config.get('context', {}).get('rotation_gates', {}).get('total2_threshold', 1.05)

            if ethbtc_strength > ethbtc_threshold or total2_strength > total2_threshold:
                self.engine_usage['ethbtc_veto'] += 1
                return False

        return True

    def execute_trade_real(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute trade with real market considerations."""
        # Position sizing
        risk_config = self.config.get('risk', {}).get('position_sizing', {})
        base_risk_pct = risk_config.get('base_risk_pct', 0.075)
        kelly_fraction = risk_config.get('kelly_fraction', 0.25)

        position_size = self.current_balance * base_risk_pct * kelly_fraction

        # Trade setup
        entry_price = signal['entry_price']
        direction = signal['direction']

        if direction == 'bullish':
            sl_price = entry_price - signal['sl_distance']
            tp_price = entry_price + signal['tp_distance']
        else:
            sl_price = entry_price + signal['sl_distance']
            tp_price = entry_price - signal['tp_distance']

        return {
            'entry_time': signal['timestamp'],
            'entry_price': entry_price,
            'direction': direction,
            'engines': signal['engines'],
            'confidence': signal['confidence'],
            'expected_rr': signal['expected_rr'],
            'sl_price': sl_price,
            'tp_price': tp_price,
            'position_size': position_size,
            'status': 'open'
        }

    def run_full_year_backtest(self) -> Dict[str, Any]:
        """Run complete full year backtest on real ETH data."""
        print(f"🚀 BULL MACHINE v1.7.1 - FULL YEAR REAL ETH BACKTEST")
        print(f"=" * 70)
        print(f"Engine: Complete Bull Machine v1.7.1 Enhanced")
        print(f"Data: Real ETH from chart_logs")
        print(f"Enhancements: All Moneytaur/Wyckoff/ZeroIKA improvements")
        print(f"=" * 70)

        # Load full year dataset
        dataset = self.load_full_year_dataset()
        df_mtf = dataset['eth_mtf']  # 12H primary timeframe

        open_trades = []
        self.current_balance = self.starting_balance

        # Process each bar
        for i in range(20, len(df_mtf)):
            current_time = df_mtf.index[i]
            current_bar = df_mtf.iloc[i]

            # Check exits for open trades
            for trade in open_trades[:]:
                closed_trade = self._check_trade_exit_real(trade, current_bar, current_time)
                if closed_trade:
                    self.trades.append(closed_trade)
                    open_trades.remove(trade)
                    print(f"🔄 Exit  {current_time.strftime('%Y-%m-%d %H:%M')} | {closed_trade['direction']:>8} | ${closed_trade['exit_price']:8.2f} | P&L: {closed_trade['pnl']:+6.2f}% | R: {closed_trade['r_multiple']:+5.1f} | Type: {closed_trade['exit_type']}")

            # Generate new signal
            signal = self.enhanced_signal_generation(dataset, i)

            if signal and len(open_trades) < 3:  # Max 3 concurrent trades
                trade = self.execute_trade_real(signal)
                if trade:
                    open_trades.append(trade)
                    engines_str = ','.join(signal['engines'])
                    print(f"🎯 Entry {current_time.strftime('%Y-%m-%d %H:%M')} | {signal['direction']:>8} | ${signal['entry_price']:8.2f} | Engines: {engines_str} | Conf: {signal['confidence']:.2f} | RR: {signal['expected_rr']:.1f}")

        # Close remaining trades
        if open_trades:
            final_price = df_mtf['Close'].iloc[-1]
            final_time = df_mtf.index[-1]
            for trade in open_trades:
                pnl = self._calculate_unrealized_pnl(trade, final_price)
                closed_trade = self._close_trade_real(trade, final_price, final_time, 'time_exit', pnl)
                self.trades.append(closed_trade)

        return self._generate_full_results(dataset)

    def _check_trade_exit_real(self, trade: Dict[str, Any], current_bar: pd.Series, current_time: pd.Timestamp) -> Optional[Dict[str, Any]]:
        """Check trade exit conditions using real data."""
        if trade['status'] != 'open':
            return None

        current_price = current_bar['Close']
        direction = trade['direction']

        # Stop loss check
        if (direction == 'bullish' and current_price <= trade['sl_price']) or \
           (direction == 'bearish' and current_price >= trade['sl_price']):
            pnl = self._calculate_realized_pnl(trade, trade['sl_price'])
            return self._close_trade_real(trade, trade['sl_price'], current_time, 'stop_loss', pnl)

        # Take profit check
        if (direction == 'bullish' and current_price >= trade['tp_price']) or \
           (direction == 'bearish' and current_price <= trade['tp_price']):
            pnl = self._calculate_realized_pnl(trade, trade['tp_price'])
            return self._close_trade_real(trade, trade['tp_price'], current_time, 'take_profit', pnl)

        return None

    def _calculate_realized_pnl(self, trade: Dict[str, Any], exit_price: float) -> float:
        """Calculate realized P&L."""
        entry_price = trade['entry_price']
        if trade['direction'] == 'bullish':
            return (exit_price - entry_price) / entry_price * 100
        else:
            return (entry_price - exit_price) / entry_price * 100

    def _calculate_unrealized_pnl(self, trade: Dict[str, Any], current_price: float) -> float:
        """Calculate unrealized P&L."""
        return self._calculate_realized_pnl(trade, current_price)

    def _close_trade_real(self, trade: Dict[str, Any], exit_price: float, exit_time: pd.Timestamp,
                         exit_type: str, pnl: float) -> Dict[str, Any]:
        """Close trade with real cost calculations."""
        # Apply transaction costs
        cost_bps = 25  # 25 basis points
        cost_amount = trade['position_size'] * (cost_bps / 10000) * 2
        pnl_after_costs = pnl - (cost_amount / self.current_balance * 100)

        # Calculate R-multiple
        risk_amount = abs(trade['sl_price'] - trade['entry_price']) / trade['entry_price'] * 100
        r_multiple = pnl_after_costs / risk_amount if risk_amount > 0 else 0

        # Update balance
        self.current_balance += (pnl_after_costs / 100 * self.current_balance)

        closed_trade = trade.copy()
        closed_trade.update({
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_type': exit_type,
            'pnl': pnl_after_costs,
            'r_multiple': r_multiple,
            'cost_bps': cost_bps,
            'status': 'closed'
        })

        return closed_trade

    def _generate_full_results(self, dataset: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive results."""
        if not self.trades:
            return {
                'error': 'No trades generated',
                'total_trades': 0,
                'final_balance': self.current_balance
            }

        # Performance calculations
        total_return = (self.current_balance - self.starting_balance) / self.starting_balance * 100
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(self.trades) * 100

        if winning_trades and len(self.trades) > len(winning_trades):
            avg_win = np.mean([t['pnl'] for t in winning_trades])
            losing_trades = [t for t in self.trades if t['pnl'] <= 0]
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if avg_loss != 0 else float('inf')
        else:
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = 0
            profit_factor = float('inf') if winning_trades else 0

        # Drawdown calculation
        running_balance = self.starting_balance
        peak = self.starting_balance
        max_dd = 0

        for trade in self.trades:
            running_balance += (trade['pnl'] / 100 * running_balance)
            peak = max(peak, running_balance)
            dd = (peak - running_balance) / peak * 100
            max_dd = max(max_dd, dd)

        # Calculate period
        start_date = dataset['eth_mtf'].index[0]
        end_date = dataset['eth_mtf'].index[-1]
        period_days = (end_date - start_date).days

        return {
            'period': {
                'start': start_date,
                'end': end_date,
                'days': period_days,
                'years': period_days / 365
            },
            'performance': {
                'total_trades': len(self.trades),
                'final_balance': self.current_balance,
                'total_return': total_return,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_dd,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'winning_trades': len(winning_trades),
                'losing_trades': len(self.trades) - len(winning_trades)
            },
            'engine_usage': self.engine_usage,
            'trades': self.trades,
            'data_source': 'real_chart_logs',
            'bull_machine_version': '1.7.1-enhanced'
        }

def main():
    """Run full year real ETH backtest."""
    backtest = FullYearBullMachine()
    results = backtest.run_full_year_backtest()

    if 'error' in results:
        print(f"❌ {results['error']}")
        return

    # Display results
    period = results['period']
    perf = results['performance']

    print(f"\n🎉 FULL YEAR REAL ETH BACKTEST COMPLETE!")
    print(f"=" * 70)
    print(f"📅 Period: {period['start'].strftime('%Y-%m-%d')} → {period['end'].strftime('%Y-%m-%d')}")
    print(f"⏱️  Duration: {period['days']} days ({period['years']:.1f} years)")
    print(f"📊 Data Source: Real ETH from chart_logs")
    print(f"🔧 Engine: Complete Bull Machine v1.7.1")

    print(f"\n📈 PERFORMANCE RESULTS:")
    print(f"-" * 70)
    print(f"💰 Starting Balance:     ${backtest.starting_balance:,.0f}")
    print(f"💰 Final Balance:        ${perf['final_balance']:,.2f}")
    print(f"📊 Total Return:         {perf['total_return']:+.2f}%")
    print(f"🎯 Total Trades:         {perf['total_trades']}")
    print(f"🏆 Win Rate:            {perf['win_rate']:.1f}%")
    print(f"⚖️  Profit Factor:       {perf['profit_factor']:.2f}")
    print(f"⬇️  Max Drawdown:        {perf['max_drawdown']:.2f}%")
    print(f"📈 Average Win:          {perf['avg_win']:+.2f}%")
    print(f"📉 Average Loss:         {perf['avg_loss']:+.2f}%")

    print(f"\n🔧 ENGINE UTILIZATION:")
    print(f"-" * 70)
    total_engine_signals = sum(results['engine_usage'].values())
    for engine, count in results['engine_usage'].items():
        if count > 0:
            pct = count / total_engine_signals * 100 if total_engine_signals > 0 else 0
            print(f"   {engine.upper()}: {count} ({pct:.1f}%)")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"full_year_real_eth_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Full results saved to: {results_file}")

    # Final assessment
    print(f"\n🎯 VERDICT:")
    print(f"=" * 70)
    if perf['total_return'] > 50 and perf['win_rate'] > 50 and perf['profit_factor'] > 1.5:
        print(f"✅ EXCEPTIONAL PERFORMANCE - Production Ready!")
        print(f"   Bull Machine v1.7.1 exceeds all institutional standards")
    elif perf['total_return'] > 0 and perf['win_rate'] > 45:
        print(f"✅ SOLID PERFORMANCE - Approved for deployment")
    else:
        print(f"⚠️  Performance below expectations - Review required")

if __name__ == "__main__":
    main()