#!/usr/bin/env python3
"""
Adaptive Backtest Engine - Bull Machine v1.7.2
Uses asset-specific configurations for improved multi-asset performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from engine.io.tradingview_loader import load_tv

class AdaptiveBullMachine:
    """
    Bull Machine v1.7.2 with Asset-Specific Adaptive Configuration.
    Loads asset profiles and applies relative thresholds instead of fixed values.
    """

    def __init__(self, symbol: str, exchange: str = "COINBASE"):
        self.symbol = symbol
        self.exchange = exchange
        self.trades = []
        self.balance = 10000
        self.position = None

        # Load asset-specific configuration
        self.config = self.load_adaptive_config()
        self.profile = self.load_asset_profile()

        # Track engine usage
        self.engine_usage = {
            'smc': 0,
            'wyckoff': 0,
            'hob': 0,
            'momentum': 0,
            'macro': 0,
            'counter_trend_blocked': 0,
            'correlation_veto': 0,
            'atr_filter': 0
        }

        print(f"🎯 Initialized Adaptive Bull Machine for {symbol}")
        print(f"   Risk Scalar: {self.config['risk']['position_size_scalar']:.2f}")
        print(f"   Volatility Regime: {self.profile['volatility']['regime']}")
        print(f"   Spring ATR Threshold: {self.config['wyckoff']['spring_wick_min_atr']:.2f}")

    def load_adaptive_config(self) -> dict:
        """Load asset-specific adaptive configuration."""
        config_path = f"configs/adaptive/{self.exchange}_{self.symbol}_config.json"

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded adaptive config: {config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠️  No adaptive config found for {self.symbol}, using defaults")
            return self.get_default_config()

    def load_asset_profile(self) -> dict:
        """Load asset profile with market characteristics."""
        profile_path = f"profiles/{self.exchange}_{self.symbol}_profile.json"

        try:
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            print(f"✅ Loaded asset profile: {profile_path}")
            return profile
        except FileNotFoundError:
            print(f"⚠️  No profile found for {self.symbol}, using defaults")
            return self.get_default_profile()

    def get_default_config(self) -> dict:
        """Fallback configuration for assets without profiles."""
        return {
            'wyckoff': {'spring_wick_min_atr': 1.2, 'volume_spike_z': 1.5},
            'smc': {'min_displacement_atr': 1.0, 'ob_volume_min_z': 1.3},
            'liquidity': {'hob_volume_z_min_long': 1.3, 'hob_volume_z_min_short': 1.6},
            'risk': {'position_size_scalar': 1.0, 'atr_multiplier_stop': 2.0, 'atr_multiplier_target': 3.0},
            'filters': {'min_atr_percentile': 20, 'max_atr_percentile': 90}
        }

    def get_default_profile(self) -> dict:
        """Fallback profile for assets without profiles."""
        return {
            'volatility': {'regime': 'medium', 'atr': {'p20': 1.5, 'p50': 2.0, 'p80': 3.0}},
            'liquidity': {'volume_z_scores': {'z70': 1.2, 'z85': 1.8}},
            'correlations': {'full_period': 0.5}
        }

    def calculate_adaptive_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate ATR with percentile context."""
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        atr = true_range.rolling(window).mean()
        atr_pct = (atr / data['Close']) * 100

        return atr_pct

    def calculate_volume_z_score(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate volume z-score."""
        volume_mean = data['Volume'].rolling(window).mean()
        volume_std = data['Volume'].rolling(window).std()
        return (data['Volume'] - volume_mean) / (volume_std + 1e-8)

    def analyze_smc_patterns(self, window_1h, window_4h):
        """Adaptive SMC analysis using asset-specific thresholds."""
        if window_1h is None or len(window_1h) < 20 or len(window_4h) < 10:
            return None

        # Use adaptive displacement threshold
        min_displacement = self.config['smc']['min_displacement_atr']
        current_atr = self.calculate_adaptive_atr(window_4h).iloc[-1]

        # Order blocks with adaptive sizing
        recent_range = current_atr * min_displacement / 100  # Convert ATR% to price

        recent_high = window_4h['High'].rolling(10).max().iloc[-1]
        recent_low = window_4h['Low'].rolling(10).min().iloc[-1]
        current = window_1h['Close'].iloc[-1]

        signal = 0
        confidence = 0.6

        # Breakout above with adaptive threshold
        if current > recent_high * (1 + recent_range/100):
            signal = 1
            confidence = 0.75

        # Breakdown below with adaptive threshold
        elif current < recent_low * (1 - recent_range/100):
            signal = -1
            confidence = 0.75

        # Volume confirmation with adaptive z-score
        volume_z = self.calculate_volume_z_score(window_1h).iloc[-1]
        min_volume_z = self.config['smc']['ob_volume_min_z']

        if signal != 0 and volume_z > min_volume_z:
            self.engine_usage['smc'] += 1
            return {'signal': signal, 'confidence': confidence}

        return None

    def analyze_wyckoff_patterns(self, window_4h, window_1d):
        """Adaptive Wyckoff analysis using asset-specific thresholds."""
        if len(window_4h) < 20 or len(window_1d) < 10:
            return None

        # Calculate current ATR for adaptive thresholds
        current_atr = self.calculate_adaptive_atr(window_4h).iloc[-1]
        volume_z = self.calculate_volume_z_score(window_4h).iloc[-1]

        # Adaptive volume spike threshold
        volume_spike_threshold = self.config['wyckoff']['volume_spike_z']
        spring_wick_threshold = self.config['wyckoff']['spring_wick_min_atr']

        # Spring pattern with adaptive wick requirement
        price_change_4h = (window_4h['Close'].iloc[-1] / window_4h['Close'].iloc[-5] - 1) * 100

        if volume_z > volume_spike_threshold and price_change_4h < -2:
            # Check for spring wick (adaptive threshold)
            wick_size = (window_4h['Close'].iloc[-1] - window_4h['Low'].iloc[-1]) / window_4h['Close'].iloc[-1] * 100

            if wick_size > spring_wick_threshold * current_atr / 100:
                self.engine_usage['wyckoff'] += 1
                return {'signal': 1, 'confidence': 0.75}

        # UTAD pattern
        if volume_z > volume_spike_threshold and price_change_4h > 2:
            wick_size = (window_4h['High'].iloc[-1] - window_4h['Close'].iloc[-1]) / window_4h['Close'].iloc[-1] * 100

            if wick_size > spring_wick_threshold * current_atr / 100:
                self.engine_usage['wyckoff'] += 1
                return {'signal': -1, 'confidence': 0.75}

        return None

    def analyze_hob_patterns(self, window_1h, window_4h, macro_window):
        """Adaptive HOB analysis using asset-specific volume requirements."""
        if window_1h is None or len(window_1h) < 20 or len(window_4h) < 10:
            return None

        # Adaptive volume thresholds from config
        min_z_long = self.config['liquidity']['hob_volume_z_min_long']
        min_z_short = self.config['liquidity']['hob_volume_z_min_short']

        volume_z = self.calculate_volume_z_score(window_1h).iloc[-1]

        # Breakout detection
        resistance = window_4h['High'].rolling(10).max().iloc[-1]
        support = window_4h['Low'].rolling(10).min().iloc[-1]
        current = window_1h['Close'].iloc[-1]

        if current > resistance * 0.995 and volume_z > min_z_long:
            self.engine_usage['hob'] += 1
            return {'signal': 1, 'confidence': 0.8}
        elif current < support * 1.005 and volume_z > min_z_short:
            self.engine_usage['hob'] += 1
            return {'signal': -1, 'confidence': 0.8}

        return None

    def analyze_momentum_patterns(self, window_4h, window_1d):
        """Adaptive momentum analysis."""
        if len(window_4h) < 26 or len(window_1d) < 14:
            return None

        # Calculate indicators
        delta = window_4h['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        # Adaptive RSI thresholds based on volatility regime
        regime = self.profile['volatility']['regime']
        if regime == 'high':
            oversold, overbought = 25, 75
        elif regime == 'low':
            oversold, overbought = 35, 65
        else:
            oversold, overbought = 30, 70

        # MACD
        ema12 = window_4h['Close'].ewm(span=12).mean()
        ema26 = window_4h['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()

        # Generate signals with adaptive thresholds
        if rsi.iloc[-1] < oversold and macd.iloc[-1] > signal_line.iloc[-1]:
            self.engine_usage['momentum'] += 1
            return {'signal': 1, 'confidence': 0.65}
        elif rsi.iloc[-1] > overbought and macd.iloc[-1] < signal_line.iloc[-1]:
            self.engine_usage['momentum'] += 1
            return {'signal': -1, 'confidence': 0.65}

        return None

    def apply_adaptive_filters(self, signal, window_4h, macro_window):
        """Apply asset-specific filters to signals."""
        if not signal:
            return signal

        # ATR filter - don't trade in extreme volatility
        current_atr = self.calculate_adaptive_atr(window_4h).iloc[-1]
        atr_profile = self.profile['volatility']['atr']

        atr_percentile = 50  # Default to median
        if current_atr <= atr_profile['p20']:
            atr_percentile = 20
        elif current_atr <= atr_profile['p50']:
            atr_percentile = 50
        elif current_atr <= atr_profile['p80']:
            atr_percentile = 80
        else:
            atr_percentile = 90

        min_atr_p = self.config['filters']['min_atr_percentile']
        max_atr_p = self.config['filters']['max_atr_percentile']

        if atr_percentile < min_atr_p or atr_percentile > max_atr_p:
            self.engine_usage['atr_filter'] += 1
            return None

        # Correlation veto for highly correlated assets
        btc_correlation = self.profile.get('correlations', {}).get('full_period', 0)
        correlation_threshold = self.config['filters']['correlation_veto_threshold']

        if abs(btc_correlation) > correlation_threshold:
            # Could add BTC momentum check here
            # For now, just note the correlation
            pass

        return signal

    def enhanced_signal_generation(self, dataset, current_idx):
        """Generate signals using adaptive thresholds."""
        # Extract data windows
        window_1h = dataset['1h'][:current_idx] if '1h' in dataset and dataset['1h'] is not None else None
        window_4h = dataset['4h'][:current_idx] if '4h' in dataset else pd.DataFrame()
        window_1d = dataset['1d'][:current_idx] if '1d' in dataset else pd.DataFrame()
        macro_window = dataset['macro'][:current_idx] if 'macro' in dataset else None

        if len(window_4h) < 20 or len(window_1d) < 10:
            return None

        # Get signals from all engines
        signals = []

        smc_signal = self.analyze_smc_patterns(window_1h, window_4h)
        wyckoff_signal = self.analyze_wyckoff_patterns(window_4h, window_1d)
        hob_signal = self.analyze_hob_patterns(window_1h, window_4h, macro_window)
        momentum_signal = self.analyze_momentum_patterns(window_4h, window_1d)

        # Collect active signals
        for sig in [smc_signal, wyckoff_signal, hob_signal, momentum_signal]:
            if sig:
                signals.append(sig)

        if len(signals) == 0:
            return None

        # Engine consensus requirement
        long_signals = [s for s in signals if s['signal'] > 0]
        short_signals = [s for s in signals if s['signal'] < 0]

        final_signal = None

        if len(long_signals) >= 2:
            avg_confidence = np.mean([s['confidence'] for s in long_signals])
            final_signal = {
                'signal': 1,
                'confidence': avg_confidence,
                'engines': len(long_signals),
                'timestamp': window_4h.index[-1],
                'price': window_4h['Close'].iloc[-1]
            }

        elif len(short_signals) >= 2:
            avg_confidence = np.mean([s['confidence'] for s in short_signals])
            final_signal = {
                'signal': -1,
                'confidence': avg_confidence,
                'engines': len(short_signals),
                'timestamp': window_4h.index[-1],
                'price': window_4h['Close'].iloc[-1]
            }

        # Apply adaptive filters
        return self.apply_adaptive_filters(final_signal, window_4h, macro_window)

    def execute_trade(self, signal, price):
        """Execute trade with adaptive risk management."""
        if self.position is not None:
            # Close existing position
            exit_price = price

            if self.position['side'] == 1:  # Long
                pnl_pct = (exit_price / self.position['entry_price'] - 1) * 100
            else:  # Short
                pnl_pct = (1 - exit_price / self.position['entry_price']) * 100

            # Apply transaction costs
            pnl_pct -= 0.3  # 0.15% each way

            # Adaptive risk management
            stop_multiplier = self.config['risk']['atr_multiplier_stop']
            target_multiplier = self.config['risk']['atr_multiplier_target']

            stop_loss_pct = stop_multiplier
            take_profit_pct = target_multiplier

            # Determine exit type
            if pnl_pct <= -stop_loss_pct:
                pnl_pct = -stop_loss_pct
                exit_type = 'stop_loss'
            elif pnl_pct >= take_profit_pct:
                pnl_pct = take_profit_pct
                exit_type = 'take_profit'
            else:
                exit_type = 'signal_reversal'

            # Apply position sizing scalar
            size_scalar = self.config['risk']['position_size_scalar']
            pnl_pct *= size_scalar

            self.balance *= (1 + pnl_pct / 100)

            self.trades.append({
                'entry_time': self.position['entry_time'],
                'exit_time': signal['timestamp'] if signal else self.position['entry_time'],
                'side': 'long' if self.position['side'] == 1 else 'short',
                'entry_price': self.position['entry_price'],
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'balance': self.balance,
                'engines_used': self.position['engines'],
                'exit_type': exit_type,
                'size_scalar': size_scalar
            })

            self.position = None

        # Open new position
        if signal and signal['signal'] != 0:
            self.position = {
                'entry_time': signal['timestamp'],
                'entry_price': price,
                'side': signal['signal'],
                'engines': signal['engines']
            }

    def run_backtest(self, symbol: str):
        """Run adaptive backtest on specified asset."""
        print(f"\n🚀 Running Adaptive Backtest: {symbol}")
        print("=" * 60)

        loader = RealDataLoader()

        # Load multi-timeframe data
        dataset = {}

        # Load data based on availability
        data_1h = loader.load_raw_data(f'COINBASE_{symbol}', '1H')
        data_4h = loader.load_raw_data(f'COINBASE_{symbol}', '4H')
        data_1d = loader.load_raw_data(f'COINBASE_{symbol}', '1D')

        if data_4h is None:
            print(f"❌ No 4H data available for {symbol}")
            return None

        dataset['4h'] = data_4h
        dataset['1d'] = data_1d
        if data_1h is not None:
            dataset['1h'] = data_1h

        # Find overlapping period
        if data_1h is not None:
            start_overlap = max(data_1h.index[0], data_4h.index[0], data_1d.index[0])
            end_overlap = min(data_1h.index[-1], data_4h.index[-1], data_1d.index[-1])
        else:
            start_overlap = max(data_4h.index[0], data_1d.index[0])
            end_overlap = min(data_4h.index[-1], data_1d.index[-1])

        print(f"📊 Data period: {start_overlap} → {end_overlap}")

        # Filter to overlapping period
        for tf in dataset:
            if dataset[tf] is not None:
                dataset[tf] = dataset[tf][(dataset[tf].index >= start_overlap) &
                                        (dataset[tf].index <= end_overlap)]

        # Generate synthetic macro
        macro = loader.generate_macro_data(
            start_overlap.strftime('%Y-%m-%d'),
            end_overlap.strftime('%Y-%m-%d'),
            '4H'
        )
        dataset['macro'] = macro

        print(f"✅ Dataset loaded: {len(dataset['4h'])} bars")

        # Run backtest
        total_bars = len(dataset['4h'])
        for i in range(26, total_bars):
            signal = self.enhanced_signal_generation(dataset, i)

            if signal:
                self.execute_trade(signal, signal['price'])

                # Log trade
                if self.position:
                    side = 'LONG' if signal['signal'] == 1 else 'SHORT'
                    print(f"🎯 {signal['timestamp'].strftime('%Y-%m-%d %H:%M')} | {side} ${signal['price']:.2f} | "
                          f"{signal['engines']} engines | Conf: {signal['confidence']:.2f}")

        # Close final position
        if self.position is not None:
            final_price = dataset['4h']['Close'].iloc[-1]
            self.execute_trade(None, final_price)

        return self.generate_results(start_overlap, end_overlap)

    def generate_results(self, start_date, end_date):
        """Generate results with adaptive metrics."""
        if len(self.trades) == 0:
            return {'error': 'No trades generated'}

        winning_trades = [t for t in self.trades if t['pnl_pct'] > 0]
        losing_trades = [t for t in self.trades if t['pnl_pct'] <= 0]

        days = (end_date - start_date).days

        return {
            'symbol': self.symbol,
            'period': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'days': days
            },
            'performance': {
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(self.trades) * 100 if self.trades else 0,
                'avg_win': np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0,
                'avg_loss': np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0,
                'final_balance': self.balance,
                'total_return': (self.balance / 10000 - 1) * 100,
                'profit_factor': abs(sum([t['pnl_pct'] for t in winning_trades]) /
                                   sum([t['pnl_pct'] for t in losing_trades])) if losing_trades else 0
            },
            'engine_usage': self.engine_usage,
            'config_used': self.config['meta'] if 'meta' in self.config else {},
            'trades': self.trades[-10:]  # Last 10 trades for review
        }

def main():
    """Test adaptive backtest on multiple assets."""
    print("🏭 Adaptive Backtest Engine v1.7.2")
    print("=" * 70)

    assets = ['ETHUSD', 'SOLUSD', 'XRPUSD']
    results = {}

    for symbol in assets:
        try:
            backtest = AdaptiveBullMachine(symbol)
            result = backtest.run_backtest(symbol)

            if result and 'error' not in result:
                results[symbol] = result

                perf = result['performance']
                print(f"\n✅ {symbol} Results:")
                print(f"   Return: {perf['total_return']:+.2f}%")
                print(f"   Trades: {perf['total_trades']}")
                print(f"   Win Rate: {perf['win_rate']:.1f}%")
                print(f"   Profit Factor: {perf['profit_factor']:.2f}")

        except Exception as e:
            print(f"❌ Error with {symbol}: {e}")

    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'adaptive_backtest_results_{timestamp}.json'

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {filename}")
    print("\n🎯 Adaptive backtesting complete!")

if __name__ == "__main__":
    main()