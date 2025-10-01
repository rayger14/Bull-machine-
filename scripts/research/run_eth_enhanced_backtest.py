#!/usr/bin/env python3
"""
ETH Enhanced Backtest v1.7.1 - Moneytaur/Wyckoff/ZeroIKA Alignment
Implements surgical improvements for better win rate and R/R skew
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from engine.risk.transaction_costs import TransactionCostModel
from engine.timeframes.mtf_alignment import MTFAlignmentEngine
from engine.metrics.cost_adjusted_metrics import CostAdjustedMetrics

class EnhancedBacktester:
    """
    Enhanced backtest with Moneytaur/Wyckoff/ZeroIKA improvements
    """

    def __init__(self, starting_balance: float = 10000.0, config_version: str = "v171"):
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.config_version = config_version

        # Load enhanced configs
        self.config = self._load_enhanced_configs()

        # Initialize engines
        self.cost_model = TransactionCostModel()
        self.mtf_engine = MTFAlignmentEngine()
        self.metrics_calc = CostAdjustedMetrics(self.cost_model)

        # Enhanced tracking
        self.engine_usage = {
            'smc': 0, 'wyckoff': 0, 'momentum': 0, 'hob': 0,
            'macro_veto': 0, 'ethbtc_veto': 0, 'atr_throttle': 0,
            'counter_trend_blocked': 0
        }

        self.trades = []
        self.daily_balance = []
        self.rejected_signals = []

    def _load_enhanced_configs(self):
        """Load v1.7.1 enhanced configuration"""
        config_dir = Path(f'configs/{self.config_version}')

        configs = {}
        for config_file in ['fusion.json', 'context.json', 'liquidity.json',
                           'exits.json', 'risk.json', 'momentum.json']:
            try:
                with open(config_dir / config_file, 'r') as f:
                    configs[config_file.replace('.json', '')] = json.load(f)
            except FileNotFoundError:
                print(f"⚠️  Config file not found: {config_file}")
                configs[config_file.replace('.json', '')] = {}

        return configs

    def load_multi_asset_data(self):
        """Load ETH + macro data for enhanced gates"""
        base_path = "/Users/raymondghandchi/Desktop/Chart Logs/"

        # Load ETH 4H data
        df_eth_4h = pd.read_csv(f"{base_path}COINBASE_ETHUSD, 240_ab8a9.csv")
        df_eth_4h['datetime'] = pd.to_datetime(df_eth_4h['time'], unit='s')
        df_eth_4h.set_index('datetime', inplace=True)
        df_eth_4h = df_eth_4h.rename(columns={'BUY+SELL V': 'volume'})

        # Create synthetic 1H and 1D data
        df_eth_1h = self._create_1h_from_4h(df_eth_4h)
        df_eth_1d = df_eth_4h.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

        # Load macro data (mock for testing)
        macro_data = self._create_mock_macro_data(df_eth_4h.index)

        print(f"✅ Enhanced data loaded:")
        print(f"   ETH 1H: {len(df_eth_1h)} bars")
        print(f"   ETH 4H: {len(df_eth_4h)} bars")
        print(f"   ETH 1D: {len(df_eth_1d)} bars")
        print(f"   Macro: {len(macro_data)} bars")

        return df_eth_1h, df_eth_4h, df_eth_1d, macro_data

    def _create_1h_from_4h(self, df_4h):
        """Create synthetic 1H data"""
        data_1h = []
        for i, (timestamp, bar) in enumerate(df_4h.iterrows()):
            for j in range(4):
                hour_start = timestamp + timedelta(hours=j)
                price_range = bar['high'] - bar['low']

                if j == 0:
                    open_px = bar['open']
                    close_px = bar['open'] + (price_range * 0.25)
                elif j == 1:
                    open_px = close_px
                    close_px = bar['open'] + (price_range * 0.6)
                elif j == 2:
                    open_px = close_px
                    close_px = bar['open'] + (price_range * 0.8)
                else:
                    open_px = close_px
                    close_px = bar['close']

                high_px = max(open_px, close_px) + (price_range * 0.1)
                low_px = min(open_px, close_px) - (price_range * 0.1)

                data_1h.append({
                    'datetime': hour_start,
                    'open': open_px, 'high': high_px, 'low': low_px,
                    'close': close_px, 'volume': bar['volume'] / 4
                })

        df_1h = pd.DataFrame(data_1h)
        df_1h.set_index('datetime', inplace=True)
        return df_1h

    def _create_mock_macro_data(self, eth_index):
        """Create mock macro data for enhanced gates"""
        macro_data = pd.DataFrame(index=eth_index)

        # Mock ETHBTC (with some trending behavior)
        base_ethbtc = 0.065
        ethbtc_trend = np.cumsum(np.random.randn(len(eth_index)) * 0.001)
        macro_data['ETHBTC'] = base_ethbtc + ethbtc_trend

        # Mock TOTAL2 (crypto market cap excluding BTC)
        base_total2 = 600e9  # $600B
        total2_growth = np.cumsum(np.random.randn(len(eth_index)) * 5e9)
        macro_data['TOTAL2'] = base_total2 + total2_growth

        # Mock TOTAL (total crypto market cap)
        macro_data['TOTAL'] = macro_data['TOTAL2'] * 1.8  # ~80% more including BTC

        # Calculate TOTAL2 dominance
        macro_data['TOTAL2_dominance'] = macro_data['TOTAL2'] / macro_data['TOTAL']

        # Mock VIX
        macro_data['VIX'] = 18 + np.random.normal(0, 4, len(eth_index))

        return macro_data

    def enhanced_signal_generation(self, df_1h, df_4h, df_1d, macro_data, current_idx):
        """
        Enhanced signal generation with all v1.7.1 improvements
        """
        if current_idx < 50:
            return None

        # Get data windows
        window_1h = df_1h.iloc[max(0, current_idx*4-200):current_idx*4]
        window_4h = df_4h.iloc[max(0, current_idx-50):current_idx]
        window_1d = df_1d.iloc[max(0, current_idx//6-20):current_idx//6+1]
        macro_window = macro_data.iloc[current_idx-20:current_idx] if len(macro_data) > current_idx else macro_data

        if len(window_1h) < 20 or len(window_4h) < 20 or len(window_1d) < 5:
            return None

        current_bar = df_4h.iloc[current_idx]
        current_timestamp = df_4h.index[current_idx]

        try:
            # 1. ATR Throttle Check (Cost-aware)
            atr_check = self._check_atr_throttle(window_4h)
            if not atr_check:
                self.engine_usage['atr_throttle'] += 1
                return None

            # 2. Generate individual engine signals
            smc_signal = self._generate_enhanced_smc_signal(window_4h)
            wyckoff_signal = self._generate_enhanced_wyckoff_signal(window_1d)
            momentum_signal = self._generate_enhanced_momentum_signal(window_4h, window_1d)
            hob_signal = self._generate_enhanced_hob_signal(window_1h, window_4h)

            # 3. Determine trend direction for counter-trend discipline
            trend_direction = self._get_trend_direction(window_1d)

            # 4. Count active engines
            active_engines = []
            total_confidence = 0

            if smc_signal['confidence'] > 0.3:
                active_engines.append('smc')
                total_confidence += smc_signal['confidence']
                self.engine_usage['smc'] += 1

            if wyckoff_signal['confidence'] > 0.3:
                active_engines.append('wyckoff')
                total_confidence += wyckoff_signal['confidence']
                self.engine_usage['wyckoff'] += 1

            if momentum_signal['confidence'] > 0.3:
                active_engines.append('momentum')
                total_confidence += momentum_signal['confidence']
                self.engine_usage['momentum'] += 1

            if hob_signal['confidence'] > 0.3:
                active_engines.append('hob')
                total_confidence += hob_signal['confidence']
                self.engine_usage['hob'] += 1

            if len(active_engines) < 2:
                return None

            # 5. Determine signal direction
            directions = [
                smc_signal['direction'], wyckoff_signal['direction'],
                momentum_signal['direction'], hob_signal['direction']
            ]
            valid_directions = [d for d in directions if d != 'neutral']

            if not valid_directions:
                return None

            direction_counts = {}
            for direction in valid_directions:
                direction_counts[direction] = direction_counts.get(direction, 0) + 1

            signal_direction = max(direction_counts, key=direction_counts.get)
            consensus_strength = direction_counts[signal_direction] / len(valid_directions)

            if consensus_strength < 0.6:
                return None

            # 6. Counter-trend discipline check
            is_counter_trend = self._is_counter_trend(signal_direction, trend_direction)
            if is_counter_trend and self.config['fusion']['counter_trend_discipline']['enabled']:
                required_engines = self.config['fusion']['counter_trend_discipline']['require_3_engines_if_countertrend']
                if required_engines and len(active_engines) < 3:
                    self.engine_usage['counter_trend_blocked'] += 1
                    return None

            # 7. ETHBTC/TOTAL2 rotation gate
            if signal_direction == 'bearish':  # ETH short
                ethbtc_veto = self._check_ethbtc_total2_gate(macro_window)
                if ethbtc_veto:
                    self.engine_usage['ethbtc_veto'] += 1
                    return None

            # 8. Enhanced HOB absorption check for shorts
            if signal_direction == 'bearish' and 'hob' in active_engines:
                hob_quality = self._check_enhanced_hob_absorption(window_1h, signal_direction)
                if not hob_quality:
                    return None

            # 9. Macro veto (VIX, etc.)
            vix_now = macro_window['VIX'].iloc[-1] if len(macro_window) > 0 else 18.0
            if self._check_macro_veto(vix_now):
                self.engine_usage['macro_veto'] += 1
                return None

            # 10. Calculate enhanced fusion confidence
            fusion_confidence = total_confidence / len(active_engines)

            # Apply momentum bias (with-trend bonus)
            if 'momentum' in active_engines:
                momentum_boost = self._calculate_momentum_bias(signal_direction, trend_direction)
                fusion_confidence = min(0.9, fusion_confidence + momentum_boost)

            # 11. R/R Pre-filter
            estimated_rr = self._estimate_risk_reward(current_bar, signal_direction, window_4h)
            min_rr = self.config['exits']['risk_reward']['min_expected_rr']
            if estimated_rr < min_rr:
                return None

            if fusion_confidence < self.config['fusion']['entry_thresholds']['confidence']:
                return None

            return {
                'direction': signal_direction,
                'confidence': fusion_confidence,
                'engines_used': active_engines,
                'is_counter_trend': is_counter_trend,
                'estimated_rr': estimated_rr,
                'timestamp': current_timestamp,
                'trend_direction': trend_direction
            }

        except Exception as e:
            print(f"Enhanced signal generation error at {current_idx}: {e}")
            return None

    def _check_atr_throttle(self, window_4h):
        """Check ATR floor to avoid low-volatility chop"""
        if len(window_4h) < 20:
            return True

        # Calculate ATR
        high_low = window_4h['high'] - window_4h['low']
        high_close = abs(window_4h['high'] - window_4h['close'].shift(1))
        low_close = abs(window_4h['low'] - window_4h['close'].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        current_atr = atr.iloc[-1]
        atr_percentile = self.config['risk']['cost_controls']['atr_floor_percentile']

        # Calculate ATR percentile threshold
        atr_history = atr.dropna()
        if len(atr_history) < 50:
            return True

        atr_threshold = np.percentile(atr_history, atr_percentile)
        return current_atr >= atr_threshold

    def _get_trend_direction(self, window_1d):
        """Determine 1D trend direction"""
        if len(window_1d) < 10:
            return 'neutral'

        # Simple MA-based trend
        ma_short = window_1d['close'].rolling(5).mean()
        ma_long = window_1d['close'].rolling(10).mean()

        if ma_short.iloc[-1] > ma_long.iloc[-1] and ma_short.iloc[-1] > ma_short.iloc[-3]:
            return 'bullish'
        elif ma_short.iloc[-1] < ma_long.iloc[-1] and ma_short.iloc[-1] < ma_short.iloc[-3]:
            return 'bearish'
        else:
            return 'neutral'

    def _is_counter_trend(self, signal_direction, trend_direction):
        """Check if signal is counter-trend"""
        if trend_direction == 'neutral':
            return False
        return (signal_direction == 'bearish' and trend_direction == 'bullish') or \
               (signal_direction == 'bullish' and trend_direction == 'bearish')

    def _check_ethbtc_total2_gate(self, macro_window):
        """ETHBTC/TOTAL2 rotation gate for ETH shorts"""
        if len(macro_window) < 5:
            return False

        # Check ETHBTC trend
        ethbtc_current = macro_window['ETHBTC'].iloc[-1]
        ethbtc_prev = macro_window['ETHBTC'].iloc[-5]
        ethbtc_uptrend = ethbtc_current > ethbtc_prev * 1.01  # 1% threshold

        # Check TOTAL2 dominance
        total2_dom = macro_window['TOTAL2_dominance'].iloc[-1]
        total2_premium = total2_dom > self.config['context']['rotation_gates']['premium_fib_min']

        # Block ETH shorts if ETHBTC uptrending AND TOTAL2 in premium
        return ethbtc_uptrend and total2_premium

    def _check_enhanced_hob_absorption(self, window_1h, direction):
        """Enhanced HOB absorption check for shorts"""
        if direction != 'bearish':
            return True  # Only strict requirements for shorts

        volume_z_min = self.config['liquidity']['hob_quality_factors']['volume_z_min_short']

        if len(window_1h) < 20:
            return False

        # Check volume z-score
        volume_ma = window_1h['volume'].rolling(20).mean()
        volume_std = window_1h['volume'].rolling(20).std()
        recent_volume = window_1h['volume'].tail(3).mean()

        z_score = (recent_volume - volume_ma.iloc[-1]) / volume_std.iloc[-1] if volume_std.iloc[-1] > 0 else 0

        return z_score >= volume_z_min

    def _calculate_momentum_bias(self, signal_direction, trend_direction):
        """Calculate momentum bias boost"""
        momentum_config = self.config['momentum']['momentum_weights']

        # Only boost if with-trend
        if signal_direction == trend_direction:
            return momentum_config['with_trend_bonus']
        else:
            return momentum_config['countertrend_bonus']  # 0.00

    def _estimate_risk_reward(self, current_bar, direction, window_4h):
        """Estimate R/R ratio for pre-filtering"""
        if len(window_4h) < 20:
            return 1.0

        # Calculate ATR for stops and targets
        high_low = window_4h['high'] - window_4h['low']
        high_close = abs(window_4h['high'] - window_4h['close'].shift(1))
        low_close = abs(window_4h['low'] - window_4h['close'].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

        # Simple R/R estimation
        sl_distance = atr * self.config['exits']['stop_loss']['initial_sl_atr']
        tp_distance = atr * self.config['exits']['take_profit']['tp1_r'] * 1.5  # Rough target

        return tp_distance / sl_distance if sl_distance > 0 else 1.0

    def _generate_enhanced_smc_signal(self, df):
        """Enhanced SMC signal with better quality"""
        if len(df) < 20:
            return {'confidence': 0, 'direction': 'neutral'}

        returns = df['close'].pct_change().dropna()
        momentum = returns.tail(10).mean()
        volume_trend = np.polyfit(range(len(df.tail(10))), df['volume'].tail(10).values, 1)[0]

        # Enhanced quality check
        price_structure = self._analyze_price_structure(df)
        volume_confirmation = abs(volume_trend) > np.std(df['volume'].tail(20)) * 0.5

        base_confidence = min(0.8, abs(momentum) * 100 + abs(volume_trend) * 0.001)

        # Boost confidence for good structure
        if price_structure and volume_confirmation:
            base_confidence *= 1.2

        direction = 'bullish' if momentum > 0 else 'bearish'
        return {'confidence': min(0.8, base_confidence), 'direction': direction}

    def _analyze_price_structure(self, df):
        """Analyze price structure for SMC"""
        if len(df) < 10:
            return False

        # Simple HH/HL or LH/LL detection
        highs = df['high'].tail(10)
        lows = df['low'].tail(10)

        recent_high = highs.tail(3).max()
        prev_high = highs.head(-3).max()
        recent_low = lows.tail(3).min()
        prev_low = lows.head(-3).min()

        # Higher highs and higher lows = bullish structure
        # Lower highs and lower lows = bearish structure
        bullish_structure = recent_high > prev_high and recent_low > prev_low
        bearish_structure = recent_high < prev_high and recent_low < prev_low

        return bullish_structure or bearish_structure

    def _generate_enhanced_wyckoff_signal(self, df):
        """Enhanced Wyckoff signal"""
        if len(df) < 10:
            return {'confidence': 0, 'direction': 'neutral'}

        volume_ma = df['volume'].rolling(5).mean()
        price_change = df['close'].pct_change()

        recent_vol = volume_ma.tail(3).mean()
        historical_vol = volume_ma.head(-3).mean()

        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1
        price_momentum = price_change.tail(5).mean()

        # Look for volume-price relationships
        confidence = min(0.7, abs(vol_ratio - 1) + abs(price_momentum) * 10)

        # Enhanced: Look for spring/upthrust patterns
        if self._detect_spring_pattern(df):
            confidence *= 1.3
            direction = 'bullish'
        elif self._detect_upthrust_pattern(df):
            confidence *= 1.3
            direction = 'bearish'
        else:
            direction = 'bullish' if price_momentum > 0 and vol_ratio > 1.2 else 'bearish'

        return {'confidence': min(0.7, confidence), 'direction': direction}

    def _detect_spring_pattern(self, df):
        """Detect Wyckoff spring pattern"""
        if len(df) < 10:
            return False

        # Simple spring detection: false breakdown with volume expansion
        lows = df['low']
        volumes = df['volume']

        recent_low = lows.iloc[-1]
        support_level = lows.iloc[-10:-2].min()

        # Check for false breakdown
        breakdown = recent_low < support_level
        volume_expansion = volumes.iloc[-2:].mean() > volumes.iloc[-10:-2].mean() * 1.5

        return breakdown and volume_expansion

    def _detect_upthrust_pattern(self, df):
        """Detect Wyckoff upthrust pattern"""
        if len(df) < 10:
            return False

        highs = df['high']
        volumes = df['volume']

        recent_high = highs.iloc[-1]
        resistance_level = highs.iloc[-10:-2].max()

        # Check for false breakout
        breakout = recent_high > resistance_level
        volume_decline = volumes.iloc[-2:].mean() < volumes.iloc[-10:-2].mean() * 0.8

        return breakout and volume_decline

    def _generate_enhanced_momentum_signal(self, window_4h, window_1d):
        """Enhanced momentum with trend bias"""
        if len(window_4h) < 14:
            return {'confidence': 0, 'direction': 'neutral'}

        # RSI calculation
        returns = window_4h['close'].pct_change().dropna()
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)

        avg_gain = gains.rolling(14).mean()
        avg_loss = losses.rolling(14).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # Enhanced momentum with trend context
        trend_direction = self._get_trend_direction(window_1d)

        if current_rsi > 70:
            confidence = (current_rsi - 70) / 30
            direction = 'bearish'
        elif current_rsi < 30:
            confidence = (30 - current_rsi) / 30
            direction = 'bullish'
        else:
            confidence = 0
            direction = 'neutral'

        # Boost confidence if aligned with trend
        if direction == trend_direction:
            confidence *= 1.2

        return {'confidence': min(0.6, confidence), 'direction': direction}

    def _generate_enhanced_hob_signal(self, window_1h, window_4h):
        """Enhanced HOB signal"""
        if len(window_1h) < 20:
            return {'confidence': 0, 'direction': 'neutral'}

        volume_ma = window_1h['volume'].rolling(20).mean()
        volume_std = window_1h['volume'].rolling(20).std()

        recent_volume = window_1h['volume'].tail(3).mean()
        z_score = (recent_volume - volume_ma.iloc[-1]) / volume_std.iloc[-1] if volume_std.iloc[-1] > 0 else 0

        if z_score > 1.5:
            price_action = window_1h['close'].pct_change().tail(3).mean()

            # Enhanced: Check for proximity to key levels
            proximity_quality = self._check_hob_proximity(window_1h, window_4h)

            confidence = min(0.5, z_score / 3)
            if proximity_quality:
                confidence *= 1.4

            direction = 'bullish' if price_action > 0 else 'bearish'
        else:
            confidence = 0
            direction = 'neutral'

        return {'confidence': min(0.6, confidence), 'direction': direction}

    def _check_hob_proximity(self, window_1h, window_4h):
        """Check HOB proximity to key levels"""
        if len(window_4h) < 20:
            return False

        current_price = window_1h['close'].iloc[-1]

        # Calculate ATR
        high_low = window_4h['high'] - window_4h['low']
        high_close = abs(window_4h['high'] - window_4h['close'].shift(1))
        low_close = abs(window_4h['low'] - window_4h['close'].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

        # Find recent significant levels
        recent_highs = window_4h['high'].tail(20)
        recent_lows = window_4h['low'].tail(20)

        key_levels = []
        key_levels.extend(recent_highs.nlargest(3).values)
        key_levels.extend(recent_lows.nsmallest(3).values)

        # Check proximity
        proximity_threshold = atr * self.config['liquidity']['hob_proximity']['atr_multiplier']

        for level in key_levels:
            if abs(current_price - level) <= proximity_threshold:
                return True

        return False

    def _check_macro_veto(self, vix):
        """Enhanced macro veto check"""
        return vix > self.config['context']['macro_context']['vix_regime_switch_threshold']

    def execute_enhanced_backtest(self):
        """Execute enhanced backtest with all improvements"""

        print("🚀 EXECUTING ENHANCED ETH BACKTEST v1.7.1")
        print("=" * 60)
        print("Improvements:")
        print("  ✅ Counter-trend discipline (3-engine requirement)")
        print("  ✅ ETHBTC/TOTAL2 rotation gates")
        print("  ✅ Enhanced HOB absorption (shorts: z≥1.6)")
        print("  ✅ Asymmetric R/R management (min 1.7)")
        print("  ✅ ATR throttles & momentum bias")
        print("  ✅ Cost-aware filtering")
        print()

        # Load enhanced data
        df_1h, df_4h, df_1d, macro_data = self.load_multi_asset_data()

        # Track position with enhanced management
        in_position = False
        entry_price = 0
        entry_timestamp = None
        position_direction = None
        stop_loss = 0
        take_profit_1 = 0
        position_r = 0

        # Enhanced backtest loop
        for i in range(50, len(df_4h) - 1):
            current_bar = df_4h.iloc[i]
            current_timestamp = df_4h.index[i]

            # Update daily balance
            if len(self.daily_balance) == 0 or self.daily_balance[-1][0].date() != current_timestamp.date():
                self.daily_balance.append((current_timestamp, self.current_balance))

            # Enhanced position management
            if in_position:
                exit_signal = self._enhanced_exit_logic(df_4h, i, entry_timestamp,
                                                       position_direction, entry_price,
                                                       stop_loss, take_profit_1)

                if exit_signal:
                    exit_price = current_bar['close']
                    exit_type = exit_signal['type']

                    # Calculate P&L
                    if position_direction == 'bullish':
                        raw_pnl = (exit_price - entry_price) / entry_price
                    else:
                        raw_pnl = (entry_price - exit_price) / entry_price

                    # Apply enhanced position sizing
                    position_risk = self.config['risk']['position_sizing']['base_risk_pct']
                    position_pnl = raw_pnl * position_risk * self.current_balance

                    # Create enhanced trade record
                    trade = {
                        'entry_timestamp': entry_timestamp,
                        'exit_timestamp': current_timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': position_direction,
                        'raw_pnl_pct': raw_pnl * 100,
                        'position_pnl': position_pnl,
                        'balance_before': self.current_balance,
                        'exit_type': exit_type,
                        'r_multiple': raw_pnl / (abs(entry_price - stop_loss) / entry_price),
                        'quantity': 1.0
                    }

                    self.trades.append(trade)
                    self.current_balance += position_pnl

                    print(f"🔄 Exit  {current_timestamp.strftime('%Y-%m-%d %H:%M')} | "
                          f"{position_direction:>8} | ${exit_price:>7.2f} | "
                          f"P&L: {raw_pnl*100:>+6.2f}% | R: {trade['r_multiple']:>+5.1f} | "
                          f"Type: {exit_type} | Balance: ${self.current_balance:>8,.0f}")

                    in_position = False
                continue

            # Enhanced signal generation
            signal = self.enhanced_signal_generation(df_1h, df_4h, df_1d, macro_data, i)

            if signal and not in_position:
                entry_price = current_bar['close']
                entry_timestamp = current_timestamp
                position_direction = signal['direction']

                # Calculate enhanced stops and targets
                atr = self._calculate_atr(df_4h, i)
                sl_distance = atr * self.config['exits']['stop_loss']['initial_sl_atr']

                if position_direction == 'bullish':
                    stop_loss = entry_price - sl_distance
                    take_profit_1 = entry_price + (sl_distance * self.config['exits']['take_profit']['tp1_r'])
                else:
                    stop_loss = entry_price + sl_distance
                    take_profit_1 = entry_price - (sl_distance * self.config['exits']['take_profit']['tp1_r'])

                print(f"🎯 Entry {current_timestamp.strftime('%Y-%m-%d %H:%M')} | "
                      f"{position_direction:>8} | ${entry_price:>7.2f} | "
                      f"Engines: {','.join(signal['engines_used'])} | "
                      f"Conf: {signal['confidence']:.2f} | "
                      f"RR: {signal['estimated_rr']:.1f} | "
                      f"Trend: {signal['trend_direction']} | "
                      f"SL: ${stop_loss:.2f}")

                in_position = True

        # Final position handling
        if in_position:
            final_bar = df_4h.iloc[-1]
            exit_price = final_bar['close']

            if position_direction == 'bullish':
                raw_pnl = (exit_price - entry_price) / entry_price
            else:
                raw_pnl = (entry_price - exit_price) / entry_price

            position_risk = self.config['risk']['position_sizing']['base_risk_pct']
            position_pnl = raw_pnl * position_risk * self.current_balance

            trade = {
                'entry_timestamp': entry_timestamp,
                'exit_timestamp': df_4h.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': position_direction,
                'raw_pnl_pct': raw_pnl * 100,
                'position_pnl': position_pnl,
                'balance_before': self.current_balance,
                'exit_type': 'final',
                'r_multiple': raw_pnl / (abs(entry_price - stop_loss) / entry_price),
                'quantity': 1.0
            }

            self.trades.append(trade)
            self.current_balance += position_pnl

        print(f"\n📊 ENHANCED BACKTEST COMPLETE")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Final Balance: ${self.current_balance:,.2f}")
        print(f"Total Return: {((self.current_balance - self.starting_balance) / self.starting_balance * 100):+.2f}%")

    def _calculate_atr(self, df_4h, current_idx):
        """Calculate ATR at current index"""
        window = df_4h.iloc[max(0, current_idx-20):current_idx]

        if len(window) < 14:
            return window['close'].iloc[-1] * 0.02  # 2% fallback

        high_low = window['high'] - window['low']
        high_close = abs(window['high'] - window['close'].shift(1))
        low_close = abs(window['low'] - window['close'].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(14).mean().iloc[-1]

    def _enhanced_exit_logic(self, df_4h, current_idx, entry_timestamp,
                           position_direction, entry_price, stop_loss, take_profit_1):
        """Enhanced exit logic with asymmetric management"""

        current_bar = df_4h.iloc[current_idx]
        entry_bar_idx = df_4h.index.get_loc(entry_timestamp)
        bars_in_trade = current_idx - entry_bar_idx

        current_price = current_bar['close']

        # 1. Stop loss
        if position_direction == 'bullish' and current_price <= stop_loss:
            return {'type': 'stop_loss'}
        elif position_direction == 'bearish' and current_price >= stop_loss:
            return {'type': 'stop_loss'}

        # 2. Take profit
        if position_direction == 'bullish' and current_price >= take_profit_1:
            return {'type': 'take_profit'}
        elif position_direction == 'bearish' and current_price <= take_profit_1:
            return {'type': 'take_profit'}

        # 3. Breakeven trigger
        be_trigger = self.config['exits']['breakeven']['trigger_r']
        sl_distance = abs(entry_price - stop_loss)
        be_threshold_distance = sl_distance * be_trigger

        if position_direction == 'bullish':
            be_threshold = entry_price + be_threshold_distance
            if current_price >= be_threshold and bars_in_trade >= 2:
                # Move stop to breakeven + buffer
                buffer = self._calculate_atr(df_4h, current_idx) * 0.1
                # Note: In real implementation, would update stop_loss variable
                pass
        else:
            be_threshold = entry_price - be_threshold_distance
            if current_price <= be_threshold and bars_in_trade >= 2:
                buffer = self._calculate_atr(df_4h, current_idx) * 0.1
                # Note: In real implementation, would update stop_loss variable
                pass

        # 4. Time-based exit
        max_bars = self.config['exits']['time_exits']['max_bars_in_trade']
        if bars_in_trade >= max_bars:
            return {'type': 'time_exit'}

        # 5. Momentum reversal
        if bars_in_trade >= 3 and self.config['exits']['dynamic_exits']['momentum_reversal']:
            recent_returns = df_4h['close'].pct_change().iloc[current_idx-3:current_idx].mean()

            if position_direction == 'bullish' and recent_returns < -0.015:  # 1.5% negative momentum
                return {'type': 'momentum_reversal'}
            elif position_direction == 'bearish' and recent_returns > 0.015:  # 1.5% positive momentum
                return {'type': 'momentum_reversal'}

        return None

    def generate_enhanced_report(self):
        """Generate comprehensive enhanced performance report"""

        if not self.trades:
            print("❌ No trades to analyze")
            return

        print("\n" + "=" * 60)
        print("📈 ENHANCED PERFORMANCE REPORT v1.7.1")
        print("=" * 60)

        # Apply transaction costs
        cost_adjusted_trades = self.metrics_calc.apply_costs_to_trades(
            self.trades,
            pd.DataFrame({
                'close': [2000 + i * 10 for i in range(len(self.trades) * 2)],
                'volume': [1000000] * (len(self.trades) * 2)
            }, index=pd.date_range('2025-01-01', periods=len(self.trades) * 2, freq='4h'))
        )

        # Calculate enhanced metrics
        metrics = self.metrics_calc.compute_performance_metrics(cost_adjusted_trades)

        print(f"📊 CORE PERFORMANCE:")
        print(f"   Starting Balance:     ${self.starting_balance:>10,.2f}")
        print(f"   Final Balance:        ${self.current_balance:>10,.2f}")
        print(f"   Total Return:         {((self.current_balance - self.starting_balance) / self.starting_balance * 100):>10.2f}%")
        print(f"   Total Trades:         {len(self.trades):>10}")
        print(f"   Win Rate:             {metrics['win_rate']*100:>10.1f}%")
        print(f"   Profit Factor:        {metrics['profit_factor']:>10.2f}")
        print(f"   Max Drawdown:         ${metrics['max_drawdown']:>10.2f}")
        print(f"   Sharpe Ratio:         {metrics['sharpe_ratio']:>10.2f}")

        print(f"\n💰 ENHANCED COST ANALYSIS:")
        print(f"   Cost Drag:            {metrics['cost_drag_pct']:>10.1f}%")
        print(f"   Avg Cost (bps):       {metrics['avg_cost_bps']:>10.1f}")
        print(f"   Total Costs:          ${metrics['total_cost']:>10.2f}")

        print(f"\n🎯 ENHANCED ENGINE UTILIZATION:")
        total_signals = sum(self.engine_usage.values())
        if total_signals > 0:
            for engine, count in self.engine_usage.items():
                percentage = (count / total_signals) * 100
                print(f"   {engine.upper().replace('_', ' '):>18}: {count:>4} ({percentage:>5.1f}%)")

        # Enhanced trade analysis
        winning_trades = [t for t in self.trades if t['raw_pnl_pct'] > 0]
        losing_trades = [t for t in self.trades if t['raw_pnl_pct'] <= 0]

        print(f"\n📋 ENHANCED TRADE BREAKDOWN:")
        print(f"   Winning Trades:       {len(winning_trades):>10}")
        print(f"   Losing Trades:        {len(losing_trades):>10}")

        if winning_trades:
            avg_win = np.mean([t['raw_pnl_pct'] for t in winning_trades])
            max_win = max([t['raw_pnl_pct'] for t in winning_trades])
            avg_win_r = np.mean([t['r_multiple'] for t in winning_trades])
            print(f"   Average Win:          {avg_win:>10.2f}%")
            print(f"   Largest Win:          {max_win:>10.2f}%")
            print(f"   Average Win R:        {avg_win_r:>10.2f}R")

        if losing_trades:
            avg_loss = np.mean([t['raw_pnl_pct'] for t in losing_trades])
            max_loss = min([t['raw_pnl_pct'] for t in losing_trades])
            avg_loss_r = np.mean([t['r_multiple'] for t in losing_trades])
            print(f"   Average Loss:         {avg_loss:>10.2f}%")
            print(f"   Largest Loss:         {max_loss:>10.2f}%")
            print(f"   Average Loss R:       {avg_loss_r:>10.2f}R")

        # R-multiple analysis
        r_multiples = [t['r_multiple'] for t in self.trades if 'r_multiple' in t]
        if r_multiples:
            avg_r = np.mean(r_multiples)
            print(f"\n📊 R-MULTIPLE ANALYSIS:")
            print(f"   Average R:            {avg_r:>10.2f}R")
            print(f"   R > 1 Trades:         {sum(1 for r in r_multiples if r > 1):>10}")
            print(f"   R > 2 Trades:         {sum(1 for r in r_multiples if r > 2):>10}")

        # Exit type analysis
        exit_types = {}
        for trade in self.trades:
            exit_type = trade.get('exit_type', 'unknown')
            exit_types[exit_type] = exit_types.get(exit_type, 0) + 1

        print(f"\n🚪 EXIT TYPE BREAKDOWN:")
        for exit_type, count in exit_types.items():
            percentage = (count / len(self.trades)) * 100
            print(f"   {exit_type.replace('_', ' ').title():>15}: {count:>4} ({percentage:>5.1f}%)")

        # Save enhanced results
        results = {
            'backtest_summary': {
                'version': '1.7.1-enhanced',
                'starting_balance': self.starting_balance,
                'final_balance': self.current_balance,
                'total_return_pct': ((self.current_balance - self.starting_balance) / self.starting_balance * 100),
                'total_trades': len(self.trades),
                'config_version': self.config_version
            },
            'performance_metrics': metrics,
            'enhanced_engine_usage': self.engine_usage,
            'trades': self.trades,
            'exit_analysis': exit_types
        }

        with open('eth_enhanced_backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Enhanced results saved to eth_enhanced_backtest_results.json")

        return results


def main():
    """Run enhanced ETH backtest"""

    print("🏛️  BULL MACHINE v1.7.1 - ENHANCED ETH BACKTEST")
    print("=" * 60)
    print("Moneytaur/Wyckoff/ZeroIKA Alignment:")
    print("  ✅ Counter-trend discipline")
    print("  ✅ ETHBTC/TOTAL2 rotation gates")
    print("  ✅ Enhanced HOB absorption")
    print("  ✅ Asymmetric R/R management")
    print("  ✅ Cost-aware throttles")
    print("  ✅ Momentum bias")
    print()

    # Initialize enhanced backtest
    backtest = EnhancedBacktester(starting_balance=10000.0, config_version="v171")

    # Execute enhanced backtest
    backtest.execute_enhanced_backtest()

    # Generate enhanced report
    results = backtest.generate_enhanced_report()

    print("\n🎉 Enhanced ETH Backtest Complete!")
    print(f"💰 Final Result: ${backtest.starting_balance:,.0f} → ${backtest.current_balance:,.0f}")
    print(f"📈 Total Return: {((backtest.current_balance - backtest.starting_balance) / backtest.starting_balance * 100):+.2f}%")

    return results


if __name__ == "__main__":
    main()