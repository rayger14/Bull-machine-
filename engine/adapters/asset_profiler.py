#!/usr/bin/env python3
"""
Asset Profiler Module - Bull Machine v1.7.2
Automatically builds asset-specific adapters for any ticker
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import json
# import yaml  # Optional dependency
from datetime import datetime, timedelta
import hashlib

class AssetProfiler:
    """
    Builds asset-specific profiles and adapters for Bull Machine.
    Computes volatility, liquidity, and microstructure characteristics.
    """

    def __init__(self, symbol: str, exchange: str = "COINBASE"):
        self.symbol = symbol
        self.exchange = exchange
        self.profile = {}
        self.config = {}

    def build_profile(self, data: pd.DataFrame,
                     benchmark_data: Optional[pd.DataFrame] = None,
                     lookback_days: int = 180) -> Dict:
        """
        Build comprehensive asset profile from historical data.

        Args:
            data: OHLCV dataframe with datetime index
            benchmark_data: Optional benchmark (BTC/SPX) for correlation
            lookback_days: Days of history to analyze

        Returns:
            Complete asset profile dictionary
        """
        # Filter to lookback period
        cutoff = datetime.now() - timedelta(days=lookback_days)
        if data.index[0] < cutoff:
            data = data[data.index >= cutoff]

        print(f"📊 Building profile for {self.symbol}")
        print(f"   Data range: {data.index[0]} → {data.index[-1]}")
        print(f"   Bars: {len(data)}")

        # Core volatility metrics
        self.profile['volatility'] = self._compute_volatility_profile(data)

        # Volume and liquidity metrics
        self.profile['liquidity'] = self._compute_liquidity_profile(data)

        # Price action characteristics
        self.profile['price_action'] = self._compute_price_action_profile(data)

        # Session characteristics
        self.profile['sessions'] = self._compute_session_profile(data)

        # Correlation profile if benchmark provided
        if benchmark_data is not None:
            self.profile['correlations'] = self._compute_correlation_profile(
                data, benchmark_data
            )

        # Microstructure
        self.profile['microstructure'] = self._compute_microstructure_profile(data)

        # Meta information
        self.profile['meta'] = {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'data_start': data.index[0].isoformat(),
            'data_end': data.index[-1].isoformat(),
            'total_bars': len(data),
            'profile_generated': datetime.now().isoformat(),
            'profile_hash': self._compute_profile_hash(data)
        }

        return self.profile

    def _compute_volatility_profile(self, data: pd.DataFrame) -> Dict:
        """Compute volatility characteristics."""
        # Returns
        returns = data['Close'].pct_change().dropna()
        log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()

        # ATR (Average True Range)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        atr_14 = true_range.rolling(14).mean()
        atr_pct = (atr_14 / data['Close']) * 100  # ATR as % of price

        # Historical Volatility (HV)
        hv_20 = returns.rolling(20).std() * np.sqrt(252) * 100  # Annualized
        hv_60 = returns.rolling(60).std() * np.sqrt(252) * 100

        # Percentile distributions
        volatility_profile = {
            'atr': {
                'p10': float(np.percentile(atr_pct.dropna(), 10)),
                'p20': float(np.percentile(atr_pct.dropna(), 20)),
                'p50': float(np.percentile(atr_pct.dropna(), 50)),
                'p80': float(np.percentile(atr_pct.dropna(), 80)),
                'p90': float(np.percentile(atr_pct.dropna(), 90)),
                'current': float(atr_pct.iloc[-1])
            },
            'returns': {
                'daily_mean': float(returns.mean() * 100),
                'daily_std': float(returns.std() * 100),
                'skew': float(returns.skew()),
                'kurtosis': float(returns.kurtosis()),
                'max_daily_gain': float(returns.max() * 100),
                'max_daily_loss': float(returns.min() * 100)
            },
            'historical_vol': {
                'hv20_mean': float(hv_20.mean()),
                'hv20_current': float(hv_20.iloc[-1]) if len(hv_20) > 0 else 0,
                'hv60_mean': float(hv_60.mean()) if len(hv_60) > 0 else 0,
                'vol_of_vol': float(hv_20.std()) if len(hv_20) > 20 else 0
            },
            'regime': self._classify_volatility_regime(atr_pct)
        }

        return volatility_profile

    def _compute_liquidity_profile(self, data: pd.DataFrame) -> Dict:
        """Compute volume and liquidity characteristics."""
        volume = data['Volume']

        # Volume statistics
        volume_mean = volume.mean()
        volume_median = volume.median()

        # Volume z-scores
        volume_z = (volume - volume_mean) / volume.std()

        # Dollar volume (price * volume)
        dollar_volume = data['Close'] * volume

        liquidity_profile = {
            'volume': {
                'mean': float(volume_mean),
                'median': float(volume_median),
                'p25': float(np.percentile(volume, 25)),
                'p75': float(np.percentile(volume, 75)),
                'p90': float(np.percentile(volume, 90)),
                'ratio_75_25': float(np.percentile(volume, 75) / np.percentile(volume, 25))
            },
            'volume_z_scores': {
                'z50': float(np.percentile(volume_z[volume_z > 0], 50)),
                'z70': float(np.percentile(volume_z[volume_z > 0], 70)),
                'z85': float(np.percentile(volume_z[volume_z > 0], 85)),
                'z95': float(np.percentile(volume_z[volume_z > 0], 95))
            },
            'dollar_volume': {
                'daily_mean': float(dollar_volume.mean()),
                'daily_median': float(dollar_volume.median())
            },
            'spread_estimate': self._estimate_spread(data),
            'slippage_factor': self._estimate_slippage(data, volume)
        }

        return liquidity_profile

    def _compute_price_action_profile(self, data: pd.DataFrame) -> Dict:
        """Compute price action characteristics."""
        # Candlestick metrics
        body = abs(data['Close'] - data['Open'])
        upper_wick = data['High'] - pd.concat([data['Open'], data['Close']], axis=1).max(axis=1)
        lower_wick = pd.concat([data['Open'], data['Close']], axis=1).min(axis=1) - data['Low']
        full_range = data['High'] - data['Low']

        # Gap analysis
        gaps = (data['Open'] - data['Close'].shift()) / data['Close'].shift() * 100
        gap_threshold = 0.5  # 0.5% gap

        price_action_profile = {
            'candle_structure': {
                'avg_body_pct': float((body / full_range).mean() * 100),
                'avg_upper_wick_pct': float((upper_wick / full_range).mean() * 100),
                'avg_lower_wick_pct': float((lower_wick / full_range).mean() * 100),
                'doji_frequency': float(((body / full_range) < 0.1).sum() / len(data) * 100)
            },
            'gaps': {
                'frequency': float((abs(gaps) > gap_threshold).sum() / len(gaps) * 100),
                'avg_gap_size': float(abs(gaps[abs(gaps) > gap_threshold]).mean()),
                'max_gap': float(abs(gaps).max())
            },
            'trend_persistence': self._compute_trend_persistence(data),
            'mean_reversion': self._compute_mean_reversion_tendency(data)
        }

        return price_action_profile

    def _compute_session_profile(self, data: pd.DataFrame) -> Dict:
        """Compute trading session characteristics."""
        # Detect if 24/7 or has sessions
        if hasattr(data.index, 'hour'):
            hourly_volume = data.groupby(data.index.hour)['Volume'].mean()
            volume_variation = hourly_volume.std() / hourly_volume.mean()
            has_sessions = volume_variation > 0.5
        else:
            has_sessions = False
            volume_variation = 0

        session_profile = {
            'type': 'sessioned' if has_sessions else 'continuous',
            'volume_variation': float(volume_variation),
            'weekend_trading': self._check_weekend_trading(data),
            'typical_bar_count_per_day': self._estimate_bars_per_day(data)
        }

        return session_profile

    def _compute_correlation_profile(self, data: pd.DataFrame,
                                    benchmark: pd.DataFrame) -> Dict:
        """Compute correlation with benchmark assets."""
        # Align data
        aligned = pd.DataFrame({
            'asset': data['Close'].pct_change(),
            'benchmark': benchmark['Close'].pct_change()
        }).dropna()

        if len(aligned) < 20:
            return {'error': 'Insufficient aligned data'}

        # Rolling correlations
        corr_20 = aligned['asset'].rolling(20).corr(aligned['benchmark'])
        corr_60 = aligned['asset'].rolling(60).corr(aligned['benchmark'])

        correlation_profile = {
            'full_period': float(aligned['asset'].corr(aligned['benchmark'])),
            'rolling_20d': {
                'mean': float(corr_20.mean()),
                'std': float(corr_20.std()),
                'current': float(corr_20.iloc[-1]) if len(corr_20) > 0 else 0
            },
            'rolling_60d': {
                'mean': float(corr_60.mean()) if len(corr_60) > 0 else 0,
                'current': float(corr_60.iloc[-1]) if len(corr_60) > 0 else 0
            },
            'beta': float(aligned['asset'].cov(aligned['benchmark']) / aligned['benchmark'].var())
        }

        return correlation_profile

    def _compute_microstructure_profile(self, data: pd.DataFrame) -> Dict:
        """Compute market microstructure characteristics."""
        # Tick size estimation
        price_diffs = data['Close'].diff().dropna()
        tick_size = self._estimate_tick_size(price_diffs)

        # Typical spread as % of price
        typical_spread_pct = self._estimate_spread(data)

        microstructure_profile = {
            'tick_size': float(tick_size),
            'tick_size_pct': float(tick_size / data['Close'].mean() * 100),
            'typical_spread_bps': float(typical_spread_pct * 100),  # basis points
            'round_number_attraction': self._compute_round_number_attraction(data),
            'typical_trade_size': self._estimate_typical_trade_size(data)
        }

        return microstructure_profile

    def generate_config(self) -> Dict:
        """
        Generate Bull Machine configuration from asset profile.
        Maps profile metrics to trading parameters.
        """
        if not self.profile:
            raise ValueError("Profile must be built before generating config")

        vol = self.profile['volatility']
        liq = self.profile['liquidity']
        pa = self.profile['price_action']

        # Adaptive scaling factors
        volatility_scalar = vol['atr']['p50'] / vol['atr']['p20']  # How much vol varies
        liquidity_scalar = liq['volume']['ratio_75_25']  # Volume consistency

        config = {
            'wyckoff': {
                'spring_wick_min_atr': round(max(1.0, 1.2 * volatility_scalar), 2),
                'utad_wick_min_atr': round(max(1.0, 1.1 * volatility_scalar), 2),
                'volume_spike_z': round(liq['volume_z_scores']['z70'], 2),
                'volume_climax_z': round(liq['volume_z_scores']['z85'], 2),
                'range_contraction_pct': round(vol['atr']['p30'] if 'p30' in vol['atr'] else vol['atr']['p20'], 2),
                'test_volume_max_z': round(liq['volume_z_scores']['z50'], 2)
            },
            'smc': {
                'min_displacement_atr': round(0.8 * vol['atr']['p50'], 2),
                'ob_volume_min_z': round(liq['volume_z_scores']['z70'], 2),
                'fvg_min_gap_pct': round(max(0.3, pa['gaps']['avg_gap_size'] * 0.5), 2),
                'bos_strength_min': round(1.2 * vol['atr']['p50'], 2),
                'choch_strength_min': round(1.5 * vol['atr']['p50'], 2)
            },
            'liquidity': {
                'hob_volume_z_min_long': round(liq['volume_z_scores']['z70'], 2),
                'hob_volume_z_min_short': round(liq['volume_z_scores']['z85'], 2),
                'absorption_time_max': self._scale_by_sessions(20),
                'sweep_wick_min_atr': round(1.0 * vol['atr']['p50'], 2),
                'magnet_distance_max_atr': round(2.0 * vol['atr']['p50'], 2)
            },
            'momentum': {
                'rsi_oversold': 30 if vol['regime'] != 'high' else 25,
                'rsi_overbought': 70 if vol['regime'] != 'high' else 75,
                'macd_signal_min_atr': round(0.5 * vol['atr']['p20'], 2),
                'momentum_lookback': self._scale_by_volatility(14)
            },
            'risk': {
                'atr_multiplier_stop': round(1.5 if vol['regime'] == 'low' else 2.0 if vol['regime'] == 'medium' else 2.5, 1),
                'atr_multiplier_target': round(2.5 if vol['regime'] == 'low' else 3.0 if vol['regime'] == 'medium' else 4.0, 1),
                'position_size_scalar': round(1.0 if vol['regime'] == 'low' else 0.75 if vol['regime'] == 'medium' else 0.5, 2),
                'max_spread_bps': round(liq['spread_estimate'] * 2, 1),
                'slippage_bps': round(liq['slippage_factor'] * 100, 1),
                'fee_bps': 10 if self.exchange == "COINBASE" else 5  # Exchange-specific
            },
            'filters': {
                'min_atr_percentile': 20,  # Don't trade in dead volatility
                'max_atr_percentile': 90,  # Don't trade in extreme volatility
                'min_volume_z': -0.5,  # Don't trade in low volume
                'correlation_veto_threshold': 0.8,  # Veto if too correlated to benchmark
                'trend_alignment_required': bool(pa['trend_persistence'] > 0.5)
            },
            'meta': {
                'generated_from_profile': self.profile['meta']['profile_hash'],
                'config_version': '1.7.2',
                'config_generated': datetime.now().isoformat()
            }
        }

        self.config = config
        return config

    def save_profile(self, output_dir: str = 'profiles') -> str:
        """Save profile to YAML file."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{output_dir}/{self.exchange}_{self.symbol}_profile.json"
        with open(filename, 'w') as f:
            json.dump(self.profile, f, indent=2)

        print(f"✅ Profile saved to {filename}")
        return filename

    def save_config(self, output_dir: str = 'configs/adaptive') -> str:
        """Save configuration to JSON file."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{output_dir}/{self.exchange}_{self.symbol}_config.json"
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"✅ Config saved to {filename}")
        return filename

    # Helper methods
    def _classify_volatility_regime(self, atr_pct: pd.Series) -> str:
        """Classify current volatility regime."""
        current = atr_pct.iloc[-1]
        p33 = np.percentile(atr_pct, 33)
        p67 = np.percentile(atr_pct, 67)

        if current < p33:
            return 'low'
        elif current < p67:
            return 'medium'
        else:
            return 'high'

    def _estimate_spread(self, data: pd.DataFrame) -> float:
        """Estimate typical spread as percentage."""
        # Use High-Low as proxy for spread
        hl_pct = ((data['High'] - data['Low']) / data['Close'] * 100).median()
        # Spread is typically 10-20% of high-low range
        return hl_pct * 0.15

    def _estimate_slippage(self, data: pd.DataFrame, volume: pd.Series) -> float:
        """Estimate slippage factor based on liquidity."""
        # Higher volume variation = higher slippage
        volume_cv = volume.std() / volume.mean()
        base_slippage = 0.1  # 10 bps base
        return base_slippage * (1 + volume_cv)

    def _compute_trend_persistence(self, data: pd.DataFrame) -> float:
        """Measure how well trends persist."""
        returns = data['Close'].pct_change()
        # Autocorrelation of returns
        autocorr = returns.autocorr(lag=1)
        # Positive autocorr = trending, negative = mean reverting
        return (autocorr + 1) / 2  # Scale to 0-1

    def _compute_mean_reversion_tendency(self, data: pd.DataFrame) -> float:
        """Measure mean reversion tendency."""
        sma20 = data['Close'].rolling(20).mean()
        deviations = (data['Close'] - sma20) / sma20
        # Count reversions
        reversions = (deviations * deviations.shift() < 0).sum()
        return reversions / len(deviations)

    def _estimate_tick_size(self, price_diffs: pd.Series) -> float:
        """Estimate minimum tick size from price differences."""
        # Get non-zero diffs
        non_zero = price_diffs[price_diffs != 0].abs()
        if len(non_zero) == 0:
            return 0.01
        # Find GCD-like pattern in price movements
        return non_zero.min()

    def _compute_round_number_attraction(self, data: pd.DataFrame) -> float:
        """Measure tendency to cluster around round numbers."""
        closes = data['Close']
        # Check proximity to round numbers (whole numbers)
        proximity = closes % 1  # Decimal part
        # More clustering = lower average proximity
        return 1 - proximity.mean()

    def _estimate_typical_trade_size(self, data: pd.DataFrame) -> float:
        """Estimate typical trade size from volume patterns."""
        # Use median volume as proxy
        return float(data['Volume'].median())

    def _check_weekend_trading(self, data: pd.DataFrame) -> bool:
        """Check if asset trades on weekends."""
        if hasattr(data.index, 'dayofweek'):
            weekend_bars = data[data.index.dayofweek.isin([5, 6])]
            return len(weekend_bars) > 0
        return True  # Assume 24/7 if can't determine

    def _estimate_bars_per_day(self, data: pd.DataFrame) -> int:
        """Estimate typical number of bars per day."""
        if len(data) < 2:
            return 24
        # Calculate based on timestamp frequency
        freq_hours = (data.index[1] - data.index[0]).total_seconds() / 3600
        return int(24 / freq_hours)

    def _scale_by_sessions(self, base_value: int) -> int:
        """Scale a parameter based on session type."""
        if self.profile.get('sessions', {}).get('type') == 'continuous':
            return base_value
        else:
            # Reduce for sessioned markets
            return int(base_value * 0.7)

    def _scale_by_volatility(self, base_value: int) -> int:
        """Scale a parameter based on volatility regime."""
        regime = self.profile.get('volatility', {}).get('regime', 'medium')
        if regime == 'high':
            return int(base_value * 0.7)
        elif regime == 'low':
            return int(base_value * 1.3)
        return base_value

    def _compute_profile_hash(self, data: pd.DataFrame) -> str:
        """Generate deterministic hash for profile reproducibility."""
        hash_input = f"{self.symbol}_{self.exchange}_{len(data)}_{data.index[0]}_{data.index[-1]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

def test_profiler():
    """Test the asset profiler with sample data."""
    print("🧪 Testing Asset Profiler")
    print("=" * 60)

    # Load sample data
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from data.real_data_loader import RealDataLoader
    loader = RealDataLoader()

    # Test with ETH
    eth_data = loader.load_eth_data('4H')
    if eth_data is not None and len(eth_data) > 100:
        profiler = AssetProfiler('ETHUSD', 'COINBASE')
        profile = profiler.build_profile(eth_data)
        config = profiler.generate_config()

        print("\n📊 ETH Profile Summary:")
        print(f"  ATR p50: {profile['volatility']['atr']['p50']:.2f}%")
        print(f"  Volume Z70: {profile['liquidity']['volume_z_scores']['z70']:.2f}")
        print(f"  Volatility Regime: {profile['volatility']['regime']}")

        print("\n⚙️ Generated Config Highlights:")
        print(f"  Spring Wick Min: {config['wyckoff']['spring_wick_min_atr']}x ATR")
        print(f"  Volume Spike Z: {config['wyckoff']['volume_spike_z']}")
        print(f"  Risk Scalar: {config['risk']['position_size_scalar']}")

        profiler.save_profile()
        profiler.save_config()

    else:
        print("❌ Insufficient ETH data for testing")

if __name__ == "__main__":
    test_profiler()