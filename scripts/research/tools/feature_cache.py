#!/usr/bin/env python3
"""
Bull Machine Feature Cache System
Precompute and cache expensive indicators to disk
Eliminates redundant calculations during grid searches
"""

import pandas as pd
import numpy as np
import json
import os
import hashlib
import time
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

class FeatureCache:
    """Persistent cache for precomputed technical indicators and features"""

    def __init__(self, cache_dir: str = "cache/features"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache metadata
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _get_cache_key(self, asset: str, timeframe: str, data_hash: str) -> str:
        """Generate unique cache key for asset/timeframe/data combination"""
        key_string = f"{asset}_{timeframe}_{data_hash}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """Generate hash of dataframe for change detection"""
        # Use first/last timestamps and length for quick hash
        if len(df) == 0:
            return "empty"

        hash_data = f"{df.index[0]}_{df.index[-1]}_{len(df)}"
        return hashlib.md5(hash_data.encode()).hexdigest()[:12]

    def _compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators for caching"""
        if len(df) < 50:  # Not enough data for meaningful indicators
            return df.copy()

        # Make a copy to avoid modifying original
        data = df.copy()

        # Basic price features
        data['hl2'] = (data['high'] + data['low']) / 2
        data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3
        data['ohlc4'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4

        # ATR family
        data['tr'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                np.abs(data['high'] - data['close'].shift(1)),
                np.abs(data['low'] - data['close'].shift(1))
            )
        )
        data['atr_14'] = data['tr'].rolling(14).mean()
        data['atr_21'] = data['tr'].rolling(21).mean()

        # RSI family
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain_14 = gain.rolling(14).mean()
        avg_loss_14 = loss.rolling(14).mean()
        rs_14 = avg_gain_14 / avg_loss_14
        data['rsi_14'] = 100 - (100 / (1 + rs_14))

        avg_gain_21 = gain.rolling(21).mean()
        avg_loss_21 = loss.rolling(21).mean()
        rs_21 = avg_gain_21 / avg_loss_21
        data['rsi_21'] = 100 - (100 / (1 + rs_21))

        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            if len(data) > window:
                data[f'sma_{window}'] = data['close'].rolling(window).mean()
                data[f'ema_{window}'] = data['close'].ewm(span=window).mean()

        # Volume indicators (if volume available)
        if 'volume' in data.columns:
            data['vwap'] = (data['volume'] * data['hlc3']).cumsum() / data['volume'].cumsum()
            data['volume_sma_20'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma_20']

        # Momentum indicators
        data['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        data['momentum_20'] = data['close'] / data['close'].shift(20) - 1

        # Bollinger Bands
        bb_window = 20
        if len(data) > bb_window:
            bb_sma = data['close'].rolling(bb_window).mean()
            bb_std = data['close'].rolling(bb_window).std()
            data['bb_upper'] = bb_sma + (bb_std * 2)
            data['bb_lower'] = bb_sma - (bb_std * 2)
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / bb_sma
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])

        # Support/Resistance levels
        window = 20
        data['local_high'] = data['high'].rolling(window, center=True).max()
        data['local_low'] = data['low'].rolling(window, center=True).min()

        # Price position relative to recent range
        data['range_position'] = (data['close'] - data['local_low']) / (data['local_high'] - data['local_low'])

        return data

    def _compute_confluence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Bull Machine specific confluence features"""
        data = df.copy()

        if len(data) < 100:
            return data

        # Wyckoff-style accumulation/distribution
        # Simplified version - full implementation would be more complex
        data['ad_line'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        data['ad_line'] = data['ad_line'].fillna(0).cumsum()

        # Momentum confluence
        # Multiple timeframe momentum alignment
        data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        data['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        data['momentum_20'] = data['close'] / data['close'].shift(20) - 1

        # Momentum alignment score
        momentum_signals = ['momentum_5', 'momentum_10', 'momentum_20']
        data['momentum_alignment'] = 0
        for signal in momentum_signals:
            if signal in data.columns:
                data['momentum_alignment'] += (data[signal] > 0).astype(int)

        # Liquidity approximation
        # High volume + tight spread suggests liquidity
        if 'volume' in data.columns:
            data['spread'] = (data['high'] - data['low']) / data['close']
            data['volume_ma'] = data['volume'].rolling(20).mean()
            data['liquidity_score'] = (data['volume'] / data['volume_ma']) / (1 + data['spread'])
        else:
            data['liquidity_score'] = 1.0

        # Temporal features
        data['hour'] = data.index.hour if hasattr(data.index, 'hour') else 0
        data['day_of_week'] = data.index.dayofweek if hasattr(data.index, 'dayofweek') else 0
        data['month'] = data.index.month if hasattr(data.index, 'month') else 1

        # Market structure breaks
        # Simplified version
        lookback = 10
        data['swing_high'] = data['high'].rolling(lookback*2+1, center=True).max() == data['high']
        data['swing_low'] = data['low'].rolling(lookback*2+1, center=True).min() == data['low']

        return data

    def get_cached_features(self, asset: str, timeframe: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get cached features if available and valid"""
        data_hash = self._get_data_hash(df)
        cache_key = self._get_cache_key(asset, timeframe, data_hash)

        cache_file = self.cache_dir / f"{cache_key}.parquet"

        if cache_file.exists():
            try:
                # Check metadata
                if cache_key in self.metadata:
                    meta = self.metadata[cache_key]
                    if meta['data_hash'] == data_hash:
                        # Load cached features
                        cached_df = pd.read_parquet(cache_file)
                        print(f"✅ Loaded cached features: {asset} {timeframe} ({len(cached_df)} bars)")
                        return cached_df

            except Exception as e:
                print(f"❌ Error loading cache for {asset} {timeframe}: {e}")

        return None

    def cache_features(self, asset: str, timeframe: str, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and cache features for given dataframe"""
        print(f"🔄 Computing features for {asset} {timeframe} ({len(df)} bars)...")

        start_time = time.time()

        # Compute all indicators
        features_df = self._compute_technical_indicators(df)
        features_df = self._compute_confluence_features(features_df)

        # Generate cache key
        data_hash = self._get_data_hash(df)
        cache_key = self._get_cache_key(asset, timeframe, data_hash)

        # Save to cache
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        try:
            features_df.to_parquet(cache_file)

            # Update metadata
            self.metadata[cache_key] = {
                'asset': asset,
                'timeframe': timeframe,
                'data_hash': data_hash,
                'num_bars': len(features_df),
                'num_features': len(features_df.columns),
                'cached_at': time.time(),
                'compute_time_s': round(time.time() - start_time, 2)
            }
            self._save_metadata()

            print(f"✅ Cached {len(features_df.columns)} features in {time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"❌ Error saving cache: {e}")

        return features_df

    def get_features(self, asset: str, timeframe: str, df: pd.DataFrame) -> pd.DataFrame:
        """Get features - from cache if available, otherwise compute and cache"""
        # Try to get from cache first
        cached = self.get_cached_features(asset, timeframe, df)
        if cached is not None:
            return cached

        # Compute and cache
        return self.cache_features(asset, timeframe, df)

    def clear_cache(self, asset: str = None, timeframe: str = None):
        """Clear cache (optionally filtered by asset/timeframe)"""
        cleared = 0

        for cache_key, meta in list(self.metadata.items()):
            should_clear = True

            if asset and meta.get('asset') != asset:
                should_clear = False
            if timeframe and meta.get('timeframe') != timeframe:
                should_clear = False

            if should_clear:
                cache_file = self.cache_dir / f"{cache_key}.parquet"
                if cache_file.exists():
                    cache_file.unlink()
                del self.metadata[cache_key]
                cleared += 1

        self._save_metadata()
        print(f"🗑️ Cleared {cleared} cache entries")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_files = len(list(self.cache_dir.glob("*.parquet")))
        total_size_mb = sum(f.stat().st_size for f in self.cache_dir.glob("*.parquet")) / 1024**2

        assets = set(meta.get('asset', 'unknown') for meta in self.metadata.values())
        timeframes = set(meta.get('timeframe', 'unknown') for meta in self.metadata.values())

        return {
            'total_entries': len(self.metadata),
            'total_files': total_files,
            'total_size_mb': round(total_size_mb, 1),
            'assets': sorted(assets),
            'timeframes': sorted(timeframes),
            'cache_dir': str(self.cache_dir)
        }

# Convenience function for backtesting integration
def get_cached_data(asset: str, timeframe: str, df: pd.DataFrame, cache_dir: str = "cache/features") -> pd.DataFrame:
    """Get cached features for backtesting - simple interface"""
    cache = FeatureCache(cache_dir)
    return cache.get_features(asset, timeframe, df)

# Example usage and testing
if __name__ == "__main__":
    print("Testing feature cache system...")

    # Create dummy data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    dummy_data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

    # Test cache
    cache = FeatureCache("test_cache")

    # First run - should compute
    features1 = cache.get_features("TEST", "1D", dummy_data)
    print(f"First run: {len(features1.columns)} features")

    # Second run - should load from cache
    features2 = cache.get_features("TEST", "1D", dummy_data)
    print(f"Second run: {len(features2.columns)} features")

    # Check stats
    stats = cache.get_cache_stats()
    print(f"Cache stats: {stats}")

    # Clean up
    cache.clear_cache()
    print("Cache cleared")