#!/usr/bin/env python3
"""
Batch Candidate Screener - Bull Machine v1.8

Vectorized, no-lookahead pre-filter for expensive domain engine execution.
Runs in seconds, outputs candidate timestamps for focused replay.

Usage:
    python3 bin/research/batch_screener.py --asset ETH --start 2025-06-15 --end 2025-09-30 \\
        --config configs/v18/ETH_comprehensive.json --output results/candidates.jsonl
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.io.tradingview_loader import load_tv


def compute_vectorized_features(df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame,
                                  macro_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features vectorized with no lookahead.

    Returns DataFrame with columns:
        - timestamp
        - close_1h, sma20_1h, sma50_1h, rsi_1h, adx_4h
        - macro_vix, macro_regime
        - sentinels: sma20_cross, sma50_cross, atr_regime_change, session_break
        - proto_fusion_score
    """
    print("📊 Computing vectorized features...")

    # Standardize column names
    for df in [df_1h, df_4h, df_1d]:
        df.columns = [c.lower() for c in df.columns]

    # Ensure datetime index
    if not isinstance(df_1h.index, pd.DatetimeIndex):
        df_1h.set_index('timestamp', inplace=True)
    if not isinstance(df_4h.index, pd.DatetimeIndex):
        df_4h.set_index('timestamp', inplace=True)
    if not isinstance(df_1d.index, pd.DatetimeIndex):
        df_1d.set_index('timestamp', inplace=True)

    # Create feature frame (1H resolution)
    features = pd.DataFrame(index=df_1h.index)
    features['close'] = df_1h['close']
    features['high'] = df_1h['high']
    features['low'] = df_1h['low']
    features['volume'] = df_1h['volume']

    # === 1H FEATURES (No Lookahead) ===

    # SMAs (shifted to avoid lookahead)
    features['sma20'] = df_1h['close'].rolling(20, min_periods=20).mean().shift(1)
    features['sma50'] = df_1h['close'].rolling(50, min_periods=50).mean().shift(1)

    # RSI (shifted)
    delta = df_1h['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = (100 - (100 / (1 + rs))).shift(1)

    # ATR (shifted)
    high = df_1h['high']
    low = df_1h['low']
    close = df_1h['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    features['atr'] = tr.rolling(14).mean().shift(1)

    # === 4H FEATURES (No Lookahead) ===

    # ADX on 4H (more stable)
    def calc_adx_vectorized(df, period=14):
        """Vectorized ADX calculation (no lookahead)."""
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()

        return adx.shift(1)  # Shift to avoid lookahead

    adx_4h = calc_adx_vectorized(df_4h, 14)
    # Resample to 1H (forward fill)
    features['adx'] = adx_4h.reindex(df_1h.index, method='ffill')

    # === 1D FEATURES (Trend Context) ===

    # Daily trend (shifted)
    ma20_1d = df_1d['close'].rolling(20).mean().shift(1)
    ma50_1d = df_1d['close'].rolling(50).mean().shift(1)
    features['trend_1d'] = (ma20_1d > ma50_1d).astype(int).reindex(df_1h.index, method='ffill')

    # === MACRO FEATURES (Daily, shifted to date boundary) ===

    if macro_data is not None and len(macro_data) > 0:
        # Join macro data by date (forward fill, shift to avoid lookahead)
        macro_daily = macro_data.copy()
        if 'timestamp' in macro_daily.columns:
            macro_daily['date'] = pd.to_datetime(macro_daily['timestamp']).dt.date
        features['date'] = features.index.date

        # VIX regime
        if 'vix' in macro_daily.columns:
            vix_regime = macro_daily.set_index('date')['vix']
            features['vix'] = features['date'].map(vix_regime).shift(1)  # Shift to prior day
        else:
            features['vix'] = 20.0  # Neutral default

        features.drop('date', axis=1, inplace=True)
    else:
        features['vix'] = 20.0

    # === SENTINEL DETECTION (No Lookahead) ===

    # SMA crossovers (compare current vs prior shifted values)
    features['sma20_cross'] = (
        (features['close'] > features['sma20']) &
        (features['close'].shift(1) <= features['sma20'].shift(1))
    ) | (
        (features['close'] < features['sma20']) &
        (features['close'].shift(1) >= features['sma20'].shift(1))
    )

    features['sma50_cross'] = (
        (features['close'] > features['sma50']) &
        (features['close'].shift(1) <= features['sma50'].shift(1))
    ) | (
        (features['close'] < features['sma50']) &
        (features['close'].shift(1) >= features['sma50'].shift(1))
    )

    # ATR regime change (percentile shift)
    atr_pct = features['atr'].rolling(100).apply(
        lambda x: (x.iloc[-1] > np.percentile(x[:-1], 75)) if len(x) > 1 else False
    )
    features['atr_regime_change'] = (atr_pct != atr_pct.shift(1)) & atr_pct.notna()

    # Volume spike detection (NEW)
    vol_ma = features['volume'].rolling(20).mean().shift(1)
    vol_std = features['volume'].rolling(20).std().shift(1)
    features['volume_spike'] = (
        (features['volume'] > vol_ma + 2 * vol_std) &
        (vol_std > 0)
    )

    # Session breaks (detect hourly gaps - simplified)
    hour_gap = features.index.to_series().diff() > pd.Timedelta(hours=2)
    features['session_break'] = hour_gap

    # Any sentinel fired (including volume spike)
    features['sentinel_fired'] = (
        features['sma20_cross'] |
        features['sma50_cross'] |
        features['atr_regime_change'] |
        features['volume_spike'] |
        features['session_break']
    )

    # === PROTO-FUSION SCORE (Simplified, No Lookahead) ===

    # Normalize features to 0-1 range for scoring
    features['rsi_norm'] = (features['rsi'] - 30) / 40  # 30-70 range
    features['adx_norm'] = features['adx'] / 50  # 0-50 range
    features['trend_align'] = (
        ((features['close'] > features['sma20']) & (features['trend_1d'] == 1)) |
        ((features['close'] < features['sma20']) & (features['trend_1d'] == 0))
    ).astype(float)
    features['vix_regime'] = (features['vix'] < 25).astype(float)  # Calm market

    # Weighted score (simple fusion proxy)
    features['proto_fusion_score'] = (
        0.3 * features['adx_norm'].clip(0, 1) +
        0.2 * features['rsi_norm'].clip(0, 1) +
        0.3 * features['trend_align'] +
        0.2 * features['vix_regime']
    )

    # Determine side hint
    features['side'] = 'neutral'
    features.loc[
        (features['close'] > features['sma20']) & (features['rsi'] < 70) & (features['adx'] > 20),
        'side'
    ] = 'long'
    features.loc[
        (features['close'] < features['sma20']) & (features['rsi'] > 30) & (features['adx'] > 20),
        'side'
    ] = 'short'

    print(f"✅ Computed features for {len(features)} bars")

    return features


def generate_candidates(features: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Filter features to generate candidate list.

    Returns DataFrame with columns:
        - timestamp
        - side
        - score
        - reason
    """
    print("\n🔍 Filtering candidates...")

    # Get thresholds from config
    batch_config = config.get('batch_mode', {})
    min_score = batch_config.get('min_fusion_score', 0.3)
    min_adx = batch_config.get('min_adx', 20)

    # Filter criteria
    candidates = features[
        (features['sentinel_fired'] == True) &
        (features['proto_fusion_score'] >= min_score) &
        (features['adx'] >= min_adx) &
        (features['side'] != 'neutral')
    ].copy()

    # Build reason string
    def build_reason(row):
        reasons = []
        if row['sma20_cross']:
            reasons.append('sma20_cross')
        if row['sma50_cross']:
            reasons.append('sma50_cross')
        if row['atr_regime_change']:
            reasons.append('atr_regime')
        if row['volume_spike']:
            reasons.append('volume_spike')
        if row['session_break']:
            reasons.append('session_break')
        return ','.join(reasons) if reasons else 'sentinel'

    candidates['reason'] = candidates.apply(build_reason, axis=1)
    candidates['score'] = candidates['proto_fusion_score']

    # Select output columns
    output = candidates[['side', 'score', 'reason']].copy()
    output['timestamp'] = candidates.index

    print(f"✅ Generated {len(output)} candidates ({len(output)/len(features)*100:.1f}% of bars)")

    # Stats
    print(f"\n📈 Candidate Statistics:")
    print(f"   Long signals:  {(output['side'] == 'long').sum()}")
    print(f"   Short signals: {(output['side'] == 'short').sum()}")
    print(f"   Avg score:     {output['score'].mean():.3f}")
    print(f"   Score range:   [{output['score'].min():.3f}, {output['score'].max():.3f}]")

    return output


def main():
    parser = argparse.ArgumentParser(description='Batch candidate screener for Bull Machine v1.8')
    parser.add_argument('--asset', required=True, help='Asset symbol (BTC, ETH, SOL)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', required=True, help='Path to config JSON')
    parser.add_argument('--output', default='results/candidates.jsonl', help='Output JSONL path')

    args = parser.parse_args()

    print("="*70)
    print("🚀 Bull Machine v1.8 - Batch Candidate Screener")
    print("="*70)
    print(f"Asset:  {args.asset}")
    print(f"Period: {args.start} → {args.end}")
    print(f"Config: {args.config}")
    print("="*70)

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Load OHLCV data using TradingView loader
    print("\n📊 Loading OHLCV data...")
    df_1h_full = load_tv(f'{args.asset}_1H')
    df_4h_full = load_tv(f'{args.asset}_4H')
    df_1d_full = load_tv(f'{args.asset}_1D')

    # Standardize column names (load_tv returns lowercase)
    for df in [df_1h_full, df_4h_full, df_1d_full]:
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                          'close': 'Close', 'volume': 'Volume'}, inplace=True)

    # Filter to date range
    if args.start:
        df_1h_full = df_1h_full[df_1h_full.index >= args.start]
        df_4h_full = df_4h_full[df_4h_full.index >= args.start]
        df_1d_full = df_1d_full[df_1d_full.index >= args.start]

    if args.end:
        df_1h_full = df_1h_full[df_1h_full.index <= args.end]
        df_4h_full = df_4h_full[df_4h_full.index <= args.end]
        df_1d_full = df_1d_full[df_1d_full.index <= args.end]

    print(f"   1H bars: {len(df_1h_full)}")
    print(f"   4H bars: {len(df_4h_full)}")
    print(f"   1D bars: {len(df_1d_full)}")

    # Load macro data
    print("\n📊 Loading macro data...")
    try:
        macro_data = pd.read_parquet('data/macro/combined_macro.parquet')
        print(f"   Loaded {len(macro_data)} macro observations")
    except Exception as e:
        print(f"   ⚠️  Could not load macro data: {e}")
        macro_data = None

    # Compute features
    features = compute_vectorized_features(df_1h_full, df_4h_full, df_1d_full, macro_data)

    # Generate candidates
    candidates = generate_candidates(features, config)

    # Save to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for _, row in candidates.iterrows():
            entry = {
                'timestamp': row['timestamp'].isoformat(),
                'side': row['side'],
                'score': float(row['score']),
                'reason': row['reason']
            }
            f.write(json.dumps(entry) + '\n')

    print(f"\n✅ Saved {len(candidates)} candidates to {output_path}")
    print("="*70)


if __name__ == '__main__':
    main()
